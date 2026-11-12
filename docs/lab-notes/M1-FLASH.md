# M1-FLASH: Qwen3.8-Flash-Next real forward pass, exact vs production, with streaming experts

Branch `flash-m1` (worktree `<engine checkout>`). Design:
the Flash engine design in the private lab notes, §5.2-5.3; M0 timing: `M0-FLASH.md`;
working log: `PROGRESS-M1.md`.

Goal (owner): replace M0's replayed routing and dummy activations with the real qwen4exp forward pass,
greedy output token-identical to production BigMoeOnEdge fixed353 (answer SHA `460169e2...0a4093`),
keeping M0's streaming machinery (R4 sidecar, CLOCK expert cache, io_uring IO thread).

## Results (2026-11-10, Claude's runs, 4 decode threads / 6 prefill threads as production)

| run | what | result |
|---|---|---|
| `flash-m1 ref` (graph port, mapped experts) | full fixed353 | **353/353 generated tokens identical to production**, then EOG 248046 (prefill 161 s, 2.37 s/token: page-faulted experts). `results/m1/full-t4.json` |
| `flash-m1 stream`, 16 GiB cache (graph port + streaming expert op) | full fixed353 | **353/353 tokens identical** (`tokens_identical: true`). `results/m1/stream-20261110-074800.json` |
| same run, check steps 1-3 | streaming vs mapped-expert context on a copy of the state | logits **bitwise identical** in all 3 steps (248,320 floats each) |
| same run, dumps vs production oracle | stream dumps of decode steps 1, 2 vs oracle u2, u3 | **110/110 common tensors byte-identical per step** (ffn_moe_up/gate/swiglu/down of layers 0, 1, 3; ffn_moe_out-0..47; l_last-0..47) |
| same run, snapshot test | save recurrent state at pos 159, decode 2, restore, decode again | logits bitwise identical; snapshot 118 MB |

Answer SHA: token identity of all 353 answer tokens with the oracle run (which itself reproduced
`460169e2...0a4093`) implies the same answer text, hence the same SHA; the engine does not detokenize.

**Streaming run numbers** (4 threads, ggml's OpenMP pool, graph port per token, IO thread unpinned,
spinning): decode **196 ms/token mean, 193 median, p90 237** (349 streaming steps; = 5.1 tok/s);
expert SSD wait seen by thread 0 **16.3 ms/token**; cache hit rate 84.5% (cold cache at the start of
decode: it is not warmed with the prefill's experts here, M0 warmed it), **144 MiB read/token**, IO
latency mean 2.3 ms / max 24.9 ms, max 10 in flight; VmHWM 23.8 GiB (includes the 593 MiB reference
context and the prefill's compute buffer), cgroup peak 25.5 GB incl. page cache.
This is the exactness path, not the speed path: M0's static 8-thread plan did 65 ms compute-only and
115-119 ms demand-streaming on the same cache/IO. The 196 ms here is ggml graph execution at production's
4 threads (production BigMoeOnEdge: 246.5 ms/token), i.e. already ~1.26x production with the same bits.

## Findings

1. ggml-cpu calls a custom op on **every** graph thread whatever its `n_tasks` (n_tasks only sizes the
   plan). The first streaming run did the cache lookups on 4 threads at once (1641 lookups for one
   token, garbage outputs). The fetch callback now acts on `ith == 0` only; the node barrier orders the rest.
2. In a process with two contexts, the **mapped-expert reference context's** dumps of expert nodes come
   out stale (its ffn_moe_gate-L does not reproduce its own ffn_moe_swiglu-L; gate-1/2/5 share values)
   although its logits are bitwise equal to the streaming context's. Single-context `flash-m1 ref` dumps
   matched the oracle 375/375 (session 1). Cause not found (graph allocator reuse is the suspect). The
   check therefore compares the streaming context's dumps with the **production oracle** and only the
   logits with the reference context (code changed after the run; not yet rebuilt, see gaps).
3. Production's flash-attention decode uses a split-KV path for n_kv >= 512 with chunk = ceil(n_kv / n_threads)
   (`ggml-cpu/ops.cpp:9501`), so **decode bits depend on the thread count after ~100 generated tokens**
   (n_cells > 256 => n_kv 512). The SHA gate is defined at production's 4 decode threads; 6/8-thread
   runs are expected to diverge from the production answer after that point. Engine rule for MTP: one
   fixed chunking (e.g. 256-cell blocks merged in order) for decode and verify, behind a flag.

## Gaps (what M1 does not have yet)

- Thread counts 6/8 not run (lease schedule). Expectation per finding 3: identical up to n_kv 512, then
  possibly different; worth one run each to confirm (and the fixed-chunk rule makes it moot later).
- Prefill is the graph with **mapped** experts (page faults through the shards: 161-283 s for 156 tokens
  cold). A streamed batched prefill (per layer, read the ubatch's distinct experts into a staging region at
  QD 32, then the same exact row-wise op) is the next prefill item; the op is already exact for any
  n_tokens, only the cache capacity limits it.
- Decode speed: ggml graph per token, 4 OpenMP threads, IO thread unpinned and spinning, no lookahead,
  cache not warmed from the prefill. M3 work: M0's static plan with these exact ops checked against the
  graph, ready-first expert dispatch, IO/crypto core placement.
- N-gram rows come from the shard mapping inside the graph (get_rows on the mapped table), not M0's
  page-granular O_DIRECT reader.
- Server adapter (`--engine flash`, feature `flash`) builds but is untested (smoke script ready).
- Uncompiled at hand-off: the oracle-based check (finding 2). Rebuild before the next run.


## How it is built

**Exactness by construction.** `crates/qwen4exp/csrc/flash_graph.cpp` is the qwen4exp graph copied op
for op from llama.cpp 04ff7df (`src/models/qwen4exp.cpp` + the llama-graph.cpp helpers it uses) with
engine-owned state (f16 KV, f16 indexer cache raw|pooled, GDN conv/ssm, PLE conv history, token
history) and the host-side inputs re-derived from llama's memory objects. It is computed by the
**production** `libggml-base.a` / `libggml-cpu.a` (copied by `tools/sync_ggml_prod.sh`, sha256 in its
MANIFEST; cpu lib `adf384be...`). Since this commit the graph port and the production ggml members it
needs are linked into one relocatable object in which every symbol except the `fg_*` C API is local
(`build.rs`: `ld -r` + `objcopy --keep-global-symbol=fg_*`), so it can share a binary with kernels-sys
(oxe-server) without symbol clashes; machine code unchanged.

**Streaming expert op** (`src/stream.rs`, graph nodes when an expert callback is set):

| node | threads | what |
|---|---|---|
| `ffn_moe_fetch-L` | 1 | CLOCK lookups of the selected experts in M0's per-layer cache; misses queued to M0's io_uring IO thread (demand class, one O_DIRECT read of the R4 sidecar record into the slot); x quantised once per token with production `from_float` |
| `ffn_moe_up-L`, `ffn_moe_gate-L` | all | 16-row units; each waits for its slot to be READY, then R4 gemv (IQ2_S / IQ3_XXS) or production `vec_dot` (layer 47 IQ4_XS) |
| `ffn_moe_swiglu-L` | all | production `ggml_swiglu_split`, unchanged |
| `ffn_moe_down-L` | all | act row quantised per thread with production `from_float`, production `vec_dot` (IQ4_NL / Q2_0) |

Why it is exact: production's `mul_mat_id` computes every output row as one `vec_dot(nrc=1)` of the
weight row with the quantised activation row (`ggml_compute_forward_mul_mat_id_one_chunk`), so any row
split, order or thread count gives the same bits; the R4 gemv is bitwise equal to `vec_dot` (vendor test
+ checked in-process here). The node names equal production's, so oracle dumps compare by name.

**State**: `fg_ctx_copy_state` (whole state), `fg_ctx_save_recurrent` / `fg_ctx_restore_recurrent`
(GDN conv/ssm + PLE history + position; KV and indexer cells stay in place).

## Commands

```bash
cd <engine checkout>
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -j 2
# reference (mapped experts; ~2.2 s/token, ~16 min): tokens vs the production oracle
flock -n -E 75 ${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock} systemd-run --user --scope --quiet -p MemoryMax=28G -p MemorySwapMax=0 -- \
  ./target/release/flash-m1 ref --oracle <BigMoeOnEdge checkout>/.bench-oracle/run1/dump --threads 4 --out results/m1/full-t4.json
# streaming (prefill mapped, 3 in-process check steps vs a mapped-expert context, snapshot test, then streaming)
tools/run_flash_m1.sh --cache-gib 16 --threads 4
python3 -I tools/fgd_compare.py DIR_A DIR_B      # FGD1 dump directories, layer order, first divergent node
```

## Codex timing commands (streaming decode, 20 GiB cache)

Preconditions: binary rebuilt from the branch head (`CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -j 2`,
not while someone else holds the lease); `results/m1/prefill-state.bin.ctx4096.tb6` present (saved by the first
`run_flash_m1.sh` run; delete it after any rebuild that changes the graph, the next run re-creates it in ~4 min).
Each command holds the lease for its whole duration and runs in the 28G / no-swap scope; exit 75 = lease busy
(retry only then), 0 = token-identical, 1 = mismatch (report, do not retry). It prints the JSON path.

```bash
cd <engine checkout>
# timing, 20 GiB cache, no reference steps (frees the reference context at once), 353 tokens:
tools/run_flash_m1.sh --cache-gib 20 --check-steps 0 --no-snapshot-test --n-ctx 4096
# same at 18 GiB (memory margin check) and with the IO thread pinned to a free core:
tools/run_flash_m1.sh --cache-gib 18 --check-steps 0 --no-snapshot-test --n-ctx 4096
tools/run_flash_m1.sh --cache-gib 20 --check-steps 0 --no-snapshot-test --n-ctx 4096 --io-cpu 11
```
`--n-ctx 4096` gives the same bits for fixed353 (n_kv is min(n_ctx, pad(cells, 256)) and the pool bound
min(.., n_ctx/4) never binds below 509 cells) and saves ~450 MiB of KV. Memory: Codex measured 20 GiB of
M0 cache as the ceiling against the 1.5 GiB host floor; this process carries ~0.6-1 GiB more than M0
(graph state + compute buffer), so watch `memory.cgroup_memory_peak` / host MemAvailable on the 20 GiB run
and fall back to 18 if the floor gets close. Report from the JSON: `ms_per_token.{mean,median,p90}`,
`expert_wait_ms_per_token_thread0`, `cache.hit_rate`, `cache.read_mib_per_token`, `io.mean_latency_us`,
`memory.*`, `tokens_identical`.
