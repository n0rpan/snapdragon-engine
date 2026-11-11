# M0-FLASH: Qwen3.8-Flash-Next timing prototype (`crates/qwen4exp`, `flash-m0`)

M0 for the Flash engine (design: the Flash engine design in the private lab notes, §5.3).
It answers one question before any exact math is written: **with the real Flash weights, the real
kernels, a static per-thread plan and real expert streaming from SSD, how fast can one decode token go?**
Numerics are not checked (activations are dummy); time, bytes, barriers, cache and SSD behaviour are real.

Gate (owner, 2026-11-09): **GO** if ≤ 90 ms/token (≥ 11 tok/s) with a streaming predictor, or ≥ 1.5× the
tuned baseline; **STOP** above 140 ms/token. Baseline (Codex P1, memlat boot): BigMoeOnEdge
**4.057 tok/s = 246.5 ms/token**, 101.8 s/request, peak 18.54 GiB, fixed353 answer SHA-identical.
So 1.5× baseline = 164 ms/token is already met far above the 90 ms line; the 90/140 ms lines decide.

## Build and test (no lease needed, but defer while Codex holds it)

```bash
cd <engine checkout>        # worktree of branch flash-m0
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -j 2
CARGO_BUILD_JOBS=2 nice -n 19 cargo test  --release -p qwen4exp -j 2   # 7 tests, synthetic + trace parse
./target/release/flash-m0 info                    # header-only: layout, quotas, PLE geometry
```

Kernel provenance and the standalone kernel test: `vendor/ggml-flash/PROVENANCE.md`
(fork `claude/flash-r4` 8009864; NEON vec_dot vs generic, R4 round trip, R4 gemv bitwise = NEON vec_dot: ALL PASSED).

## One-time sidecar (done 2026-11-10, 86 s, 50.36 GB)

`~/.cache/local-ai/snapdragon-engine/flash/experts-r4.bin` (+ `.json` manifest with shard sizes/mtimes and layout).
One 4 KiB-aligned record per (layer, expert): `gate | up | down`, gate/up already in R4 layout for IQ2_S / IQ3_XXS
layers, stride = record rounded to 4 KiB (1,974,272 B IQ2_S layers, 1,511,424 B Q2_0-down layers, 2,179,072 B
IQ3_XXS layers, 2,662,400 B layer 47 IQ4_XS). A miss is one O_DIRECT read straight into the cache slot.
Rebuild (heavy I/O, hold the lease): `flock -n -E 75 ${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock} ./target/release/flash-m0 sidecar`;
re-check: `flash-m0 verify-sidecar --n 256` (re-reads records, compares with records rebuilt from the shards).

## The command for Codex (authoritative timing)

Preconditions: memlat boot; nothing else holding memory (llama-server / bmoe-server stopped); quiet machine.
`tools/run_flash_m0.sh` takes the lease for the whole run (`flock -n -E 75`, retry only on 75) and runs inside
`systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=0`. It prints the JSON path; one summary line goes to stderr.

Each run is ~45-60 s wall (load 3-5 s, prefault 5 s, warm-up 8.5 s, 353 tokens), peak ~20.6 GiB.
Run in this order (the first three are the gate set; 3 repetitions each if time allows, ABBA not needed):

```bash
cd <engine checkout>
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -j 2   # only if target/release/flash-m0 is missing/stale
tools/run_flash_m0.sh --io none                                         # (i)  compute only, no SSD
tools/run_flash_m0.sh --io demand                                       # (ii) streaming, demand reads only
tools/run_flash_m0.sh --io lookahead --recall 0.9 --depth 2 --false-pos off   # (iii) GATE: predictor, 2 layers ahead
tools/run_flash_m0.sh --io lookahead --recall 0.7 --depth 2 --false-pos off   # predictor sensitivity
tools/run_flash_m0.sh --io lookahead --recall 1.0 --depth 2 --false-pos off   # oracle (upper bound)
tools/run_flash_m0.sh --io lookahead --recall 0.9 --depth 2 --false-pos on    # with wasted reads (pessimistic)
tools/run_flash_m0.sh --io demand --cpu-list 0-7 --io-cpu 8                   # packed layout (grey-zone rule)
tools/run_flash_m0.sh --io demand --cache-gib 18                              # cache size (watch memory.peak)
```

`tools/run_flash_m0.sh` exits 75 at once if the lease is busy (retry only on 75); any other non-zero exit is a
failure to report, not to retry. Summary line on stderr; JSON path on stdout. Report per run: `ms_per_token.median`,
`breakdown_ms`, `cache.hit_rate_resident`, `cache.read_mib_per_token`, `io[].decode_mean_latency_us`,
`memory.cgroup_memory_peak`.

Exactly what the default runs:

```
flash-m0 run --cpu-list 0,1,2,4,5,6,8,9 --io-cpu 3 --cache-gib 16 --quota prefill --io lookahead --recall 0.9 --false-pos off \
  --depth 2 --tokens 353 --warmup 8 --out results/flash-m0-<ts>.json
```
(defaults also: `--dense vecdot --qd 32 --prefetch-qd 16 --ngram on --ngram-cache-mib 64 --ctx 2048 --pos0 156`)

Sequence per run: dense arena load (~4.5 GB, buffered + FADV_DONTNEED) → 16 GiB cache arena allocated and prefaulted →
cache warmed with the trace's **prefill** routes (what the 156-token prompt leaves behind) → 353 replayed decode tokens,
the first 8 excluded from the statistics.

## Results (Claude's indicative runs, 2026-11-10 02:50-03:25, NOT the authoritative timing)

Same boot, same binary (`3175eb8` + the warm-up/IO-stat fixes in the next commit), each run under the lease and the
28G/no-swap scope, 353 replayed tokens (first 8 excluded), 8 spread threads `0,1,2,4,5,6,8,9`, IO thread on core 3,
16 GiB cache (prefill quotas), dense `vecdot`. Raw JSON/logs: `results/flash-m0-<name>.{json,log}` (git-ignored).
"nf" = no false positives (the predictor's misses cost nothing); otherwise the predictor's non-recalled slots are
filled with random experts (pessimistic waste). Breakdown columns are ms/token; the sum is the token wall.

| run | flags | median ms | tok/s | dense | experts | SSD wait | N-gram | sync | resident hit | demand misses/tok | MiB read/tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (i) none | `--io none` | **65.1** | 15.4 | 45.5 | 17.9 | 0.5 | 1.5 | 0.2 | 87.3% | (61.1, not read) | 0.9 |
| (ii) demand | `--io demand` | **115.5** | 8.66 | 45.8 | 18.0 | 55.2 | 0.0 | 0.2 | 87.3% | 61.1 | 118.2 |
| la r0.7 d1 | `--recall 0.7` | 218.6 | 4.58 | 56.6 | 21.1 | 144.8 | 0.5 | 0.4 | 86.7% | 30.2 | 363.5 |
| la r0.5 d1 | `--recall 0.5` | 266.0 | 3.76 | 55.6 | 21.4 | 191.7 | 0.2 | 1.7 | 84.8% | 54.7 | 509.5 |
| la r0.9 d1 | `--recall 0.9` | 132.5 | 7.55 | 54.8 | 20.1 | 62.3 | 0.1 | 0.3 | 89.2% | 10.2 | 207.7 |
| la r0.7 d2 | `--recall 0.7 --depth 2` | 284.8 | 3.51 | 58.7 | 22.4 | 205.7 | 0.1 | 0.4 | 89.1% | 13.5 | 579.0 |
| la r0.7 d1 nf | `--recall 0.7 --false-pos off` | 114.0 | 8.77 | 51.6 | 19.1 | 46.3 | 0.2 | 0.4 | 91.5% | 20.6 | 118.2 |
| la r0.9 d1 nf | `--recall 0.9 --false-pos off` | 96.2 | 10.40 | 50.8 | 18.7 | 30.3 | 0.0 | 0.3 | 92.0% | 8.8 | 118.1 |
| la r1.0 d1 (oracle) | `--recall 1.0 --false-pos off` | 96.4 | 10.38 | 52.2 | 19.0 | 28.7 | 0.0 | 0.3 | 91.5% | 2.9 | 118.2 |
| la r0.7 d2 nf | `--recall 0.7 --depth 2 --false-pos off` | 92.7 | 10.79 | 51.6 | 19.1 | 25.4 | 0.1 | 0.3 | 95.3% | 8.8 | 118.0 |
| **la r0.9 d2 nf** | `--recall 0.9 --depth 2 --false-pos off` | **88.7** | **11.27** | 52.4 | 19.4 | 20.5 | 0.1 | 0.3 | 95.9% | 3.7 | 118.1 |

Common to all: 484 barriers/token (10/layer + PLE + head + join); cgroup `memory.peak` 20.50-20.65 GiB (VmHWM 20.47
GiB) with the 16 GiB cache, swap 0; dense arena 4.52 GB loaded in 2.6-5.1 s; cache prefault ~5 s; warm-up from the
prefill routes 20.8 GB in 8.3-8.5 s; cache-management time on thread 0 0.03-0.12 ms/token. Hit rate (87.3% demand-only,
per-layer CLOCK) matches the offline replay of the same policy (87.9%; global LRU would give 88.7%).
Per-token bytes: dense 3.739 GB + F32 router 0.252 GB + 10 routed experts × 48 = 0.982 GB.

### Reading the numbers

1. **Compute floor (mode i) = 65.1 ms** (design target ≤ 62 ms): dense 45.5 ms (3.99 GB → 88 GB/s, as predicted),
   experts 17.9 ms (0.98 GB; predicted 12: the R4/IQ4_NL/Q2_0 expert kernels run ~55 GB/s, ALU-bound), sync only
   0.2 ms (the static plan's barriers are nearly free; the 56 ms sync bucket of the llama.cpp path is gone).
2. **Streaming costs compute too:** with SSD traffic, dense goes 45.5 → 51-59 ms (+6-13 ms): DMA + dm-crypt decryption
   (kcryptd on the same cores/memory bus) eat bandwidth. More bytes read → slower dense (r0.7 d2 with waste: 58.7).
3. **Latency, not bandwidth, is the SSD problem.** A 2 MB record takes 2.5-5 ms from submit to completion at the low
   queue depth decode produces (1.0-1.3 GB/s used of 4.6), while one layer of compute is ~1.4 ms. Demand-only exposes
   ~1.15 ms per layer (55 ms). A one-layer lookahead cannot hide a read even with a perfect predictor (oracle d1: 28.7 ms
   left); **two layers of lookahead** is required (r0.7 d2 nf: 25.4 ms, r0.9 d2 nf: 20.5 ms).
4. **Prefetch waste is fatal:** wrong guesses that cost reads (random false positives) make lookahead worse than
   demand-only (218-285 ms): they triple the SSD bytes, queue ahead of demand reads in the device and evict useful
   experts. A real predictor must be cheap when wrong (only prefetch confident guesses / staging area, prefetch QD cap).

### GO / STOP indication

- By the letter of the gate: **GO** — every realistic configuration is ≥ 1.5× the 246.5 ms baseline (demand-only
  115.5 ms = 2.13×), and the predictor case r0.9 / depth 2 / no waste reaches **88.7 ms (11.3 tok/s) ≤ 90 ms**.
- Honest reading: the ≤ 90 ms line is met only under an **optimistic, unmeasured predictor** (90% recall two layers
  ahead with no wasted reads). Without a predictor the engine sits at **115 ms (8.7 tok/s), in the 90-140 grey zone**,
  still 2.1× the tuned baseline. The first M1-phase item is therefore to measure the real router-lookahead recall at
  depth 2 (the P2 trace has no hidden states; needs `x_{L-1}` dumps) — and the next speed levers are dense-vs-IO
  contention (pin kcryptd away from the compute cores, `--cpu-list`/`--io-cpu` layout), and the expert kernels (18 ms).
- The `gate.verdict` field in each JSON applies the rule literally to that run (it also says GO for `--io none`,
  which is not a gate run); use the table above.

## Design as built

| Part | What |
|---|---|
| Dense | every non-expert tensor except the N-gram table (4.52 GB incl. token_embd) in one anonymous THP arena, file layout, production NEON `vec_dot` (Q6_K, Q8_0), `oxe_vec_dot_f32` (F32 router, alpha/beta, inject), bf16 indexer in Rust |
| Experts | R4 gemv (IQ2_S / IQ3_XXS gate/up), NEON vec_dot IQ4_XS (layer 47 gate/up), IQ4_NL / Q2_0 down vs q8_0 |
| Plan | 8 pinned threads, 10 barriers per layer (11 on layer 1) + head, ~485/token; rows split in 8-row groups (4 for expert gate/up), GDN by value head, attention by query head; every thread quantises its activations redundantly |
| Cache | per-layer slot pools, quotas ∝ prefill distinct experts × slot size (floor 64; layer 47's cropped prefill gets the mean), CLOCK, PENDING/READY words; slots stamped with the current token are never evicted |
| IO | one io_uring (raw syscalls), dedicated spinning IO thread on its own core, O_DIRECT READ into the slot, QD 32, classes demand > N-gram > prefetch (prefetch capped at 16 in flight) |
| Lookahead | at layer L (after the router barrier) thread 0 issues layer L's demand misses, then prefetches layer L+1's predicted top-10: each true expert with probability `--recall`, the rest random (false positives pollute the cache) |
| Expert compute | resident experts first, misses as they land; time spinning on a PENDING slot = SSD wait |
| N-gram | 16 rows/token (orders 2 and 3 × 8 heads), page-granular O_DIRECT reads (4 KiB, 8 KiB when a 90 B row straddles), 64 MiB page CLOCK cache, issued at token start, consumed at layer 1 |
| Breakdown | per phase: wall (thread 0 barrier exits) = slowest thread's compute + its SSD / N-gram wait + sync (rest) |

## Faithful vs approximated

| Faithful | Approximated (M0 only) |
|---|---|
| every weight byte a decode token reads (dense 3.97 GB + router 0.25 GB + 10 experts × 48), by the production kernels | activations dummy (norm-bounded); argmax meaningless |
| expert bytes come from SSD through the cache exactly as the trace's routing demands them | routing replayed from the P2 trace (fixed353 answer, 353 tokens), not computed |
| work partition, barriers, pinned team, dedicated IO core | hyper-connection mixing math, PLE combination, GDN gating constants |
| GDN state r+w (36 × 3 MiB), conv state, KV (f32) at ctx ≤ 509 | attention: no RoPE, no compression (ratio 4) / indexer selection (all keys < 2048 kept; indexer scores computed for cost) |
| N-gram table geometry (P4 correction: 320 M rows × 90 B) and page reads | N-gram row hash is a stand-in (not the model's); token ids synthetic → pessimistic page hit rate |
| | lookahead predictor is an oracle with a recall knob (real router-lookahead recall is unmeasured) |
