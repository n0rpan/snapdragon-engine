# M3-ATTN: long-context attention, bit-identical to llama.cpp FA-off

Branch `attn-longctx`. Decode and prefill attention are rebuilt around one kernel that
computes **blocks** of `ggml_vec_dot_f16` results with the reference's exact per-element
arithmetic. The work is split over all threads by KV position / V^T row instead of by query
head, and every K and V^T row is loaded once for all query heads that share it (GQA 8) and,
in prefill, for 4 tokens at a time. **The result is bit-identical to the per-head M1 path and
to llama.cpp FA-off** (native-layout oracle, f16 KV, n_kv padded to 256). The old path stays
selectable (`OXE_ATTN=legacy`) for A/B timing on one binary.

**Gates: all passed** (tiled path, 8 threads, cpus 0-7, under the lease):

| Gate | Ornith-1.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| **NEW long-context fixture `*-lc8k`**: llama.cpp batched prefill of 8180 tokens (32 ubatches) + 32 greedy decode tokens at ctx 8448 (n_kv 8192 -> 8448 = cache size) | last logits bit-exact, KV + conv + S state identical, **31/31 decode rows bit-exact**, greedy identical | same: bit-exact, identical, **31/31**, identical |
| `tools/run_prefill_verify.sh` pf64 / pf512 / pf2048 | all three: last logits exact, state identical, 7/7 decode rows, greedy identical | all three: same |
| M2 `oxe verify` p1-code (token-by-token, 308 rows) | 308/308 bit-exact rows, greedy identical, KLD 0 | 308/308, identical, KLD 0 |
| `oxe mtp-check` p1/p2/p3, depths 1-3, 256 tokens | ids identical, 256/256 rows bit-identical in all 9 runs | same, all 9 runs |
| Unit tests (`cargo test --release -p kernels-sys -p model -p server -p sched`) | all pass, incl. 5 new (section 3) | |
| Legacy arm (`OXE_ATTN=legacy`) vs the long-context oracle | | `qwen36-lc8k`: bit-exact, state identical, 31/31, greedy identical |

The gate binary was 2a86348; the follow-up commit changes only the legacy path (a slice length
that the old 1-KV-head mini model could not catch) and the tests, so the tiled results carry
over; the legacy arm was then checked against the 8K oracle separately (last row).

## 1. Analysis

### Shapes (both models, from the GGUF)

`n_head` 16, `n_head_kv` 2 (GQA group 8), `head_dim` 256, 10 full-attention layers of 40,
f16 K cache `[kv head][pos][256]`, f16 V^T cache `[kv head][dim][pos]`.

### Decode, per token at context C (10 layers)

| | C = 2K | C = 8K | C = 16K |
|---|---:|---:|---:|
| KV bytes, each row read once (20 KiB x C) | 40 MiB | 160 MiB | 320 MiB |
| KV bytes streamed by main (each of 16 query heads streams its KV head: 8x) | 320 MiB | 1.25 GiB | 2.5 GiB |
| f16 FMAs, K^T q + V^T p (2 x 16 x 256 x C x 10) | 168 M | 671 M | 1.34 G |
| main: V^T p over the padded n_kv instead of pos+1 | up to +255 positions per head | +0-3 % | +0-1.5 % |
| for scale: weights per token (Qwen3.6, ~3B active) | ~1.8 GB | ~1.8 GB | ~1.8 GB |

The KV rows of one KV head and layer are 4 MiB (K + V) at 8K, so the 4 query-head owners of a KV
head (on different clusters with the production cpu list 0,1,2,4,5,6,8,9) re-read them mostly
from the system cache / DRAM: at 8K main streams ~0.7x the weight bytes again, at 16K ~1.4x.

**Main's thread split.** `AttnCore` gives each thread its query heads: 16 heads over 8 threads
= 2 each (over 10 or 12 threads 2 vs 1: 67-80 % efficiency). Each (head, position) is one
`ggml_vec_dot_f16(256)` call and each output dim one `ggml_vec_dot_f16(n_kv)` call. On this
build (`+fp16`) that kernel accumulates in **f16** in four `float16x8` accumulators, i.e. four
dependent FMA chains per call: latency-bound at ~1 vector FMA per cycle per core out of 4 FP
pipes. Barriers: 3 per attention layer (AttnQkv, AttnCore, AttnOut), 211 per token.

Rough decode cost of attention at 8K on 8 threads (~8 f16 FMA/cycle/core, 3.8 GHz): 671 M /
(8 x 30 G/s) = ~2.8 ms compute, overlapped with ~1.25 GiB of KV streaming, against ~30 ms per
token at short context. That is the order of the observed slowdown (llama.cpp Ornith 30.5 ->
26 tok/s at 5.5K = +5.7 ms; llama.cpp's own FA-off path has the same per-element structure).

### Prefill, per token with a prefix of P positions

| | P = 2K | P = 7K (2K on top of 6K) | P = 16K |
|---|---:|---:|---:|
| f16 FMAs per token (2 x 16 x 256 x P x 10) | 168 M | 587 M | 1.34 G |
| main: KV bytes streamed per token (each (token, head) streams its KV head) | 320 MiB | 1.1 GiB | 2.5 GiB |
| for scale: dense + MoE MACs per token (i8mm) | ~2.5 G | ~2.5 G | ~2.5 G |

Main splits prefill attention into (query head, 16-token block) units over the team, each
(token, head) running the decode kernel: the same latency-bound f16 dot chains, with no reuse
of a K or V^T row across heads or tokens. At ~30 G f16 FMA/s per core x 12 cores, 587 M/token
costs ~1.6 ms per token at P = 7K. That matches the reported Qwen3.6 slowdown on real agent
tasks (155 vs 212 tok/s = 6.45 vs 4.72 ms/token, +1.7 ms at ~3-6K context): long-context
prefill is dominated by attention, not by the MoE.

### What llama.cpp computes (FA off), checked in the pinned source (ee6b9a1)

- `kq = mul_mat(k, q)`, `soft_max_ext(kq, mask, scale, 0)`, `kqv = mul_mat(v^T, kq)`
  (llama-graph.cpp `build_attn_mha`). `ggml-cpu.c` undefines `GGML_USE_LLAMAFILE` when
  `__ARM_FEATURE_MATMUL_INT8` is set, and the F16 type traits have `nrows = 1`: **each output
  element is exactly one `ggml_vec_dot_f16(ne00, ..., nrc = 1)` call** on a K row / V^T row and
  the f16-converted q / probability column (`from_float = ggml_cpu_fp32_to_fp16`). The split over
  threads never changes an element.
- `ggml_compute_forward_soft_max_f32`: rows interleaved over threads, each row by one thread:
  copy, `ggml_vec_scale_f32`, `wp[i] += slope*mask[i]` (f32 mask, 0 / -inf, slope 1),
  `ggml_vec_max_f32`, `ggml_vec_soft_max_f32` (NEON: groups of 4 from i = 0, `sum +=
  (ggml_float)vaddvq_f32(exp)`), `1.0/sum`, `ggml_vec_scale_f32`. Row length = padded n_kv.
- `ggml_v_expf(-inf)`: the |n| > 192 branch returns `s1*s1` = 2^-125 * 2^-125, which flushes
  to **+0.0f** (derivation from the source; confirmed by the unit tests below, which compare the
  full softmax over n_kv).

## 2. Levers implemented (all exact)

### (a) One kernel: blocks of `ggml_vec_dot_f16`, same arithmetic per element

`oxe_dot_f16_tile(n, x rows, y columns, out, n_full)` (`kernels-sys/csrc/oxe_ops.cpp`).
For every (row, column) it performs exactly the reference's operations: four `float16x8`
accumulators, accumulator j takes elements `[i+8j, i+8j+8)` of every 32-element step i, in
increasing i, with `vfmaq_f16`; then `GGML_F16x8_REDUCE` (a0+a2, a1+a3, sum; widen;
`vaddvq_f32`) and the `(float)` store. Only the interleaving of *independent* outputs and
accumulators changes: a 4 x 4 register tile has 16 independent chains (throughput- instead of
latency-bound) and loads each x / y vector once per 16 FMAs; positions go in blocks of 256 so a
4-row slice stays in L1 across the four accumulator passes. No spills in the inner loops
(checked in the object code).

### (b) Decode: split by KV position and V^T row over all threads, GQA rows loaded once

`Engine::attn_core_tiled` (`engine.rs`), inside the existing `AttnCore` phase:
1. Every thread computes the K rows of this run's positions (k-norm, RoPE, f16: the same
   redundant prologue as before); the designated thread stores them in the cache.
2. **Scores**: the (KV head, position) rows `0..=pos` are split evenly over all threads; each
   64-row block is one tile call against the 8 query heads x lanes of that KV head
   (q is converted to f16 once by the query-head owner in `AttnQkv`). -B-
3. **Softmax**: the (lane, head) rows are split over the threads. -B-
4. **Outputs**: the 512 V^T rows are split over the threads; each 32-row block is one tile call
   against all 8 x lanes probability columns; the sigmoid gate (computed once in `AttnQkv`) is
   applied and O written.

Every K and V^T row is read once per token instead of 8 times, every thread works at every
context length, and the f16 FMAs run at tile throughput. Cost: +2 barriers per attention
layer (231 instead of 211 per token, ~+20-40 us). The same code serves verify / MTP runs
(up to 8 lanes; a row of this run's position i is scored only for lanes >= i).

### (c) Prefill: (KV head, 4 tokens) units share every K and V^T row over 32 columns

`Engine::pf_attention_tiled` (`engine/prefill.rs`): dynamic units of (KV head, 4 query
tokens) = 32 query columns (128 units per 256-token ubatch). Per unit: one tile call for the
scores over K rows `0..p0+t1` (rows beyond a token's position are masked anyway), 32 softmax
rows, one tile call over the 256 V^T rows. K / V^T traffic drops 32x and the FMAs run at tile
throughput. The per-token n_kv is a closure (`n_kv_of(t)`), ready for the multi-ubatch branch.
Thread scratch: 32 x n_ctx f32 + f16 per thread (touched only up to the prefix length).

### (d) Padded-n_kv waste: skipped only where provably identical

- **Softmax** runs on `n4 = round_up(pos+1, 4)` entries instead of n_kv. The entries beyond are
  -inf; `ggml_vec_max_f32` ignores them, each further group of four is `exp(-inf) = +0` and adds
  `+0.0` to the (positive) double sum, and their probabilities are +0. Identical for every entry.
- **V^T p** stops at `round_up(pos+1, 32)` (a step boundary). A skipped step is
  `acc = fma(acc, v, +0) = acc + (v * +0)`: a non-zero lane is unchanged, a +0 lane stays +0,
  but **a lane holding -0 becomes +0 when v >= +0**. -0 lanes do occur at long context (a tiny
  negative product `v * p` with p = 2^-24 rounds to -0; the unit test builds one), and would
  change the sign of an exact-zero output. So the kernel checks the accumulators for -0 lanes
  at the cut and, if there is one, **replays the reference's remaining steps** for that output.
  The result equals the full-length dot bit for bit whenever the V cache entries beyond pos are
  finite (they are written values or zeros; `restore_recurrent` zeroes rolled-back V rows; the
  per-head path already assumed finite K entries the same way).
- The masked K rows were already skipped by the M1 kernel.

### (e) Smaller items checked

- **f32 -> f16 of q**: 16 x 256 conversions per token, once per head by its owner
  (`GGML_CPU_FP32_TO_FP16`, as `from_float`); negligible.
- **V^T layout**: the write side is 512 scattered 2-byte stores per token per layer (cheap); the
  read side is what makes V^T p a row-contiguous dot, so the tile kernel streams it.
- **Trace mode** (`oxe trace` / `compare`) keeps the per-head path (it emits per-head tensors).

### Not exact (would need a KLD gate; not implemented)

- f32 accumulation (FMLAL widening) or an online-softmax (flash) formulation: different
  rounding from llama.cpp's f16-accumulating FA-off path; would also avoid materialising the
  n_kv scores. llama.cpp's own FA-on path is a different computation as well.
- Quantised (q8_0) KV cache: halves KV bytes, different numerics.

## 3. Tests

`cargo test --release -p kernels-sys -p model` (no model needed):
- `dot_f16_tile_matches_vec_dot`: every tile shape (1-9 rows, 1-70 columns, n up to 1024,
  n % 32 != 0 fallback) equals `ggml_vec_dot_f16` bit for bit.
- `attn_tiled_matches_attn_head`: scores + softmax rows + V^T tile vs `oxe_attn_head_f16`
  at 20 positions (0..2047 around every 4 / 32 / 256 boundary): probabilities and outputs
  bit-identical.
- `dot_f16_tile_tail_rule`: crafted all -0 / mixed lanes with positive, +0, negative and -0
  tail entries: equals the full-length dot; without the replay the all -0 column would come out
  -0.0 instead of +0.0 (asserted, so the test proves the case occurs).
- `mini_gguf_tiled_attention_equals_legacy`: 300 decode steps (n_kv 256 -> 512) at 1, 3 and 4
  threads, every logit row equal to the legacy path.
- `mini_gguf_tiled_prefill_equals_legacy`: 700-token prompt in ubatches of 256 (n_kv 256 / 512
  / 768) + 40 decode steps: logits, full state and decode rows equal to the legacy path at 1 and
  3 threads.
- The mini model now has GQA (4 query heads over 2 KV heads; it had 1 KV head, which hid a
  slice-length bug in the legacy path during development).
- Existing: decode thread invariance (barrier count now 2 more per attention layer), verify
  lanes with rollback, MTP greedy (plain and batched prefill) == plain, prefill invariance.

## 4. Rough sanity timing (not a benchmark; Codex measures)

One run each, Qwen3.6, same binary, `OXE_ATTN=legacy` = main's attention path. Machine shared
with other lanes (lease held for each run), so treat as indicative only.

| Run | legacy (main's path) | tiled | Same output |
|---|---:|---:|---|
| decode at ctx 8K (`bench --prompt-len 8192 --batched-prompt --tokens 32`, cpus 0,1,2,4,5,6,8,9) | 22.3 tok/s (44.9 ms) | **30.4-31.0 tok/s** (32.3-32.9 ms) | ids hash identical |
| prefill 2048 tokens on a 6144-token prefix (`prefill-bench --prefix-len 6144`, cpus 0-11) | 121.5 tok/s | **169.2 tok/s** | first token identical |
| `prefill-verify qwen36-lc8k` (8180 tokens from 0, 8 threads, nice 19, first touch) | 62.1 s | 60.1 s | both bit-exact vs llama.cpp |

The 8K decode is back near the short-context ~33 tok/s. The last row is a wall time of a
verification run (8 threads, nice 19, average prefix 4K, scratch allocation and first touch
included), not a prefill measurement; it is listed because the legacy arm was also checked
against the long-context oracle there (bit-exact, 31/31 rows).

## Merging with `prefill-multi-ubatch`

`git merge-tree` against `origin/prefill-multi-ubatch` (725e751): `engine.rs`, `prefill.rs` and
`tests.rs` auto-merge; `oxe.rs` conflicts textually in `prefill-bench` (keep both `--group` and
`--prefix-len`). After the merge `prefill.rs` does not compile until the tiled call gets that
branch's per-token n_kv: replace `self.pf_attention_tiled(pf, ts, l, n, p0, |_t| n_kv)` with
`self.pf_attention_tiled(pf, ts, l, n, p0, n_kv_of)` (and drop the now unused `n_kv`). The tail
rule then compares against each token's own ubatch n_kv, as the legacy loop there does. One
difference from llama.cpp in grouped mode, which the legacy loop shares: the V cache beyond a
token's ubatch already holds the next ubatch's values, so a replayed -0 lane can turn out with a
different zero sign than llama.cpp. Only the sign of an exact-zero attention output can change,
and the next op quantises it (`wo` activation rows) to the same bits, so logits and state stay
identical. Strictly, though, it is not the same intermediate.

## 5. Codex timing commands

One binary; `OXE_ATTN=legacy` runs main's attention code path; `OXE_ATTN_TILED_MIN_KV=N` sets
the decode cutoff (default 4096; use 0 to force the tiled decode at every context) (everything else in the tree is
main b477aec), so each pair is a clean A/B. Every run holds the lease.

`$MODELS` is the directory that holds the GGUF files.

```bash
# worktree of branch attn-longctx (the main checkout is on prefill-multi-ubatch; do not switch it)
cd <repo> && git pull && CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release
mkdir -p results
QW=$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
OR=$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf
L=${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock}
# decode tok/s at context 1K / 4K / 8K / 16K (prompt prefilled batched, then 64 timed tokens)
for M in QW OR; do for C in 1024 4096 8192 16384; do for A in tiled legacy; do
  OXE_ATTN=$A flock -n $L ./target/release/oxe bench --model ${!M} --cpu-list 0,1,2,4,5,6,8,9 \
      --prompt-len $C --batched-prompt --warmup 8 --tokens 64 --ctx $((C + 512)) \
      > results/attn-decode-$M-c$C-$A.json
done; done; done
# prefill of 2K tokens on top of 6K existing context (prefix prefilled once, untimed)
for M in QW OR; do for A in tiled legacy; do
  OXE_ATTN=$A flock -n $L ./target/release/oxe prefill-bench --model ${!M} --cpu-list 0-11 \
      --prefix-len 6144 --prompt-len 2048 --ubatch 256 --reps 3 --ctx 8448 \
      --out results/attn-prefill-$M-6k+2k-$A.json
done; done
```

Read `tok_s` / `ms_per_token.median` (bench) and `tok_s_median` (prefill-bench); `attn` in the
JSON says which path ran. `generated_ids` / `first_token` must be identical between the tiled and
legacy arm of a model. For a check against main itself, the same commands without `OXE_ATTN`
on main need `--batched-prompt` / `--prefix-len`, which main lacks; use the legacy arm.
Worth also: decode with 12 threads (`--cpu-list 0-11`): the per-head split was 16 heads over
12 threads; the tiled split has no such step.

## 6. Remaining

- Speed: only the rough sanity runs above (decode 8K +37 %, prefill 2K-on-6K +39 %). The
  proper sweep (1K / 4K / 8K / 16K, both models, 8 and 12 threads) is Codex's, commands in 5.
- **Decode cutoff (added at integration).** Codex's first timing (packed cores, same binary,
  identical tokens): decode at 8K Ornith 37.48 -> 41.55 tok/s (+10.8 %), Qwen3.6 28.68 -> 30.16
  (+5.2 %); prefill 2K on a 6K prefix Ornith 126 -> 164 (+30 %), Qwen3.6 ~126 -> ~167; but decode
  at 1K Qwen3.6 32.82 -> 28.87 (-12 %, being repeated), Ornith flat. So decode / verify runs now
  take the tiled path only when the run's padded n_kv >= `OXE_ATTN_TILED_MIN_KV` (default
  **4096**, `Engine::set_attn_tiled_min_kv`; 0 = always tiled) and the per-head path below it.
  Prefill always uses the tiled path. Both paths are bit-identical, so the cutoff is a pure
  speed knob for Codex to tune (e.g. sweep 1024 / 2048 / 4096 / 8192 with the decode commands in
  section 5). The likely cost at short context is the 20 extra barriers per token plus three
  short phases with little work each; a fused single-barrier variant (scores + softmax per
  thread-owned row block) is the follow-up if the crossover needs to move lower.
- Unit tests pin the cutoff to 0 where they compare paths; the MTP / verify tests now run the
  MTP engine tiled against a per-head plain reference.
- If the softmax rows (16 per token, n_kv exps each) show up at 16K+, they can be fused into the
  score step per thread with a partial max (still exact: max is order-independent; the sum must
  stay sequential per row).
- The multi-ubatch prefill branch (`prefill-multi-ubatch`) touches the same function: on merge
  its `n_kv_of(t)` goes into `pf_attention_tiled`'s closure argument.

## 7. Integration on main (mtp-seed-fast + attn-longctx + prefix-cache-tools)

main = b477aec + `mtp-seed-fast` (c73a628) + `attn-longctx` + `prefix-cache-tools` (cdc64e3) + the
decode cutoff. `prefill-multi-ubatch` is not included. One conflict: mtp-seed-fast factored the
designated K writes into `write_k`, and the tiled path widened the K scratch to
n_head_kv x head_dim, so `write_k` now uses the first head_dim halves. The tiled attention keeps
the KV cache layout (its buffers are separate), so `state_layout` / `write_state` / `read_state`
of the prefix store are unchanged.

Gates on the merged tree (one run, lease held per run, 8 threads cpus 0-7, default cutoff 4096):

| Gate | Result |
|---|---|
| lc8k (Ornith, Qwen3.6): 8180-token prefill + 31 decode rows at n_kv 8192 / 8448 (tiled) | both: last logits bit-exact, state identical, 31/31 rows, greedy identical |
| pf64 / pf512 / pf2048, both models | all six: exact, state identical, 7/7, greedy identical |
| M2 `oxe verify` p1-code | Ornith 308/308, Qwen3.6 308/308 bit-exact, KLD 0 |
| `mtp-check` 3 prompts x depths 1-3 | Ornith and Qwen3.6: all 9 runs ids identical, 256/256 rows |
| forced tiled decode at short context (`OXE_ATTN_TILED_MIN_KV=0`), Qwen3.6 | `oxe verify` 308/308; `mtp-check` all 9 runs identical (tiled multi-lane verify) |
| `mtp_multiturn_greedy_is_identical_to_plain` (ignored, real model) | Ornith pass, Qwen3.6 pass |
| `prefix_store_real_model` (ignored, real model) | Ornith pass, Qwen3.6 pass |
| unit tests, whole workspace (incl. stub server tests) | all pass |
