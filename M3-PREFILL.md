# M3.2: batched prefill, bit-identical to llama.cpp's batched prefill

`Engine::prefill(pool, tokens, ubatch)` replaces the token-by-token prompt path. A prompt is
processed in ubatches (production `--ubatch-size 256`). After the prefill, the **KV cache, the
conv and delta-net state and the last-token logits are bit-identical to llama.cpp's own
batched prefill** (native-layout oracle: `<llama.cpp checkout>` @ `ee6b9a1`, repack on,
KleidiAI off, FA off, f16 KV, same ubatch), and decode continues from that state exactly as
llama.cpp does.

The reference is llama.cpp's *batched* path, not our decode: the two differ on Ornith (KLD
0.027, M2.md) because the dense Q4_K GEMM rounds differently from the GEMV.

**Gate: passed on both models at 64, 512 and 2048 tokens.**

## Gate results

Oracle: `dump_ref --batch --ubatch 256 --last-only --state F --logits --gen 8` (llama.cpp's
batched prefill with production ubatch 256 and logits for the last token only, as
llama-server requests them; 6 threads), prompt = the first N tokens of
`snapdragon-local-ai-lab/benchmarks/roofline/quality-corpus.txt`. oxe: `oxe prefill-verify`
(8 threads, cpus 0-7). The state comparison parses llama.cpp's `llama_state_seq_get_data`
(every f16 K row and V^T element of the 10 attention layers, conv state and delta-net S of the
30 recurrent layers) and compares bit for bit. After the prefill both engines decode the
oracle's 8 greedy tokens (teacher-forced) and every logit row is compared.

| Model | Prompt | ubatches | Last-token logits | KV + conv + S state | Decode rows after prefill | Greedy (8) |
|---|---:|---:|---|---|---:|---|
| Ornith-1.5-35B-A3B | 64 | 1 | bit-identical | identical | 7 / 7 bit-identical | identical |
| Ornith | 512 | 2 | bit-identical | identical | 7 / 7 | identical |
| Ornith | 2048 | 8 | bit-identical | identical | 7 / 7 | identical |
| Qwen3.6-35B-A3B | 64 | 1 | bit-identical | identical | 7 / 7 | identical |
| Qwen3.6 | 512 | 2 | bit-identical | identical | 7 / 7 | identical |
| Qwen3.6 | 2048 | 8 | bit-identical | identical | 7 / 7 | identical |

Per-layer: on the 64-token Ornith fixture, `oxe compare` of the prefill trace against the
oracle's batched dump finds 831 / 832 named tensors bit-exact over all 40 layers (the one
other is `attn_residual-39`, which llama.cpp crops to the output row, so the sizes differ).

Reproduce: `tools/run_prefill_verify.sh [--cpu-list 0-7] [fixture ...]` (holds the inference
lease per run; JSON in `results/prefill-verify-<fixture>.json`). Fixtures (logits + state,
~70-350 MB each): `~/.cache/local-ai/snapdragon-engine/oracle/{ornith,qwen36}-pf{64,512,2048}`.

### How the first attempt failed (for the record)

The first port used llamafile's tinyBLAS semantics for the F32 router and the f16 attention
matmuls (`n >= 4` / `n >= 8`). `oxe compare` localised the first difference to
`ffn_moe_logits-0` (router), with every input exact. Cause: `ggml-cpu.c` undefines
`GGML_USE_LLAMAFILE` when `__ARM_FEATURE_MATMUL_INT8` is set, so on this build llama.cpp uses
`ggml_vec_dot_f32` / `ggml_vec_dot_f16` per element, i.e. the decode kernels. With that, all
six fixtures matched.

## What llama.cpp computes for a ubatch of N tokens (and oxe mirrors)

Read in the pinned source; each point was confirmed by the per-tensor oracle comparison.

| Op | llama.cpp path (N tokens) | oxe |
|---|---|---|
| Dense projections (Q4_K/Q6_K/Q8_0 repacked: qkv, z, alpha/beta, ssm_out, wq/wk/wv/wo, shared experts) | repack `forward_mul_mat`: rows `0..N-N%4` quantised as 4-row interleaved groups (`ggml_quantize_mat_q8_K_4x8` generic / `ggml_quantize_mat_q8_0_4x8` NEON) through `gemm<>` (`ggml_gemm_q4_K_8x8_q8_K`, `..q6_K..`, `ggml_gemm_q8_0_4x8_q8_0`, i8mm); the last `N%4` rows quantised with `from_float` through the GEMV | same kernels, vendored verbatim (extract.py), same row grouping; units of 64 weight rows, dynamically scheduled |
| F32 weights (router `ffn_gate_inp`, shared gate, Qwen3.6's F32 alpha/beta) | `ggml_vec_dot_f32` per element. **Not tinyBLAS:** `ggml-cpu.c` does `#undef GGML_USE_LLAMAFILE` when `__ARM_FEATURE_MATMUL_INT8` is defined | `oxe_vec_dot_f32` (decode's) |
| Routed experts (MUL_MAT_ID) | every row quantised with `from_float`; per expert, rows in (token, slot) order, 4 at a time interleaved (`ggml_repack_interleave_q8_rows_x4`) through `gemm_mmid` (Q4_K/Q5_K: `_gemv_order` kernels; Q3_K/Q6_K/Q8_0: the MUL_MAT gemm), the rest via GEMV | same; for every type in both models `gemm_mmid` is bitwise equal to the GEMV (unit test) |
| Delta-net | `ssm_conv` over `concat(state, x)`, fused `GGML_OP_GATED_DELTA_NET` (`GDN_CH`): the same sequential per-token recurrence as decode | decode's ported conv step and delta-rule step, token by token; channels / value heads split over threads |
| Attention (FA off) | K^T Q and V^T P via `ggml_vec_dot_f16` per element (no llamafile, see above); softmax over `n_kv = min(size, max(256, PAD(p0+N, 256)))` with the causal mask | decode's `oxe_attn_head_f16` per (token, head), with the ubatch's n_kv and the token's mask |
| Last layer, last ubatch | `crop_before_nextn`: with logits for the last token only, the residual, FFN, output norm and head run on that one row (N = 1: GEMV / vec_dot) | same (`Rows { r0: N-1, n: 1 }`); non-final ubatches skip the last layer's FFN |

Everything else (norms, l2 norm, gates, RoPE, softmax/top-k routing, swiglu, the expert add
chain) is row-wise and identical to decode.

Consequences:
- `prefill` of 1-3 tokens equals decode bit for bit (all GEMV); unit test.
- For Qwen3.6 (dense Q8_0, Q8_0 shared experts) the GEMMs are bitwise equal to the GEMV, so
  its batched prefill should also equal its decode; for Ornith the dense Q4_K GEMM does not.
- The result does not depend on the thread count or the work split (unit test at 1-4 threads,
  real model at 8 threads vs the oracle's 6).

## Code

| Piece | Where |
|---|---|
| Prefill engine | `crates/model/src/engine/prefill.rs` (`Engine::prefill`, scratch allocated on first use: ~70 MB at ubatch 256 and 4K ctx) |
| State export in llama.cpp's order | `Engine::dump_state` (`StateDump`) |
| Vendored kernels | `vendor/ggml-subset/extract.py` (+20 items: quantize_mat 4x8, interleave tile, gemm q3/q4/q5/q6/q8_0 + gemv_order + generics), `kernels-sys` wrappers `gemm_raw`, `gemm_mmid_raw`, `quantize_mat4_raw`, `interleave_rows4_raw` |
| Oracle | `tools/oracle/dump_ref.cpp`: `--ubatch`, `--last-only`, `--state FILE` (`llama_state_seq_get_data`), `--n-tokens`, `--dump-layers` |
| Gate | `oxe prefill-verify --model M --ref DIR` (parses llama.cpp's state file: KV + conv + S; logits; teacher-forced decode after the prefill); `tools/run_prefill_verify.sh` runs all six fixtures under the lease |
| Bench | `oxe prefill-bench` |
| Server | `crates/server/src/oxe_engine.rs`: prompts of 2+ tokens use `Engine::prefill` (ubatch from `--ubatch-size`, default 256; optional `--oxe-prefill-cpus` for a separate, larger prefill team); single tokens use the decode step |

## MTP interface (`Engine::prefill_ext`)

`prefill_ext(pool, tokens, ubatch, Some(&mut hidden))` additionally returns the output-normed
hidden state (`h_nextn`) of every prompt position, for MTP catch-up after a prompt. This is
llama.cpp's `embeddings_nextn` graph (`crop_after_nextn`): the last layer's residual and FFN
run on all rows, as in a llama-server with `--spec-type draft-mtp`. KV and recurrent state are
identical to `prefill` (unit test); the last-token logits may differ from `prefill`'s in the
last bits because the last layer's FFN for that row then goes through the GEMM. Not yet gated
against a llama.cpp `embeddings_nextn` oracle (dump_ref would need `cparams.embeddings_nextn`).

## Speed

### Codex's measurement and the earlier estimate (corrected)

Codex timed `oxe prefill-bench` (pseudo-random prompt, after one warm-up rep): Ornith pp512
**179 / 206 / 209 tok/s at 8 / 10 / 12 threads**, pp4096 **151 tok/s at 8 threads**, i.e. at
parity with llama.cpp's ~185 pp512.

The "~650 tok/s" I wrote earlier was a reading error, not a different measurement: I mapped
`prefill_seconds` values to fixtures by output order, and the 3.1 s belonged to
`qwen36-pf64` (64 tokens), not to Ornith 2048. The actual `prefill-verify` timings (one
`Engine::prefill` call around the whole prompt, all ubatches, nice 19, 8 threads, first
touch of file-backed weights, scratch allocation included) were Ornith 2048: 14.7 s
(139 tok/s), Ornith 512: 2.95 s (174 tok/s), Qwen3.6 2048: 16.1 s (127 tok/s), consistent with
Codex's numbers. `prefill-bench` times only `Engine::prefill` (scratch allocated in the warm-up
rep, `reset()` outside the timer, no logits copy, no state dump).

### Where the time goes for a 256-token ubatch (by reasoning, Ornith)

Multiply-accumulates per token:

| Part | MAC / token | Kernel | Notes |
|---|---:|---|---|
| Dense projections (qkv, z, alpha/beta, ssm_out; wq/wk/wv/wo) | ~1.28 G | i8mm GEMM (`gemm<>`), 64-row units over all threads | quantised once per input (llama.cpp quantises per op) |
| Shared experts + router | ~0.15 G | GEMM (Q6_K) / f32 vec_dot | |
| Routed experts (8 x gate/up/down) | ~1.0 G | `gemm_mmid` 4-row tiles + GEMV remainders | ~8 rows per expert on average at ubatch 256 |
| Delta-net recurrence (30 layers x 32 heads x 3 passes over 128x128) | ~0.05 G | f32 vec_dot / mad, serial over tokens | split by value head (32 units) over threads |
| Attention (10 layers) | ~0.02 G at pp512, ~0.25 G at pp4096 | f16 vec_dot per (token, head, position) | decode kernels (exactness); grows with the prefix |
| Barriers | ~12 per layer, ~500 per ubatch | spin barrier, ~1 us | negligible |

Observations:
- **MoE remainders.** A 4-row tile decodes the expert weights once per 4 rows; a GEMV
  remainder row decodes them once per row. With ~8 routed rows per expert the cnt % 4 rows
  (~1.5 per expert on average) cost about as much as all the tiled rows. llama.cpp has the same
  cost (same path), which is consistent with parity.
- **Delta-net.** ~3 % of the MACs, but scalar-heavy and token-serial per head; 32 units over
  10-12 threads leaves a 3 vs 2.67 imbalance. Estimated 5-8 % of the ubatch time, more at
  higher thread counts. It is already parallel over all heads; a head cannot be split across
  threads without a barrier per token.
- **Attention** is why pp4096 (151) is below pp512 (179): the f16 vec_dot work per token grows
  with the position, ~15 % of the MACs at a 4K prefix, and vec_dot is far from GEMM efficiency.
- **Thread count**: compute bound, so 12 threads beat 8 (+17 %); prefill should run on its own
  12-thread team (`--oxe-prefill-cpus 0-11`) while decode keeps 8.

### Speedups that keep bit-exactness

1. **Pad MoE remainder rows into a tile** (branch `prefill-mmid-pad`: implemented and gated, all
   six fixtures still bit-identical; new kernel unit test for the padded tile). For
   every expert type of both models `gemm_mmid` row j is bitwise the GEMV of row j, so the
   cnt % 4 rows go through one more 4-row tile padded with copies of the last row, and only the
   real rows are kept. Expected: the MoE gate/up/down step ~35-45 % cheaper, roughly 15-20 % of
   the prefill (estimate; needs Codex's timing). Indication only (prefill-verify wall times,
   nice 19, 8 threads, not a benchmark): Ornith 2048 tokens 14.7 s -> 11.6 s, Qwen3.6 2048
   tokens 16.1 s -> 11.5 s.
2. **Split the biggest experts' gate/up units by columns** (implemented on the same branch:
   experts with >= 16 routed rows become two 256-column units). Exact (column split never
   changes results); gated with item 1.
3. **Run prefill on 12 threads** (thread-count invariant, already supported by the server flag).
4. **Several ubatches per pass, layer by layer.** Dense GEMM results depend only on whether a
   row sits in a 4-row group (all rows of full ubatches do) and MoE grouping is free (item 1), so
   processing e.g. 2 x 256 tokens through each layer together is exact if every token keeps its
   own ubatch's padded n_kv for attention and the delta-net runs the tokens in order. Halves
   barriers and per-step overheads and doubles rows per expert. Medium effort, modest gain.
5. Interleave the z projection with the delta-net recurrence (z is only needed in the gated
   norm). Small.

### Speedups that would need a KLD gate (not bit-exact vs llama.cpp)

- GEMM instead of GEMV for the final ubatch's N % 4 dense rows (Q4_K dense GEMM != GEMV).
- Tiled/flash attention or tinyBLAS-style f16 GEMM for K^T Q and V^T P (different accumulation).
- Chunked (WY / parallel-scan) gated delta net: parallel over tokens, different rounding.
- KleidiAI kernels for Q6_K / Q8_0 / dense Q4_K (the production build's numerics).

### Commands for Codex

`oxe prefill-bench` runs `--reps` timed prefills of a fixed pseudo-random prompt from
position 0 (after one warm-up rep, `reset()` between reps) and writes JSON with
`seconds`, `median_seconds`, `tok_s_median`, `first_token` (must be identical across arms of
one model) and memory.

`$MODELS` is the directory that holds the GGUF files.

```bash
cd <repo> && git pull && CARGO_BUILD_JOBS=2 cargo build --release
OR=$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf
QW=$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
L=${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock}
for M in OR QW; do for C in 0-9 0-11; do for P in 512 4096; do
  flock -n $L ./target/release/oxe prefill-bench --model ${!M} --cpu-list $C --prompt-len $P --ctx 8192 \
      --ubatch 256 --reps 3 --out results/prefill-bench-$M-pp$P-cpus$C.json
done; done; done
```

- `--cpu-list 0-9` matches llama-server's `--threads-batch 10`; `0-11` is all cores.
- Compare with llama-bench `-p 512` / `-p 4096 -ub 256 -b 256` on the patched build (the
  production numbers Codex already has: pp512 ~185 tok/s Ornith, `ornith-patched-final-prefill4096`).
- Exactness of a timed arm: `tools/run_prefill_verify.sh --cpu-list <same list>` (the result is
  thread-count invariant, so any list must pass).
- Server: `oxe-server --engine oxe --oxe-cpus 0,1,2,4,5,6,8,9 --oxe-prefill-cpus 0-11
  --ubatch-size 256 --model <gguf>` (prefill on its own 12-thread team, decode on the 8 spread
  cores).

## Remaining gaps

1. Speed is unmeasured (above). Obvious levers if it is short: fuse the norm/quantise steps into
   the preceding barrier phase, split the expert gate/up units by columns for the most popular
   experts, and overlap the delta-net recurrence (32 heads) with the shared-expert GEMMs.
2. Exactness is against the native-layout llama.cpp (KleidiAI off), like M1/M2. The production
   build (KleidiAI on) prefills Q6_K/Q8_0/dense Q4_K with different kernels.
3. The oracle branch is `claude/q3k-moe-gemm @ ee6b9a1`. `claude/profile-kernels` adds a Q3_K
   SMMLA kernel for 2-3 row expert remainders and a reordered gated-delta-net update; if the
   production build adopts those, the prefill reference changes and must be re-gated.
4. `prefill_ext` (MTP hidden states) is unit-tested only (no llama.cpp oracle for
   `embeddings_nextn` yet).
5. Prompts whose llama-server split differs from `head (n-1) + last token` (the server lane's
   checkpoint scheme) are exact per call, but llama-server itself batches the whole prompt; the
   server-level token identity vs llama-server depends on using the same split.
