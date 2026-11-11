# Image engine (`oxi`): design and M0 plan

Purpose-built image engine for this laptop (Snapdragon X Elite, 12 Oryon cores, i8mm), first model
FLUX.2 klein 4B at 4 steps, CFG 1. Lives in the engine repo next to `oxe` and shares `gguf`,
`sched` and the new `int8-gemm`. Tags: [M] measured, [E] estimate/arithmetic, [D] design decision.

## 1. Where the time goes (arithmetic from the GGUF header)

klein 4B [M, header]: hidden 3072, 24 heads x 128, 5 double + 20 single blocks, all linears Q8_0, no biases.
Single block: `linear1` 3072 -> 27648 (q, k, v and a 2 x 9216 SiLU-gated MLP), `linear2` 12288 -> 3072.
Every block (single or double) costs 122.7 M MAC per token in linears [E, from shapes].
Sequence = 512 text tokens + image tokens: 1536 (512 px), 2560 (512 px edit), 4608 (1024 px).

| per block             | 1536 tokens | 4608 tokens |
|-----------------------|------------:|------------:|
| linear MAC            | 188 G       | 565 G       |
| attention MAC (QK+PV) | 14 G        | 130 G       |
| linear1 output as f32 | 170 MB      | 510 MB      |

sd.cpp reaches about 0.9 TOPS on this [M, earlier report; its real per-block time comes from ticket 030].
CORRECTION 2026-11-10: the "about 10 TOPS int8 SMMLA peak" this document started from is wrong for this CPU kernel.
Measured [M, `int8-gemm` test `kernel_peak_single_core`, one core at 3.42 GHz, L1- or L2-resident operands]: the 8x8
SMMLA tile runs 207 GMAC/s = 0.41 TOPS per core (about 1.9 SMMLA per cycle), 159 GMAC/s at chunk 32, 128 GMAC/s at the
attention q.k shape (k = 128). So the CPU roofline of this kernel is about 4.9 TOPS on 12 cores (12 x 0.41), not 10.
An SDOT tile does not change this [M, test `kernel_peak_sdot_vs_smmla`]: SDOT runs 13.1 per ns at 16 MAC each =
209 GMAC/s, SMMLA 6.7 per ns at 32 MAC each = 216 GMAC/s (integer kernels only): both are about 64 int8 MAC per cycle,
which is the per-core limit of this CPU for int8 products. (About 10 TOPS is what was measured on the NPU for resident matmuls, not on the CPU.)
A block at 4608 tokens is 1.39 T ops: about 1.5 s at 0.9 TOPS, about 0.28 s at the 4.9 TOPS roofline [E].
Indicative, not a gate number [M, DEV window, shared machine, random weights]: the block prototype reaches 3.0-3.4
effective TOPS (0.12 s at 1536 tokens, 0.44-0.46 s at 4608), i.e. 60-70% of the roofline in a noisy window.
Ticket 013 measures sd.cpp's real per-block wall time and the perf split; that is the gate's denominator.
SECOND CORRECTION 2026-11-10 (team DiT-LINEAR) [M, all build locks held, `kernel_peak_all_cores` and `prod_shape_model_linears`]:
12 x one core is not what 12 cores give for longer than a few seconds. The same L1-resident tile kernel on all 12 cores at
once runs 4.6-4.7 TOPS on a cool machine (chunk 256; the chunk fix-up alone costs 5%: 198 against 207 GMAC/s per core) and
3.5-3.6 TOPS after about 30 s of load at 97 C (throttling that cpufreq does not show). An image is 14 s or more of DiT, so
the SUSTAINED CPU roofline is about 3.5 TOPS, and the model path's linears now run 3.2-3.6 TOPS in that state (4.06 cool):
at the sustained kernel rate within a few percent. What got them there was the split, not the kernel: a static per-thread
plan left 16-25% of the thread time waiting at barriers (slowest thread 1.14-1.37x the mean); work sharing (`Share` in
fast_model.rs: fixed steps, claimed by whichever thread is free, helpers finish the last items) brought that to 0.4-1% and
the linears from 2.75-3.02 to 3.3-3.6 TOPS (1536 tokens) and 2.56-2.75 to 3.2-3.3 (4608), same output bit for bit.
Measured and NOT worth it at these shapes: streaming W from L2 (no cost: 199.8 against 197.7 GMAC/s), merging units into
384-row passes, 128-row linear2 blocks, batch 2 or 8, software prefetch of the next token batch (all equal or slower),
NEON sinks (bitwise equal, no measurable gain; kept because they are simpler than the loops they replace), per-block
buffer reuse (allocation is 0.6-2 ms per forward). Attention q.k^T at k = 128: a row kernel without the per-tile set-up
runs 195 against 151 GMAC/s per core, attention stage +7-10%.

## 2. Why sd.cpp/ggml is slow here, and what the engine does instead [D]

1. **No cache blocking over tokens.** ggml's repack GEMM takes 4 tokens at a time through the whole
   weight matrix (85 MB for linear1), so the matrix is re-read from DRAM about 1150 times per block.
   Here a thread keeps a weight block in its L2 share and walks all tokens past it; one token group
   (8 tokens x K bytes) is the L1 operand.
2. **A float fix-up every 32 MACs** (Q8_0 scale per 32 on both sides). Here the scale granularity is a
   parameter, the *chunk*: 32 reproduces Q8_0 exactly, larger chunks do one fix-up per chunk.
   An 8x8 tile does 16 SMMLA per 8 loads; a fix-up costs about 8 uops per accumulator, so chunk 32
   roughly doubles the instruction count, chunk 256 adds about 15% [E].
3. **Huge f32 intermediates.** linear1's output (510 MB at 4608 tokens) is written, sliced, copied and
   re-quantised. Here it never exists: the weight rows are re-ordered at load into units that are
   complete for their consumer (one head of q/k/v, or a block of gate rows with their partner rows),
   and each unit's f32 tile is consumed in cache (norm + RoPE + int8; SiLU gate + int8).
4. **f32 attention.** Here both products are int8: q.k^T by SMMLA; probabilities, pre-multiplied by the
   per-key value scale, as u8 per (query, key chunk), times int8 values by USMMLA, flash-style with a
   running max. A scale per key chunk keeps the tail of the distribution that one scale per row would
   round to zero.
5. **Dynamic OpenMP scheduling.** Here: `sched::Pool` (pinned, spin barrier), five parallel stages per
   block, each thread's work list fixed at load.

## 3. Number format and the precision question [D, to be settled by measurement]

Optimal uniform 8-bit quantisation of a Gaussian vector has about 0.9-1.0% relative RMSE; a max-based
scale over n values has about 0.48% (n = 32), 0.61% (128), 0.68% (256), 0.89% (3072) for a clean
Gaussian, and much worse with outlier channels [E]. Two independent sides add in quadrature, and a
block chains linear1 -> attention/MLP -> linear2. So a pure per-row x per-token int8 format cannot be
expected to meet a 1% bound on the block's delta; chunk scales can trade speed for precision without
changing the kernel. sd.cpp itself (Q8_0 x q8_0) sits at about 0.5% per linear by the same arithmetic.
M0 therefore measures speed AND error over chunk in {32, 64, 128, 256, 512, 1024} (and per-token for
linear1) on real activations, and reports the error on the block's delta `out - x`, not on `out`
(where the residual hides it).

## 4. Shared crate `crates/int8-gemm`

See the crate docs (`src/lib.rs`) for the layout and the float contract. In short:
- `Packed`: int8 vectors in 8x8-tile layout with one f32 scale per (vector, chunk); constructors from
  f32 (quantise), from codes, from ggml Q8_0 (verbatim at chunk 32, re-quantised otherwise, optional
  row permutation); `quantize_unsync` for static plans that fill disjoint parts from several threads.
- `gemm` / `gemm_blocked`: (token groups) x (row groups) -> f32, single-threaded; threading and block
  sizes are the caller's static plan. Fixed accumulation order: per output element, chunks ascending,
  `acc + f32(isum) * (sx * sw)` (or one FMA), independent of tile shape and thread count; `gemm_ref`
  is the scalar definition and the SMMLA path is tested bitwise against it.
- **K-quant block scales (for the Flash prefill, requested by the Flash lead):** `Packed.sub` adds an
  integer scale per 16 (or 32) weights under the f32 scale per 256 super-block; the integer part
  `sum_j sub_scale_j * isum_j` is exact, then the same float contract applies per super-block.
  Implemented correct-first (per-sub-block kernel calls); a fused kernel follows if ticket 011 shows
  dense matmuls dominate. Not yet covered: the additive `min` term of Q4_K/Q5_K (needs activation
  block sums; Q6_K, Q3_K and Q8_0 need none), and bit-identity with ggml's NEON `vec_dot_q8_0_q8_0`
  (which adds per-lane partial sums in float: a different order, needs its own kernel on the same
  packed data).
- Kernels are C with NEON intrinsics (`csrc/i8g_kernels.c`), built with `-ffp-contract=off`.

## 5. Backend boundary and the NPU [D]

`image_dit::backend`: a backend owns a window of consecutive blocks with weights in its own format and
maps f32 token matrices to f32 token matrices. Weights are handed over as file byte ranges. CPU keeps
all 25 blocks resident (int8, about 3.2 GB plus scales). An NPU backend maps windows that fit the
measured 1 GiB session budget (about 7 blocks of 130 MB) and pays the dispatch floor once per window,
not per op; its natural format is per-row int8 weights with 16-bit activations, which is why the
boundary is float and the CPU's chunk-scale format stays private to the CPU backend. Per-row weight
scales (chunk = K) are the HMX-friendly special case of the same packer. Nothing NPU is touched in M0.

## 6. M0 gate

GO: >= 2.0x sd.cpp's own block time at 4096 image tokens and >= 1.7x at 1024, with relative RMSE of
the block delta against the f32 reference on dumped real activations <= max(1%, about 2x sd.cpp's own
error against the same reference on the same activations) (decision 2026-11-10: never stricter than the
reference engine's own numeric noise). The image-level M1 gate (PSNR >= 30 dB vs sd.cpp on 5 seeds plus
an eyeball check) is the final word. 1.4-2.0x: patch the kernel
into sd.cpp instead. Below 1.4x: stop. Tickets: 013 (sd.cpp baseline + dumps), 015 (prototype).
