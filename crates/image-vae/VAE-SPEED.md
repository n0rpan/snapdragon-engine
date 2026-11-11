# VAE decoder speed (flux2 decoder, 512 px output, 12 threads, random weights at the real channel layout)

## Update 21:15 (branch `image-vae-conv`, team VAE-CONV): channel-last decoder, fused upsample, Winograd, int8 without repacking

All numbers below are [M], ONE process, the five arms interleaved (warm-up pass, then best of 2), 12 threads, random weights,
with ALL THREE build locks held (BUILD-RULES 10). Other lanes' non-build work still ran (load average 5-9), so expect a few
percent of noise: four such runs gave 4.64-5.58 s for the unchanged 8531151 path. Fixture: `tests/prod_shape.rs`
(`prod_shape_decode_512`), kernels: `bench_nhwc_ab` in `src/decoder.rs`.

### 512 px decode, per stage (seconds)

| stage | 8531151 f32 (channel-major) | f32 channel-last | f32 channel-last + Winograd | 8531151 int8 | int8 channel-last |
|---|---:|---:|---:|---:|---:|
| unpatchify + post_quant + conv_in | 0.003 | 0.003 | 0.002 | 0.003 | 0.002 |
| mid resnet 1 | 0.057 | 0.052 | 0.023 | 0.025 | 0.015 |
| mid attention (4096 positions x 512) | 0.218 | 0.055 | 0.058 | 0.233 | 0.060 |
| mid resnet 2 | 0.052 | 0.052 | 0.023 | 0.025 | 0.015 |
| up level 0 resnets (64x64) | 0.165 | 0.161 | 0.069 | 0.074 | 0.044 |
| up level 0 upsample + conv | 0.127 | 0.038 | 0.042 | 0.055 | 0.012 |
| up level 1 resnets (128x128) | 0.741 | 0.546 | 0.290 | 0.339 | 0.180 |
| up level 1 upsample + conv | 0.520 | 0.149 | 0.177 | 0.265 | 0.050 |
| up level 2 resnets (256x256) | 0.982 | 0.661 | 0.411 | 0.564 | 0.264 |
| up level 2 upsample + conv | 0.544 | 0.154 | 0.159 | 0.331 | 0.055 |
| up level 3 resnets (512x512) | 1.179 | 0.683 | 0.370 | 0.834 | 0.331 |
| norm_out + conv_out | 0.057 | 0.015 | 0.017 | 0.060 | 0.017 |
| **total** | **4.64** | **2.57** | **1.64** | **2.81** | **1.04** |

The first f32 port was 20.67 s (IM-5 baseline). By kind of work, channel-last: direct f32 = convolution 2.42 s, GroupNorm+SiLU
0.11 s, attention core 0.04 s; Winograd = convolution 1.49 s; int8 = int8 convolution 0.80 s, norm+quantise 0.12 s, f32
convolutions (1x1, conv_in, conv_out) 0.08 s, attention core 0.05 s.

1024 px, one run each (same rules): f32 channel-last 12.15 s, Winograd 7.79 s, int8 5.36 s; the mid attention is 0.79-0.84 s of each.

### Distance to the 8531151 f32 decode (same random weights, whole 512 px output)

| arm | relative RMSE | max abs diff | PSNR as 8-bit pixels | bitwise |
|---|---:|---:|---:|---|
| f32 channel-last | 3.3e-6 | 1.5e-5 | 119.2 dB | no |
| f32 channel-last + Winograd | 4.0e-6 | 1.8e-5 | 117.7 dB | no |
| int8 channel-last (vs f32 channel-last) | 2.4% | | 42.0 dB | no (8531151 int8 vs its f32: 42.2 dB) |

Not bitwise, and why: the plain f32 and int8 convolutions ARE bitwise equal to the 8531151 kernels (unit tests
`nhwc_conv_is_bitwise_the_channel_major_gemm`, `nhwc_int8_conv_matches_channel_major_int8`). The differences come from
(1) GroupNorm statistics summed in another order (per-row f64 partial sums, `var = E[v^2] - mean^2`), (2) the upsample
convolutions using folded weights (sums of taps), (3) the attention as a GEMM with `exp_fast` instead of libm `exp`,
(4) for Winograd, the transforms. All of it is float rounding; against the f64 reference decoder the direct and the
Winograd decode both stay below the existing 1e-5 bound (`tests/decoder_reference.rs`). Every arm is deterministic and
independent of the thread count (tests).

### What changed (all in `src/decoder/nhwc.rs`; the 8531151 path is kept as `decode_timed_channel_major` for A/B)

1. Channel-last `[h][w][c]` maps from un-patchify to conv_out: no layout copy around any stage. The f32 convolution is an
   implicit GEMM straight on the map (no padded copy; border tiles read a 4-pixel patch), 4 pixels x 16 output channels in
   registers, input loaded as vectors and multiplied by lane (20 loads per 64 FMAs instead of 32), swept a few taps at a
   time along a row so the weights of a sweep sit in L1. 500-560 GMAC/s on one 38.7 GMAC conv (8531151 kernel in the same
   process: 290-440).
2. Nearest x2 upsample + 3x3 conv = four 2x2 convs on the small map (weights of coinciding taps summed once): 16 instead
   of 36 products per source pixel and no upsampled tensor. 256->256 from 256x256: 0.48 s -> 0.13 s (f32), 0.06 s (int8).
3. Winograd F(2x2, 3x3) for the plain 3x3 f32 convs (`set_winograd(true)`, `oxi --vae-conv winograd`; NOT the default):
   16 products per 2x2 outputs instead of 36, transforms in NEON, the 16 products as 1x1 GEMMs on the same tile kernel.
   Where V and M (4x the map) live decides everything: thread-local 1 MB blocks when the transformed weights fit L2
   (128 channels), map-wide super-bands ordered by transform position otherwise. One conv of 38.7 GMAC: 0.040-0.052 s
   against 0.074-0.080 s direct. First attempts that were SLOWER than direct, kept here as a warning: position planes at a
   pitch that is a multiple of 4 KB (16 streams in the same L1 sets), tile-major records (strided GEMM operand), and
   map-wide V/M at 128 channels (2 GB of traffic per conv).
4. GroupNorm: one statistics pass + one NEON affine/SiLU pass, in place where the input is dead; the residual add runs
   inside the convolution's row band.
5. Attention: two GEMMs per block of 128 queries on the conv tile kernel + vectorised softmax: 0.22 s -> 0.05 s.
6. int8: GroupNorm + SiLU + quantisation in one pass that writes int8 codes directly in the SMMLA tile layout, three
   copies shifted by one pixel, so every tap of every 8-pixel group is a contiguous operand; the conv calls the SMMLA
   tile per tap, no per-pixel repacking. 128->128 at 512x512 (the weak case): 0.095-0.13 s -> 0.043 s including the
   quantisation (880-900 GMAC/s); 512->512 at 128x128: 1500-1580 GMAC/s.
7. One set of parked worker threads per decode instead of scoped spawns per stage; large buffers reused through a pool.

### Open
- Real weights: nothing here has seen them. Ticket 058 (G6) measures f32 vs sd.cpp, Winograd vs f32 and int8 vs f32.
  Winograd becomes the default only after that (one line: `WINOGRAD` in `decoder.rs`).
- Direct f32 is at about 80% of the FMA roofline in the kernel bench; the rest of the f32 time is MACs.
- Not done: Winograd-style folding for the 2x2 upsample convs (9 instead of 16 products, about 0.15 s), GroupNorm
  statistics inside the producing convolution (about 0.05 s), 1024 px attention (0.8 s).

---

## Earlier state (image lead, commit 8531151)

All numbers are [M] on a shared machine in a DEV window with other lanes building (load average 5-8), so they are
indicative, not gate numbers. Fixture: `tests/prod_shape.rs` (`prod_shape_decode_512`), kernel A/B: `bench_conv_ab`.

## Whole decode, 512 px

| stage | f32 port (first version) | GEMM conv (current) |
|---|---:|---:|
| total | 20.67 s (load about 1) | 5.98 s (load about 7) |
| mid attention (4096 positions x 512) | 0.25 s | 0.24 s |
| up level at 64x64 (resnets + upsample conv) | | 0.48 s |
| up level at 128x128 | | 1.62 s |
| up level at 256x256 | | 1.93 s |
| up level at 512x512 | | 1.42 s |
| everything else | | 0.29 s |

The first version spent about 15 s in the 3x3 convolutions of the 128x128 to 512x512 levels (worker's measurement).
The decoder is about 1.22 TMAC of convolution at 512 px (arithmetic from the tensor shapes); at 1024 px four times that.

## Convolution kernels (one 3x3 conv of 38.7 GMAC, 12 threads, interleaved in one process)

| kernel | 512->512 at 128x128 | 256->256 at 256x256 | 128->128 at 512x512 | exactness |
|---|---:|---:|---:|---|
| reference loop (per output channel, axpy rows) | 0.58 s | 0.56-0.72 s | 0.71-0.78 s | definition |
| channel-major blocked tiles (4 output channels x 16 columns) | 1.17-1.20 s | 1.24-1.53 s | 1.27-1.28 s | bitwise equal to the reference |
| GEMM tiles (channel-last input, 4 pixels x 16 output channels, FMA) | 0.11-0.15 s | 0.12-0.13 s | 0.14 s | rel RMSE 0.8e-6 to 1.6e-6 vs reference |

GEMM tiles: 264-341 GMAC/s. The f32 roofline of this CPU is about 16 MAC per cycle per core (4 FMLA x 4 lanes),
about 650 GMAC/s on 12 cores at 3.42 GHz [E], so the kernel sits at 40-50% of it in a loaded window.

Decision: the GEMM tile kernel is `Conv::forward`. The channel-major blocked kernel is slower than the plain reference
loop (its 512 input planes are 512 separate streams per tile) and is kept only as a second implementation in tests.

## Update 20:00: vectorised GroupNorm + SiLU, and the int8 convolution path (flag, lossy)

| 512 px decode, 12 threads, random weights, load average about 10 | time |
|---|---:|
| f32 GEMM conv + vectorised GroupNorm/SiLU (default) | 4.98 s |
| int8 SMMLA conv for the heavy 3x3 convolutions (`set_int8_conv(true)`, `oxi --vae-conv int8`) | 3.00 s |

int8 against the f32 decode on the same random weights: relative RMSE 2.3% of the decoder output, **PSNR 42.2 dB** as 8-bit
pixels. Random weights say little about real images: the bound that counts is PSNR on real weights (hardware check:
`oxi vae-decode` on sd.cpp's dumped latent with both settings, then `oxi compare`).

One 38.7 GMAC conv, int8 path: 0.056 s at 512->512 / 128x128 (690 GMAC/s, twice the f32 tile), 0.07-0.10 s at
256->256 / 256x256, 0.11 s at 128->128 / 512x512 (355 GMAC/s); per-conv relative RMSE 0.37-0.49%.
The int8 path loses its lead at few channels and many pixels: the per-pixel work that is not MACs (channel-last copy,
quantisation, the 9-tap operand build, the copy back) is paid per pixel while the MACs per pixel shrink with the
channel count. Format: activations int8 with one scale per pixel per 64 channels, weights int8 with one scale per
output channel per 64 inputs of a tap; only 3x3 convs with at least 64 input channels take this path.

## What is left, in order of size
1. The int8 path exists (above) but its PSNR on real weights is unmeasured, so f32 stays the default.
2. About 1.5-2 s of the decode is not convolution MACs: GroupNorm statistics in f64, the channel-major <-> channel-last
   copies around every conv, SiLU, nearest upsampling, and allocation of 134 MB tensors at 512x512x128. Keeping the
   whole decoder channel-last removes the copies.
3. At 1024 px the mid attention has 16384 positions (16x the work of 512 px) and becomes visible again.
