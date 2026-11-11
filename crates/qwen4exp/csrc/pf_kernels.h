// pf_kernels: multi-column int8 SMMLA tiles for the Flash prefill (dense Q6_K / Q8_0 matmuls).
// See pf_kernels.c for the layouts and the float contract. Plain C API, no ggml headers.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tile geometry: 8 activation vectors ("tokens") x 4 weight vectors ("rows"), k consumed 8 at a time.
enum { PF_XG = 8, PF_WG = 4, PF_KS = 8 };

// Copy one vector of n int8 codes (n % 8 == 0) into slot `j` of an interleaved group of `g` vectors:
// piece s (8 codes) goes to dst[s * 8 * g + 8 * j ..].
void pf_interleave(int8_t * dst, int g, int j, const int8_t * src, size_t n);

// Decode `nblk` Q6_K blocks (210 bytes each, production layout) of one row:
//   codes[256 * b + i] = q6(i) - 32          (int8, -32..31)
//   scales[16 * b + s] = block.scales[s]     (int16)
//   d[b * dstride]     = f32(block.d)        (exact f16 -> f32)
void pf_q6k_decode_row(const void * blocks, size_t nblk, int8_t * codes, int16_t * scales, ptrdiff_t sstride, float * d, ptrdiff_t dstride);

// Two-level tile (K-quant shape): chunk = nsub sub-blocks of sub_steps*8 codes, one int16 scale per
// (row, sub-block), one f32 scale per (vector, chunk).
//   isum[t][r]  = sum_{s < nsub} wsc[c][s][r] * dot(x[t], w[r] over sub-block s)      int32, wrapping
//   y[t][r]     = fold over chunks c ascending, from +0.0:
//                   fused == 0: acc = acc + f32(isum) * (dx[c][t] * dw[c][r])
//                   fused != 0: acc = fma(f32(isum), dx[c][t] * dw[c][r], acc)
// x:   [chunk][sub][step] 64 bytes (8 tokens x 8 codes)      w: [chunk][sub][step] 32 bytes (4 rows x 8 codes)
// wsc: [chunk][sub][4] int16     dx: [chunk][8] f32     dw: [chunk][4] f32     y[t * ystride + r]
void pf_tile8x4_two(const int8_t * x, const int8_t * w, const int16_t * wsc, size_t nchunk, size_t nsub, size_t sub_steps,
                    const float * dx, const float * dw, float * y, ptrdiff_t ystride, int fused);

// One-level tile (Q8_0 shape): chunk = steps*8 codes, one f32 scale per (vector, chunk); same fold.
void pf_tile8x4_one(const int8_t * x, const int8_t * w, size_t nchunk, size_t steps, const float * dx, const float * dw, float * y,
                    ptrdiff_t ystride, int fused);

// Scalar definitions of the two tiles (same arguments, same bits): the contract and the test oracle.
void pf_tile8x4_two_ref(const int8_t * x, const int8_t * w, const int16_t * wsc, size_t nchunk, size_t nsub, size_t sub_steps,
                        const float * dx, const float * dw, float * y, ptrdiff_t ystride, int fused);
void pf_tile8x4_one_ref(const int8_t * x, const int8_t * w, size_t nchunk, size_t steps, const float * dx, const float * dw, float * y,
                        ptrdiff_t ystride, int fused);

#ifdef __cplusplus
}
#endif
