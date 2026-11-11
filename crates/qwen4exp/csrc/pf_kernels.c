// pf_kernels: multi-column int8 SMMLA tiles for the Flash prefill.
//
// Why: production ggml (ARM, i8mm, no repack, the tiled GEMM is x86-only) computes a batched Q6_K or
// Q8_0 mul_mat with 2x2 SMMLA vec_dot calls: every (row pair, token pair) re-reads, and for Q6_K
// re-decodes, the weight blocks. Here a weight row block is decoded ONCE into int8 codes and then
// multiplied with every token of the ubatch in 8-token x 4-row tiles (8 SMMLA per 8 k, 256 MACs).
//
// Exactness. Production's kernels (arch/arm/quants.c, objects of libggml-cpu.a adf384be, disassembled):
//   Q6_K nrc=2:  per 256-block: visum = sum_s scale[s] * dot16(q6, q8) - 32 * sum_s scale[s] * bsum[s]   (int32)
//                vfsum = vfsum + f32(visum) * (f32(dw) * dx)                    fmul, fadd (NOT fused)
//   Q6_K nrc=1:  sum   = fma(f32(dw) * dx, f32(isum - 32 * isum_mins), sum)     fmadd (fused)
//   Q8_0 nrc=2:  per 32-block: sumv = sumv + f32(dot32) * (f32(dw) * f32(dx))   fmul, fadd (NOT fused)
// i.e. per output element: an exact integer per chunk (256 for Q6_K, 32 for Q8_0), one f32 scale
// product, and a left fold over the chunks from +0.0. The integer is exact (mod 2^32), so how it is
// formed is free: codes are stored as q6 - 32, which folds the bsum term in exactly, because
// quantize_row_q8_K sets bsums[s] to the exact sum of the 16 codes. The fold order and the
// fused / not-fused choice are reproduced by `fused`. => bitwise equal to ggml's mul_mat.
// (Shares the idea and the fold contract with crates/int8-gemm of the image lane: chunk scales,
// integer part order-free, fixed float fold. Differences proposed for upstreaming there: the fused
// two-level tile, and the 8x4 shape, which keeps every accumulator in a register.)
//
// Compiled with -ffp-contract=off: a multiply and an add fuse only where the code says vfmaq / fmaf.
//
// Layouts. A group of g vectors of K codes is step-major: step s (8 consecutive k) holds g * 8 bytes,
// vector j at bytes [8j, 8j + 8). X groups have 8 vectors (4 registers per step, each a vector PAIR),
// W groups have 4 (2 registers). smmla(acc, X pair p, W pair q) adds
//   lane0 += x[2p] . w[2q]     lane1 += x[2p] . w[2q+1]     lane2 += x[2p+1] . w[2q]     lane3 += x[2p+1] . w[2q+1]
#include "pf_kernels.h"

#include <arm_neon.h>
#include <math.h>
#include <string.h>

void pf_interleave(int8_t * dst, int g, int j, const int8_t * src, size_t n) {
    const size_t stride = (size_t) g * 8;
    int8_t * o = dst + (size_t) j * 8;
    for (size_t s = 0; s < n / 8; s++) {
        memcpy(o + s * stride, src + s * 8, 8);
    }
}

void pf_q6k_decode_row(const void * blocks, size_t nblk, int8_t * codes, int16_t * scales, ptrdiff_t sstride, float * d, ptrdiff_t dstride) {
    // block_q6_K: ql[128] (low 4 bits), qh[64] (high 2 bits), int8 scales[16], f16 d  = 210 bytes
    const uint8_t * p = (const uint8_t *) blocks;
    for (size_t b = 0; b < nblk; b++, p += 210) {
        const uint8_t * ql = p;
        const uint8_t * qh = p + 128;
        const int8_t * sc = (const int8_t *) (p + 192);
        __fp16 h;
        memcpy(&h, p + 208, 2);
        d[(ptrdiff_t) b * dstride] = (float) h;
        for (int s = 0; s < 16; s++) {
            scales[((ptrdiff_t) b * 16 + s) * sstride] = (int16_t) sc[s];
        }
        int8_t * y = codes + b * 256;
        for (int n = 0; n < 2; n++, y += 128, ql += 64, qh += 32) {
            for (int l = 0; l < 32; l++) {
                y[l]      = (int8_t) (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                y[l + 32] = (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                y[l + 64] = (int8_t) (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                y[l + 96] = (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
        }
    }
}

#define PF_INLINE static inline __attribute__((always_inline))

// one chunk's float fold: f[2p+q] (+)= f32(I[2p+q]) * (dx pair p (x) dw pair q)
PF_INLINE void pf_fold(float32x4_t f[8], const int32x4_t I[8], const float * dx, const float * dw, int fused) {
    const float32x2_t w01 = vld1_f32(dw), w23 = vld1_f32(dw + 2);
    const float32x4_t sw0 = vcombine_f32(w01, w01);  // [a, b, a, b]
    const float32x4_t sw1 = vcombine_f32(w23, w23);
    for (int p = 0; p < 4; p++) {
        const float32x2_t v = vld1_f32(dx + 2 * p);
        const float32x4_t sx = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a, a, b, b]
        const float32x4_t s0 = vmulq_f32(sx, sw0);
        const float32x4_t s1 = vmulq_f32(sx, sw1);
        if (fused) {
            f[2 * p]     = vfmaq_f32(f[2 * p], vcvtq_f32_s32(I[2 * p]), s0);
            f[2 * p + 1] = vfmaq_f32(f[2 * p + 1], vcvtq_f32_s32(I[2 * p + 1]), s1);
        } else {
            f[2 * p]     = vaddq_f32(f[2 * p], vmulq_f32(vcvtq_f32_s32(I[2 * p]), s0));
            f[2 * p + 1] = vaddq_f32(f[2 * p + 1], vmulq_f32(vcvtq_f32_s32(I[2 * p + 1]), s1));
        }
    }
}

PF_INLINE void pf_store(const float32x4_t f[8], float * y, ptrdiff_t ystride) {
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 2; q++) {
            vst1_f32(y + (2 * p) * ystride + 2 * q, vget_low_f32(f[2 * p + q]));
            vst1_f32(y + (2 * p + 1) * ystride + 2 * q, vget_high_f32(f[2 * p + q]));
        }
    }
}

// integer dots of `steps` k-steps into t[2p+q] (8 SMMLA per step against 6 loads)
#define PF_STEPS(t, x, w, steps)                                                                             \
    for (size_t k_ = 0; k_ < (steps); k_++) {                                                                \
        const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16), x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48); \
        const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);                                             \
        x += 64;                                                                                             \
        w += 32;                                                                                             \
        t[0] = vmmlaq_s32(t[0], x0, w0);                                                                     \
        t[1] = vmmlaq_s32(t[1], x0, w1);                                                                     \
        t[2] = vmmlaq_s32(t[2], x1, w0);                                                                     \
        t[3] = vmmlaq_s32(t[3], x1, w1);                                                                     \
        t[4] = vmmlaq_s32(t[4], x2, w0);                                                                     \
        t[5] = vmmlaq_s32(t[5], x2, w1);                                                                     \
        t[6] = vmmlaq_s32(t[6], x3, w0);                                                                     \
        t[7] = vmmlaq_s32(t[7], x3, w1);                                                                     \
    }

PF_INLINE void pf_two_core(const int8_t * restrict x, const int8_t * restrict w, const int16_t * restrict wsc, size_t nchunk, size_t nsub,
                           size_t sub_steps, const float * dx, const float * dw, float * y, ptrdiff_t ystride, int fused) {
    float32x4_t f[8];
    for (int i = 0; i < 8; i++) f[i] = vdupq_n_f32(0.0f);
    for (size_t c = 0; c < nchunk; c++) {
        int32x4_t I[8];
        for (int i = 0; i < 8; i++) I[i] = vdupq_n_s32(0);
        for (size_t s = 0; s < nsub; s++) {
            int32x4_t t[8];
            for (int i = 0; i < 8; i++) t[i] = vdupq_n_s32(0);
            PF_STEPS(t, x, w, sub_steps)
            const int32x4_t sc = vmovl_s16(vld1_s16(wsc));  // [s0, s1, s2, s3]
            wsc += 4;
            const int32x4_t S0 = vcombine_s32(vget_low_s32(sc), vget_low_s32(sc));    // rows 0, 1
            const int32x4_t S1 = vcombine_s32(vget_high_s32(sc), vget_high_s32(sc));  // rows 2, 3
            I[0] = vmlaq_s32(I[0], t[0], S0);
            I[1] = vmlaq_s32(I[1], t[1], S1);
            I[2] = vmlaq_s32(I[2], t[2], S0);
            I[3] = vmlaq_s32(I[3], t[3], S1);
            I[4] = vmlaq_s32(I[4], t[4], S0);
            I[5] = vmlaq_s32(I[5], t[5], S1);
            I[6] = vmlaq_s32(I[6], t[6], S0);
            I[7] = vmlaq_s32(I[7], t[7], S1);
        }
        pf_fold(f, I, dx, dw, fused);
        dx += 8;
        dw += 4;
    }
    pf_store(f, y, ystride);
}

void pf_tile8x4_two(const int8_t * x, const int8_t * w, const int16_t * wsc, size_t nchunk, size_t nsub, size_t sub_steps,
                    const float * dx, const float * dw, float * y, ptrdiff_t ystride, int fused) {
    // constant-propagated copies for the shapes in use (Q6_K: 16 sub-blocks of 2 steps)
    if (nsub == 16 && sub_steps == 2) {
        if (fused) pf_two_core(x, w, wsc, nchunk, 16, 2, dx, dw, y, ystride, 1);
        else       pf_two_core(x, w, wsc, nchunk, 16, 2, dx, dw, y, ystride, 0);
    } else if (nsub == 8 && sub_steps == 4) {
        if (fused) pf_two_core(x, w, wsc, nchunk, 8, 4, dx, dw, y, ystride, 1);
        else       pf_two_core(x, w, wsc, nchunk, 8, 4, dx, dw, y, ystride, 0);
    } else {
        pf_two_core(x, w, wsc, nchunk, nsub, sub_steps, dx, dw, y, ystride, fused);
    }
}

PF_INLINE void pf_one_core(const int8_t * restrict x, const int8_t * restrict w, size_t nchunk, size_t steps, const float * dx,
                           const float * dw, float * y, ptrdiff_t ystride, int fused) {
    float32x4_t f[8];
    for (int i = 0; i < 8; i++) f[i] = vdupq_n_f32(0.0f);
    for (size_t c = 0; c < nchunk; c++) {
        int32x4_t t[8];
        for (int i = 0; i < 8; i++) t[i] = vdupq_n_s32(0);
        PF_STEPS(t, x, w, steps)
        pf_fold(f, t, dx, dw, fused);
        dx += 8;
        dw += 4;
    }
    pf_store(f, y, ystride);
}

void pf_tile8x4_one(const int8_t * x, const int8_t * w, size_t nchunk, size_t steps, const float * dx, const float * dw, float * y,
                    ptrdiff_t ystride, int fused) {
    if (steps == 4) {
        if (fused) pf_one_core(x, w, nchunk, 4, dx, dw, y, ystride, 1);
        else       pf_one_core(x, w, nchunk, 4, dx, dw, y, ystride, 0);
    } else {
        pf_one_core(x, w, nchunk, steps, dx, dw, y, ystride, fused);
    }
}

// ---- scalar definitions ----

static inline float pf_fold1(float acc, int32_t isum, float sx, float sw, int fused) {
    const float s = sx * sw;
    const float f = (float) isum;
    return fused ? fmaf(f, s, acc) : acc + f * s;
}

void pf_tile8x4_two_ref(const int8_t * x, const int8_t * w, const int16_t * wsc, size_t nchunk, size_t nsub, size_t sub_steps,
                        const float * dx, const float * dw, float * y, ptrdiff_t ystride, int fused) {
    const size_t sub = sub_steps * 8;
    for (int t = 0; t < 8; t++) {
        for (int r = 0; r < 4; r++) {
            float acc = 0.0f;
            for (size_t c = 0; c < nchunk; c++) {
                uint32_t isum = 0;
                for (size_t s = 0; s < nsub; s++) {
                    uint32_t part = 0;
                    for (size_t i = 0; i < sub; i++) {
                        const size_t k = (c * nsub + s) * sub + i;
                        const int32_t xv = x[(k / 8) * 64 + t * 8 + k % 8];
                        const int32_t wv = w[(k / 8) * 32 + r * 8 + k % 8];
                        part += (uint32_t) (xv * wv);
                    }
                    isum += part * (uint32_t) (int32_t) wsc[(c * nsub + s) * 4 + r];
                }
                acc = pf_fold1(acc, (int32_t) isum, dx[c * 8 + t], dw[c * 4 + r], fused);
            }
            y[t * ystride + r] = acc;
        }
    }
}

void pf_tile8x4_one_ref(const int8_t * x, const int8_t * w, size_t nchunk, size_t steps, const float * dx, const float * dw, float * y,
                        ptrdiff_t ystride, int fused) {
    const size_t chunk = steps * 8;
    for (int t = 0; t < 8; t++) {
        for (int r = 0; r < 4; r++) {
            float acc = 0.0f;
            for (size_t c = 0; c < nchunk; c++) {
                uint32_t isum = 0;
                for (size_t i = 0; i < chunk; i++) {
                    const size_t k = c * chunk + i;
                    const int32_t xv = x[(k / 8) * 64 + t * 8 + k % 8];
                    const int32_t wv = w[(k / 8) * 32 + r * 8 + k % 8];
                    isum += (uint32_t) (xv * wv);
                }
                acc = pf_fold1(acc, (int32_t) isum, dx[c * 8 + t], dw[c * 4 + r], fused);
            }
            y[t * ystride + r] = acc;
        }
    }
}
