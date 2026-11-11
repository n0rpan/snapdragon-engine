// pf_expert_down: 8-token tiles for the IQ4_NL down rows of the batched expert op (see pf_expert.c).
//
// Exactness: ggml_vec_dot_iq4_nl_q8_0 (NEON) computes per (row, token), two 32-weight blocks A, B per step
// [source + production object: fmul sB, scvtf x2, fmul sA, fmul f(iB) * sB, fmadd sA * f(iA) + that, fadd]:
//   sA = f32(dw_A) * f32(dx_A);  sB = f32(dw_B) * f32(dx_B);   iA, iB = exact int32 dots of the blocks
//   sumf = sumf + fma(sA, f32(iA), sB * f32(iB))          (GCC fuses the FIRST product of the source expression)
// (block count even: 640 / 32 = 20, no scalar tail; an odd count is not supported here). Activations are Q8_0 (f16 d, int8 qs[32]).
// Compiled with -ffp-contract=off; the one fma is explicit.
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "pf_kernels.h"

#include <arm_neon.h>
#include <string.h>

static inline float pf_h2f(ggml_half h) {  // exact f16 -> f32
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float) v;
}

// Pack up to 8 q8_0 rows of n values into one X group: xc [n / 8 steps][8 tokens][8], xd [n / 32][8].
void pf_expert_pack_x8_q8_0(int64_t n, int cnt, const void * const vy[8], int8_t * xc, float * xd) {
    const int64_t nb = n / 32;
    if (cnt < 8) {
        memset(xc, 0, (size_t) n * 8);
        memset(xd, 0, (size_t) nb * 8 * sizeof(float));
    }
    for (int j = 0; j < cnt; ++j) {
        const block_q8_0 * y = (const block_q8_0 *) vy[j];
        for (int64_t b = 0; b < nb; ++b) {
            xd[b * 8 + j] = pf_h2f(y[b].d);
            pf_interleave(xc + (size_t) b * 32 * 8, 8, j, y[b].qs, 32);
        }
    }
}

// 8 tokens x 4 rows, npair pairs of 32-weight chunks. x [step][8][8], w [step][4][8], dx [chunk][8], dw [chunk][4]
static void pf_tile8x4_pair(const int8_t * restrict x, const int8_t * restrict w, size_t npair, const float * dx, const float * dw, float * y) {
    float32x4_t acc[8];
    for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.0f);
    for (size_t pr = 0; pr < npair; pr++) {
        float32x4_t a[8];
        for (int half = 0; half < 2; half++) {
            int32x4_t t0 = vdupq_n_s32(0), t1 = t0, t2 = t0, t3 = t0, t4 = t0, t5 = t0, t6 = t0, t7 = t0;
            // block B (the second of the pair) first: its product is the plain multiply, A's is the fused one
            const int blk = 1 - half;
            const int8_t * xb = x + blk * 4 * 64;
            const int8_t * wb = w + blk * 4 * 32;
            const float * dxb = dx + blk * 8;
            const float * dwb = dw + blk * 4;
            for (int k = 0; k < 4; k++) {
                const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16), x2 = vld1q_s8(xb + 32), x3 = vld1q_s8(xb + 48);
                const int8x16_t w0 = vld1q_s8(wb), w1 = vld1q_s8(wb + 16);
                xb += 64;
                wb += 32;
                t0 = vmmlaq_s32(t0, x0, w0);
                t1 = vmmlaq_s32(t1, x0, w1);
                t2 = vmmlaq_s32(t2, x1, w0);
                t3 = vmmlaq_s32(t3, x1, w1);
                t4 = vmmlaq_s32(t4, x2, w0);
                t5 = vmmlaq_s32(t5, x2, w1);
                t6 = vmmlaq_s32(t6, x3, w0);
                t7 = vmmlaq_s32(t7, x3, w1);
            }
            const int32x4_t t[8] = {t0, t1, t2, t3, t4, t5, t6, t7};
            const float32x2_t w01 = vld1_f32(dwb), w23 = vld1_f32(dwb + 2);
            const float32x4_t sw0 = vcombine_f32(w01, w01), sw1 = vcombine_f32(w23, w23);
            for (int p = 0; p < 4; p++) {
                const float32x2_t v = vld1_f32(dxb + 2 * p);
                const float32x4_t sx = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));
                const float32x4_t s0 = vmulq_f32(sw0, sx), s1 = vmulq_f32(sw1, sx);
                if (half == 0) {
                    a[2 * p]     = vmulq_f32(s0, vcvtq_f32_s32(t[2 * p]));
                    a[2 * p + 1] = vmulq_f32(s1, vcvtq_f32_s32(t[2 * p + 1]));
                } else {
                    acc[2 * p]     = vaddq_f32(acc[2 * p], vfmaq_f32(a[2 * p], s0, vcvtq_f32_s32(t[2 * p])));
                    acc[2 * p + 1] = vaddq_f32(acc[2 * p + 1], vfmaq_f32(a[2 * p + 1], s1, vcvtq_f32_s32(t[2 * p + 1])));
                }
            }
        }
        x += 8 * 64;
        w += 8 * 32;
        dx += 16;
        dw += 8;
    }
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 2; q++) {
            vst1_f32(y + (2 * p) * 4 + 2 * q, vget_low_f32(acc[2 * p + q]));
            vst1_f32(y + (2 * p + 1) * 4 + 2 * q, vget_high_f32(acc[2 * p + q]));
        }
    }
}

size_t pf_expert_iq4nl_scratch(int64_t ne00, int64_t nrows) {
    const size_t g = (size_t) nrows / 4, nb = (size_t) ne00 / 32;
    return g * ((size_t) ne00 * 4 + nb * 4 * sizeof(float)) + (size_t) ne00 + 64;
}

// Rows [0, nrows) (nrows % 4 == 0) of a plain IQ4_NL matrix vx (rows of ne00 / 32 block_iq4_nl, ne00 % 64 == 0)
// against `ngroups` packed q8_0 X groups: s[8 * g + t][r] = W[r] . x[g][t] for every non-NULL s[8 * g + t].
void pf_expert_iq4nl_rows(int64_t ne00, const void * vx, int64_t nrows, const int8_t * xc, const float * xd, int64_t ngroups,
                          float * const * s, void * scratch) {
    const int64_t nb = ne00 / 32, ng = nrows / 4;
    float * wd = (float *) scratch;
    int8_t * wc = (int8_t *) (wd + (size_t) ng * nb * 4);
    int8_t * row = wc + (size_t) ng * ne00 * 4;
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const block_iq4_nl * b = (const block_iq4_nl *) vx;
    for (int64_t x = 0; x < ng; ++x) {
        for (int r = 0; r < 4; ++r, b += nb) {
            for (int64_t i = 0; i < nb; ++i) {
                wd[(x * nb + i) * 4 + r] = pf_h2f(b[i].d);
                const uint8x16_t q = vld1q_u8(b[i].qs);
                vst1q_s8(row + i * 32, vqtbl1q_s8(values, vandq_u8(q, m4b)));
                vst1q_s8(row + i * 32 + 16, vqtbl1q_s8(values, vshrq_n_u8(q, 4)));
            }
            pf_interleave(wc + (size_t) x * ne00 * 4, 4, r, row, (size_t) ne00);
        }
    }
    for (int64_t g = 0; g < ngroups; ++g) {
        const int8_t * gx = xc + (size_t) g * ne00 * 8;
        const float * gd = xd + (size_t) g * nb * 8;
        float * const * gs = s + g * 8;
        for (int64_t x = 0; x < ng; ++x) {
            float tile[32];
            pf_tile8x4_pair(gx, wc + (size_t) x * ne00 * 4, (size_t) nb / 2, gd, wd + (size_t) x * nb * 4, tile);
            for (int t = 0; t < 8; ++t) {
                if (gs[t]) vst1q_f32(gs[t] + 4 * x, vld1q_f32(tile + 4 * t));
            }
        }
    }
}
