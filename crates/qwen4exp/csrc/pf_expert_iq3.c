// pf_expert_iq3: 8-token tiles for the IQ3_XXS R4 gate / up rows of the batched expert op (see pf_expert.c).
//
// Exactness: ggml_vec_dot_iq3_xxs_q8_K / the R4 gemv / gemm4 compute per (row, token) [source + production objects]
//   per 256-block:  d = f32(d_w) * d_x;  s1 = s2 = 0
//     per 32-weight sub-block ib:  p = dot32 (exact int32);  scf = f32(scale) + 0.5
//                                  ib even: s1 = fma(f32(p), scf, s1)     ib odd: s2 = fma(f32(p), scf, s2)
//     sumf = fma(s1 + s2, d, sumf)
//   result = 0.5f * sumf
// The float steps are per sub-block here (unlike IQ2_S / Q6_K), so this tile keeps two float accumulators per
// element and folds them per block. Compiled with -ffp-contract=off; every fma below is explicit.
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "expert-r4.h"
#include "pf_kernels.h"

#include <arm_neon.h>
#include <string.h>

// one group of 4 IQ3_XXS R4 rows (nb block_iq3_xxs_r4) -> wc [ne00 / 8 steps][4 rows][8] signed codes,
// wscf [nb][8 sub-blocks][4] = f32(scale) + 0.5, wd [nb][4]
static void pf_iq3_r4_decode(const block_iq3_xxs_r4 * b, int64_t nb, int8_t * wc, float * wscf, float * wd) {
    const uint32_t * grid = iq3xxs_grid;
    for (int64_t i = 0; i < nb; ++i) {
        vst1q_f32(wd + i * 4, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b[i].d))));
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t * qi = b[i].qs + ib * 32;    // [row][8] grid indices
            const uint8_t * sas = b[i].sas + ib * 16;  // [row][4]: 7 sign bits | scale bit
            for (int r = 0; r < 4; ++r) {
                int sc = 0;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t v = sas[4 * r + l];
                    sc |= (v >> 7) << l;
                    const uint8_t s7 = v & 0x7f;
                    const uint8_t sg = (uint8_t) (s7 | ((__builtin_popcount(s7) & 1) << 7));  // 8th sign = parity of the other 7
                    // weights 8l .. 8l+7 of the sub-block = grid[index 2l] (4 bytes) | grid[index 2l + 1]
                    const uint64_t g64 = (uint64_t) grid[qi[8 * r + 2 * l]] | ((uint64_t) grid[qi[8 * r + 2 * l + 1]] << 32);
                    const uint8x8_t g = vcreate_u8(g64);
                    const uint8x8_t m = vtst_u8(vdup_n_u8(sg), vcreate_u8(0x8040201008040201ull));
                    vst1_u8((uint8_t *) wc + ((size_t) (i * 32 + ib * 4 + l) * 32 + r * 8), vsub_u8(veor_u8(g, m), m));
                }
                wscf[(i * 8 + ib) * 4 + r] = (float) sc + 0.5f;
            }
        }
    }
}

// 8 tokens x 4 rows over nb blocks: y[t * 4 + r] = 0.5 * fold described above.
// x [step][8 tokens][8], w [step][4 rows][8], wscf [nb][8][4], dx [nb][8], dw [nb][4]
static void pf_tile8x4_iq3(const int8_t * restrict x, const int8_t * restrict w, const float * restrict wscf, size_t nb, const float * dx,
                           const float * dw, float * y) {
    float32x4_t sumf[8];
    for (int i = 0; i < 8; i++) sumf[i] = vdupq_n_f32(0.0f);
    for (size_t b = 0; b < nb; b++) {
        float32x4_t s[2][8];
        for (int i = 0; i < 8; i++) s[0][i] = s[1][i] = vdupq_n_f32(0.0f);
        for (int ib = 0; ib < 8; ib++) {
            int32x4_t t0 = vdupq_n_s32(0), t1 = t0, t2 = t0, t3 = t0, t4 = t0, t5 = t0, t6 = t0, t7 = t0;
            for (int k = 0; k < 4; k++) {
                const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16), x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48);
                const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
                x += 64;
                w += 32;
                t0 = vmmlaq_s32(t0, x0, w0);
                t1 = vmmlaq_s32(t1, x0, w1);
                t2 = vmmlaq_s32(t2, x1, w0);
                t3 = vmmlaq_s32(t3, x1, w1);
                t4 = vmmlaq_s32(t4, x2, w0);
                t5 = vmmlaq_s32(t5, x2, w1);
                t6 = vmmlaq_s32(t6, x3, w0);
                t7 = vmmlaq_s32(t7, x3, w1);
            }
            // lanes of t[2p + q]: [tok 2p row 2q, tok 2p row 2q+1, tok 2p+1 row 2q, tok 2p+1 row 2q+1]: scale by row
            const float32x4_t sc = vld1q_f32(wscf);
            wscf += 4;
            const float32x4_t S0 = vcombine_f32(vget_low_f32(sc), vget_low_f32(sc));
            const float32x4_t S1 = vcombine_f32(vget_high_f32(sc), vget_high_f32(sc));
            float32x4_t * a = s[ib & 1];
            a[0] = vfmaq_f32(a[0], vcvtq_f32_s32(t0), S0);
            a[1] = vfmaq_f32(a[1], vcvtq_f32_s32(t1), S1);
            a[2] = vfmaq_f32(a[2], vcvtq_f32_s32(t2), S0);
            a[3] = vfmaq_f32(a[3], vcvtq_f32_s32(t3), S1);
            a[4] = vfmaq_f32(a[4], vcvtq_f32_s32(t4), S0);
            a[5] = vfmaq_f32(a[5], vcvtq_f32_s32(t5), S1);
            a[6] = vfmaq_f32(a[6], vcvtq_f32_s32(t6), S0);
            a[7] = vfmaq_f32(a[7], vcvtq_f32_s32(t7), S1);
        }
        const float32x2_t w01 = vld1_f32(dw), w23 = vld1_f32(dw + 2);
        const float32x4_t sw0 = vcombine_f32(w01, w01), sw1 = vcombine_f32(w23, w23);
        for (int p = 0; p < 4; p++) {
            const float32x2_t v = vld1_f32(dx + 2 * p);
            const float32x4_t sx = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));
            sumf[2 * p]     = vfmaq_f32(sumf[2 * p], vaddq_f32(s[0][2 * p], s[1][2 * p]), vmulq_f32(sw0, sx));
            sumf[2 * p + 1] = vfmaq_f32(sumf[2 * p + 1], vaddq_f32(s[0][2 * p + 1], s[1][2 * p + 1]), vmulq_f32(sw1, sx));
        }
        dx += 8;
        dw += 4;
    }
    for (int p = 0; p < 4; p++) {
        for (int q = 0; q < 2; q++) {
            const float32x4_t v = vmulq_n_f32(sumf[2 * p + q], 0.5f);
            vst1_f32(y + (2 * p) * 4 + 2 * q, vget_low_f32(v));
            vst1_f32(y + (2 * p + 1) * 4 + 2 * q, vget_high_f32(v));
        }
    }
}

size_t pf_expert_iq3_scratch(int64_t ne00, int64_t nrows) {
    const size_t g = (size_t) nrows / 4, nb = (size_t) ne00 / QK_K;
    return g * ((size_t) ne00 * 4 + nb * 8 * 4 * sizeof(float) + nb * 4 * sizeof(float)) + 64;
}

// As pf_expert_iq2s_rows, for an IQ3_XXS R4 matrix.
void pf_expert_iq3_rows(int64_t ne00, const void * vx, int64_t nrows, const int8_t * xc, const float * xd, int64_t ngroups,
                        float * const * s, void * scratch) {
    const int64_t nb = ne00 / QK_K, ng = nrows / 4;
    float * wd = (float *) scratch;
    float * wscf = wd + (size_t) ng * nb * 4;
    int8_t * wc = (int8_t *) (wscf + (size_t) ng * nb * 32);
    for (int64_t x = 0; x < ng; ++x) {
        pf_iq3_r4_decode((const block_iq3_xxs_r4 *) vx + x * nb, nb, wc + (size_t) x * ne00 * 4, wscf + (size_t) x * nb * 32, wd + (size_t) x * nb * 4);
    }
    for (int64_t g = 0; g < ngroups; ++g) {
        const int8_t * gx = xc + (size_t) g * ne00 * 8;
        const float * gd = xd + (size_t) g * nb * 8;
        float * const * gs = s + g * 8;
        for (int64_t x = 0; x < ng; ++x) {
            float tile[32];
            pf_tile8x4_iq3(gx, wc + (size_t) x * ne00 * 4, wscf + (size_t) x * nb * 32, (size_t) nb, gd, wd + (size_t) x * nb * 4, tile);
            for (int t = 0; t < 8; ++t) {
                if (gs[t]) vst1q_f32(gs[t] + 4 * x, vld1q_f32(tile + 4 * t));
            }
        }
    }
}
