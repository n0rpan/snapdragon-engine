// pf_expert: 8-token tiles for the batched expert op of the prefill (gate / up of the IQ2_S R4 layers).
//
// Today (Stream::run_batch): ggml_cpu_expert_r4_gemm4 decodes every 32-weight sub-block of 4 rows again for
// every group of 4 tokens. Here a row chunk of one expert is decoded ONCE into int8 codes (+-{8, 25, 43}) with
// int16 sub-block scales (1 + 2 * nibble, per 16 weights) and multiplied with all of the expert's tokens in
// 8-token x 4-row SMMLA tiles (pf_kernels.c, two-level tile).
//
// Exactness: ggml_vec_dot_iq2_s_q8_K / the R4 gemv / gemm4 compute per (row, token)
//   per 256-block: sumi = sum over 16-weight halves of dot16 * (1 + 2 * nibble)    exact int32
//                  sumf = fma(f32(sumi), f32(d_w) * d_x, sumf)                       one fmadd per block
//   result = 0.125f * sumf
// = the two-level tile with fused = 1, then one multiply. Bitwise equal (test: ffi.rs pf_expert_rows_bitwise).
// IQ3_XXS folds in float per 32-weight sub-block (two interleaved accumulators): its own tile, pf_expert_iq3.c.
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "expert-r4.h"
#include "pf_kernels.h"

#include <arm_neon.h>
#include <string.h>

// Pack up to 8 q8_K rows (block_q8_K: f32 d, int8 qs[256], int16 bsums[16]) into one X group:
// xc [ne00 / 8 steps][8 tokens][8], xd [ne00 / 256][8]. Absent tokens (n < 8) are zero codes with zero scales.
void pf_expert_pack_x8(int64_t ne00, int n, const void * const vy[8], int8_t * xc, float * xd) {
    const int64_t nb = ne00 / QK_K;
    if (n < 8) {
        memset(xc, 0, (size_t) ne00 * 8);
        memset(xd, 0, (size_t) nb * 8 * sizeof(float));
    }
    for (int j = 0; j < n; ++j) {
        const block_q8_K * y = (const block_q8_K *) vy[j];
        for (int64_t b = 0; b < nb; ++b) {
            xd[b * 8 + j] = y[b].d;
            pf_interleave(xc + (size_t) b * QK_K * 8, 8, j, y[b].qs, QK_K);
        }
    }
}

// one group of 4 IQ2_S R4 rows (nb block_iq2_s_r4) -> wc [ne00 / 8 steps][4 rows][8], wsc [nb][16][4], wd [nb][4]
static void pf_iq2s_r4_decode(const block_iq2_s_r4 * b, int64_t nb, int8_t * wc, int16_t * wsc, float * wd) {
    const uint64_t * grid = iq2s_grid;
    for (int64_t i = 0; i < nb; ++i) {
        const float32x4_t d = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b[i].d)));
        vst1q_f32(wd + i * 4, d);
        for (int ib = 0; ib < 8; ++ib) {
            const block_iq2_s_r4_sub * s = &b[i].sub[ib];
            for (int r = 0; r < 4; ++r) {
                uint64_t v;
                memcpy(&v, s->qi + 5 * r, 8);  // 5 bytes used; the 3 after stay inside the sub-block
                for (int l = 0; l < 4; ++l) {
                    // 8 unsigned grid values, sign byte: bit j set => weight j negative ((g ^ m) - m)
                    const uint8x8_t g = vcreate_u8(grid[(v >> (10 * l)) & 1023]);
                    const uint8x8_t m = vtst_u8(vdup_n_u8(s->sg[4 * r + l]), vcreate_u8(0x8040201008040201ull));
                    const uint8x8_t w = vsub_u8(veor_u8(g, m), m);
                    vst1_u8((uint8_t *) wc + ((size_t) (i * 32 + ib * 4 + l) * 32 + r * 8), w);
                }
                const int sc = s->sc[r];
                wsc[(i * 16 + 2 * ib) * 4 + r]     = (int16_t) (1 + 2 * (sc & 15));
                wsc[(i * 16 + 2 * ib + 1) * 4 + r] = (int16_t) (1 + 2 * (sc >> 4));
            }
        }
    }
}

// bytes of scratch pf_expert_iq2s_rows needs for `nrows` rows of length ne00
size_t pf_expert_iq2s_scratch(int64_t ne00, int64_t nrows) {
    const size_t g = (size_t) nrows / 4, nb = (size_t) ne00 / QK_K;
    return g * ((size_t) ne00 * 4 + nb * 16 * 4 * sizeof(int16_t) + nb * 4 * sizeof(float)) + 64;
}

// Rows [0, nrows) (nrows % 4 == 0) of the R4 matrix vx against `ngroups` packed X groups (consecutive in xc / xd,
// as written by pf_expert_pack_x8): s[8 * g + t][r] = W[r] . x[g][t] for every non-NULL s[8 * g + t].
// scratch: pf_expert_iq2s_scratch(ne00, nrows) bytes, 4-byte aligned.
void pf_expert_iq2s_rows(int64_t ne00, const void * vx, int64_t nrows, const int8_t * xc, const float * xd, int64_t ngroups,
                         float * const * s, void * scratch) {
    const int64_t nb = ne00 / QK_K, ng = nrows / 4;
    float * wd = (float *) scratch;
    int16_t * wsc = (int16_t *) (wd + (size_t) ng * nb * 4);
    int8_t * wc = (int8_t *) (wsc + (size_t) ng * nb * 16 * 4);
    for (int64_t x = 0; x < ng; ++x) {
        pf_iq2s_r4_decode((const block_iq2_s_r4 *) vx + x * nb, nb, wc + (size_t) x * ne00 * 4, wsc + (size_t) x * nb * 64, wd + (size_t) x * nb * 4);
    }
    for (int64_t g = 0; g < ngroups; ++g) {
        const int8_t * gx = xc + (size_t) g * ne00 * 8;
        const float * gd = xd + (size_t) g * nb * 8;
        float * const * gs = s + g * 8;
        for (int64_t x = 0; x < ng; ++x) {
            float tile[32];
            pf_tile8x4_two(gx, wc + (size_t) x * ne00 * 4, wsc + (size_t) x * nb * 64, (size_t) nb, 16, 2, gd, wd + (size_t) x * nb * 4, tile, 4, 1);
            for (int t = 0; t < 8; ++t) {
                if (gs[t]) vst1q_f32(gs[t] + 4 * x, vmulq_n_f32(vld1q_f32(tile + 4 * t), 0.125f));
            }
        }
    }
}
