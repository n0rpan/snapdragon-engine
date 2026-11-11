// expert-r4.c -- R4 layouts, in-place slice transforms and ARM kernels for streamed IQ3_XXS / IQ2_S experts.
// See expert-r4.h for the layouts and the bitwise-identity contract.
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "expert-r4.h"
#include "ggml-cpu.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define R4_HAVE_GEMV 1
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define R4_HAVE_GEMM 1
#endif
#endif

_Static_assert(sizeof(block_iq3_xxs_r4) == 4 * sizeof(block_iq3_xxs), "IQ3_XXS R4 must be size-neutral");
_Static_assert(sizeof(block_iq2_s_r4) == 4 * sizeof(block_iq2_s), "IQ2_S R4 must be size-neutral");

// ---------------------------------------------------------------------------------------------------------------
// marker

static int         g_r4_tag_storage;
static atomic_bool g_r4_in_use = false;

void * ggml_cpu_expert_r4_tag(void) {
    atomic_store_explicit(&g_r4_in_use, true, memory_order_relaxed);
    return &g_r4_tag_storage;
}

bool ggml_cpu_expert_r4_is(const struct ggml_tensor * t) {
    return t != NULL && t->extra == (void *) &g_r4_tag_storage;
}

bool ggml_cpu_expert_r4_in_use(void) {
    return atomic_load_explicit(&g_r4_in_use, memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------
// type / shape gate

static size_t r4_block_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        default:                return 0;
    }
}

static bool r4_shape_ok(enum ggml_type type, int64_t ne00, int64_t ne01) {
    return r4_block_size(type) != 0 && ne00 > 0 && ne00 % QK_K == 0 && ne01 > 0 && ne01 % 4 == 0;
}

bool ggml_cpu_expert_r4_type_ok(enum ggml_type type, int64_t ne00, int64_t ne01) {
#if defined(R4_HAVE_GEMV)
    return r4_shape_ok(type, ne00, ne01);
#else
    // no kernel that matches this build's vec_dot rounding: never offer the layout
    (void) type; (void) ne00; (void) ne01;
    return false;
#endif
}

bool ggml_cpu_expert_r4_supported(enum ggml_type type, int64_t ne00, int64_t ne01) {
    return ggml_cpu_expert_r4_type_ok(type, ne00, ne01);
}

bool ggml_cpu_expert_r4_has_gemm(void) {
#if defined(R4_HAVE_GEMM)
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// repack of one group of 4 rows (src: 4 rows of nb blocks, row r at src + r * nb) and its exact inverse

static void iq3_xxs_group_to_r4(block_iq3_xxs_r4 * dst, const block_iq3_xxs * src, int nb) {
    for (int x = 0; x < nb; ++x) {
        block_iq3_xxs_r4 * o = &dst[x];
        for (int r = 0; r < 4; ++r) {
            const block_iq3_xxs * b = &src[r * nb + x];
            o->d[r] = b->d;
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux;
                memcpy(&aux, b->qs + 64 + 4 * ib, 4);
                memcpy(o->qs + (ib * 4 + r) * 8, b->qs + 8 * ib, 8);
                for (int l = 0; l < 4; ++l) {
                    o->sas[ib * 16 + r * 4 + l] = (uint8_t) (((aux >> (7 * l)) & 127) | (((aux >> (28 + l)) & 1) << 7));
                }
            }
        }
    }
}

static void iq3_xxs_group_from_r4(block_iq3_xxs * dst, const block_iq3_xxs_r4 * src, int nb) {
    for (int x = 0; x < nb; ++x) {
        const block_iq3_xxs_r4 * o = &src[x];
        for (int r = 0; r < 4; ++r) {
            block_iq3_xxs * b = &dst[r * nb + x];
            b->d = o->d[r];
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux = 0;
                memcpy(b->qs + 8 * ib, o->qs + (ib * 4 + r) * 8, 8);
                for (int l = 0; l < 4; ++l) {
                    const uint8_t v = o->sas[ib * 16 + r * 4 + l];
                    aux |= (uint32_t) (v & 127) << (7 * l);
                    aux |= (uint32_t) (v >> 7) << (28 + l);
                }
                memcpy(b->qs + 64 + 4 * ib, &aux, 4);
            }
        }
    }
}

// IQ2_S: index l of sub-block ib = qs[4ib + l] | ((qh[ib] >> 2l) & 3) << 8; signs qs[32 + 4ib + l]; scale scales[ib]
static void iq2_s_group_to_r4(block_iq2_s_r4 * dst, const block_iq2_s * src, int nb) {
    for (int x = 0; x < nb; ++x) {
        block_iq2_s_r4 * o = &dst[x];
        for (int r = 0; r < 4; ++r) {
            const block_iq2_s * b = &src[r * nb + x];
            o->d[r] = b->d;
            for (int ib = 0; ib < 8; ++ib) {
                block_iq2_s_r4_sub * s = &o->sub[ib];
                uint64_t packed = 0;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t idx = (uint64_t) b->qs[4 * ib + l] | ((uint64_t) ((b->qh[ib] >> (2 * l)) & 3) << 8);
                    packed |= idx << (10 * l);
                }
                for (int k = 0; k < 5; ++k) s->qi[5 * r + k] = (uint8_t) (packed >> (8 * k));
                memcpy(s->sg + 4 * r, b->qs + QK_K / 8 + 4 * ib, 4);
                s->sc[r] = b->scales[ib];
            }
        }
    }
}

static void iq2_s_group_from_r4(block_iq2_s * dst, const block_iq2_s_r4 * src, int nb) {
    for (int x = 0; x < nb; ++x) {
        const block_iq2_s_r4 * o = &src[x];
        for (int r = 0; r < 4; ++r) {
            block_iq2_s * b = &dst[r * nb + x];
            b->d = o->d[r];
            for (int ib = 0; ib < 8; ++ib) {
                const block_iq2_s_r4_sub * s = &o->sub[ib];
                uint64_t packed = 0;
                for (int k = 0; k < 5; ++k) packed |= (uint64_t) s->qi[5 * r + k] << (8 * k);
                uint8_t qh = 0;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t idx = (uint32_t) (packed >> (10 * l)) & 1023;
                    b->qs[4 * ib + l] = (uint8_t) (idx & 255);
                    qh |= (uint8_t) ((idx >> 8) << (2 * l));
                }
                b->qh[ib] = qh;
                memcpy(b->qs + QK_K / 8 + 4 * ib, s->sg + 4 * r, 4);
                b->scales[ib] = s->sc[r];
            }
        }
    }
}

// in place, one 4-row group at a time through a temp of 4 rows
static bool r4_slice_convert(enum ggml_type type, void * data, int64_t ne00, int64_t ne01, bool to_r4) {
    if (!r4_shape_ok(type, ne00, ne01) || data == NULL) return false;
    const int    nb    = (int) (ne00 / QK_K);
    const size_t group = 4 * (size_t) nb * r4_block_size(type);
    uint8_t      stack_tmp[16384];
    uint8_t *    tmp = group <= sizeof(stack_tmp) ? stack_tmp : (uint8_t *) malloc(group);
    if (tmp == NULL) return false;
    uint8_t * p = (uint8_t *) data;
    for (int64_t g = 0; g < ne01 / 4; ++g, p += group) {
        memcpy(tmp, p, group);
        if (type == GGML_TYPE_IQ3_XXS) {
            if (to_r4) iq3_xxs_group_to_r4((block_iq3_xxs_r4 *) p, (const block_iq3_xxs *) tmp, nb);
            else       iq3_xxs_group_from_r4((block_iq3_xxs *) p, (const block_iq3_xxs_r4 *) tmp, nb);
        } else {
            if (to_r4) iq2_s_group_to_r4((block_iq2_s_r4 *) p, (const block_iq2_s *) tmp, nb);
            else       iq2_s_group_from_r4((block_iq2_s *) p, (const block_iq2_s_r4 *) tmp, nb);
        }
    }
    if (tmp != stack_tmp) free(tmp);
    return true;
}

bool ggml_cpu_expert_r4_slice_to_r4(enum ggml_type type, void * data, int64_t ne00, int64_t ne01) {
    return r4_slice_convert(type, data, ne00, ne01, true);
}

bool ggml_cpu_expert_r4_slice_from_r4(enum ggml_type type, void * data, int64_t ne00, int64_t ne01) {
    return r4_slice_convert(type, data, ne00, ne01, false);
}

bool ggml_cpu_expert_r4_transform(enum ggml_type type, void * data, int64_t ne00, int64_t ne01) {
    // the public entry the streamer calls: only for what the kernels can consume in this build
    return ggml_cpu_expert_r4_type_ok(type, ne00, ne01) && r4_slice_convert(type, data, ne00, ne01, true);
}

// ---------------------------------------------------------------------------------------------------------------
// kernels

#if defined(R4_HAVE_GEMV)

static const uint8_t r4_k_bit[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
// byte selectors: row r, half h -> sign bytes 4r + 2h (weights 0-7 of the half) and 4r + 2h + 1 (weights 8-15)
static const uint8_t r4_k_pat[4][2][16] = {
    { { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 }, { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 } },
    { { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5 }, { 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 } },
    { { 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 }, { 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11 } },
    { { 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13 }, { 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15 } },
};

// apply 16 sign bytes ([row][4], byte = 8 weights) to the 4 rows' unsigned grid values: (g ^ m) - m = m ? -g : g
static inline __attribute__((always_inline)) void r4_apply_signs(uint8x16_t sg, int8x16_t w[4][2]) {
    const uint8x16_t bitm = vld1q_u8(r4_k_bit);
    for (int r = 0; r < 4; ++r) {
        const uint8x16_t m0 = vtstq_u8(vqtbl1q_u8(sg, vld1q_u8(r4_k_pat[r][0])), bitm);
        const uint8x16_t m1 = vtstq_u8(vqtbl1q_u8(sg, vld1q_u8(r4_k_pat[r][1])), bitm);
        w[r][0] = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(vreinterpretq_u8_s8(w[r][0]), m0), m0));
        w[r][1] = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(vreinterpretq_u8_s8(w[r][1]), m1), m1));
    }
}

// IQ3_XXS: one 32-weight sub-block of 4 rows -> signed int8 weights, and (0.5 + scale) per row as floats
static inline __attribute__((always_inline)) float32x4_t iq3r4_decode_sub(const uint8_t * GGML_RESTRICT qi,
                                                                          const uint8_t * GGML_RESTRICT sas,
                                                                          int8x16_t w[4][2]) {
    const uint32_t * GGML_RESTRICT grid = iq3xxs_grid;
    const uint8x16_t sv = vld1q_u8(sas);
    const uint32x4_t sc = vshrq_n_u32(vmulq_u32(vreinterpretq_u32_u8(vshrq_n_u8(sv, 7)), vdupq_n_u32(0x01020408u)), 24);
    const uint8x16_t s7 = vandq_u8(sv, vdupq_n_u8(0x7f));
    const uint8x16_t sg = vorrq_u8(s7, vshlq_n_u8(vcntq_u8(s7), 7)); // 8th sign = parity of the other 7
    for (int r = 0; r < 4; ++r) {
        uint64_t ix;
        memcpy(&ix, qi + 8 * r, 8);
        const uint32x4_t g0 = { grid[ix & 255], grid[(ix >> 8) & 255], grid[(ix >> 16) & 255], grid[(ix >> 24) & 255] };
        const uint32x4_t g1 = { grid[(ix >> 32) & 255], grid[(ix >> 40) & 255], grid[(ix >> 48) & 255], grid[ix >> 56] };
        w[r][0] = vreinterpretq_s8_u32(g0);
        w[r][1] = vreinterpretq_s8_u32(g1);
    }
    r4_apply_signs(sg, w);
    return vaddq_f32(vcvtq_f32_u32(sc), vdupq_n_f32(0.5f));
}

// IQ2_S: one 32-weight sub-block of 4 rows -> signed int8 weights; ls_lo/ls_hi = 1 + 2 * scale nibble per row
static inline __attribute__((always_inline)) void iq2sr4_decode_sub(const block_iq2_s_r4_sub * GGML_RESTRICT s,
                                                                    int8x16_t w[4][2], int32x4_t * ls_lo,
                                                                    int32x4_t * ls_hi) {
    const uint64_t * GGML_RESTRICT grid = iq2s_grid;
    for (int r = 0; r < 4; ++r) {
        uint64_t v;
        memcpy(&v, s->qi + 5 * r, 8); // 5 bytes used; the 3 after stay inside the sub-block (sg follows qi)
        w[r][0] = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(grid[v & 1023]), vcreate_u64(grid[(v >> 10) & 1023])));
        w[r][1] = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(grid[(v >> 20) & 1023]), vcreate_u64(grid[(v >> 30) & 1023])));
    }
    r4_apply_signs(vld1q_u8(s->sg), w);
    uint32_t sc4;
    memcpy(&sc4, s->sc, 4);
    const uint32x4_t sc = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8((uint64_t) sc4))));
    const uint32x4_t one = vdupq_n_u32(1);
    *ls_lo = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vandq_u32(sc, vdupq_n_u32(15)), 1), one));
    *ls_hi = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vshrq_n_u32(sc, 4), 1), one));
}

// ---- gemv (decode)

static void gemv_iq3_xxs_r4(int nb, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx,
                            const block_q8_K * GGML_RESTRICT y, int64_t nrows) {
    for (int64_t x = 0; x < nrows / 4; ++x) {
        const block_iq3_xxs_r4 * GGML_RESTRICT b = (const block_iq3_xxs_r4 *) vx + x * nb;
        float32x4_t sumf = vdupq_n_f32(0.f);
        for (int i = 0; i < nb; ++i) {
            const float32x4_t d = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b[i].d))), y[i].d);
            float32x4_t sumf1 = vdupq_n_f32(0.f), sumf2 = vdupq_n_f32(0.f);
            const int8_t * GGML_RESTRICT q8 = y[i].qs;
            for (int ib = 0; ib < 8; ib += 2) {
                const int8x16x4_t q8b = vld1q_s8_x4(q8);
                q8 += 64;
                for (int h = 0; h < 2; ++h) {
                    int8x16_t w[4][2];
                    const float32x4_t scf = iq3r4_decode_sub(b[i].qs + (ib + h) * 32, b[i].sas + (ib + h) * 16, w);
                    int32x4_t acc[4];
                    for (int r = 0; r < 4; ++r) {
                        acc[r] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w[r][0], q8b.val[2 * h + 0]), w[r][1], q8b.val[2 * h + 1]);
                    }
                    const int32x4_t p = vpaddq_s32(vpaddq_s32(acc[0], acc[1]), vpaddq_s32(acc[2], acc[3]));
                    if (h == 0) sumf1 = vfmaq_f32(sumf1, vcvtq_f32_s32(p), scf);
                    else        sumf2 = vfmaq_f32(sumf2, vcvtq_f32_s32(p), scf);
                }
            }
            sumf = vfmaq_f32(sumf, vaddq_f32(sumf1, sumf2), d);
        }
        vst1q_f32(s + 4 * x, vmulq_n_f32(sumf, 0.5f));
    }
}

static void gemv_iq2_s_r4(int nb, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx,
                          const block_q8_K * GGML_RESTRICT y, int64_t nrows) {
    for (int64_t x = 0; x < nrows / 4; ++x) {
        const block_iq2_s_r4 * GGML_RESTRICT b = (const block_iq2_s_r4 *) vx + x * nb;
        float32x4_t sumf = vdupq_n_f32(0.f);
        for (int i = 0; i < nb; ++i) {
            const float32x4_t d = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(b[i].d))), y[i].d);
            int32x4_t sumi = vdupq_n_s32(0);
            const int8_t * GGML_RESTRICT q8 = y[i].qs;
            for (int ib = 0; ib < 8; ++ib) {
                const int8x16_t q8a = vld1q_s8(q8 + 32 * ib);
                const int8x16_t q8c = vld1q_s8(q8 + 32 * ib + 16);
                int8x16_t w[4][2];
                int32x4_t ls_lo, ls_hi;
                iq2sr4_decode_sub(&b[i].sub[ib], w, &ls_lo, &ls_hi);
                int32x4_t lo[4], hi[4];
                for (int r = 0; r < 4; ++r) {
                    lo[r] = vdotq_s32(vdupq_n_s32(0), w[r][0], q8a);
                    hi[r] = vdotq_s32(vdupq_n_s32(0), w[r][1], q8c);
                }
                const int32x4_t plo = vpaddq_s32(vpaddq_s32(lo[0], lo[1]), vpaddq_s32(lo[2], lo[3]));
                const int32x4_t phi = vpaddq_s32(vpaddq_s32(hi[0], hi[1]), vpaddq_s32(hi[2], hi[3]));
                sumi = vmlaq_s32(vmlaq_s32(sumi, plo, ls_lo), phi, ls_hi);
            }
            // upstream: sumf += d * (sumi1 + sumi2), contracted to one fmadd; the integer sum is exact either way
            sumf = vfmaq_f32(sumf, vcvtq_f32_s32(sumi), d);
        }
        vst1q_f32(s + 4 * x, vmulq_n_f32(sumf, 0.125f));
    }
}

#endif // R4_HAVE_GEMV

#if defined(R4_HAVE_GEMM)

// ---- gemm (prefill): 4 rows x 4 tokens per SMMLA tile set. Tile (rp, cp) lanes:
// [row 2rp tok 2cp, row 2rp tok 2cp+1, row 2rp+1 tok 2cp, row 2rp+1 tok 2cp+1]. The decode of each sub-block is shared
// by the 4 tokens; the tokens are plain q8_K rows (MUL_MAT_ID's wdata as is).

// A[rp][k]: k-chunk k (8 weights) of rows 2rp, 2rp+1; B[cp][k]: the same for tokens 2cp, 2cp+1
static inline __attribute__((always_inline)) void r4_pack_a(const int8x16_t w[4][2], int8x16_t A[2][4]) {
    for (int rp = 0; rp < 2; ++rp) {
        for (int hh = 0; hh < 2; ++hh) {
            const int64x2_t a0 = vreinterpretq_s64_s8(w[2 * rp][hh]), a1 = vreinterpretq_s64_s8(w[2 * rp + 1][hh]);
            A[rp][2 * hh + 0] = vreinterpretq_s8_s64(vzip1q_s64(a0, a1));
            A[rp][2 * hh + 1] = vreinterpretq_s8_s64(vzip2q_s64(a0, a1));
        }
    }
}

static inline __attribute__((always_inline)) void r4_pack_b(const block_q8_K * const yt[4], int i, int ib, int8x16_t B[2][4]) {
    for (int cp = 0; cp < 2; ++cp) {
        for (int hh = 0; hh < 2; ++hh) {
            const int64x2_t c0 = vreinterpretq_s64_s8(vld1q_s8(yt[2 * cp + 0][i].qs + ib * 32 + 16 * hh));
            const int64x2_t c1 = vreinterpretq_s64_s8(vld1q_s8(yt[2 * cp + 1][i].qs + ib * 32 + 16 * hh));
            B[cp][2 * hh + 0] = vreinterpretq_s8_s64(vzip1q_s64(c0, c1));
            B[cp][2 * hh + 1] = vreinterpretq_s8_s64(vzip2q_s64(c0, c1));
        }
    }
}

static inline __attribute__((always_inline)) void r4_store_tile(float * const s[4], int64_t row0, float32x4_t sumf[2][2], float scale) {
    for (int rp = 0; rp < 2; ++rp) {
        for (int cp = 0; cp < 2; ++cp) {
            float o[4];
            vst1q_f32(o, vmulq_n_f32(sumf[rp][cp], scale));
            s[2 * cp + 0][row0 + 2 * rp + 0] = o[0];
            s[2 * cp + 1][row0 + 2 * rp + 0] = o[1];
            s[2 * cp + 0][row0 + 2 * rp + 1] = o[2];
            s[2 * cp + 1][row0 + 2 * rp + 1] = o[3];
        }
    }
}

static inline __attribute__((always_inline)) void r4_block_d(const uint16_t * dxh, const block_q8_K * const yt[4], int i,
                                                             float32x4_t d[2][2]) {
    const float32x4_t dx = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(dxh)));
    const float dyv[4] = { yt[0][i].d, yt[1][i].d, yt[2][i].d, yt[3][i].d };
    const float32x4_t dy = vld1q_f32(dyv);
    const float32x4_t dxl[2] = { vzip1q_f32(dx, dx), vzip2q_f32(dx, dx) };
    const float32x4_t dyl[2] = { vcombine_f32(vget_low_f32(dy), vget_low_f32(dy)),
                                 vcombine_f32(vget_high_f32(dy), vget_high_f32(dy)) };
    for (int rp = 0; rp < 2; ++rp)
        for (int cp = 0; cp < 2; ++cp) d[rp][cp] = vmulq_f32(dxl[rp], dyl[cp]); // = fp16(x.d) * y.d, one rounding
}

static void gemm4_iq3_xxs_r4(int nb, float * const s[4], const void * GGML_RESTRICT vx, const block_q8_K * const yt[4],
                             int64_t nrows) {
    for (int64_t x = 0; x < nrows / 4; ++x) {
        const block_iq3_xxs_r4 * GGML_RESTRICT b = (const block_iq3_xxs_r4 *) vx + x * nb;
        float32x4_t sumf[2][2] = { { vdupq_n_f32(0.f), vdupq_n_f32(0.f) }, { vdupq_n_f32(0.f), vdupq_n_f32(0.f) } };
        for (int i = 0; i < nb; ++i) {
            float32x4_t d[2][2];
            r4_block_d(b[i].d, yt, i, d);
            float32x4_t s1[2][2], s2[2][2];
            for (int rp = 0; rp < 2; ++rp)
                for (int cp = 0; cp < 2; ++cp) { s1[rp][cp] = vdupq_n_f32(0.f); s2[rp][cp] = vdupq_n_f32(0.f); }
            for (int ib = 0; ib < 8; ++ib) {
                int8x16_t w[4][2];
                const float32x4_t scf = iq3r4_decode_sub(b[i].qs + ib * 32, b[i].sas + ib * 16, w);
                int8x16_t A[2][4], B[2][4];
                r4_pack_a(w, A);
                r4_pack_b(yt, i, ib, B);
                const float32x4_t scl[2] = { vzip1q_f32(scf, scf), vzip2q_f32(scf, scf) };
                for (int rp = 0; rp < 2; ++rp) {
                    for (int cp = 0; cp < 2; ++cp) {
                        int32x4_t acc = vmmlaq_s32(vdupq_n_s32(0), A[rp][0], B[cp][0]);
                        acc = vmmlaq_s32(acc, A[rp][1], B[cp][1]);
                        acc = vmmlaq_s32(acc, A[rp][2], B[cp][2]);
                        acc = vmmlaq_s32(acc, A[rp][3], B[cp][3]);
                        if (ib & 1) s2[rp][cp] = vfmaq_f32(s2[rp][cp], vcvtq_f32_s32(acc), scl[rp]);
                        else        s1[rp][cp] = vfmaq_f32(s1[rp][cp], vcvtq_f32_s32(acc), scl[rp]);
                    }
                }
            }
            for (int rp = 0; rp < 2; ++rp)
                for (int cp = 0; cp < 2; ++cp) sumf[rp][cp] = vfmaq_f32(sumf[rp][cp], vaddq_f32(s1[rp][cp], s2[rp][cp]), d[rp][cp]);
        }
        r4_store_tile(s, 4 * x, sumf, 0.5f);
    }
}

static void gemm4_iq2_s_r4(int nb, float * const s[4], const void * GGML_RESTRICT vx, const block_q8_K * const yt[4],
                           int64_t nrows) {
    for (int64_t x = 0; x < nrows / 4; ++x) {
        const block_iq2_s_r4 * GGML_RESTRICT b = (const block_iq2_s_r4 *) vx + x * nb;
        float32x4_t sumf[2][2] = { { vdupq_n_f32(0.f), vdupq_n_f32(0.f) }, { vdupq_n_f32(0.f), vdupq_n_f32(0.f) } };
        for (int i = 0; i < nb; ++i) {
            float32x4_t d[2][2];
            r4_block_d(b[i].d, yt, i, d);
            int32x4_t sumi[2][2] = { { vdupq_n_s32(0), vdupq_n_s32(0) }, { vdupq_n_s32(0), vdupq_n_s32(0) } };
            for (int ib = 0; ib < 8; ++ib) {
                int8x16_t w[4][2];
                int32x4_t ls_lo, ls_hi;
                iq2sr4_decode_sub(&b[i].sub[ib], w, &ls_lo, &ls_hi);
                int8x16_t A[2][4], B[2][4];
                r4_pack_a(w, A);
                r4_pack_b(yt, i, ib, B);
                const int32x4_t llo[2] = { vzip1q_s32(ls_lo, ls_lo), vzip2q_s32(ls_lo, ls_lo) };
                const int32x4_t lhi[2] = { vzip1q_s32(ls_hi, ls_hi), vzip2q_s32(ls_hi, ls_hi) };
                for (int rp = 0; rp < 2; ++rp) {
                    for (int cp = 0; cp < 2; ++cp) {
                        const int32x4_t alo = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), A[rp][0], B[cp][0]), A[rp][1], B[cp][1]);
                        const int32x4_t ahi = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), A[rp][2], B[cp][2]), A[rp][3], B[cp][3]);
                        sumi[rp][cp] = vmlaq_s32(vmlaq_s32(sumi[rp][cp], alo, llo[rp]), ahi, lhi[rp]);
                    }
                }
            }
            for (int rp = 0; rp < 2; ++rp)
                for (int cp = 0; cp < 2; ++cp) sumf[rp][cp] = vfmaq_f32(sumf[rp][cp], vcvtq_f32_s32(sumi[rp][cp]), d[rp][cp]);
        }
        r4_store_tile(s, 4 * x, sumf, 0.125f);
    }
}

#endif // R4_HAVE_GEMM

void ggml_cpu_expert_r4_gemv(enum ggml_type type, int64_t ne00, float * s, const void * vx, const void * vy, int64_t nrows) {
#if defined(R4_HAVE_GEMV)
    const int nb = (int) (ne00 / QK_K);
    if (type == GGML_TYPE_IQ3_XXS) gemv_iq3_xxs_r4(nb, s, vx, (const block_q8_K *) vy, nrows);
    else if (type == GGML_TYPE_IQ2_S) gemv_iq2_s_r4(nb, s, vx, (const block_q8_K *) vy, nrows);
    else abort();
#else
    (void) type; (void) ne00; (void) s; (void) vx; (void) vy; (void) nrows;
    abort(); // unreachable: type_ok() is false in this build, so nothing is ever marked
#endif
}

void ggml_cpu_expert_r4_gemm4(enum ggml_type type, int64_t ne00, float * const s[4], const void * vx,
                              const void * const vy[4], int64_t nrows) {
#if defined(R4_HAVE_GEMM)
    const int nb = (int) (ne00 / QK_K);
    const block_q8_K * const yt[4] = { (const block_q8_K *) vy[0], (const block_q8_K *) vy[1],
                                       (const block_q8_K *) vy[2], (const block_q8_K *) vy[3] };
    if (type == GGML_TYPE_IQ3_XXS) gemm4_iq3_xxs_r4(nb, s, vx, yt, nrows);
    else if (type == GGML_TYPE_IQ2_S) gemm4_iq2_s_r4(nb, s, vx, yt, nrows);
    else abort();
#else
    (void) type; (void) ne00; (void) s; (void) vx; (void) vy; (void) nrows;
    abort(); // callers check has_gemm()
#endif
}
