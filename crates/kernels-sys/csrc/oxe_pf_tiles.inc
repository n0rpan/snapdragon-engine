// oxe_pf_tiles.inc: multi-token SMMLA tiles for the MoE prefill (hand-written, not upstream).
// Included at the end of oxe_shim.cpp, i.e. in the translation unit and with the flags of the
// vendored repack kernels, so every float intrinsic below is compiled exactly like the same
// intrinsic in the reference kernel it mirrors.
//
// Why. llama.cpp's repacked GEMMs (`ggml_gemm_q4_K_8x8_q8_K`, ...) walk 4 tokens x 8 weight rows
// and unpack the weight block (nibbles, 6-bit scales, mins, bsum products) again for every
// 4-token group: 256 SMMLA per 8192 multiply-accumulates plus ~3x that in decode work. Here a
// chunk of weight rows is decoded ONCE into int8 codes and then multiplied with every token in
// 8-token x 4-row tiles whose 16 accumulators stay in registers (the shape the Flash prefill
// team measured, docs/lab-notes/PROGRESS-PF.md), with the loops blocked so that the activations of one
// 256-weight block for 64 tokens, the codes and the float accumulators all stay in L1.
//
// Exactness contract (Q4_K dense, reference `ggml_gemm_q4_K_8x8_q8_K`, i8mm branch).
// Per output element (token t, weight row r) the reference computes, for block b = 0..nb-1:
//     I_b = sum over the 8 sub-blocks s of scale[r][s] * dot32(q4[r][s], q8[t][s])      (int32)
//     B_b = sum over s of min[r][s] * bsum32[t][s]                                       (int32)
//     acc = vmlsq_f32(acc, f32(B_b), f32(dmin[r]) * d[t])
//     acc = vmlaq_f32(acc, f32(I_b), f32(d[r]) * d[t])
// from acc = +0. The integers are exact (mod 2^32), so how they are formed is free; this kernel
// forms the same I_b and B_b (B_b with one USMMLA + one SMMLA on the bsum bytes) and applies
// the same two float updates with the same intrinsics, per element, in the same block order.
// => bitwise equal to the reference for every (t, r). Unit test: kernels-sys tests
// `pf_tile_q4k_dense_bitwise`.
//
// Lane layout of an accumulator for token pair p (tokens 2p, 2p+1) and row pair q (rows 2q,
// 2q+1 of the quad): smmla(acc, X pair, W pair) adds
//     lane0 += x[2p].w[2q]   lane1 += x[2p].w[2q+1]   lane2 += x[2p+1].w[2q]   lane3 += x[2p+1].w[2q+1]

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)

#include <stdlib.h>
#include <string.h>
#include <type_traits>

#define OXE_PF_INLINE static inline __attribute__((always_inline))

// rows decoded per chunk and tokens per block (multiples of 8)
#define OXE_PF_RC 64
#define OXE_PF_TB 64

namespace {

struct oxe_pf_scratch {
    void *  mem = nullptr;
    size_t  cap = 0;
    ~oxe_pf_scratch() { free(mem); }
    void * get(size_t n) {
        if (n > cap) {
            free(mem);
            mem = nullptr;
            if (posix_memalign(&mem, 64, n) != 0) {
                abort();
            }
            cap = n;
        }
        return mem;
    }
};

// Decoded weights of one row quad for one 256-block.
struct oxe_pf_q4k_wblk {
    int8_t  codes[32 * 32];  // step-major: step = sub * 4 + k; [row0 x8, row1 x8 | row2 x8, row3 x8]
    int16_t sc[8 * 4];       // [sub][row]: 6-bit sub-block scales
    int8_t  mins[2 * 16];    // row pair q: [min(row 2q)[sub 0..8], min(row 2q+1)[sub 0..8]]
    float   d[4];            // f32(d) of the 4 rows
    float   dmin[4];         // f32(dmin) of the 4 rows
};

// decode one block_q4_Kx8 (8 interleaved rows) into two quads
OXE_PF_INLINE void oxe_pf_q4k_decode(const block_q4_Kx8 * w, oxe_pf_q4k_wblk * o /* [2] */) {
    const uint8x16_t  m4b = vdupq_n_u8(0x0f);
    const float32x4_t d0  = vcvt_f32_f16(vld1_f16((const __fp16 *) w->d));
    const float32x4_t d1  = vcvt_f32_f16(vld1_f16((const __fp16 *) w->d + 4));
    const float32x4_t m0  = vcvt_f32_f16(vld1_f16((const __fp16 *) w->dmin));
    const float32x4_t m1  = vcvt_f32_f16(vld1_f16((const __fp16 *) w->dmin + 4));
    vst1q_f32(o[0].d, d0);
    vst1q_f32(o[1].d, d1);
    vst1q_f32(o[0].dmin, m0);
    vst1q_f32(o[1].dmin, m1);
    for (int sb = 0; sb < 4; sb++) {
        for (int blk = 0; blk < 2; blk++) {
            const int sub = sb * 2 + blk;
            int16x8_t mins16;
            int8_t    sc8[8];
            decode_q_Kx8_6bit_scales(&w->scales[sb * 24 + blk * 12], &mins16, sc8);
            int16_t mn[8];
            vst1q_s16(mn, mins16);
            for (int r = 0; r < 8; r++) {
                o[r >> 2].sc[sub * 4 + (r & 3)]                      = (int16_t) sc8[r];
                o[r >> 2].mins[((r >> 1) & 1) * 16 + (r & 1) * 8 + sub] = (int8_t) mn[r];
            }
        }
        for (int k = 0; k < 4; k++) {
            for (int cp = 0; cp < 4; cp++) {
                // [row 2cp: weights sb*64 + k*8 .. +8 (low nibbles) and +32 (high nibbles), row 2cp+1: same]
                const uint8x16_t v  = vld1q_u8(w->qs + sb * QK_K + k * 64 + cp * 16);
                oxe_pf_q4k_wblk * q = &o[cp >> 1];
                const int       off = (cp & 1) * 16;
                vst1q_s8(q->codes + ((sb * 2 + 0) * 4 + k) * 32 + off, vreinterpretq_s8_u8(vandq_u8(v, m4b)));
                vst1q_s8(q->codes + ((sb * 2 + 1) * 4 + k) * 32 + off, vreinterpretq_s8_u8(vshrq_n_u8(v, 4)));
            }
        }
    }
}

// bsum bytes of one 4-token group for one block: for token pair p, lo[p] / hi[p] hold
// [token 2p: sub 0..8, token 2p+1: sub 0..8] of the low (unsigned) and high (signed) byte of
// the 32-value sums, so that bsum32 = 256 * hi + lo.
OXE_PF_INLINE void oxe_pf_q8k_bsum_bytes(const block_q8_Kx4 * a, uint8x16_t lo[2], int8x16_t hi[2]) {
    // P[sbp] = [r0h0 r0h1 r1h0 r1h1 r2h0 r2h1 r3h0 r3h1] (the reference's vpaddq pairing)
    const int16x8_t P0 = vpaddq_s16(vld1q_s16(a->bsums + 0), vld1q_s16(a->bsums + 8));
    const int16x8_t P1 = vpaddq_s16(vld1q_s16(a->bsums + 16), vld1q_s16(a->bsums + 24));
    const int16x8_t P2 = vpaddq_s16(vld1q_s16(a->bsums + 32), vld1q_s16(a->bsums + 40));
    const int16x8_t P3 = vpaddq_s16(vld1q_s16(a->bsums + 48), vld1q_s16(a->bsums + 56));
    const int32x4_t p0 = vreinterpretq_s32_s16(P0), p1 = vreinterpretq_s32_s16(P1);
    const int32x4_t p2 = vreinterpretq_s32_s16(P2), p3 = vreinterpretq_s32_s16(P3);
    const int64x2_t a01 = vreinterpretq_s64_s32(vzip1q_s32(p0, p1));  // [P0r0 P1r0 P0r1 P1r1]
    const int64x2_t b01 = vreinterpretq_s64_s32(vzip1q_s32(p2, p3));
    const int64x2_t a23 = vreinterpretq_s64_s32(vzip2q_s32(p0, p1));  // rows 2, 3
    const int64x2_t b23 = vreinterpretq_s64_s32(vzip2q_s32(p2, p3));
    const int16x8_t r0 = vreinterpretq_s16_s64(vzip1q_s64(a01, b01));  // row 0: sub 0..8
    const int16x8_t r1 = vreinterpretq_s16_s64(vzip2q_s64(a01, b01));
    const int16x8_t r2 = vreinterpretq_s16_s64(vzip1q_s64(a23, b23));
    const int16x8_t r3 = vreinterpretq_s16_s64(vzip2q_s64(a23, b23));
    lo[0] = vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(r0)), vmovn_u16(vreinterpretq_u16_s16(r1)));
    lo[1] = vcombine_u8(vmovn_u16(vreinterpretq_u16_s16(r2)), vmovn_u16(vreinterpretq_u16_s16(r3)));
    hi[0] = vcombine_s8(vshrn_n_s16(r0, 8), vshrn_n_s16(r1, 8));
    hi[1] = vcombine_s8(vshrn_n_s16(r2, 8), vshrn_n_s16(r3, 8));
}

// One tile: NP token pairs (4 = two 4-token groups xa, xb; 2 = xa only) x one row quad, one
// 256-block. f[2p+q] is the float accumulator of (token pair p, row pair q) in lane layout.
template <int NP>
OXE_PF_INLINE void oxe_pf_q4k_tile(const int8_t * xa, const int8_t * xb, const uint8x16_t * blo, const int8x16_t * bhi, const float * dx,
                                   const oxe_pf_q4k_wblk * w, float * f) {
    int32x4_t I[2 * NP];
    for (int i = 0; i < 2 * NP; i++) {
        I[i] = vdupq_n_s32(0);
    }
    const int8_t * wc = w->codes;
    for (int sub = 0; sub < 8; sub++) {
        int32x4_t t[2 * NP];
        for (int i = 0; i < 2 * NP; i++) {
            t[i] = vdupq_n_s32(0);
        }
        for (int k = 0; k < 4; k++) {
            const int8x16_t w0 = vld1q_s8(wc), w1 = vld1q_s8(wc + 16);
            const int8x16_t x0 = vld1q_s8(xa), x1 = vld1q_s8(xa + 16);
            t[0] = vmmlaq_s32(t[0], x0, w0);
            t[1] = vmmlaq_s32(t[1], x0, w1);
            t[2] = vmmlaq_s32(t[2], x1, w0);
            t[3] = vmmlaq_s32(t[3], x1, w1);
            if (NP == 4) {
                const int8x16_t x2 = vld1q_s8(xb), x3 = vld1q_s8(xb + 16);
                t[2 * NP - 4] = vmmlaq_s32(t[2 * NP - 4], x2, w0);
                t[2 * NP - 3] = vmmlaq_s32(t[2 * NP - 3], x2, w1);
                t[2 * NP - 2] = vmmlaq_s32(t[2 * NP - 2], x3, w0);
                t[2 * NP - 1] = vmmlaq_s32(t[2 * NP - 1], x3, w1);
                xb += 32;
            }
            xa += 32;
            wc += 32;
        }
        const int32x4_t sc = vmovl_s16(vld1_s16(w->sc + sub * 4));               // [s0 s1 s2 s3]
        const int32x4_t S0 = vcombine_s32(vget_low_s32(sc), vget_low_s32(sc));    // row pair 0
        const int32x4_t S1 = vcombine_s32(vget_high_s32(sc), vget_high_s32(sc));  // row pair 1
        for (int p = 0; p < NP; p++) {
            I[2 * p]     = vmlaq_s32(I[2 * p], t[2 * p], S0);
            I[2 * p + 1] = vmlaq_s32(I[2 * p + 1], t[2 * p + 1], S1);
        }
    }
    const int8x16_t   mn0 = vld1q_s8(w->mins), mn1 = vld1q_s8(w->mins + 16);
    const float32x2_t d01 = vld1_f32(w->d), d23 = vld1_f32(w->d + 2);
    const float32x2_t m01 = vld1_f32(w->dmin), m23 = vld1_f32(w->dmin + 2);
    const float32x4_t wd[2] = { vcombine_f32(d01, d01), vcombine_f32(d23, d23) };  // [a b a b]
    const float32x4_t wm[2] = { vcombine_f32(m01, m01), vcombine_f32(m23, m23) };
    const int8x16_t   mn[2] = { mn0, mn1 };
    for (int p = 0; p < NP; p++) {
        const float32x2_t v    = vld1_f32(dx + 2 * p);
        const float32x4_t q8_d = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a a b b]
        for (int q = 0; q < 2; q++) {
            // B = sum_s min[r][s] * bsum32[t][s], bsum32 = 256 * hi + lo
            const int32x4_t   bl    = vusmmlaq_s32(vdupq_n_s32(0), blo[p], mn[q]);
            const int32x4_t   bh    = vmmlaq_s32(vdupq_n_s32(0), bhi[p], mn[q]);
            const int32x4_t   bias  = vaddq_s32(bl, vshlq_n_s32(bh, 8));
            const float32x4_t dmins = vmulq_f32(wm[q], q8_d);
            const float32x4_t scale = vmulq_f32(wd[q], q8_d);
            float32x4_t       a     = vld1q_f32(f + (2 * p + q) * 4);
            a = vmlsq_f32(a, vcvtq_f32_s32(bias), dmins);
            a = vmlaq_f32(a, vcvtq_f32_s32(I[2 * p + q]), scale);
            vst1q_f32(f + (2 * p + q) * 4, a);
        }
    }
}

}  // namespace

extern "C" {

// Drop-in for ggml_gemm_q4_K_8x8_q8_K (same arguments, same bits):
// s[t * bs + r] = W[r] . a[t] for t < nr (multiple of 4), r < nc (multiple of 8); vx = nc/8 row
// groups of n/256 block_q4_Kx8, vy = nr/4 token groups of n/256 block_q8_Kx4.
void oxe_pf_gemm_q4_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    static thread_local oxe_pf_scratch scratch;
    const int nb = n / QK_K;
    if (nr <= 0 || nc <= 0) {
        return;
    }
    const size_t wbytes = (size_t) (OXE_PF_RC / 4) * nb * sizeof(oxe_pf_q4k_wblk);
    const size_t wpad   = (wbytes + 63) & ~(size_t) 63;
    const size_t fbytes = (size_t) OXE_PF_TB * OXE_PF_RC * sizeof(float);
    char *       mem    = (char *) scratch.get(wpad + fbytes);
    oxe_pf_q4k_wblk * wd = (oxe_pf_q4k_wblk *) mem;  // [b][quad]
    float *           F  = (float *) (mem + wpad);   // [token quad][row quad][16]: tile layout
    const block_q4_Kx8 * W = (const block_q4_Kx8 *) vx;
    const block_q8_Kx4 * A = (const block_q8_Kx4 *) vy;

    for (int c0 = 0; c0 < nc; c0 += OXE_PF_RC) {
        const int ncq = ((nc - c0) < OXE_PF_RC ? (nc - c0) : OXE_PF_RC) / 4;  // row quads in this chunk
        // decode the chunk once: wd[b * ncq + quad]
        for (int g = 0; g < ncq / 2; g++) {
            const block_q4_Kx8 * wg = W + (size_t) (c0 / 8 + g) * nb;
            for (int b = 0; b < nb; b++) {
                oxe_pf_q4k_wblk tmp[2];
                oxe_pf_q4k_decode(wg + b, tmp);
                wd[(size_t) b * ncq + 2 * g]     = tmp[0];
                wd[(size_t) b * ncq + 2 * g + 1] = tmp[1];
            }
        }
        for (int t0 = 0; t0 < nr; t0 += OXE_PF_TB) {
            const int ntq = ((nr - t0) < OXE_PF_TB ? (nr - t0) : OXE_PF_TB) / 4;  // token quads in this block
            for (int i = 0; i < ntq * ncq * 16; i += 4) {
                vst1q_f32(F + i, vdupq_n_f32(0));
            }
            for (int b = 0; b < nb; b++) {
                const oxe_pf_q4k_wblk * wb = wd + (size_t) b * ncq;
                int                     tq = 0;
                for (; tq + 2 <= ntq; tq += 2) {
                    const block_q8_Kx4 * a0 = A + (size_t) (t0 / 4 + tq) * nb + b;
                    const block_q8_Kx4 * a1 = A + (size_t) (t0 / 4 + tq + 1) * nb + b;
                    uint8x16_t           blo[4];
                    int8x16_t            bhi[4];
                    float                dx[8];
                    oxe_pf_q8k_bsum_bytes(a0, blo, bhi);
                    oxe_pf_q8k_bsum_bytes(a1, blo + 2, bhi + 2);
                    vst1q_f32(dx, vld1q_f32(a0->d));
                    vst1q_f32(dx + 4, vld1q_f32(a1->d));
                    for (int cq = 0; cq < ncq; cq++) {
                        // token pairs 0,1 live in F quad tq, pairs 2,3 in quad tq+1 (contiguous per row quad)
                        float f[32];
                        float * fa = F + ((size_t) tq * ncq + cq) * 16;
                        float * fb = F + ((size_t) (tq + 1) * ncq + cq) * 16;
                        for (int i = 0; i < 16; i += 4) {
                            vst1q_f32(f + i, vld1q_f32(fa + i));
                            vst1q_f32(f + 16 + i, vld1q_f32(fb + i));
                        }
                        oxe_pf_q4k_tile<4>(a0->qs, a1->qs, blo, bhi, dx, wb + cq, f);
                        for (int i = 0; i < 16; i += 4) {
                            vst1q_f32(fa + i, vld1q_f32(f + i));
                            vst1q_f32(fb + i, vld1q_f32(f + 16 + i));
                        }
                    }
                }
                if (tq < ntq) {
                    const block_q8_Kx4 * a0 = A + (size_t) (t0 / 4 + tq) * nb + b;
                    uint8x16_t           blo[2];
                    int8x16_t            bhi[2];
                    oxe_pf_q8k_bsum_bytes(a0, blo, bhi);
                    for (int cq = 0; cq < ncq; cq++) {
                        oxe_pf_q4k_tile<2>(a0->qs, a0->qs, blo, bhi, a0->d, wb + cq, F + ((size_t) tq * ncq + cq) * 16);
                    }
                }
            }
            // tile layout -> s: accumulator (p, q) = [t(2p) r(2q), t(2p) r(2q+1), t(2p+1) r(2q), t(2p+1) r(2q+1)]
            for (int tq = 0; tq < ntq; tq++) {
                for (int cq = 0; cq < ncq; cq++) {
                    const float * f = F + ((size_t) tq * ncq + cq) * 16;
                    for (int p = 0; p < 2; p++) {
                        float * r0 = s + (size_t) (t0 + tq * 4 + 2 * p) * bs + c0 + cq * 4;
                        float * r1 = r0 + bs;
                        const float32x4_t v0 = vld1q_f32(f + (2 * p) * 4);      // row pair 0
                        const float32x4_t v1 = vld1q_f32(f + (2 * p + 1) * 4);  // row pair 1
                        vst1q_f32(r0, vcombine_f32(vget_low_f32(v0), vget_low_f32(v1)));
                        vst1q_f32(r1, vcombine_f32(vget_high_f32(v0), vget_high_f32(v1)));
                    }
                }
            }
        }
    }
}

int oxe_pf_tiles_available(void) { return 1; }

}  // extern "C"

#else

extern "C" {
void oxe_pf_gemm_q4_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}
int oxe_pf_tiles_available(void) { return 0; }
}

#endif

// =============================================================================================
// Routed experts (MUL_MAT_ID): multi-token tiles in the GEMV's float order.
//
// llama.cpp sends an expert's routed rows 4 at a time through `gemm_mmid` (and the last cnt % 4
// through the GEMV); for every type below each row of that gemm is bitwise the GEMV of the row
// (kernels-sys test gemm_kernels_vs_generic_and_gemv). So the reference per (token, weight row)
// is a pure function of that pair, and any grouping of tokens that reproduces it is exact.
// Here an expert's rows are packed in PAIRS (oxe_pf_xblk), the weight rows are decoded once per
// 64-row chunk, and up to 4 pairs (8 tokens) x 4 rows share a tile. An odd row count pads one
// token; 2-7 leftover rows use 1-3 pair tiles instead of padded 4-row tiles.
//
// Float contracts reproduced (per element, block b = 0..nb-1, from acc = +0):
//   Q3_K  (ggml_gemm_q3_K_8x8_q8_K):   I = sum_16 sub-blocks sc * dot16;
//          acc = vfmaq_f32(acc, f32(I), d_x * d_w)
//   Q4_K  (ggml_gemm_qK_8x8_q8_K_gemv_order<q4>): per 32-weight sub-block s = 0..7:
//          acc = vfmaq_f32(acc, d_w * d_x, f32(sc[s] * dot32[s]));
//          then once per block: acc = vmlsq_f32(acc, f32(sum_s min[s] * bsum32[s]), dmin_w * d_x)
//   Q5_K  (same template, IS_Q5): the two fma of a 64-weight pair of sub-blocks, then
//          acc = vmlsq_f32(acc, dmin_w * d_x, f32(min[2j] * bsum32[2j] + min[2j+1] * bsum32[2j+1]))
// Unit test: kernels-sys `pf_tile_mmid_bitwise` (every row vs the GEMV and vs gemm_mmid).
// =============================================================================================
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)

extern "C" {
// One 256-block of a PAIR of from_float q8_K rows (a, b).
struct oxe_pf_xblk {
    int8_t  qs[32 * 16];  // step-major: [a: 8 codes, b: 8 codes] per 8-weight step
    float   d[2];         // d of a, b
    int32_t pad0[2];
    uint8_t blo[16];      // low bytes of the 32-value sums: [a: sub 0..8, b: sub 0..8]
    int8_t  bhi[16];      // high bytes (bsum32 = 256 * hi + lo)
    int32_t bs32[8][4];   // [sub]: [a, a, b, b]
};
}

namespace {

OXE_PF_INLINE void oxe_pf_pack_pair_blk(const block_q8_K * a, const block_q8_K * b, oxe_pf_xblk * o) {
    for (int st = 0; st < 32; st++) {
        memcpy(o->qs + st * 16, a->qs + st * 8, 8);
        memcpy(o->qs + st * 16 + 8, b->qs + st * 8, 8);
    }
    o->d[0]    = a->d;
    o->d[1]    = b->d;
    o->pad0[0] = o->pad0[1] = 0;
    for (int sub = 0; sub < 8; sub++) {
        const int32_t sa = (int32_t) a->bsums[2 * sub] + (int32_t) a->bsums[2 * sub + 1];
        const int32_t sb = (int32_t) b->bsums[2 * sub] + (int32_t) b->bsums[2 * sub + 1];
        o->blo[sub]     = (uint8_t) (sa & 0xff);
        o->bhi[sub]     = (int8_t) (sa >> 8);
        o->blo[8 + sub] = (uint8_t) (sb & 0xff);
        o->bhi[8 + sub] = (int8_t) (sb >> 8);
        o->bs32[sub][0] = sa;
        o->bs32[sub][1] = sa;
        o->bs32[sub][2] = sb;
        o->bs32[sub][3] = sb;
    }
}

// ---------------------------------------------------------------- Q3_K
struct oxe_pf_q3k_wblk {
    int8_t  codes[32 * 32];  // step-major (step = 8 weights): [row0 x8, row1 x8 | row2 x8, row3 x8]
    int16_t sc[16 * 4];      // [sub-block of 16 weights][row]
    float   d[4];
    float   pad[4];
};

// The tile of the "16 sub-blocks of 16 weights, one integer scale each, one float update per
// block" types (Q3_K, Q6_K). Token pair p: codes at xq[p] + step * XS ([a: 8 codes, b: 8 codes]),
// d of a, b at xd[p]. FUSED: vfmaq_f32 (Q3_K reference) or vmlaq_f32 (Q6_K reference).
template <int NP, int XS, bool FUSED>
OXE_PF_INLINE void oxe_pf_s16_tile(const int8_t * const * xq, const float * const * xd, const oxe_pf_q3k_wblk * w, float * f) {
    int32x4_t I[2 * NP];
    for (int i = 0; i < 2 * NP; i++) {
        I[i] = vdupq_n_s32(0);
    }
    const int8_t * wc = w->codes;
    for (int sub = 0; sub < 16; sub++) {
        int32x4_t t[2 * NP];
        for (int i = 0; i < 2 * NP; i++) {
            t[i] = vdupq_n_s32(0);
        }
        for (int k = 0; k < 2; k++) {
            const int8x16_t w0 = vld1q_s8(wc), w1 = vld1q_s8(wc + 16);
            for (int p = 0; p < NP; p++) {
                const int8x16_t x = vld1q_s8(xq[p] + (sub * 2 + k) * XS);
                t[2 * p]     = vmmlaq_s32(t[2 * p], x, w0);
                t[2 * p + 1] = vmmlaq_s32(t[2 * p + 1], x, w1);
            }
            wc += 32;
        }
        const int32x4_t sc = vmovl_s16(vld1_s16(w->sc + sub * 4));
        const int32x4_t S0 = vcombine_s32(vget_low_s32(sc), vget_low_s32(sc));
        const int32x4_t S1 = vcombine_s32(vget_high_s32(sc), vget_high_s32(sc));
        for (int p = 0; p < NP; p++) {
            I[2 * p]     = vmlaq_s32(I[2 * p], t[2 * p], S0);
            I[2 * p + 1] = vmlaq_s32(I[2 * p + 1], t[2 * p + 1], S1);
        }
    }
    const float32x2_t d01 = vld1_f32(w->d), d23 = vld1_f32(w->d + 2);
    const float32x4_t dw[2] = { vcombine_f32(d01, d01), vcombine_f32(d23, d23) };  // [a b a b]
    for (int p = 0; p < NP; p++) {
        const float32x2_t v  = vld1_f32(xd[p]);
        const float32x4_t da = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a a b b]
        for (int q = 0; q < 2; q++) {
            float32x4_t a = vld1q_f32(f + (2 * p + q) * 4);
            if (FUSED) {
                a = vfmaq_f32(a, vcvtq_f32_s32(I[2 * p + q]), vmulq_f32(da, dw[q]));
            } else {
                a = vmlaq_f32(a, vcvtq_f32_s32(I[2 * p + q]), vmulq_f32(dw[q], da));
            }
            vst1q_f32(f + (2 * p + q) * 4, a);
        }
    }
}

struct oxe_pf_q3k {
    typedef block_q3_Kx8    xblock;
    typedef oxe_pf_q3k_wblk wblk;

    // o0: rows 0..3, o1: rows 4..7
    OXE_PF_INLINE void decode(const block_q3_Kx8 * w, oxe_pf_q3k_wblk * o0, oxe_pf_q3k_wblk * o1) {
        vst1q_f32(o0->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d)));
        vst1q_f32(o1->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d + 4)));
        int8x16_t S[8];
        ggml_q3x8_decode_scales(w->scales, S);
        for (int g = 0; g < 16; g++) {
            // sub-block g: bytes (g & 1) * 8 + r of S[g / 2] = scale of row r
            const int16x8_t s16 = vmovl_s8((g & 1) ? vget_high_s8(S[g / 2]) : vget_low_s8(S[g / 2]));
            vst1_s16(o0->sc + g * 4, vget_low_s16(s16));
            vst1_s16(o1->sc + g * 4, vget_high_s16(s16));
        }
        for (int t = 0; t < 4; t++) {
            for (int cp = 0; cp < 4; cp++) {
                int8x16_t R[8];  // R[j]: [row 2cp, row 2cp+1] x weights t*64 + j*8 .. +8
                GGML_Q3X8_DECODE_TRIPLE(w->qs + (t * 4 + cp) * 48, R);
                int8_t * c = ((cp >> 1) ? o1 : o0)->codes + t * 8 * 32 + (cp & 1) * 16;
                for (int j = 0; j < 8; j++) {
                    vst1q_s8(c + j * 32, R[j]);
                }
            }
        }
    }

    template <int NP>
    OXE_PF_INLINE void tile(const oxe_pf_xblk * const * xp, const oxe_pf_q3k_wblk * w, float * f) {
        const int8_t * xq[NP];
        const float *  xd[NP];
        for (int p = 0; p < NP; p++) {
            xq[p] = xp[p]->qs;
            xd[p] = xp[p]->d;
        }
        oxe_pf_s16_tile<NP, 16, true>(xq, xd, w, f);
    }
    // tokens straight from 4-row interleaved q8_K groups (pair p at xq[p], 32 bytes per step)
    template <int NP>
    OXE_PF_INLINE void tile_k4(const int8_t * const * xq, const float * const * xd, const oxe_pf_q3k_wblk * w, float * f) {
        oxe_pf_s16_tile<NP, 32, true>(xq, xd, w, f);
    }
};

// ---------------------------------------------------------------- Q6_K
// Reference `ggml_gemm_q6_K_8x8_q8_K` (i8mm branch), which is also the MUL_MAT_ID kernel of Q6_K
// and bitwise the GEMV per row: per 256-block I = sum over the 16 sub-blocks of scale * dot16 with
// codes q6 - 32, then acc = vmlaq_f32(acc, f32(I), f32(d_w) * d_x). Same block shape as Q3_K (16
// sub-blocks of 16 weights, one int8 scale each, one float update per block), so it shares the
// tile; only the decode and the float intrinsic differ.
struct oxe_pf_q6k {
    typedef block_q6_Kx8    xblock;
    typedef oxe_pf_q3k_wblk wblk;

    // o0: rows 0..3, o1: rows 4..7
    OXE_PF_INLINE void decode(const block_q6_Kx8 * w, oxe_pf_q3k_wblk * o0, oxe_pf_q3k_wblk * o1) {
        const uint8x16_t m4b     = vdupq_n_u8(0x0f);
        const uint8x16_t mask_lo = vdupq_n_u8(0x03);
        const uint8x16_t mask_hi = vdupq_n_u8(0x30);
        const int8x16_t  m32s    = vdupq_n_s8(32);
        vst1q_f32(o0->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d)));
        vst1q_f32(o1->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d + 4)));
        for (int g = 0; g < 16; g++) {
            const int16x8_t s16 = vmovl_s8(vld1_s8(w->scales + g * 8));  // sub-block g, rows 0..7
            vst1_s16(o0->sc + g * 4, vget_low_s16(s16));
            vst1_s16(o1->sc + g * 4, vget_high_s16(s16));
        }
        for (int half = 0; half < 2; half++) {
            const uint8_t * ql_base = w->ql + half * 512;
            const uint8_t * qh_base = w->qh + half * 256;
            for (int sb = 0; sb < 4; sb++) {
                const int ql_off = sb * QK_K / 2;
                const int qh_off = (sb * QK_K / 2) & 255;
                for (int cp = 0; cp < 4; cp++) {
                    const uint8x16_t l0 = vld1q_u8(ql_base + ql_off + 16 * cp);
                    const uint8x16_t l1 = vld1q_u8(ql_base + ql_off + 64 + 16 * cp);
                    uint8x16_t       h0 = vld1q_u8(qh_base + qh_off + 16 * cp);
                    uint8x16_t       h1 = vld1q_u8(qh_base + qh_off + 64 + 16 * cp);
                    if (sb > 1) {
                        h0 = vshrq_n_u8(h0, 2);
                        h1 = vshrq_n_u8(h1, 2);
                    }
                    // q6 = (low4 | high2 << 4) - 32, as the reference forms it
                    const int8x16_t q6_l0 = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(l0, m4b), vandq_u8(h0, mask_lo), 4)), m32s);
                    const int8x16_t q6_l1 = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(l1, m4b), vandq_u8(h1, mask_lo), 4)), m32s);
                    const int8x16_t q6_h0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(h0, mask_hi))), m32s);
                    const int8x16_t q6_h1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(h1, mask_hi))), m32s);
                    // weights half*128 + sb*16 (+8) and half*128 + 64 + sb*16 (+8): steps of 8 weights
                    int8_t *  c  = ((cp >> 1) ? o1 : o0)->codes + (cp & 1) * 16;
                    const int st = half * 16 + sb * 2;
                    vst1q_s8(c + (st + 0) * 32, q6_l0);
                    vst1q_s8(c + (st + 1) * 32, q6_l1);
                    vst1q_s8(c + (st + 8) * 32, q6_h0);
                    vst1q_s8(c + (st + 9) * 32, q6_h1);
                }
            }
        }
    }

    template <int NP>
    OXE_PF_INLINE void tile(const oxe_pf_xblk * const * xp, const oxe_pf_q3k_wblk * w, float * f) {
        const int8_t * xq[NP];
        const float *  xd[NP];
        for (int p = 0; p < NP; p++) {
            xq[p] = xp[p]->qs;
            xd[p] = xp[p]->d;
        }
        oxe_pf_s16_tile<NP, 16, false>(xq, xd, w, f);
    }
    template <int NP>
    OXE_PF_INLINE void tile_k4(const int8_t * const * xq, const float * const * xd, const oxe_pf_q3k_wblk * w, float * f) {
        oxe_pf_s16_tile<NP, 32, false>(xq, xd, w, f);
    }
};

// ---------------------------------------------------------------- Q4_K / Q5_K (gemv order)
struct oxe_pf_qk_wblk {
    int8_t  codes[32 * 32];  // step-major, sub-block of 32 weights = 4 steps
    int16_t sc[8 * 4];       // [sub][row]
    int16_t mn16[8 * 4];     // [sub][row] (Q5_K bias)
    int8_t  mins[2 * 16];    // row pair q: [min(row 2q)[sub 0..8], min(row 2q+1)[sub 0..8]] (Q4_K bias)
    float   d[4];
    float   dmin[4];
};

template <bool IS_Q5> struct oxe_pf_qk {
    typedef typename std::conditional<IS_Q5, block_q5_Kx8, block_q4_Kx8>::type xblock;
    typedef oxe_pf_qk_wblk wblk;

    // o0: rows 0..3, o1: rows 4..7
    OXE_PF_INLINE void decode(const xblock * w, oxe_pf_qk_wblk * o0, oxe_pf_qk_wblk * o1) {
        const uint8x16_t m4b  = vdupq_n_u8(0x0f);
        const uint8x16_t mone = vdupq_n_u8(1);
        const uint8x16_t mtwo = vdupq_n_u8(2);
        vst1q_f32(o0->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d)));
        vst1q_f32(o1->d, vcvt_f32_f16(vld1_f16((const __fp16 *) w->d + 4)));
        vst1q_f32(o0->dmin, vcvt_f32_f16(vld1_f16((const __fp16 *) w->dmin)));
        vst1q_f32(o1->dmin, vcvt_f32_f16(vld1_f16((const __fp16 *) w->dmin + 4)));
        uint8x16_t qh[4][4];
        if constexpr (IS_Q5) {
            for (int cp = 0; cp < 4; cp++) {
                for (int k = 0; k < 4; k++) {
                    qh[cp][k] = vld1q_u8(w->qh + 16 * cp + 64 * k);
                }
            }
        }
        for (int sb = 0; sb < 4; sb++) {
            for (int blk = 0; blk < 2; blk++) {
                const int sub = sb * 2 + blk;
                int16x8_t mins16;
                int8_t    sc8[8];
                decode_q_Kx8_6bit_scales(&w->scales[sb * 24 + blk * 12], &mins16, sc8);
                const int16x8_t sc16 = vmovl_s8(vld1_s8(sc8));
                vst1_s16(o0->sc + sub * 4, vget_low_s16(sc16));
                vst1_s16(o1->sc + sub * 4, vget_high_s16(sc16));
                if constexpr (IS_Q5) {
                    vst1_s16(o0->mn16 + sub * 4, vget_low_s16(mins16));
                    vst1_s16(o1->mn16 + sub * 4, vget_high_s16(mins16));
                } else {
                    // mins[(row pair) * 16 + (row & 1) * 8 + sub] (0..63: the narrowing is exact)
                    uint8_t m8[8];
                    vst1_u8(m8, vmovn_u16(vreinterpretq_u16_s16(mins16)));
                    o0->mins[sub]      = (int8_t) m8[0];
                    o0->mins[8 + sub]  = (int8_t) m8[1];
                    o0->mins[16 + sub] = (int8_t) m8[2];
                    o0->mins[24 + sub] = (int8_t) m8[3];
                    o1->mins[sub]      = (int8_t) m8[4];
                    o1->mins[8 + sub]  = (int8_t) m8[5];
                    o1->mins[16 + sub] = (int8_t) m8[6];
                    o1->mins[24 + sub] = (int8_t) m8[7];
                }
            }
            for (int cp = 0; cp < 4; cp++) {
                int8_t * c = ((cp >> 1) ? o1 : o0)->codes + (cp & 1) * 16;
                for (int k = 0; k < 4; k++) {
                    const uint8x16_t qs = vld1q_u8(w->qs + sb * QK_K + 16 * cp + 64 * k);
                    int8x16_t        q_lo, q_hi;
                    if constexpr (IS_Q5) {
                        const uint8x16_t hbit_lo = vandq_u8(qh[cp][k], mone);
                        const uint8x16_t hbit_hi = vshlq_n_u8(vandq_u8(qh[cp][k], mtwo), 3);
                        qh[cp][k]                = vshrq_n_u8(qh[cp][k], 2);
                        q_lo = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(qs, m4b), hbit_lo, 4));
                        q_hi = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), hbit_hi));
                    } else {
                        q_lo = vreinterpretq_s8_u8(vandq_u8(qs, m4b));
                        q_hi = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));
                    }
                    vst1q_s8(c + ((sb * 2 + 0) * 4 + k) * 32, q_lo);
                    vst1q_s8(c + ((sb * 2 + 1) * 4 + k) * 32, q_hi);
                }
            }
        }
    }

    template <int NP>
    OXE_PF_INLINE void tile(const oxe_pf_xblk * const * xp, const oxe_pf_qk_wblk * w, float * f) {
        const float32x2_t d01 = vld1_f32(w->d), d23 = vld1_f32(w->d + 2);
        const float32x2_t m01 = vld1_f32(w->dmin), m23 = vld1_f32(w->dmin + 2);
        const float32x4_t q_d[2]    = { vcombine_f32(d01, d01), vcombine_f32(d23, d23) };  // [a b a b]
        const float32x4_t q_dmin[2] = { vcombine_f32(m01, m01), vcombine_f32(m23, m23) };
        float32x4_t       acc[2 * NP], sb_scale[2 * NP], sb_min[2 * NP];
        for (int p = 0; p < NP; p++) {
            const float32x2_t v    = vld1_f32(xp[p]->d);
            const float32x4_t q8_d = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a a b b]
            for (int q = 0; q < 2; q++) {
                sb_scale[2 * p + q] = vmulq_f32(q_d[q], q8_d);
                sb_min[2 * p + q]   = vmulq_f32(q_dmin[q], q8_d);
                acc[2 * p + q]      = vld1q_f32(f + (2 * p + q) * 4);
            }
        }
        const int8_t * wc = w->codes;
        for (int sub = 0; sub < 8; sub++) {
            int32x4_t t[2 * NP];
            for (int i = 0; i < 2 * NP; i++) {
                t[i] = vdupq_n_s32(0);
            }
            for (int k = 0; k < 4; k++) {
                const int8x16_t w0 = vld1q_s8(wc), w1 = vld1q_s8(wc + 16);
                for (int p = 0; p < NP; p++) {
                    const int8x16_t x = vld1q_s8(xp[p]->qs + (sub * 4 + k) * 16);
                    t[2 * p]     = vmmlaq_s32(t[2 * p], x, w0);
                    t[2 * p + 1] = vmmlaq_s32(t[2 * p + 1], x, w1);
                }
                wc += 32;
            }
            const int32x4_t sc = vmovl_s16(vld1_s16(w->sc + sub * 4));
            const int32x4_t S[2] = { vcombine_s32(vget_low_s32(sc), vget_low_s32(sc)), vcombine_s32(vget_high_s32(sc), vget_high_s32(sc)) };
            for (int i = 0; i < 2 * NP; i++) {
                acc[i] = vfmaq_f32(acc[i], sb_scale[i], vcvtq_f32_s32(vmulq_s32(t[i], S[i & 1])));
            }
            if constexpr (IS_Q5) {
                if (sub & 1) {
                    // min correction of the 64-weight pair (sub - 1, sub)
                    const int32x4_t ma = vmovl_s16(vld1_s16(w->mn16 + (sub - 1) * 4));
                    const int32x4_t mb = vmovl_s16(vld1_s16(w->mn16 + sub * 4));
                    const int32x4_t MA[2] = { vcombine_s32(vget_low_s32(ma), vget_low_s32(ma)), vcombine_s32(vget_high_s32(ma), vget_high_s32(ma)) };
                    const int32x4_t MB[2] = { vcombine_s32(vget_low_s32(mb), vget_low_s32(mb)), vcombine_s32(vget_high_s32(mb), vget_high_s32(mb)) };
                    for (int p = 0; p < NP; p++) {
                        const int32x4_t ba = vld1q_s32(xp[p]->bs32[sub - 1]);
                        const int32x4_t bb = vld1q_s32(xp[p]->bs32[sub]);
                        for (int q = 0; q < 2; q++) {
                            const int32x4_t bias = vmlaq_s32(vmulq_s32(ba, MA[q]), bb, MB[q]);
                            acc[2 * p + q]       = vmlsq_f32(acc[2 * p + q], sb_min[2 * p + q], vcvtq_f32_s32(bias));
                        }
                    }
                }
            }
        }
        if constexpr (!IS_Q5) {
            const int8x16_t mn[2] = { vld1q_s8(w->mins), vld1q_s8(w->mins + 16) };
            for (int p = 0; p < NP; p++) {
                const uint8x16_t blo = vld1q_u8(xp[p]->blo);
                const int8x16_t  bhi = vld1q_s8(xp[p]->bhi);
                for (int q = 0; q < 2; q++) {
                    const int32x4_t bl   = vusmmlaq_s32(vdupq_n_s32(0), blo, mn[q]);
                    const int32x4_t bh   = vmmlaq_s32(vdupq_n_s32(0), bhi, mn[q]);
                    const int32x4_t bias = vaddq_s32(bl, vshlq_n_s32(bh, 8));
                    acc[2 * p + q]       = vmlsq_f32(acc[2 * p + q], vcvtq_f32_s32(bias), sb_min[2 * p + q]);
                }
            }
        }
        for (int i = 0; i < 2 * NP; i++) {
            vst1q_f32(f + i * 4, acc[i]);
        }
    }
};

// Driver: orow[t] + c = W[c] . row t for t < nr, c < nc (nc % 8 == 0); xq = ceil(nr / 2) pairs x nb blocks.
// Per 64-row chunk the loop is block-major: one 256-block of the chunk is decoded (19 KB, stays in
// L1) and used at once by every pair group; the float accumulators of all (pair group, row quad)
// live in F between blocks.
template <class T>
static void oxe_pf_mmid_run(int n, float * const * orow, const void * vx, const oxe_pf_xblk * xq, int nr, int nc) {
    static thread_local oxe_pf_scratch scratch;
    typedef typename T::wblk wblk;
    const int nb     = n / QK_K;
    const int npairs = (nr + 1) / 2;
    const int ngrp   = (npairs + 3) / 4;
    if (nr <= 0 || nc <= 0) {
        return;
    }
    const size_t wbytes = (((size_t) (OXE_PF_RC / 4) * sizeof(wblk)) + 63) & ~(size_t) 63;
    char *       mem    = (char *) scratch.get(wbytes + (size_t) ngrp * (OXE_PF_RC / 4) * 32 * sizeof(float));
    wblk *       wd     = (wblk *) mem;              // [quad] of the current block
    float *      F      = (float *) (mem + wbytes);  // [group][quad][(2p + q) * 4]
    const typename T::xblock * W = (const typename T::xblock *) vx;
    for (int c0 = 0; c0 < nc; c0 += OXE_PF_RC) {
        const int ncq = ((nc - c0) < OXE_PF_RC ? (nc - c0) : OXE_PF_RC) / 4;
        for (int i = 0; i < ngrp * ncq * 32; i += 4) {
            vst1q_f32(F + i, vdupq_n_f32(0));
        }
        for (int b = 0; b < nb; b++) {
            // (A software prefetch of block b + 1 of every row group was measured here on 2026-11-10 with
            // cold weights, 96 experts per pass: 8 rows 115 -> 104 GMAC/s (Q3_K), slower at every count
            // and type. The hardware prefetcher follows the 8 strided streams; leave it alone.)
            for (int g = 0; g < ncq / 2; g++) {
                T::decode(W + (size_t) (c0 / 8 + g) * nb + b, wd + 2 * g, wd + 2 * g + 1);
            }
            for (int gi = 0; gi < ngrp; gi++) {
                const int           p0 = gi * 4;
                const int           np = (npairs - p0) < 4 ? (npairs - p0) : 4;
                const oxe_pf_xblk * xp[4];
                for (int p = 0; p < np; p++) {
                    xp[p] = xq + (size_t) (p0 + p) * nb + b;
                }
                float * f = F + (size_t) gi * ncq * 32;
                switch (np) {
                    case 4: for (int cq = 0; cq < ncq; cq++) { T::template tile<4>(xp, wd + cq, f + cq * 32); } break;
                    case 3: for (int cq = 0; cq < ncq; cq++) { T::template tile<3>(xp, wd + cq, f + cq * 32); } break;
                    case 2: for (int cq = 0; cq < ncq; cq++) { T::template tile<2>(xp, wd + cq, f + cq * 32); } break;
                    default: for (int cq = 0; cq < ncq; cq++) { T::template tile<1>(xp, wd + cq, f + cq * 32); } break;
                }
            }
        }
        for (int p = 0; p < npairs; p++) {
            const int     t0 = 2 * p;
            const float * f  = F + (size_t) (p / 4) * ncq * 32 + (2 * (p % 4)) * 4;
            for (int cq = 0; cq < ncq; cq++) {
                const float32x4_t v0 = vld1q_f32(f + cq * 32);      // [t0 r0, t0 r1, t1 r0, t1 r1]
                const float32x4_t v1 = vld1q_f32(f + cq * 32 + 4);  // rows 2, 3
                vst1q_f32(orow[t0] + c0 + cq * 4, vcombine_f32(vget_low_f32(v0), vget_low_f32(v1)));
                if (t0 + 1 < nr) {
                    vst1q_f32(orow[t0 + 1] + c0 + cq * 4, vcombine_f32(vget_high_f32(v0), vget_high_f32(v1)));
                }
            }
        }
    }
}

// Dense driver: s[t * bs + c] = W[c] . a[t] for t < nr (multiple of 4), c < nc (multiple of 8), tokens
// in 4-row interleaved q8_K groups (the forward_mul_mat layout). Block-major per (64-row chunk,
// 64-token block): one decoded 256-block of the chunk serves all token groups of the block.
template <class T>
static void oxe_pf_dense_run(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    static thread_local oxe_pf_scratch scratch;
    typedef typename T::wblk wblk;
    const int nb = n / QK_K;
    if (nr <= 0 || nc <= 0) {
        return;
    }
    const size_t wbytes = (((size_t) (OXE_PF_RC / 4) * sizeof(wblk)) + 63) & ~(size_t) 63;
    char *       mem    = (char *) scratch.get(wbytes + (size_t) (OXE_PF_TB / 8) * (OXE_PF_RC / 4) * 32 * sizeof(float));
    wblk *       wd     = (wblk *) mem;              // [quad] of the current block
    float *      F      = (float *) (mem + wbytes);  // [token octet][quad][(2p + q) * 4]
    const typename T::xblock * W = (const typename T::xblock *) vx;
    const block_q8_Kx4 *       A = (const block_q8_Kx4 *) vy;
    for (int c0 = 0; c0 < nc; c0 += OXE_PF_RC) {
        const int ncq = ((nc - c0) < OXE_PF_RC ? (nc - c0) : OXE_PF_RC) / 4;
        for (int t0 = 0; t0 < nr; t0 += OXE_PF_TB) {
            const int ntq  = ((nr - t0) < OXE_PF_TB ? (nr - t0) : OXE_PF_TB) / 4;  // 4-token groups
            const int ngrp = (ntq + 1) / 2;                                           // octets (the last may be a quad)
            for (int i = 0; i < ngrp * ncq * 32; i += 4) {
                vst1q_f32(F + i, vdupq_n_f32(0));
            }
            for (int b = 0; b < nb; b++) {
                for (int g = 0; g < ncq / 2; g++) {
                    T::decode(W + (size_t) (c0 / 8 + g) * nb + b, wd + 2 * g, wd + 2 * g + 1);
                }
                for (int gi = 0; gi < ngrp; gi++) {
                    const int      np = (2 * gi + 1 < ntq) ? 4 : 2;
                    const int8_t * xq[4];
                    const float *  xd[4];
                    for (int h = 0; h < np / 2; h++) {
                        const block_q8_Kx4 * a = A + (size_t) (t0 / 4 + 2 * gi + h) * nb + b;
                        xq[2 * h]     = a->qs;       // tokens 0, 1 of the group
                        xq[2 * h + 1] = a->qs + 16;  // tokens 2, 3
                        xd[2 * h]     = a->d;
                        xd[2 * h + 1] = a->d + 2;
                    }
                    float * f = F + (size_t) gi * ncq * 32;
                    if (np == 4) {
                        for (int cq = 0; cq < ncq; cq++) { T::template tile_k4<4>(xq, xd, wd + cq, f + cq * 32); }
                    } else {
                        for (int cq = 0; cq < ncq; cq++) { T::template tile_k4<2>(xq, xd, wd + cq, f + cq * 32); }
                    }
                }
            }
            for (int tp = 0; tp < ntq * 2; tp++) {  // token pairs of the block
                const float * f = F + (size_t) (tp / 4) * ncq * 32 + (2 * (tp % 4)) * 4;
                float *       r0 = s + (size_t) (t0 + 2 * tp) * bs + c0;
                float *       r1 = r0 + bs;
                for (int cq = 0; cq < ncq; cq++) {
                    const float32x4_t v0 = vld1q_f32(f + cq * 32);
                    const float32x4_t v1 = vld1q_f32(f + cq * 32 + 4);
                    vst1q_f32(r0 + cq * 4, vcombine_f32(vget_low_f32(v0), vget_low_f32(v1)));
                    vst1q_f32(r1 + cq * 4, vcombine_f32(vget_high_f32(v0), vget_high_f32(v1)));
                }
            }
        }
    }
}

}  // namespace

extern "C" {

size_t oxe_pf_xblk_bytes(void) { return sizeof(oxe_pf_xblk); }

// Pack `nr` from_float q8_K rows (rows[t] -> n / 256 block_q8_K) into ceil(nr / 2) pairs x n / 256
// oxe_pf_xblk (an odd count repeats the last row).
void oxe_pf_pack_q8K_pairs(const void * const * rows, int nr, int n, void * out) {
    const int     nb = n / QK_K;
    oxe_pf_xblk * o  = (oxe_pf_xblk *) out;
    for (int p = 0; p < (nr + 1) / 2; p++) {
        const block_q8_K * a = (const block_q8_K *) rows[2 * p];
        const block_q8_K * b = (const block_q8_K *) rows[(2 * p + 1 < nr) ? 2 * p + 1 : 2 * p];
        for (int i = 0; i < nb; i++) {
            oxe_pf_pack_pair_blk(a + i, b + i, o + (size_t) p * nb + i);
        }
    }
}

// type: 3 = Q3_K, 4 = Q4_K, 5 = Q5_K, 6 = Q6_K (the MUL_MAT_ID kernels of those types). Returns 0 if the
// type has no tile kernel (nothing written).
int oxe_pf_mmid_tile(int type, int n, float * const * orow, const void * vx, const void * xq, int nr, int nc) {
    switch (type) {
        case 3: oxe_pf_mmid_run<oxe_pf_q3k>(n, orow, vx, (const oxe_pf_xblk *) xq, nr, nc); return 1;
        case 4: oxe_pf_mmid_run<oxe_pf_qk<false>>(n, orow, vx, (const oxe_pf_xblk *) xq, nr, nc); return 1;
        case 5: oxe_pf_mmid_run<oxe_pf_qk<true>>(n, orow, vx, (const oxe_pf_xblk *) xq, nr, nc); return 1;
        case 6: oxe_pf_mmid_run<oxe_pf_q6k>(n, orow, vx, (const oxe_pf_xblk *) xq, nr, nc); return 1;
        default: return 0;
    }
}

// Drop-in for ggml_gemm_q6_K_8x8_q8_K (same arguments, same bits).
void oxe_pf_gemm_q6_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    oxe_pf_dense_run<oxe_pf_q6k>(n, s, bs, vx, vy, nr, nc);
}

}  // extern "C"

#else

extern "C" {
void oxe_pf_gemm_q6_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}
size_t oxe_pf_xblk_bytes(void) { return 688; }
void oxe_pf_pack_q8K_pairs(const void * const * rows, int nr, int n, void * out) {}
int oxe_pf_mmid_tile(int type, int n, float * const * orow, const void * vx, const void * xq, int nr, int nc) { return 0; }
}

#endif
