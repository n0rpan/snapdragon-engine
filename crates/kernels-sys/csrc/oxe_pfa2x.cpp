// v0.1.3 APPEND-PREFILL (fa2x): register-blocked flash-attention-2 prefill unit for the f16 KV
// cache (hand-written, not upstream). One call computes one unit of `pfattn::Tile`: ncol query
// columns of ONE KV head (8 tokens x the 8 query heads of the group in the engine: every K / V^T
// row is read once per unit and used by all 64 columns, the GQA reuse) against positions
// [0, last + 1), walking fixed absolute chunks of `ch` positions.
//
// Per chunk:
//   S[r][c]  = K[s0 + r] . q_c                     (8x16 micro-kernel, k = head dim)
//   online softmax per column (vertical over r, 4 columns per f32 vector): m' = max(m, max_r S*scale),
//   P[r][c]  = f16(exp(S*scale - m')) (+0 when masked), l = l * exp(m - m') + sum_r P (f32)
//   O[d][c]  = O[d][c] * exp(m - m') + V^T[d][s0 .. s0 + len) . P[.][c]   (8x16 micro-kernel, k = position)
// and out[c][d] = O[d][c] * (1 / l[c]) at the end.
//
// Micro-kernel (both products): C[i][c] += sum_k B[i][k] * A[k][c] for 8 rows i x 16 columns c:
// 16 f16x8 accumulators, B row i loaded as 8 consecutive k (K row / V^T row segment, as the cache
// stores them) and broadcast by lane (FMLA by element), A = Q^T / P^T rows of 16 columns
// (2 vector loads per k, reused by the 8 rows). 2 loads + 1/8 B load per 16 FMLA.
//
// Numerics (class of O-70 / fa2: f16 products, f32 combination; NOT the ggml FA-off order):
// every output accumulates in f16 over `fold` consecutive k only (8, 16 or 32), then the f16
// partial is added to the f32 accumulator (FMLAL by 1.0 where available: the f16 product is exact,
// so it equals vcvt + vaddq bit for bit and the fallback gives the same bits). fold = 8 is a shorter
// f16 chain than ggml_vec_dot_f16 (8 terms per accumulator, then an f16 tree of 32) for the scores
// and than fa2's chunk-local V^T . p (the same tile kernel over 256 positions).
//
// Invariance: a unit is one thread, chunks in order from 0: thread-count invariant. A column's
// result does not depend on the unit it is in: chunk boundaries are absolute, every chunk's
// V^T . P runs over the whole chunk (len, up to n_ctx: the tail rule; V^T must be finite there)
// with exact +0 weights beyond the column's position, and a chunk in which a column has no valid
// position leaves its O, m, l untouched (select, not arithmetic). Scores of rows beyond the
// unit's last position are never computed; K rows are read on [0, round_up(last + 1, 8)) only
// within the chunk (masked by select: their values never reach a weight).
#include "ggml-cpu/vec.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr int PX_MR = 8;       // rows per micro tile
constexpr int PX_NR = 16;      // columns per micro tile (2 x f16x8)
constexpr int PX_MAXCOL = 256;
constexpr int PX_MAXCH = 512;

static inline int64_t px_al(int64_t b) { return (b + 63) & ~(int64_t) 63; }

// c + f32(a.lo) / c + f32(a.hi), exact product then one rounding: equal to vaddq(c, vcvt(a)).
static inline float32x4_t px_fold_lo(float32x4_t c, float16x8_t a) {
#if defined(__ARM_FEATURE_FP16_FML)
    return vfmlalq_low_f16(c, a, vdupq_n_f16((__fp16) 1.0f));
#else
    return vaddq_f32(c, vcvt_f32_f16(vget_low_f16(a)));
#endif
}
static inline float32x4_t px_fold_hi(float32x4_t c, float16x8_t a) {
#if defined(__ARM_FEATURE_FP16_FML)
    return vfmlalq_high_f16(c, a, vdupq_n_f16((__fp16) 1.0f));
#else
    return vaddq_f32(c, vcvt_high_f32_f16(a));
#endif
}

// one k of the 8 x 16 tile: lane L of the 8 B vectors times the A row's two vectors
template <int L, bool FIRST>
static inline void px_step(float16x8_t (&acc)[PX_MR][2], const float16x8_t (&b)[PX_MR], const __fp16 * a) {
    const float16x8_t a0 = vld1q_f16(a);
    const float16x8_t a1 = vld1q_f16(a + 8);
    for (int i = 0; i < PX_MR; i++) {
        if (FIRST) {
            acc[i][0] = vmulq_laneq_f16(a0, b[i], L);
            acc[i][1] = vmulq_laneq_f16(a1, b[i], L);
        } else {
            acc[i][0] = vfmaq_laneq_f16(acc[i][0], a0, b[i], L);
            acc[i][1] = vfmaq_laneq_f16(acc[i][1], a1, b[i], L);
        }
    }
}

// 8 consecutive k (one B vector per row)
template <bool FIRST>
static inline void px_k8(float16x8_t (&acc)[PX_MR][2], const __fp16 * B, int64_t bs, const __fp16 * A, int64_t as, int k) {
    float16x8_t b[PX_MR];
    for (int i = 0; i < PX_MR; i++) {
        b[i] = vld1q_f16(B + i * bs + k);
    }
    const __fp16 * a = A + (int64_t) k * as;
    px_step<0, FIRST>(acc, b, a);
    px_step<1, false>(acc, b, a + as);
    px_step<2, false>(acc, b, a + 2 * as);
    px_step<3, false>(acc, b, a + 3 * as);
    px_step<4, false>(acc, b, a + 4 * as);
    px_step<5, false>(acc, b, a + 5 * as);
    px_step<6, false>(acc, b, a + 6 * as);
    px_step<7, false>(acc, b, a + 7 * as);
}

// Scores: C[i][c] (f32, row stride cs) = sum over k < K of B[i][k] * A[k][c], f16 chains of FOLD.
template <int FOLD>
static inline void px_mk_qk(int K, const __fp16 * B, int64_t bs, const __fp16 * A, int64_t as, float * C, int64_t cs) {
    for (int k0 = 0; k0 < K; k0 += FOLD) {
        float16x8_t acc[PX_MR][2];
        px_k8<true>(acc, B, bs, A, as, k0);
        for (int k = k0 + 8; k < k0 + FOLD; k += 8) {
            px_k8<false>(acc, B, bs, A, as, k);
        }
        for (int i = 0; i < PX_MR; i++) {
            float * c = C + i * cs;
            for (int h = 0; h < 2; h++) {
                if (k0 == 0) {
                    vst1q_f32(c + 8 * h, vcvt_f32_f16(vget_low_f16(acc[i][h])));
                    vst1q_f32(c + 8 * h + 4, vcvt_high_f32_f16(acc[i][h]));
                } else {
                    vst1q_f32(c + 8 * h, px_fold_lo(vld1q_f32(c + 8 * h), acc[i][h]));
                    vst1q_f32(c + 8 * h + 4, px_fold_hi(vld1q_f32(c + 8 * h + 4), acc[i][h]));
                }
            }
        }
    }
}

// Outputs: O[i][c] (f32, row stride cs) = valid_c ? O * alpha_c + sum_k B[i][k] * A[k][c] : O,
// f16 chains of FOLD, the rescale on the first fold (alpha / valid: 16 columns).
template <int FOLD>
static inline void px_mk_pv(int K, const __fp16 * B, int64_t bs, const __fp16 * A, int64_t as, float * C, int64_t cs,
                            const float * alpha, const uint32_t * valid) {
    for (int k0 = 0; k0 < K; k0 += FOLD) {
        float16x8_t acc[PX_MR][2];
        px_k8<true>(acc, B, bs, A, as, k0);
        for (int k = k0 + 8; k < k0 + FOLD; k += 8) {
            px_k8<false>(acc, B, bs, A, as, k);
        }
        for (int q = 0; q < 4; q++) {
            const uint32x4_t vm = vld1q_u32(valid + 4 * q);
            const float32x4_t va = vld1q_f32(alpha + 4 * q);
            for (int i = 0; i < PX_MR; i++) {
                float * c = C + i * cs + 4 * q;
                const float32x4_t o = vld1q_f32(c);
                float32x4_t n = k0 == 0 ? vmulq_f32(o, va) : o;
                n = (q & 1) ? px_fold_hi(n, acc[i][q >> 1]) : px_fold_lo(n, acc[i][q >> 1]);
                vst1q_f32(c, vbslq_f32(vm, n, o));
            }
        }
    }
}

typedef void (*px_qk_fn)(int, const __fp16 *, int64_t, const __fp16 *, int64_t, float *, int64_t);
typedef void (*px_pv_fn)(int, const __fp16 *, int64_t, const __fp16 *, int64_t, float *, int64_t, const float *, const uint32_t *);

struct PxLayout {
    int ncp;
    int64_t qt, s, p, o, m, l, al, vm, lim, total;
};

static PxLayout px_layout(int hd, int ncol, int ch) {
    PxLayout L;
    L.ncp = (ncol + PX_NR - 1) / PX_NR * PX_NR;
    int64_t at = 0;
    L.qt = at;  at += px_al((int64_t) hd * L.ncp * 2);   // Q^T [hd][ncp] f16
    L.s = at;   at += px_al((int64_t) ch * L.ncp * 4);   // S [ch][ncp] f32
    L.p = at;   at += px_al((int64_t) ch * L.ncp * 2);   // P [ch][ncp] f16
    L.o = at;   at += px_al((int64_t) hd * L.ncp * 4);   // O [hd][ncp] f32
    L.m = at;   at += px_al((int64_t) L.ncp * 4);
    L.l = at;   at += px_al((int64_t) L.ncp * 4);
    L.al = at;  at += px_al((int64_t) L.ncp * 4);
    L.vm = at;  at += px_al((int64_t) L.ncp * 4);
    L.lim = at; at += px_al((int64_t) L.ncp * 4);
    L.total = at + 64;  // the caller's buffer is aligned up to 64 inside
    return L;
}

}  // namespace

extern "C" {

// Scratch bytes of oxe_pfa2x_unit for up to ncol columns of hd dims and chunks of up to ch positions.
int64_t oxe_pfa2x_scratch_bytes(int hd, int ncol, int ch) {
    return px_layout(hd, std::max(ncol, 1), ch).total;
}

// One fa2x unit (see the header). Columns c = 0..ncol: token pos0 + c / group, query q16 + c * hd
// (f16); K rows k + p * hd; V^T rows vt + d * v_stride (finite up to n_ctx); out[c * hd + d]
// (ungated, normalised). ch: chunk positions (multiple of 32, <= 512); fold: 8, 16 or 32.
// Requires hd % 32 == 0, n_ctx % 32 == 0, last < n_ctx, ncol <= 256; returns 0, or -1 when a
// requirement fails (nothing written).
int oxe_pfa2x_unit(int hd, int ncol, int group, int pos0, float scale, const ggml_fp16_t * k, const ggml_fp16_t * vt,
                   int64_t v_stride, int n_ctx, const ggml_fp16_t * q16, int ch, int fold, void * scratch, int64_t scratch_bytes,
                   float * out) {
    if (hd <= 0 || hd % 32 || ncol < 1 || ncol > PX_MAXCOL || group < 1 || pos0 < 0 || n_ctx % 32 || ch < 32 || ch % 32 ||
        ch > PX_MAXCH || !(fold == 8 || fold == 16 || fold == 32)) {
        return -1;
    }
    const int last = pos0 + (ncol - 1) / group;
    if (last >= n_ctx) {
        return -1;
    }
    const PxLayout L = px_layout(hd, ncol, ch);
    if (scratch_bytes < L.total) {
        return -1;
    }
    uint8_t * base = (uint8_t *) (((uintptr_t) scratch + 63) & ~(uintptr_t) 63);
    const int ncp = L.ncp;
    __fp16 * qt = (__fp16 *) (base + L.qt);
    float * S = (float *) (base + L.s);
    __fp16 * P = (__fp16 *) (base + L.p);
    float * O = (float *) (base + L.o);
    float * M = (float *) (base + L.m);
    float * Ls = (float *) (base + L.l);
    float * AL = (float *) (base + L.al);
    uint32_t * VM = (uint32_t *) (base + L.vm);
    int32_t * LIM = (int32_t *) (base + L.lim);

    const px_qk_fn qk = fold == 8 ? px_mk_qk<8> : fold == 16 ? px_mk_qk<16> : px_mk_qk<32>;
    const px_pv_fn pv = fold == 8 ? px_mk_pv<8> : fold == 16 ? px_mk_pv<16> : px_mk_pv<32>;

    // Q^T (pad columns 0), per-column position, running max / sum, O = 0
    const __fp16 * q = (const __fp16 *) q16;
    for (int d = 0; d < hd; d++) {
        __fp16 * row = qt + (int64_t) d * ncp;
        for (int c = 0; c < ncol; c++) {
            row[c] = q[(int64_t) c * hd + d];
        }
        for (int c = ncol; c < ncp; c++) {
            row[c] = (__fp16) 0.0f;
        }
    }
    for (int c = 0; c < ncp; c++) {
        LIM[c] = c < ncol ? pos0 + c / group : -1;
        M[c] = -INFINITY;
        Ls[c] = 0.0f;
    }
    memset(O, 0, (size_t) hd * ncp * 4);

    const __fp16 * K = (const __fp16 *) k;
    const __fp16 * VT = (const __fp16 *) vt;
    const int rows = last + 1;
    const float32x4_t vscale = vdupq_n_f32(scale);
    const float32x4_t ninf = vdupq_n_f32(-INFINITY);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    for (int s0 = 0; s0 < rows; s0 += ch) {
        const int len = std::min(ch, n_ctx - s0);                 // V^T . P length (tail rule)
        const int rq = std::min((rows - s0 + 7) / 8 * 8, len);    // score rows computed
        const int nx = std::min(ch, std::max(0, n_ctx - s0 - ch)); // next chunk's positions
        // [1] scores, K rows of 8 positions outer (read once), 16-column tiles inner (Q^T in L1)
        for (int r0 = 0; r0 < rq; r0 += PX_MR) {
            const __fp16 * kr = K + (int64_t) (s0 + r0) * hd;
            if (r0 < nx && s0 + ch + r0 < rows) {
                const char * pf = (const char *) (K + (int64_t) (s0 + ch + r0) * hd);
                for (int b = 0; b < PX_MR * hd * 2; b += 64) {
                    __builtin_prefetch(pf + b, 0, 3);
                }
            }
            for (int cb = 0; cb < ncp; cb += PX_NR) {
                qk(hd, kr, hd, qt + cb, ncp, S + (int64_t) r0 * ncp + cb, ncp);
            }
        }
        // [2] online softmax, 4 columns at a time, vertical over the chunk's rows
        for (int c = 0; c < ncp; c += 4) {
            const int32x4_t lim = vsubq_s32(vld1q_s32(LIM + c), vdupq_n_s32(s0));
            float32x4_t mx = ninf;
            for (int r = 0; r < rq; r++) {
                const uint32x4_t ok = vcleq_s32(vdupq_n_s32(r), lim);
                const float32x4_t x = vmulq_f32(vld1q_f32(S + (int64_t) r * ncp + c), vscale);
                vst1q_f32(S + (int64_t) r * ncp + c, x);
                mx = vmaxq_f32(mx, vbslq_f32(ok, x, ninf));
            }
            const float32x4_t mo = vld1q_f32(M + c);
            const float32x4_t mn = vmaxq_f32(mo, mx);
            // a column with no position yet (pad) keeps alpha = 1 (no -inf - -inf)
            const uint32x4_t fin = vcgtq_f32(mn, ninf);
            const float32x4_t al = vbslq_f32(fin, ggml_v_expf(vsubq_f32(mo, mn)), one);
            const float32x4_t mexp = vbslq_f32(fin, mn, zero);
            float32x4_t ls = zero;
            for (int r = 0; r < rq; r++) {
                const uint32x4_t ok = vcleq_s32(vdupq_n_s32(r), lim);
                const float32x4_t e = vbslq_f32(ok, ggml_v_expf(vsubq_f32(vld1q_f32(S + (int64_t) r * ncp + c), mexp)), zero);
                const float16x4_t h = vcvt_f16_f32(e);
                vst1_f16(P + (int64_t) r * ncp + c, h);
                ls = vaddq_f32(ls, vcvt_f32_f16(h));
            }
            for (int r = rq; r < len; r++) {
                vst1_f16(P + (int64_t) r * ncp + c, vdup_n_f16((__fp16) 0.0f));
            }
            // valid in this chunk: the column's position reaches s0
            const uint32x4_t any = vcgeq_s32(lim, vdupq_n_s32(0));
            vst1q_u32(VM + c, any);
            vst1q_f32(AL + c, vbslq_f32(any, al, one));
            vst1q_f32(Ls + c, vbslq_f32(any, vaddq_f32(vmulq_f32(vld1q_f32(Ls + c), al), ls), vld1q_f32(Ls + c)));
            vst1q_f32(M + c, vbslq_f32(any, mn, mo));
        }
        // [3] O = O * alpha + V^T . P over the whole chunk, 8 V^T rows outer (read once)
        for (int d0 = 0; d0 < hd; d0 += PX_MR) {
            const __fp16 * vr = VT + (int64_t) d0 * v_stride + s0;
            if (nx > 0 && s0 + ch < rows) {
                for (int i = 0; i < PX_MR; i++) {
                    const char * pf = (const char *) (vr + i * v_stride + ch);
                    for (int b = 0; b < nx * 2; b += 64) {
                        __builtin_prefetch(pf + b, 0, 3);
                    }
                }
            }
            for (int cb = 0; cb < ncp; cb += PX_NR) {
                pv(len, vr, v_stride, P + cb, ncp, O + (int64_t) d0 * ncp + cb, ncp, AL + cb, VM + cb);
            }
        }
    }
    // [4] normalise and transpose: out[c][d] = O[d][c] * (1 / l)
    for (int c = 0; c < ncol; c++) {
        const float inv = 1.0f / Ls[c];
        float * o = out + (int64_t) c * hd;
        for (int d = 0; d < hd; d++) {
            o[d] = O[(int64_t) d * ncp + c] * inv;
        }
    }
    return 0;
}

}  // extern "C"
