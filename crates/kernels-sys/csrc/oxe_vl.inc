// oxe_vl.inc: verify-lane GEMVs (hand-written, not upstream). OD-6b.
// Included at the end of oxe_shim.cpp, i.e. in the translation unit and with the flags of the
// vendored repack kernels, so every float intrinsic is compiled like the same intrinsic in the
// reference GEMV it mirrors.
//
// Why. A speculative verify runs 2-4 lanes (tokens) through the same weight rows. Today every lane
// calls the one-column GEMV on a cache-warm chunk, so the weight block is loaded, unpacked (nibbles,
// 6-bit scales, mins) and dotted once per lane. Here the weight rows are read and unpacked ONCE and
// dotted with a PAIR of lanes per instruction: the interleaved formats hold [row r: 8 codes | row
// r+1: 8 codes] per 16 bytes, and with X = [lane a: 8 codes | lane b: 8 codes]
//     vmmlaq_s32(acc, W, X)  adds  [r.a, r.b, r'.a, r'.b]
// i.e. one SMMLA does for two lanes what one SDOT does for one lane in the reference. The float
// work (scale products, one or two fma per sub-block or block) stays per lane.
//
// Exactness contract: per (lane, weight row) the integers formed are the same integers as in the
// reference GEMV (exact arithmetic, so their grouping is free) and the float updates are the same
// intrinsics with the same operand values in the same order, from acc = +0. => bitwise equal to
// `ggml_gemv_<type>` called once per lane. Checked by tools/vl_check and the kernels-sys test
// `vl_gemv_bitwise`.
//
// Lane layout of every 4-lane vector below, for lane pair (a, b) and row pair (r, r') = (2q, 2q+1):
//     [r.a, r.b, r'.a, r'.b]
// An odd lane count pairs the last lane with itself and drops the copy.
//
// One file per weight type. OXE_VL_ONLY=<type> (3, 4, 5, 6, 80 for Q8_0) compiles one of them alone
// (tools/vl_check while the others are being written).

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)

#define OXE_VL_INLINE static inline __attribute__((always_inline))
#define OXE_VL_MAX_LANES 4

namespace {
// X operand of lane pair (a, b) for one 8-weight step: [a: 8 codes | b: 8 codes]
OXE_VL_INLINE int8x16_t oxe_vl_x(const int8_t * a, const int8_t * b) { return vcombine_s8(vld1_s8(a), vld1_s8(b)); }
// [v0, v0, v1, v1] and [v2, v2, v3, v3] of a 4-float vector (per-row values in the lane layout)
OXE_VL_INLINE float32x4_t oxe_vl_rr_lo(float32x4_t v) { return vzip1q_f32(v, v); }
OXE_VL_INLINE float32x4_t oxe_vl_rr_hi(float32x4_t v) { return vzip2q_f32(v, v); }
// store the 8 rows of one row group for lane pair p: f[q] = [r(2q).a, r(2q).b, r(2q+1).a, r(2q+1).b]
OXE_VL_INLINE void oxe_vl_store8(const float32x4_t * f, float * oa, float * ob) {
    vst1q_f32(oa, vuzp1q_f32(f[0], f[1]));
    vst1q_f32(oa + 4, vuzp1q_f32(f[2], f[3]));
    if (ob) {
        vst1q_f32(ob, vuzp2q_f32(f[0], f[1]));
        vst1q_f32(ob + 4, vuzp2q_f32(f[2], f[3]));
    }
}
}  // namespace

#define OXE_VL_HAVE 1
#if !defined(OXE_VL_ONLY) || OXE_VL_ONLY == 4
#include "oxe_vl_q4k.inc"
#define OXE_VL_HAVE_Q4K 1
#endif
#if (!defined(OXE_VL_ONLY) || OXE_VL_ONLY == 6) && __has_include("oxe_vl_q6k.inc")
#include "oxe_vl_q6k.inc"
#define OXE_VL_HAVE_Q6K 1
#endif
#if (!defined(OXE_VL_ONLY) || OXE_VL_ONLY == 3) && __has_include("oxe_vl_q3k.inc")
#include "oxe_vl_q3k.inc"
#define OXE_VL_HAVE_Q3K 1
#endif
#if (!defined(OXE_VL_ONLY) || OXE_VL_ONLY == 5) && __has_include("oxe_vl_q5k.inc")
#include "oxe_vl_q5k.inc"
#define OXE_VL_HAVE_Q5K 1
#endif
#if (!defined(OXE_VL_ONLY) || OXE_VL_ONLY == 80) && __has_include("oxe_vl_q80.inc")
#include "oxe_vl_q80.inc"
#define OXE_VL_HAVE_Q80 1
#endif

#endif  // aarch64 + NEON + i8mm

extern "C" {
// Lane GEMV: s[l][c] = W[c] . y[l] for l < nl (1..4; one lane is paired with itself, which only pays
// for Q6_K, whose reference GEMV is slow), c < nc (multiple of the row group); vx as for
// ggml_gemv_<type>, y[l] the lane's activation blocks (from_float q8_K, or q8_0 for type 80).
// type: 3 = Q3_K, 4 = Q4_K, 5 = Q5_K, 6 = Q6_K, 80 = Q8_0. Returns 0 (nothing written) if this build
// has no lane kernel for the type or nl is out of range; the caller then runs the GEMV per lane.
int oxe_vl_gemv(int type, int n, float * const * s, const void * vx, const void * const * y, int nl, int nc) {
#ifdef OXE_VL_HAVE
    if (nl < 1 || nl > OXE_VL_MAX_LANES || nc <= 0) {
        return 0;
    }
    switch (type) {
#ifdef OXE_VL_HAVE_Q4K
        case 4: oxe_vl_gemv_q4_K(n, s, vx, y, nl, nc); return 1;
#endif
#ifdef OXE_VL_HAVE_Q6K
        case 6: oxe_vl_gemv_q6_K(n, s, vx, y, nl, nc); return 1;
#endif
#ifdef OXE_VL_HAVE_Q3K
        case 3: oxe_vl_gemv_q3_K(n, s, vx, y, nl, nc); return 1;
#endif
#ifdef OXE_VL_HAVE_Q5K
        case 5: oxe_vl_gemv_q5_K(n, s, vx, y, nl, nc); return 1;
#endif
#ifdef OXE_VL_HAVE_Q80
        case 80: oxe_vl_gemv_q8_0(n, s, vx, y, nl, nc); return 1;
#endif
        default: return 0;
    }
#else
    return 0;
#endif
}
}  // extern "C"
