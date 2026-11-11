// Float flash-attention micro-kernels (F-IMG-ATTN): f32, or f16 operands with f32 accumulation.
//
// Layouts (one head, `hd` a multiple of 8, tokens padded to a multiple of 8):
//   qt  [query group][d][8 queries]   q already RMS-normed, scaled, RoPE'd and times 1/sqrt(hd)
//   kt  [key step]   [d][8 keys]
//   vt  [dim tile]   [key][8 dims]    dim tile = 8 consecutive dims; tile stride = n_pad * 8 elements
//   pt  [key][8 queries]              probabilities of one key chunk
//   s   [8 queries][stride]           scores (f32)
//   o   [8 queries][hd]               output accumulators (f32)
//
// f16 kernels use FMLAL/FMLAL2 (FEAT_FHM): every product is exact in f32 and is added in f32, so the
// only f16 rounding is that of the stored operands. Summation order: sequential over d (q.k) and
// over keys (p.v), one fused multiply-add per term. This file is compiled on its own with
// +fp16fml (see build.rs) so the flags of the int8 kernels do not change.
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#define ZERO4 vdupq_n_f32(0.0f)

// ---- q.k^T -------------------------------------------------------------------------------------
// s[qi*ss + 8*st + j] = sum_d qt[d*8+qi] * kt[st*hd*8 + d*8 + j], for st in 0..steps.
void i8g_fattn_qk_f32(const float *restrict qt, const float *restrict kt, size_t hd, size_t steps, float *restrict s,
                      size_t ss) {
    for (size_t st = 0; st < steps; st++) {
        const float *k = kt + st * hd * 8;
        float32x4_t a00 = ZERO4, a01 = ZERO4, a10 = ZERO4, a11 = ZERO4, a20 = ZERO4, a21 = ZERO4, a30 = ZERO4, a31 = ZERO4;
        float32x4_t a40 = ZERO4, a41 = ZERO4, a50 = ZERO4, a51 = ZERO4, a60 = ZERO4, a61 = ZERO4, a70 = ZERO4, a71 = ZERO4;
        for (size_t d = 0; d < hd; d++) {
            const float32x4_t k0 = vld1q_f32(k + d * 8), k1 = vld1q_f32(k + d * 8 + 4);
            const float32x4_t q0 = vld1q_f32(qt + d * 8), q1 = vld1q_f32(qt + d * 8 + 4);
            a00 = vfmaq_laneq_f32(a00, k0, q0, 0);
            a01 = vfmaq_laneq_f32(a01, k1, q0, 0);
            a10 = vfmaq_laneq_f32(a10, k0, q0, 1);
            a11 = vfmaq_laneq_f32(a11, k1, q0, 1);
            a20 = vfmaq_laneq_f32(a20, k0, q0, 2);
            a21 = vfmaq_laneq_f32(a21, k1, q0, 2);
            a30 = vfmaq_laneq_f32(a30, k0, q0, 3);
            a31 = vfmaq_laneq_f32(a31, k1, q0, 3);
            a40 = vfmaq_laneq_f32(a40, k0, q1, 0);
            a41 = vfmaq_laneq_f32(a41, k1, q1, 0);
            a50 = vfmaq_laneq_f32(a50, k0, q1, 1);
            a51 = vfmaq_laneq_f32(a51, k1, q1, 1);
            a60 = vfmaq_laneq_f32(a60, k0, q1, 2);
            a61 = vfmaq_laneq_f32(a61, k1, q1, 2);
            a70 = vfmaq_laneq_f32(a70, k0, q1, 3);
            a71 = vfmaq_laneq_f32(a71, k1, q1, 3);
        }
        float *o = s + st * 8;
        vst1q_f32(o + 0 * ss, a00); vst1q_f32(o + 0 * ss + 4, a01);
        vst1q_f32(o + 1 * ss, a10); vst1q_f32(o + 1 * ss + 4, a11);
        vst1q_f32(o + 2 * ss, a20); vst1q_f32(o + 2 * ss + 4, a21);
        vst1q_f32(o + 3 * ss, a30); vst1q_f32(o + 3 * ss + 4, a31);
        vst1q_f32(o + 4 * ss, a40); vst1q_f32(o + 4 * ss + 4, a41);
        vst1q_f32(o + 5 * ss, a50); vst1q_f32(o + 5 * ss + 4, a51);
        vst1q_f32(o + 6 * ss, a60); vst1q_f32(o + 6 * ss + 4, a61);
        vst1q_f32(o + 7 * ss, a70); vst1q_f32(o + 7 * ss + 4, a71);
    }
}

#define FML(lo, hi, kv, qv, lane)                     \
    lo = vfmlalq_laneq_low_f16(lo, kv, qv, lane);     \
    hi = vfmlalq_laneq_high_f16(hi, kv, qv, lane);

void i8g_fattn_qk_f16(const float16_t *restrict qt, const float16_t *restrict kt, size_t hd, size_t steps,
                      float *restrict s, size_t ss) {
    for (size_t st = 0; st < steps; st++) {
        const float16_t *k = kt + st * hd * 8;
        float32x4_t a00 = ZERO4, a01 = ZERO4, a10 = ZERO4, a11 = ZERO4, a20 = ZERO4, a21 = ZERO4, a30 = ZERO4, a31 = ZERO4;
        float32x4_t a40 = ZERO4, a41 = ZERO4, a50 = ZERO4, a51 = ZERO4, a60 = ZERO4, a61 = ZERO4, a70 = ZERO4, a71 = ZERO4;
        for (size_t d = 0; d < hd; d++) {
            const float16x8_t kv = vld1q_f16(k + d * 8);
            const float16x8_t qv = vld1q_f16(qt + d * 8);
            FML(a00, a01, kv, qv, 0)
            FML(a10, a11, kv, qv, 1)
            FML(a20, a21, kv, qv, 2)
            FML(a30, a31, kv, qv, 3)
            FML(a40, a41, kv, qv, 4)
            FML(a50, a51, kv, qv, 5)
            FML(a60, a61, kv, qv, 6)
            FML(a70, a71, kv, qv, 7)
        }
        float *o = s + st * 8;
        vst1q_f32(o + 0 * ss, a00); vst1q_f32(o + 0 * ss + 4, a01);
        vst1q_f32(o + 1 * ss, a10); vst1q_f32(o + 1 * ss + 4, a11);
        vst1q_f32(o + 2 * ss, a20); vst1q_f32(o + 2 * ss + 4, a21);
        vst1q_f32(o + 3 * ss, a30); vst1q_f32(o + 3 * ss + 4, a31);
        vst1q_f32(o + 4 * ss, a40); vst1q_f32(o + 4 * ss + 4, a41);
        vst1q_f32(o + 5 * ss, a50); vst1q_f32(o + 5 * ss + 4, a51);
        vst1q_f32(o + 6 * ss, a60); vst1q_f32(o + 6 * ss + 4, a61);
        vst1q_f32(o + 7 * ss, a70); vst1q_f32(o + 7 * ss + 4, a71);
    }
}

// ---- softmax of one query row of one key chunk -------------------------------------------------
// exp(x) for x <= 0 (clamped at -87): Cephes expf form (two-constant ln2 reduction, degree-5
// polynomial), relative error about 1e-7; the same formula as `image_dit::fast::exp_fast`.
static inline __attribute__((always_inline)) float32x4_t exp_ps(float32x4_t x) {
    x                   = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r       = vsubq_f32(x, vmulq_n_f32(n, 0.693359375f));
    r                   = vaddq_f32(r, vmulq_n_f32(n, 2.12194440e-4f));
    float32x4_t p       = vdupq_n_f32(1.9875691500e-4f);
    p                   = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(1.3981999507e-3f));
    p                   = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(8.3334519073e-3f));
    p                   = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(4.1665795894e-2f));
    p                   = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(1.6666665459e-1f));
    p                   = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(0.5f));
    const float32x4_t y = vaddq_f32(vaddq_f32(vmulq_f32(vmulq_f32(p, r), r), r), vdupq_n_f32(1.0f));
    const int32x4_t e   = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

// p[j] = exp(s[j] - m) for j < nvalid, 0 for nvalid <= j < kc; written to pt[j*8 + qi].
// Returns the sum of the STORED values (f32, sequential per 4 lanes, lanes added at the end).
// kc is a multiple of 8.
float i8g_fattn_softmax_f32(const float *restrict s, size_t kc, size_t nvalid, float m, float *restrict pt, size_t qi) {
    const float32x4_t vm = vdupq_n_f32(m);
    float32x4_t acc      = ZERO4;
    for (size_t j = 0; j < kc; j += 4) {
        float32x4_t e = exp_ps(vsubq_f32(vld1q_f32(s + j), vm));
        if (j + 4 > nvalid) {
            float tmp[4];
            vst1q_f32(tmp, e);
            for (size_t i = 0; i < 4; i++)
                if (j + i >= nvalid) tmp[i] = 0.0f;
            e = vld1q_f32(tmp);
        }
        acc = vaddq_f32(acc, e);
        pt[(j + 0) * 8 + qi] = vgetq_lane_f32(e, 0);
        pt[(j + 1) * 8 + qi] = vgetq_lane_f32(e, 1);
        pt[(j + 2) * 8 + qi] = vgetq_lane_f32(e, 2);
        pt[(j + 3) * 8 + qi] = vgetq_lane_f32(e, 3);
    }
    return vaddvq_f32(acc);
}

float i8g_fattn_softmax_f16(const float *restrict s, size_t kc, size_t nvalid, float m, float16_t *restrict pt,
                            size_t qi) {
    const float32x4_t vm = vdupq_n_f32(m);
    float32x4_t acc      = ZERO4;
    for (size_t j = 0; j < kc; j += 4) {
        float32x4_t e = exp_ps(vsubq_f32(vld1q_f32(s + j), vm));
        if (j + 4 > nvalid) {
            float tmp[4];
            vst1q_f32(tmp, e);
            for (size_t i = 0; i < 4; i++)
                if (j + i >= nvalid) tmp[i] = 0.0f;
            e = vld1q_f32(tmp);
        }
        const float16x4_t h = vcvt_f16_f32(e);
        acc                 = vaddq_f32(acc, vcvt_f32_f16(h)); // sum of what P.V will actually use
        pt[(j + 0) * 8 + qi] = vget_lane_f16(h, 0);
        pt[(j + 1) * 8 + qi] = vget_lane_f16(h, 1);
        pt[(j + 2) * 8 + qi] = vget_lane_f16(h, 2);
        pt[(j + 3) * 8 + qi] = vget_lane_f16(h, 3);
    }
    return vaddvq_f32(acc);
}

// ---- p.v ---------------------------------------------------------------------------------------
// o[qi*hd + dt*8 + i] += sum_j pt[j*8+qi] * vt[dt*vstride + j*8 + i], for every dim tile dt.
void i8g_fattn_pv_f32(const float *restrict pt, const float *restrict vt, size_t kc, size_t hd, size_t vstride,
                      float *restrict o) {
    for (size_t dt = 0; dt < hd / 8; dt++) {
        const float *v = vt + dt * vstride;
        float *ob      = o + dt * 8;
        float32x4_t a00 = vld1q_f32(ob + 0 * hd), a01 = vld1q_f32(ob + 0 * hd + 4);
        float32x4_t a10 = vld1q_f32(ob + 1 * hd), a11 = vld1q_f32(ob + 1 * hd + 4);
        float32x4_t a20 = vld1q_f32(ob + 2 * hd), a21 = vld1q_f32(ob + 2 * hd + 4);
        float32x4_t a30 = vld1q_f32(ob + 3 * hd), a31 = vld1q_f32(ob + 3 * hd + 4);
        float32x4_t a40 = vld1q_f32(ob + 4 * hd), a41 = vld1q_f32(ob + 4 * hd + 4);
        float32x4_t a50 = vld1q_f32(ob + 5 * hd), a51 = vld1q_f32(ob + 5 * hd + 4);
        float32x4_t a60 = vld1q_f32(ob + 6 * hd), a61 = vld1q_f32(ob + 6 * hd + 4);
        float32x4_t a70 = vld1q_f32(ob + 7 * hd), a71 = vld1q_f32(ob + 7 * hd + 4);
        for (size_t j = 0; j < kc; j++) {
            const float32x4_t v0 = vld1q_f32(v + j * 8), v1 = vld1q_f32(v + j * 8 + 4);
            const float32x4_t p0 = vld1q_f32(pt + j * 8), p1 = vld1q_f32(pt + j * 8 + 4);
            a00 = vfmaq_laneq_f32(a00, v0, p0, 0);
            a01 = vfmaq_laneq_f32(a01, v1, p0, 0);
            a10 = vfmaq_laneq_f32(a10, v0, p0, 1);
            a11 = vfmaq_laneq_f32(a11, v1, p0, 1);
            a20 = vfmaq_laneq_f32(a20, v0, p0, 2);
            a21 = vfmaq_laneq_f32(a21, v1, p0, 2);
            a30 = vfmaq_laneq_f32(a30, v0, p0, 3);
            a31 = vfmaq_laneq_f32(a31, v1, p0, 3);
            a40 = vfmaq_laneq_f32(a40, v0, p1, 0);
            a41 = vfmaq_laneq_f32(a41, v1, p1, 0);
            a50 = vfmaq_laneq_f32(a50, v0, p1, 1);
            a51 = vfmaq_laneq_f32(a51, v1, p1, 1);
            a60 = vfmaq_laneq_f32(a60, v0, p1, 2);
            a61 = vfmaq_laneq_f32(a61, v1, p1, 2);
            a70 = vfmaq_laneq_f32(a70, v0, p1, 3);
            a71 = vfmaq_laneq_f32(a71, v1, p1, 3);
        }
        vst1q_f32(ob + 0 * hd, a00); vst1q_f32(ob + 0 * hd + 4, a01);
        vst1q_f32(ob + 1 * hd, a10); vst1q_f32(ob + 1 * hd + 4, a11);
        vst1q_f32(ob + 2 * hd, a20); vst1q_f32(ob + 2 * hd + 4, a21);
        vst1q_f32(ob + 3 * hd, a30); vst1q_f32(ob + 3 * hd + 4, a31);
        vst1q_f32(ob + 4 * hd, a40); vst1q_f32(ob + 4 * hd + 4, a41);
        vst1q_f32(ob + 5 * hd, a50); vst1q_f32(ob + 5 * hd + 4, a51);
        vst1q_f32(ob + 6 * hd, a60); vst1q_f32(ob + 6 * hd + 4, a61);
        vst1q_f32(ob + 7 * hd, a70); vst1q_f32(ob + 7 * hd + 4, a71);
    }
}

void i8g_fattn_pv_f16(const float16_t *restrict pt, const float16_t *restrict vt, size_t kc, size_t hd, size_t vstride,
                      float *restrict o) {
    for (size_t dt = 0; dt < hd / 8; dt++) {
        const float16_t *v = vt + dt * vstride;
        float *ob          = o + dt * 8;
        float32x4_t a00 = vld1q_f32(ob + 0 * hd), a01 = vld1q_f32(ob + 0 * hd + 4);
        float32x4_t a10 = vld1q_f32(ob + 1 * hd), a11 = vld1q_f32(ob + 1 * hd + 4);
        float32x4_t a20 = vld1q_f32(ob + 2 * hd), a21 = vld1q_f32(ob + 2 * hd + 4);
        float32x4_t a30 = vld1q_f32(ob + 3 * hd), a31 = vld1q_f32(ob + 3 * hd + 4);
        float32x4_t a40 = vld1q_f32(ob + 4 * hd), a41 = vld1q_f32(ob + 4 * hd + 4);
        float32x4_t a50 = vld1q_f32(ob + 5 * hd), a51 = vld1q_f32(ob + 5 * hd + 4);
        float32x4_t a60 = vld1q_f32(ob + 6 * hd), a61 = vld1q_f32(ob + 6 * hd + 4);
        float32x4_t a70 = vld1q_f32(ob + 7 * hd), a71 = vld1q_f32(ob + 7 * hd + 4);
        for (size_t j = 0; j < kc; j++) {
            const float16x8_t vv = vld1q_f16(v + j * 8);
            const float16x8_t pv = vld1q_f16(pt + j * 8);
            FML(a00, a01, vv, pv, 0)
            FML(a10, a11, vv, pv, 1)
            FML(a20, a21, vv, pv, 2)
            FML(a30, a31, vv, pv, 3)
            FML(a40, a41, vv, pv, 4)
            FML(a50, a51, vv, pv, 5)
            FML(a60, a61, vv, pv, 6)
            FML(a70, a71, vv, pv, 7)
        }
        vst1q_f32(ob + 0 * hd, a00); vst1q_f32(ob + 0 * hd + 4, a01);
        vst1q_f32(ob + 1 * hd, a10); vst1q_f32(ob + 1 * hd + 4, a11);
        vst1q_f32(ob + 2 * hd, a20); vst1q_f32(ob + 2 * hd + 4, a21);
        vst1q_f32(ob + 3 * hd, a30); vst1q_f32(ob + 3 * hd + 4, a31);
        vst1q_f32(ob + 4 * hd, a40); vst1q_f32(ob + 4 * hd + 4, a41);
        vst1q_f32(ob + 5 * hd, a50); vst1q_f32(ob + 5 * hd + 4, a51);
        vst1q_f32(ob + 6 * hd, a60); vst1q_f32(ob + 6 * hd + 4, a61);
        vst1q_f32(ob + 7 * hd, a70); vst1q_f32(ob + 7 * hd + 4, a71);
    }
}

// ---- conversion ----------------------------------------------------------------------------------
// dst[i * stride] = f16(x[i]) (round to nearest even; overflow -> inf), n any.
void i8g_fattn_to_f16(const float *restrict x, size_t n, float16_t *restrict dst, size_t stride) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float16x4_t h = vcvt_f16_f32(vld1q_f32(x + i));
        dst[(i + 0) * stride] = vget_lane_f16(h, 0);
        dst[(i + 1) * stride] = vget_lane_f16(h, 1);
        dst[(i + 2) * stride] = vget_lane_f16(h, 2);
        dst[(i + 3) * stride] = vget_lane_f16(h, 3);
    }
    for (; i < n; i++) dst[i * stride] = (float16_t)x[i];
}

// x[i] = f32(src[i]) (exact).
void i8g_fattn_from_f16(const float16_t *restrict src, size_t n, float *restrict x) {
    for (size_t i = 0; i < n; i++) x[i] = (float)src[i];
}
