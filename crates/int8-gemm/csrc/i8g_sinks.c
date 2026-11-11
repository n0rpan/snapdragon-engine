// NEON sink kernels for crates/int8-gemm (see src/sinks.rs for the contracts and the scalar definitions).
// Built with -ffp-contract=off: every multiply and add below is a separate IEEE operation, on purpose.
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------------------------
// 1. Quantise one chunk (n floats, one scale) to int8 codes, 8 codes per store.
// ---------------------------------------------------------------------------------------------
float i8g_quant_chunk(const float *x, size_t n, int8_t *dst, ptrdiff_t stride) {
    // max(|x|) is order independent; maxNum ignores NaN like f32::max. Start at 0 like the scalar code.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < n; i += 8) {
        acc0 = vmaxnmq_f32(acc0, vabsq_f32(vld1q_f32(x + i)));
        acc1 = vmaxnmq_f32(acc1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    float amax = vmaxnmvq_f32(vmaxnmq_f32(acc0, acc1));

    float scale = 0.0f;
    float inv = 0.0f;
    if (amax > 0.0f && isfinite(amax)) {
        scale = amax / 127.0f;
        inv = 127.0f / amax;
    }

    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t lo = vdupq_n_f32(-127.0f);
    const float32x4_t hi = vdupq_n_f32(127.0f);
    size_t s = 0;
    for (size_t i = 0; i < n; i += 8, s++) {
        float32x4_t a = vmulq_f32(vld1q_f32(x + i), vinv);
        float32x4_t b = vmulq_f32(vld1q_f32(x + i + 4), vinv);
        // round half away from zero, then clamp with the NaN-propagating forms (Rust clamp keeps NaN).
        a = vmaxq_f32(vminq_f32(vrndaq_f32(a), hi), lo);
        b = vmaxq_f32(vminq_f32(vrndaq_f32(b), hi), lo);
        // Values are integers in [-127, 127] or NaN; vcvt maps NaN to 0, as Rust `as i8` does.
        int16x8_t h = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(a)), vqmovn_s32(vcvtq_s32_f32(b)));
        vst1_s8(dst + (ptrdiff_t)s * stride, vqmovn_s16(h));
    }
    return scale;
}

// ---------------------------------------------------------------------------------------------
// 2. SiLU gate: out[j] = silu(row[j]) * row[u + j], with the crate's exp_fast (see sinks.rs).
// ---------------------------------------------------------------------------------------------
static inline float32x4_t i8g_exp_fast4(float32x4_t x) {
    const float32x4_t lo = vdupq_n_f32(-87.0f);
    const float32x4_t hi = vdupq_n_f32(87.0f);
    x = vmaxq_f32(vminq_f32(x, hi), lo);
    // n = round(x * LOG2_E)
    float32x4_t n = vrndaq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896340736f)));
    // r = (x - n*c1) + n*c2
    float32x4_t r = vaddq_f32(vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(0.6933594f))),
                              vmulq_f32(n, vdupq_n_f32(2.1219444e-4f)));
    // p = ((((c5*r + c4)*r + c3)*r + c2)*r + c1)*r + 0.5
    float32x4_t p = vdupq_n_f32(1.9875691e-4f);
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(1.3982e-3f));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(8.333452e-3f));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(4.1665796e-2f));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(1.6666665e-1f));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(0.5f));
    // y = p*r*r + r + 1
    float32x4_t y = vaddq_f32(vmulq_f32(vmulq_f32(p, r), r), r);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));
    // 2^n from the exponent bits: ((n + 127) << 23)
    int32x4_t e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    float32x4_t pow2 = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    return vmulq_f32(y, pow2);
}

static inline float32x4_t i8g_silu4(float32x4_t a) {
    float32x4_t den = vaddq_f32(vdupq_n_f32(1.0f), i8g_exp_fast4(vnegq_f32(a)));
    return vdivq_f32(a, den);
}

void i8g_silu_gate(const float *row, size_t u, float *out) {
    for (size_t j = 0; j < u; j += 4) {
        float32x4_t a = vld1q_f32(row + j);
        float32x4_t g = vld1q_f32(row + u + j);
        vst1q_f32(out + j, vmulq_f32(i8g_silu4(a), g));
    }
}

// ---------------------------------------------------------------------------------------------
// 3. RMS-norm scale + RoPE:
//    a = (row[2i] * r) * scale[2i];  b = (row[2i+1] * r) * scale[2i+1];
//    vals[2i]   = pe[4i] * a + pe[4i+1] * b;
//    vals[2i+1] = pe[4i+2] * a + pe[4i+3] * b;
// ---------------------------------------------------------------------------------------------
void i8g_rope_scale(const float *row, size_t hd, float r, const float *scale, const float *pe, float *vals) {
    const float32x4_t rv = vdupq_n_f32(r);
    // Four pairs (i = 2p) per iteration: row/scale/vals offsets are i, pe offset is 2i.
    for (size_t i = 0; i < hd; i += 8) {
        float32x4x2_t x = vld2q_f32(row + i);
        float32x4x2_t s = vld2q_f32(scale + i);
        float32x4_t a = vmulq_f32(vmulq_f32(x.val[0], rv), s.val[0]);
        float32x4_t b = vmulq_f32(vmulq_f32(x.val[1], rv), s.val[1]);
        float32x4x4_t q = vld4q_f32(pe + 2 * i);
        float32x4x2_t o;
        o.val[0] = vaddq_f32(vmulq_f32(q.val[0], a), vmulq_f32(q.val[1], b));
        o.val[1] = vaddq_f32(vmulq_f32(q.val[2], a), vmulq_f32(q.val[3], b));
        vst2q_f32(vals + i, o);
    }
}
