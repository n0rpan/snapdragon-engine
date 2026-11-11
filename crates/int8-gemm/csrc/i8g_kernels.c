// int8 SMMLA micro-kernels for crates/int8-gemm. See src/lib.rs for the layouts and the contract.
//
// Tile = 8 "tokens" (activation vectors) x 8 "rows" (weight vectors), K consumed 8 at a time.
// Packed layout, identical for both operands: per k-step 64 bytes = 8 vectors x 8 consecutive k,
// vector-major: bytes [8*j, 8*j+8) of a step belong to vector j. So one 16-byte register holds a
// vector PAIR, which is exactly SMMLA's 2x8 operand:
//   smmla(acc, X(pair p), W(pair q)) adds
//     lane0 += x[2p]  .w[2q]    lane1 += x[2p]  .w[2q+1]
//     lane2 += x[2p+1].w[2q]    lane3 += x[2p+1].w[2q+1]
// 16 SMMLA per step = 512 MACs per step, against 8 loads.
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Integer part: acc[p][q] over `steps` k-steps. Exact (int32), so order-free.
static inline __attribute__((always_inline)) void dot8x8(const int8_t *restrict x, const int8_t *restrict w, size_t steps,
                                                         int32x4_t acc[16]) {
    int32x4_t a00 = vdupq_n_s32(0), a01 = a00, a02 = a00, a03 = a00;
    int32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    int32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    int32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    for (size_t s = 0; s < steps; s++) {
        const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16), x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48);
        const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
        x += 64;
        w += 64;
        a00 = vmmlaq_s32(a00, x0, w0);
        a01 = vmmlaq_s32(a01, x0, w1);
        a02 = vmmlaq_s32(a02, x0, w2);
        a03 = vmmlaq_s32(a03, x0, w3);
        a10 = vmmlaq_s32(a10, x1, w0);
        a11 = vmmlaq_s32(a11, x1, w1);
        a12 = vmmlaq_s32(a12, x1, w2);
        a13 = vmmlaq_s32(a13, x1, w3);
        a20 = vmmlaq_s32(a20, x2, w0);
        a21 = vmmlaq_s32(a21, x2, w1);
        a22 = vmmlaq_s32(a22, x2, w2);
        a23 = vmmlaq_s32(a23, x2, w3);
        a30 = vmmlaq_s32(a30, x3, w0);
        a31 = vmmlaq_s32(a31, x3, w1);
        a32 = vmmlaq_s32(a32, x3, w2);
        a33 = vmmlaq_s32(a33, x3, w3);
    }
    acc[0] = a00, acc[1] = a01, acc[2] = a02, acc[3] = a03;
    acc[4] = a10, acc[5] = a11, acc[6] = a12, acc[7] = a13;
    acc[8] = a20, acc[9] = a21, acc[10] = a22, acc[11] = a23;
    acc[12] = a30, acc[13] = a31, acc[14] = a32, acc[15] = a33;
}

// Same with an UNSIGNED x operand (USMMLA): for probabilities x weights (attention P.V).
static inline __attribute__((always_inline)) void udot8x8(const uint8_t *restrict x, const int8_t *restrict w, size_t steps,
                                                          int32x4_t acc[16]) {
    int32x4_t a[16];
    for (int i = 0; i < 16; i++) a[i] = vdupq_n_s32(0);
    for (size_t s = 0; s < steps; s++) {
        const uint8x16_t x0 = vld1q_u8(x), x1 = vld1q_u8(x + 16), x2 = vld1q_u8(x + 32), x3 = vld1q_u8(x + 48);
        const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
        x += 64;
        w += 64;
        a[0] = vusmmlaq_s32(a[0], x0, w0);
        a[1] = vusmmlaq_s32(a[1], x0, w1);
        a[2] = vusmmlaq_s32(a[2], x0, w2);
        a[3] = vusmmlaq_s32(a[3], x0, w3);
        a[4] = vusmmlaq_s32(a[4], x1, w0);
        a[5] = vusmmlaq_s32(a[5], x1, w1);
        a[6] = vusmmlaq_s32(a[6], x1, w2);
        a[7] = vusmmlaq_s32(a[7], x1, w3);
        a[8] = vusmmlaq_s32(a[8], x2, w0);
        a[9] = vusmmlaq_s32(a[9], x2, w1);
        a[10] = vusmmlaq_s32(a[10], x2, w2);
        a[11] = vusmmlaq_s32(a[11], x2, w3);
        a[12] = vusmmlaq_s32(a[12], x3, w0);
        a[13] = vusmmlaq_s32(a[13], x3, w1);
        a[14] = vusmmlaq_s32(a[14], x3, w2);
        a[15] = vusmmlaq_s32(a[15], x3, w3);
    }
    for (int i = 0; i < 16; i++) acc[i] = a[i];
}

// acc[p][q] (pair layout) -> out[t*8 + r] (token-major 8x8).
static inline __attribute__((always_inline)) void unshuffle_s32(const int32x4_t acc[16], int32_t *out) {
    for (int p = 0; p < 4; p++)
        for (int q = 0; q < 4; q++) {
            const int32x4_t v = acc[p * 4 + q];
            vst1_s32(out + (2 * p) * 8 + 2 * q, vget_low_s32(v));
            vst1_s32(out + (2 * p + 1) * 8 + 2 * q, vget_high_s32(v));
        }
}

// Pure integer tile: out[t*8+r] = sum_k x[t][k]*w[r][k], k over steps*8.
void i8g_dot8x8(const int8_t *x, const int8_t *w, size_t steps, int32_t *out) {
    int32x4_t acc[16];
    dot8x8(x, w, steps, acc);
    unshuffle_s32(acc, out);
}

void i8g_udot8x8(const uint8_t *x, const int8_t *w, size_t steps, int32_t *out) {
    int32x4_t acc[16];
    udot8x8(x, w, steps, acc);
    unshuffle_s32(acc, out);
}

// Chunk-scaled tile, the fast path and the Q8_0-exact path (chunk = 32):
//   y[t*ystride + r] (+)= sum_c  f32(isum_c[t][r]) * (sx[c*8+t] * sw[c*8+r])     c ascending
// isum_c is the exact int32 dot over chunk c (`steps_per_chunk` k-steps).
// Float contract (fixed, documented in lib.rs): per element one f32 accumulator starting at +0.0
// (or the old y when `accumulate`), for c ascending: s = sx*sw (one f32 multiply), then
//   fused == 0:  acc = acc + (f32(isum) * s)    (multiply, round, add, round)
//   fused != 0:  acc = fma(f32(isum), s, acc)
// int32 -> f32 is round-to-nearest-even (scvtf).
void i8g_tile8x8(const int8_t *x, const int8_t *w, size_t steps_per_chunk, size_t nchunks, const float *sx, const float *sw,
                 float *y, ptrdiff_t ystride, int accumulate, int fused) {
    float32x4_t f[16];
    if (accumulate) {
        for (int p = 0; p < 4; p++)
            for (int q = 0; q < 4; q++)
                f[p * 4 + q] = vcombine_f32(vld1_f32(y + (2 * p) * ystride + 2 * q), vld1_f32(y + (2 * p + 1) * ystride + 2 * q));
    } else {
        for (int i = 0; i < 16; i++) f[i] = vdupq_n_f32(0.0f);
    }
    for (size_t c = 0; c < nchunks; c++) {
        int32x4_t acc[16];
        dot8x8(x, w, steps_per_chunk, acc);
        x += steps_per_chunk * 64;
        w += steps_per_chunk * 64;
        float32x4_t sxp[4], swq[4];
        for (int p = 0; p < 4; p++) {
            const float32x2_t v = vld1_f32(sx + 2 * p);  // [a, b]
            sxp[p]              = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a, a, b, b]
        }
        for (int q = 0; q < 4; q++) {
            const float32x2_t v = vld1_f32(sw + 2 * q);  // [a, b]
            swq[q]              = vcombine_f32(v, v);    // [a, b, a, b]
        }
        sx += 8;
        sw += 8;
        if (fused) {
            for (int p = 0; p < 4; p++)
                for (int q = 0; q < 4; q++)
                    f[p * 4 + q] = vfmaq_f32(f[p * 4 + q], vcvtq_f32_s32(acc[p * 4 + q]), vmulq_f32(sxp[p], swq[q]));
        } else {
            for (int p = 0; p < 4; p++)
                for (int q = 0; q < 4; q++)
                    f[p * 4 + q] = vaddq_f32(f[p * 4 + q], vmulq_f32(vcvtq_f32_s32(acc[p * 4 + q]), vmulq_f32(sxp[p], swq[q])));
        }
    }
    for (int p = 0; p < 4; p++)
        for (int q = 0; q < 4; q++) {
            vst1_f32(y + (2 * p) * ystride + 2 * q, vget_low_f32(f[p * 4 + q]));
            vst1_f32(y + (2 * p + 1) * ystride + 2 * q, vget_high_f32(f[p * 4 + q]));
        }
}

// Unsigned-x variant of the chunk-scaled tile (attention P.V; non-fused order only).
void i8g_utile8x8(const uint8_t *x, const int8_t *w, size_t steps_per_chunk, size_t nchunks, const float *sx, const float *sw,
                  float *y, ptrdiff_t ystride, int accumulate) {
    float32x4_t f[16];
    if (accumulate) {
        for (int p = 0; p < 4; p++)
            for (int q = 0; q < 4; q++)
                f[p * 4 + q] = vcombine_f32(vld1_f32(y + (2 * p) * ystride + 2 * q), vld1_f32(y + (2 * p + 1) * ystride + 2 * q));
    } else {
        for (int i = 0; i < 16; i++) f[i] = vdupq_n_f32(0.0f);
    }
    for (size_t c = 0; c < nchunks; c++) {
        int32x4_t acc[16];
        udot8x8(x, w, steps_per_chunk, acc);
        x += steps_per_chunk * 64;
        w += steps_per_chunk * 64;
        for (int p = 0; p < 4; p++) {
            const float32x2_t vx  = vld1_f32(sx + 2 * p);
            const float32x4_t sxp = vcombine_f32(vdup_lane_f32(vx, 0), vdup_lane_f32(vx, 1));
            for (int q = 0; q < 4; q++) {
                const float32x2_t vw = vld1_f32(sw + 2 * q);
                f[p * 4 + q] = vaddq_f32(f[p * 4 + q], vmulq_f32(vcvtq_f32_s32(acc[p * 4 + q]), vmulq_f32(sxp, vcombine_f32(vw, vw))));
            }
        }
        sx += 8;
        sw += 8;
    }
    for (int p = 0; p < 4; p++)
        for (int q = 0; q < 4; q++) {
            vst1_f32(y + (2 * p) * ystride + 2 * q, vget_low_f32(f[p * 4 + q]));
            vst1_f32(y + (2 * p + 1) * ystride + 2 * q, vget_high_f32(f[p * 4 + q]));
        }
}

// ---- attention helpers: row max, and exp + value-scale + u8 packing of one probability row ----

// max of n floats (n multiple of 4, n >= 4).
float i8g_row_max(const float *s, size_t n) {
    float32x4_t m = vld1q_f32(s);
    for (size_t i = 4; i < n; i += 4) m = vmaxq_f32(m, vld1q_f32(s + i));
    return vmaxvq_f32(m);
}

// exp(x) for x <= 0, x clamped below at -87. Range reduction to |r| <= ln2/2 and a degree-4
// Taylor polynomial with explicit FMAs: relative error below 5e-5, which is far inside what the
// consumer keeps (probabilities become u8 codes relative to the row maximum, 1/255 steps; the
// row sum feeds a division of the same precision class). Half the instructions of the Cephes form.
static inline __attribute__((always_inline)) float32x4_t exp_neg_ps(float32x4_t x) {
    x                   = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    const float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    const float32x4_t r = vfmsq_n_f32(x, n, 0.69314718055994531f);
    float32x4_t p       = vdupq_n_f32(1.0f / 24.0f);
    p                   = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p                   = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p                   = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p                   = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    const int32x4_t e   = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// One query row of one key chunk of the int8 flash attention (kc keys, kc multiple of 8):
//   e[j]  = exp(s[j] - m)                 (m >= max of the row seen so far, so the argument is <= 0;
//                                          -inf scores of padding keys give about 1.6e-38, i.e. 0)
//   *sum  = sum_j e[j]                    (4 float lanes, then added: NOT the scalar order)
//   pv[j] = e[j] * sv[j]                  (sv = int8 scale of value row j, >= 0)
//   *scale = max_j pv[j] / 255
//   code[j] = round(pv[j] / *scale) as u8, written in the USMMLA operand layout:
//             pcodes[(j / 8) * 64 + qi * 8 + j % 8]
// s is overwritten (it holds pv on return).
void i8g_softmax_pack(float *s, size_t kc, float m, const float *sv, uint8_t *pcodes, size_t qi, float *sum, float *scale) {
    const float32x4_t vm = vdupq_n_f32(m);
    float32x4_t acc      = vdupq_n_f32(0.0f);
    float32x4_t pmax     = vdupq_n_f32(0.0f);
    for (size_t j = 0; j < kc; j += 4) {
        const float32x4_t e  = exp_neg_ps(vsubq_f32(vld1q_f32(s + j), vm));
        acc                  = vaddq_f32(acc, e);
        const float32x4_t pv = vmulq_f32(e, vld1q_f32(sv + j));
        pmax                 = vmaxq_f32(pmax, pv);
        vst1q_f32(s + j, pv);
    }
    *sum           = vaddvq_f32(acc);
    const float mx = vmaxvq_f32(pmax);
    if (!(mx > 0.0f)) {
        *scale = 0.0f;
        for (size_t j = 0; j < kc; j += 8) memset(pcodes + (j / 8) * 64 + qi * 8, 0, 8);
        return;
    }
    *scale          = mx / 255.0f;
    const float inv = 255.0f / mx;
    for (size_t j = 0; j < kc; j += 8) {
        const uint32x4_t a = vcvtnq_u32_f32(vmulq_n_f32(vld1q_f32(s + j), inv));
        const uint32x4_t b = vcvtnq_u32_f32(vmulq_n_f32(vld1q_f32(s + j + 4), inv));
        const uint16x8_t w = vcombine_u16(vqmovn_u32(a), vqmovn_u32(b));
        vst1_u8(pcodes + (j / 8) * 64 + qi * 8, vqmovn_u16(w));
    }
}

// Roofline probe only (not used by any GEMM path): a 4x4 SDOT tile, 16 SDOT per 16 k against 8 loads.
// x and w: 4 vectors each, per step 64 bytes = 4 vectors x 16 consecutive k. Returns the 16 dots in out[t*4+r].
void i8g_probe_sdot4x4(const int8_t *x, const int8_t *w, size_t steps, int32_t *out) {
    int32x4_t a[16];
    for (int i = 0; i < 16; i++) a[i] = vdupq_n_s32(0);
    for (size_t s = 0; s < steps; s++) {
        const int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16), x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48);
        const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
        x += 64;
        w += 64;
        a[0] = vdotq_s32(a[0], x0, w0);
        a[1] = vdotq_s32(a[1], x0, w1);
        a[2] = vdotq_s32(a[2], x0, w2);
        a[3] = vdotq_s32(a[3], x0, w3);
        a[4] = vdotq_s32(a[4], x1, w0);
        a[5] = vdotq_s32(a[5], x1, w1);
        a[6] = vdotq_s32(a[6], x1, w2);
        a[7] = vdotq_s32(a[7], x1, w3);
        a[8] = vdotq_s32(a[8], x2, w0);
        a[9] = vdotq_s32(a[9], x2, w1);
        a[10] = vdotq_s32(a[10], x2, w2);
        a[11] = vdotq_s32(a[11], x2, w3);
        a[12] = vdotq_s32(a[12], x3, w0);
        a[13] = vdotq_s32(a[13], x3, w1);
        a[14] = vdotq_s32(a[14], x3, w2);
        a[15] = vdotq_s32(a[15], x3, w3);
    }
    for (int i = 0; i < 16; i++) out[i] = vaddvq_s32(a[i]);
}

// Short-vector row kernel (attention q.k: k = head_dim, ONE chunk): one token group against `ngroups`
// consecutive row groups in one call.
//   y[t*ystride + g*8 + r] = +0.0 + f32(isum[t][g*8+r]) * (sx[t] * sw[g*8+r])
// Bitwise the same as i8g_tile8x8(nchunks = 1, accumulate = 0, fused = 0) per row group; what it saves
// is the per-tile set-up (call, branches, scale shuffles of the token side, half-vector stores),
// which at k = 128 is a large part of a tile.
void i8g_rows1(const int8_t *x, const int8_t *w, size_t steps, size_t ngroups, const float *sx, const float *sw, float *y,
               ptrdiff_t ystride) {
    float32x4_t sxp[4];
    for (int p = 0; p < 4; p++) {
        const float32x2_t v = vld1_f32(sx + 2 * p);
        sxp[p]              = vcombine_f32(vdup_lane_f32(v, 0), vdup_lane_f32(v, 1));  // [a, a, b, b]
    }
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (size_t g = 0; g < ngroups; g++) {
        int32x4_t acc[16];
        dot8x8(x, w, steps, acc);
        w += steps * 64;
        float32x4_t swq[4];
        for (int q = 0; q < 4; q++) {
            const float32x2_t v = vld1_f32(sw + 2 * q);
            swq[q]              = vcombine_f32(v, v);  // [a, b, a, b]
        }
        sw += 8;
        float *yg = y + g * 8;
        for (int p = 0; p < 4; p++) {
            float32x4_t f[4];
            for (int q = 0; q < 4; q++) f[q] = vaddq_f32(zero, vmulq_f32(vcvtq_f32_s32(acc[p * 4 + q]), vmulq_f32(sxp[p], swq[q])));
            // f[q] = [t=2p: r=2q, 2q+1 | t=2p+1: r=2q, 2q+1]
            float *r0 = yg + (2 * p) * ystride, *r1 = yg + (2 * p + 1) * ystride;
            vst1q_f32(r0, vcombine_f32(vget_low_f32(f[0]), vget_low_f32(f[1])));
            vst1q_f32(r0 + 4, vcombine_f32(vget_low_f32(f[2]), vget_low_f32(f[3])));
            vst1q_f32(r1, vcombine_f32(vget_high_f32(f[0]), vget_high_f32(f[1])));
            vst1q_f32(r1 + 4, vcombine_f32(vget_high_f32(f[2]), vget_high_f32(f[3])));
        }
    }
}

int i8g_kernel_abi(void) { return 1; }
