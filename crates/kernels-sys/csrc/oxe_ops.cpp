// Hand-written (not upstream): the f32/f16 ops of the qwen35moe decode graph, ported from
// llama.cpp ggml-cpu at the pinned commit (see vendor/ggml-subset/PROVENANCE.md) for ONE token.
//
// Exactness rules (M1):
// - Vector helpers are ggml's own (`vec.h` / `vec.cpp`, vendored verbatim and compiled with the
//   same flags as llama.cpp's ggml-cpu: -O3 -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme).
// - Scalar loops keep the upstream expression shape, because GCC contracts a*b+c into FMA
//   (-ffp-contract=fast is the GNU default) and the result depends on the expression tree.
// - Every function names the upstream function it mirrors.
#include "ggml-cpu/vec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" {

// ggml_compute_forward_rms_norm_f32 (one row), unfused: y = x * scale
void oxe_rms_norm_f32(const float * x, float * y, int64_t ne00, float eps) {
    ggml_float sum = 0.0;
    for (int64_t i00 = 0; i00 < ne00; i00++) {
        sum += (ggml_float)(x[i00] * x[i00]);
    }
    const float mean  = sum/ne00;
    const float scale = 1.0f/sqrtf(mean + eps);
    memcpy(y, x, ne00 * sizeof(float));
    ggml_vec_scale_f32(ne00, y, scale);
}

// ggml_compute_forward_rms_norm_f32 fused with GGML_OP_MUL (ggml_cpu_try_fuse_ops):
// y = x * scale * w. Numerically identical to rms_norm then mul (two roundings either way).
void oxe_rms_norm_mul_f32(const float * x, const float * w, float * y, int64_t ne00, float eps) {
    ggml_float sum = 0.0;
    for (int64_t i00 = 0; i00 < ne00; i00++) {
        sum += (ggml_float)(x[i00] * x[i00]);
    }
    const float mean  = sum/ne00;
    const float scale = 1.0f/sqrtf(mean + eps);
    for (int64_t i00 = 0; i00 < ne00; i00++) {
        y[i00] = x[i00] * scale * w[i00];
    }
}

// build_gdn_l2_norm: ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrtf(n)), in place on one row.
void oxe_gdn_l2_norm_f32(float * x, int64_t ne00, float eps) {
    const float n = ne00;
    oxe_rms_norm_f32(x, x, ne00, eps/n);
    ggml_vec_scale_f32(ne00, x, 1.0f/sqrtf(n));  // ggml_compute_forward_scale_f32, b == 0
}

void oxe_vec_scale_f32(int n, float * y, float v) { ggml_vec_scale_f32(n, y, v); }

float oxe_vec_dot_f32(int n, const float * x, const float * y) {
    float s = 0.0f;
    ggml_vec_dot_f32(n, &s, 0, x, 0, y, 0, 1);
    return s;
}

float oxe_vec_dot_f16(int n, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    float s = 0.0f;
    ggml_vec_dot_f16(n, &s, 0, (ggml_fp16_t *) x, 0, (ggml_fp16_t *) y, 0, 1);
    return s;
}

// ggml_cpu_fp32_to_fp16 (from_float of GGML_TYPE_F16; also the f32 -> f16 KV-cache copy)
void oxe_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void oxe_silu_f32(int n, float * y, const float * x) { ggml_vec_silu_f32(n, y, x); }

// ggml_swiglu_split(a = gate, b = up): y = silu(gate) * up
void oxe_swiglu_f32(int n, float * y, const float * gate, const float * up) {
    ggml_vec_swiglu_f32(n, y, gate, up);
}

// unary-ops.cpp op_sigmoid / op_softplus / op_exp, elementwise
void oxe_sigmoid_f32(int64_t n, float * y, const float * x) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = 1.f / (1.f + expf(-x[i]));
    }
}
void oxe_softplus_f32(int64_t n, float * y, const float * x) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = (x[i] > 20.0f) ? x[i] : log1pf(expf(x[i]));
    }
}
float oxe_expf(float x) { return expf(x); }

// ggml_vec_sum_f32 (ggml_sum_rows on one row)
float oxe_vec_sum_f32(int n, const float * x) {
    float s = 0.0f;
    ggml_vec_sum_f32(n, &s, x);
    return s;
}

// ggml_compute_forward_soft_max_f32 for one row: dp = softmax(sp * scale + mask)
// (max_bias = 0 so slope = 1; no sinks). wp is scratch of ne00 floats. mask may be NULL.
void oxe_soft_max_f32(int ne00, float * dp, const float * sp, float scale, const float * mp_f32, float * wp) {
    const float slope = 1.0f;
    ggml_vec_cpy_f32  (ne00, wp, sp);
    ggml_vec_scale_f32(ne00, wp, scale);
    if (mp_f32) {
        for (int i = 0; i < ne00; ++i) {
            wp[i] += slope*mp_f32[i];
        }
    }
    float max = -INFINITY;
    ggml_vec_max_f32(ne00, &max, wp);
    ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
    sum = 1.0/sum;
    ggml_vec_scale_f32(ne00, dp, sum);
}

// ggml_argsort_top_k(x, k) on one row: std::sort of all indices by x descending, first k.
// Fast path: if the k+1 largest values are pairwise distinct and distinct from every other
// value, std::sort's order of the first k is the strict descending order, so a selection
// gives the identical result. Otherwise run the exact upstream std::sort.
void oxe_argsort_top_k_f32(const float * x, int n, int k, int32_t * out, int32_t * scratch) {
    // selection of the top k+1 (k small): insertion into a sorted prefix
    const int kk = std::min(n, k + 1);
    int m = 0;
    for (int i = 0; i < n; i++) {
        const float v = x[i];
        if (m == kk && !(v > x[scratch[m - 1]])) {
            continue;
        }
        int j = (m < kk) ? m++ : kk - 1;
        while (j > 0 && v > x[scratch[j - 1]]) {
            scratch[j] = scratch[j - 1];
            j--;
        }
        scratch[j] = i;
    }
    bool ties = false;
    for (int j = 1; j < m; j++) {
        ties |= !(x[scratch[j - 1]] > x[scratch[j]]);
    }
    if (m == kk && kk == k + 1) {
        // the (k+1)-th value must not tie with any value outside the selection either
        const float t = x[scratch[k]];
        int eq = 0;
        for (int i = 0; i < n; i++) {
            eq += (x[i] == t);
            ties |= std::isnan(x[i]);
        }
        ties |= eq != 1;
    }
    if (!ties) {
        for (int j = 0; j < k; j++) {
            out[j] = scratch[j];
        }
        return;
    }
    // exact upstream path (ggml_compute_forward_argsort_f32, GGML_SORT_ORDER_DESC)
    for (int j = 0; j < n; j++) {
        scratch[j] = j;
    }
    std::sort(scratch, scratch + n, [x](int32_t a, int32_t b) { return x[a] > x[b]; });
    for (int j = 0; j < k; j++) {
        out[j] = scratch[j];
    }
}

// ggml_compute_forward_ssm_conv_f32 for one token and channels [0, nr): conv state rows are
// (nc - 1) floats per channel (the conv window minus the new column), x_new the new column.
// Writes the conv output and shifts the state (build_conv_state_store keeps the last nc-1).
//
// The rounding is made explicit instead of leaving it to the compiler: llama.cpp's build of
//     for (i0 = 0; i0 < nc; ++i0) sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
// is vectorised by GCC 16 into blocks of 4 (fmul v.4s, then four in-order scalar fadd, no
// fusion) with an fmadd tail for nc % 4 (verified by disassembling ops.cpp.o of the oracle).
void oxe_ssm_conv_step_f32(float * state, const float * x_new, const float * c, int nr, int nc, float * out) {
    float s[16];
    for (int i1 = 0; i1 < nr; ++i1) {
        float * st = state + i1*(nc - 1);
        for (int i0 = 0; i0 < nc - 1; ++i0) {
            s[i0] = st[i0];
        }
        s[nc - 1] = x_new[i1];
        const float * ci = c + i1*nc;
        float sumf = 0.0f;
        int i0 = 0;
        for (; i0 + 4 <= nc; i0 += 4) {
            const float32x4_t p = vmulq_f32(vld1q_f32(s + i0), vld1q_f32(ci + i0));
            sumf = sumf + vgetq_lane_f32(p, 0);
            sumf = sumf + vgetq_lane_f32(p, 1);
            sumf = sumf + vgetq_lane_f32(p, 2);
            sumf = sumf + vgetq_lane_f32(p, 3);
        }
        for (; i0 < nc; ++i0) {
            sumf = fmaf(s[i0], ci[i0], sumf);
        }
        out[i1] = sumf;
        for (int k = 0; k < nc - 1; ++k) {
            st[k] = s[k + 1];
        }
    }
}

// ggml_compute_forward_gated_delta_net_one_chunk, K = 1, one token, one value head, GDA
// (scalar gate). State is stored transposed (s[j*S_v + i] = S[i][j]) and updated in place.
// q, k are the (l2-normalised) key-head vectors, v the value-head vector, g the log-gate,
// beta the sigmoid beta. delta is scratch of S_v floats; attn receives S_v floats.
void oxe_gated_delta_net_step_f32(float * s_out, const float * q_d, const float * k_d, const float * v_d,
                                  float g, float beta_val, int64_t S_v, float * delta, float * attn_data) {
    const float scale = 1.0f / sqrtf((float) S_v);
    ggml_vec_scale_f32(S_v * S_v, s_out, expf(g));
    for (int64_t j = 0; j < S_v; ++j) {
        float sum = 0.0f;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
        delta[j] = (v_d[j] - sum) * beta_val;
    }
    for (int64_t j = 0; j < S_v; ++j) {
        ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
    }
    for (int64_t j = 0; j < S_v; ++j) {
        float sum = 0.0f;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
        attn_data[j] = sum * scale;
    }
}

// OD-6b: oxe_gated_delta_net_step_f32 reading the state from s_in and writing the updated state to
// s_out (a verify lane continues from the previous lane's slot; this saves the separate copy of
// the state). ggml_vec_scale_f32 is one IEEE multiply per element (vector body and leftovers
// alike), so s_out = s_in * e^g here gives the bits of "copy, then scale in place"; the rest is the
// reference on s_out. s_in == s_out is allowed. Unit test gdn_step_from_bitwise.
void oxe_gated_delta_net_step_f32_from(float * s_out, const float * s_in, const float * q_d, const float * k_d, const float * v_d,
                                       float g, float beta_val, int64_t S_v, float * delta, float * attn_data) {
    const float scale = 1.0f / sqrtf((float) S_v);
    const float eg    = expf(g);
    for (int64_t i = 0; i < S_v * S_v; ++i) {
        s_out[i] = s_in[i] * eg;
    }
    for (int64_t j = 0; j < S_v; ++j) {
        float sum = 0.0f;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
        delta[j] = (v_d[j] - sum) * beta_val;
    }
    for (int64_t j = 0; j < S_v; ++j) {
        ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
    }
    for (int64_t j = 0; j < S_v; ++j) {
        float sum = 0.0f;
        ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
        attn_data[j] = sum * scale;
    }
}

// moe-prefill: the same step with the four passes over the state fused into two and 4 state rows
// interleaved. Per element and per row it performs exactly the operations of
// oxe_gated_delta_net_step_f32 on this build (GGML_SIMD, NEON, GGML_F32_STEP 16, GGML_F32_ARR 4):
//   s = s * e^g                      ggml_vec_scale_f32:  GGML_F32_VEC_MUL per element
//   sum_j = dot(s_j, k)              ggml_vec_dot_f32:    accumulator a takes elements [i + 4a, i + 4a + 4)
//                                    of every 16-element step i, GGML_F32_VEC_FMA(sum, s, k), then
//                                    GGML_F32_VEC_REDUCE and the float store
//   delta_j = (v_j - sum_j) * beta
//   s_j = s_j + k * delta_j          ggml_vec_mad_f32:    GGML_F32_VEC_FMA(s, k, delta) per element
//   attn_j = dot(s_j, q) * scale     ggml_vec_dot_f32 as above
// Row j's delta depends on row j only, so rows are independent: the reference's "all rows scaled,
// then all dots, then all updates, then all dots" and this "row group by row group" order give
// the same bits (unit test pf_gdn_step_x4_bitwise). Needs S_v % 16 == 0 (no leftover path).
// OD-6b: `_from` reads the old state from s_in (pass A loads s_in, stores s_out; s_in == s_out is the
// in-place step), bit for bit the in-place step on a copy.
void oxe_gated_delta_net_step_f32_x4_from(float * s_out, const float * s_in, const float * q_d, const float * k_d, const float * v_d,
                                          float g, float beta_val, int64_t S_v, float * delta, float * attn_data) {
#if defined(GGML_SIMD) && defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
    if (S_v % GGML_F32_STEP != 0 || GGML_F32_ARR != 4 || GGML_F32_EPR != 4) {
        oxe_gated_delta_net_step_f32_from(s_out, s_in, q_d, k_d, v_d, g, beta_val, S_v, delta, attn_data);
        return;
    }
    const float        scale = 1.0f / sqrtf((float) S_v);
    const GGML_F32_VEC vg    = GGML_F32_VEC_SET1(expf(g));
    const int          n     = (int) S_v;
    for (int64_t j0 = 0; j0 < S_v; j0 += 4) {
        const int nr = (int) ((S_v - j0) < 4 ? (S_v - j0) : 4);
        float *       row[4];
        const float * rin[4];
        for (int r = 0; r < 4; r++) {
            row[r] = s_out + (j0 + (r < nr ? r : nr - 1)) * S_v;  // unused slots alias the last row (never written)
            rin[r] = s_in + (j0 + (r < nr ? r : nr - 1)) * S_v;
        }
        // pass A: scale + dot with k
        GGML_F32_VEC sum[4][GGML_F32_ARR];
        for (int r = 0; r < 4; r++) {
            for (int a = 0; a < GGML_F32_ARR; a++) {
                sum[r][a] = GGML_F32_VEC_ZERO;
            }
        }
        for (int i = 0; i < n; i += GGML_F32_STEP) {
            for (int a = 0; a < GGML_F32_ARR; a++) {
                const GGML_F32_VEC kv = GGML_F32_VEC_LOAD(k_d + i + a * GGML_F32_EPR);
                for (int r = 0; r < nr; r++) {
                    GGML_F32_VEC ay = GGML_F32_VEC_LOAD(rin[r] + i + a * GGML_F32_EPR);
                    ay = GGML_F32_VEC_MUL(ay, vg);
                    GGML_F32_VEC_STORE(row[r] + i + a * GGML_F32_EPR, ay);
                    sum[r][a] = GGML_F32_VEC_FMA(sum[r][a], ay, kv);
                }
            }
        }
        GGML_F32_VEC vd[4];
        for (int r = 0; r < nr; r++) {
            float sumf = 0.0f;
            GGML_F32_VEC_REDUCE(sumf, sum[r]);
            float s1 = sumf;
            delta[j0 + r] = (v_d[j0 + r] - s1) * beta_val;
            vd[r]         = GGML_F32_VEC_SET1(delta[j0 + r]);
        }
        // pass B: rank-1 update + dot with q
        for (int r = 0; r < 4; r++) {
            for (int a = 0; a < GGML_F32_ARR; a++) {
                sum[r][a] = GGML_F32_VEC_ZERO;
            }
        }
        for (int i = 0; i < n; i += GGML_F32_STEP) {
            for (int a = 0; a < GGML_F32_ARR; a++) {
                const GGML_F32_VEC kv = GGML_F32_VEC_LOAD(k_d + i + a * GGML_F32_EPR);
                const GGML_F32_VEC qv = GGML_F32_VEC_LOAD(q_d + i + a * GGML_F32_EPR);
                for (int r = 0; r < nr; r++) {
                    GGML_F32_VEC ay = GGML_F32_VEC_LOAD(row[r] + i + a * GGML_F32_EPR);
                    ay = GGML_F32_VEC_FMA(ay, kv, vd[r]);
                    GGML_F32_VEC_STORE(row[r] + i + a * GGML_F32_EPR, ay);
                    sum[r][a] = GGML_F32_VEC_FMA(sum[r][a], ay, qv);
                }
            }
        }
        for (int r = 0; r < nr; r++) {
            float sumf = 0.0f;
            GGML_F32_VEC_REDUCE(sumf, sum[r]);
            float s2 = sumf;
            attn_data[j0 + r] = s2 * scale;
        }
    }
#else
    oxe_gated_delta_net_step_f32_from(s_out, s_in, q_d, k_d, v_d, g, beta_val, S_v, delta, attn_data);
#endif
}

void oxe_gated_delta_net_step_f32_x4(float * s_out, const float * q_d, const float * k_d, const float * v_d,
                                     float g, float beta_val, int64_t S_v, float * delta, float * attn_data) {
    oxe_gated_delta_net_step_f32_x4_from(s_out, s_out, q_d, k_d, v_d, g, beta_val, S_v, delta, attn_data);
}

// ---- RoPE (ops.cpp rope_yarn_ramp / rope_yarn / ggml_mrope_cache_init / rotate_pairs,
//      ggml.c ggml_rope_yarn_corr_dim(s))

static float oxe_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

static void oxe_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow,
                                    bool truncate, float dims[2]) {
    float start = oxe_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base);
    float end   = oxe_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base);
    if (truncate) {
        start = floorf(start);
        end   = ceilf(end);
    }
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

static void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
                      float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

// ggml_mrope_cache_init for a text token at position p (positions [p, p, p, 0]); fills
// cache[0..ne0) (pairs cos, sin). freq_factors = NULL, forward (sin_sign = 1).
void oxe_mrope_cache_init(int64_t p, const int32_t * sections_in, int is_imrope, int n_dims, int64_t ne0,
                          float freq_base, float freq_scale, float ext_factor, float attn_factor,
                          float beta_fast, float beta_slow, int n_ctx_orig, int truncate, float * cache) {
    int sections[4] = { sections_in[0], sections_in[1], sections_in[2], sections_in[3] };
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    float corr_dims[2];
    oxe_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, truncate != 0, corr_dims);
    const float sin_sign = 1.0f;
    const float * freq_factors = NULL;
    const float mscale = attn_factor;
    const bool indep_sects = false;  // is_vision

    float theta_base_t = p, theta_base_h = p, theta_base_w = p, theta_base_e = 0;
    float theta_t = theta_base_t;
    float theta_h = theta_base_h;
    float theta_w = theta_base_w;
    float theta_e = theta_base_e;
    int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int sec_w = sections[1] + sections[0];
    int sec_e = sections[2] + sec_w;
    (void) sec_e;
    (void) indep_sects;

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        int sector = (i0 / 2) % sect_dims;
        float theta = theta_t;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            } else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }
        rope_yarn(theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;
        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

// rotate_pairs<float>(n_dims, n_dims/2, cache, src, dst) in place (NEOX / MROPE / IMROPE);
// the channels past n_dims are left untouched (upstream copies them unchanged).
void oxe_rope_neox_inplace(float * x, int n_dims, const float * cache) {
    const int64_t n_offset = n_dims/2;
    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
        const int64_t ic = i0/2;
        const float cos_theta = cache[i0 + 0];
        const float sin_theta = cache[i0 + 1];
        float * src = x + ic;
        const float x0 = src[0];
        const float x1 = src[n_offset];
        src[0]        = x0*cos_theta - x1*sin_theta;
        src[n_offset] = x0*sin_theta + x1*cos_theta;
    }
}

// One attention head, FA off, single query (llm_graph_context::build_attn_mha, v_trans):
//   kq = mul_mat(K f16, q -> f16)  (ggml_vec_dot_f16 per cached row)
//   kq = soft_max_ext(kq, mask, scale)      over n_kv (llama.cpp pads n_kv to 256; mask -inf)
//   out = mul_mat(V^T f16, kq -> f16)       (ggml_vec_dot_f16 per head dim, over n_kv)
// K rows: k + p*k_stride (head_dim halves) for p < pos_last; the row of pos_last is k_last (the
// caller's own copy of what is being written to the cache). V^T rows: vt + d*v_stride (n_kv
// halves, zero past the valid positions). Positions > pos_last are masked. q16: head_dim halves
// of scratch, kq/mask/wp: n_kv floats each, p16: n_kv halves.
void oxe_attn_head_f16(const float * q, const ggml_fp16_t * k, int64_t k_stride, const ggml_fp16_t * k_last,
                       const ggml_fp16_t * vt, int64_t v_stride,
                       int n_kv, int pos_last, int head_dim, float scale, ggml_fp16_t * q16, float * kq, float * mask,
                       float * wp, ggml_fp16_t * p16, float * out) {
    for (int i = 0; i < head_dim; i++) {
        q16[i] = GGML_CPU_FP32_TO_FP16(q[i]);
    }
    for (int p = 0; p < n_kv; p++) {
        if (p <= pos_last) {
            float s = 0.0f;
            const ggml_fp16_t * kr = p == pos_last ? k_last : k + p * k_stride;
            ggml_vec_dot_f16(head_dim, &s, 0, (ggml_fp16_t *) kr, 0, q16, 0, 1);
            kq[p] = s;
            mask[p] = 0.0f;
        } else {
            kq[p] = 0.0f;
            mask[p] = -INFINITY;
        }
    }
    oxe_soft_max_f32(n_kv, kq, kq, scale, mask, wp);
    for (int p = 0; p < n_kv; p++) {
        p16[p] = GGML_CPU_FP32_TO_FP16(kq[p]);
    }
    for (int d = 0; d < head_dim; d++) {
        float s = 0.0f;
        ggml_vec_dot_f16(n_kv, &s, 0, (ggml_fp16_t *) (vt + d * v_stride), 0, p16, 0, 1);
        out[d] = s;
    }
}

// ---------------------------------------------------------------------------------------------
// M3-ATTN: long-context attention building blocks, bit-identical to the per-element path above
// (ggml_vec_dot_f16 per K row / V^T row, ggml_compute_forward_soft_max_f32 over n_kv).
//
// oxe_dot_f16_tile computes a block of independent ggml_vec_dot_f16 results. Each output is
// produced by exactly the operations ggml_vec_dot_f16 performs for it on this build (NEON with
// FP16 vector arithmetic: four float16x8 accumulators, accumulator j takes elements
// [i + 8j, i + 8j + 8) of every 32-element step i in increasing i, vfmaq_f16; then
// GGML_F16x8_REDUCE and the (float) store). Only the interleaving of independent outputs and of
// independent accumulators differs, which cannot change any rounding.
//
// Tail rule (`n_full`): the caller may stop the dot at n < n_full when every element y[n..n_full)
// is +0 (masked softmax probabilities). For such a step the reference performs
// acc = fma(acc, x, +0) = acc + (x * +0): this leaves every non-zero lane unchanged and every +0
// lane +0 (+0 + ±0 = +0 under round-to-nearest); only a lane holding -0 can change (to +0 when
// x >= +0). So after step n we check for -0 lanes and, if there is one, run the remaining steps
// exactly as the reference would (with y = +0). The result equals the full-length dot bit for
// bit whenever x[n..n_full) is finite (it always is: those are V cache entries).
// ---------------------------------------------------------------------------------------------
}  // extern "C"

#include <arm_neon.h>

namespace {

constexpr int OXE_DT_BLK  = 256;  // positions per block (8 steps of 32): rows stay in L1 across j
constexpr int OXE_DT_MAXC = 64;   // columns per chunk (accumulator state 4 x 64 x 4 x 16 B)

// GGML_F16x8_REDUCE with GGML_F16_ARR = 4, then ggml_vec_dot_f16's double sumf and float store.
static inline float oxe_f16_reduce(float16x8_t a0, float16x8_t a1, float16x8_t a2, float16x8_t a3) {
    a0 = vaddq_f16(a0, a2);
    a1 = vaddq_f16(a1, a3);
    a0 = vaddq_f16(a0, a1);
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(a0));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(a0));
    const ggml_float sumf = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));
    return (float) sumf;
}

// Steps [i0, i1) (multiples of 32) of R x C dots; st[r*rs + c*4 + j] holds accumulator j of (r, c).
template <int R, int C>
static inline void oxe_tile_steps(int i0, int i1, const __fp16 * const * xr, const __fp16 * const * yc,
                                  float16x8_t * st, int rs) {
    for (int j = 0; j < 4; j++) {
        float16x8_t a[R][C];
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                a[r][c] = st[r * rs + c * 4 + j];
            }
        }
        for (int i = i0 + 8 * j; i < i1; i += 32) {
            float16x8_t xv[R];
            float16x8_t yv[C];
            for (int r = 0; r < R; r++) {
                xv[r] = vld1q_f16(xr[r] + i);
            }
            for (int c = 0; c < C; c++) {
                yv[c] = vld1q_f16(yc[c] + i);
            }
            for (int r = 0; r < R; r++) {
                for (int c = 0; c < C; c++) {
                    a[r][c] = vfmaq_f16(a[r][c], xv[r], yv[c]);
                }
            }
        }
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                st[r * rs + c * 4 + j] = a[r][c];
            }
        }
    }
}

template <int R>
static inline void oxe_tile_steps_c(int nc, int i0, int i1, const __fp16 * const * xr, const __fp16 * const * yc,
                                    float16x8_t * st, int rs) {
    switch (nc) {
        case 1: oxe_tile_steps<R, 1>(i0, i1, xr, yc, st, rs); break;
        case 2: oxe_tile_steps<R, 2>(i0, i1, xr, yc, st, rs); break;
        case 3: oxe_tile_steps<R, 3>(i0, i1, xr, yc, st, rs); break;
        default: oxe_tile_steps<R, 4>(i0, i1, xr, yc, st, rs); break;
    }
}

static inline bool oxe_has_neg_zero(const float16x8_t * a) {
    const uint16x8_t nz = vdupq_n_u16(0x8000);
    uint16x8_t m = vceqq_u16(vreinterpretq_u16_f16(a[0]), nz);
    m = vorrq_u16(m, vceqq_u16(vreinterpretq_u16_f16(a[1]), nz));
    m = vorrq_u16(m, vceqq_u16(vreinterpretq_u16_f16(a[2]), nz));
    m = vorrq_u16(m, vceqq_u16(vreinterpretq_u16_f16(a[3]), nz));
    return vmaxvq_u16(m) != 0;
}

}  // namespace

extern "C" {

// out[r*ox + c*oy] = ggml_vec_dot_f16(n, x + r*xs, y + c*ys) for r < nx, c < ny, bit for bit.
// n_full (optional, per column, multiples of 32, >= n): the reference length when y[c] is +0
// beyond n (see the tail rule above); x rows must be readable up to max(n_full).
void oxe_dot_f16_tile(int n, const ggml_fp16_t * x, int64_t xs, int nx, const ggml_fp16_t * y, int64_t ys, int ny,
                      float * out, int64_t ox, int64_t oy, const int * n_full) {
    if (n % 32 != 0) {
        // ggml's scalar leftover path; not used by the engine (n is a multiple of 32 there)
        for (int r = 0; r < nx; r++) {
            for (int c = 0; c < ny; c++) {
                float s = 0.0f;
                ggml_vec_dot_f16(n, &s, 0, (ggml_fp16_t *) (x + r * xs), 0, (ggml_fp16_t *) (y + c * ys), 0, 1);
                out[r * ox + c * oy] = s;
            }
        }
        return;
    }
    float16x8_t st[4 * OXE_DT_MAXC * 4];
    const __fp16 * yc[OXE_DT_MAXC];
    const float16x8_t zero = vdupq_n_f16(0.0f);
    for (int c0 = 0; c0 < ny; c0 += OXE_DT_MAXC) {
        const int nc = std::min(OXE_DT_MAXC, ny - c0);
        const int rs = nc * 4;
        for (int c = 0; c < nc; c++) {
            yc[c] = (const __fp16 *) (y + (int64_t) (c0 + c) * ys);
        }
        for (int r0 = 0; r0 < nx; r0 += 4) {
            const int nr = std::min(4, nx - r0);
            const __fp16 * xr[4];
            for (int r = 0; r < nr; r++) {
                xr[r] = (const __fp16 *) (x + (int64_t) (r0 + r) * xs);
            }
            for (int i = 0; i < nr * rs; i++) {
                st[i] = zero;
            }
            for (int b = 0; b < n; b += OXE_DT_BLK) {
                const int b1 = std::min(n, b + OXE_DT_BLK);
                for (int c = 0; c < nc; c += 4) {
                    const int ncc = std::min(4, nc - c);
                    switch (nr) {
                        case 1: oxe_tile_steps_c<1>(ncc, b, b1, xr, yc + c, st + c * 4, rs); break;
                        case 2: oxe_tile_steps_c<2>(ncc, b, b1, xr, yc + c, st + c * 4, rs); break;
                        case 3: oxe_tile_steps_c<3>(ncc, b, b1, xr, yc + c, st + c * 4, rs); break;
                        default: oxe_tile_steps_c<4>(ncc, b, b1, xr, yc + c, st + c * 4, rs); break;
                    }
                }
            }
            for (int r = 0; r < nr; r++) {
                for (int c = 0; c < nc; c++) {
                    float16x8_t * a = st + r * rs + c * 4;
                    if (n_full && n_full[c0 + c] > n && oxe_has_neg_zero(a)) {
                        for (int j = 0; j < 4; j++) {
                            for (int i = n + 8 * j; i < n_full[c0 + c]; i += 32) {
                                a[j] = vfmaq_f16(a[j], vld1q_f16(xr[r] + i), zero);
                            }
                        }
                    }
                    out[(int64_t) (r0 + r) * ox + (int64_t) (c0 + c) * oy] = oxe_f16_reduce(a[0], a[1], a[2], a[3]);
                }
            }
        }
    }
}

// moe-prefill: oxe_dot_f16_tile with the loops blocked by POSITION for long dots (the V^T . p
// product of the prefill attention at a long prefix: n = prefix length, nx = 256 V^T rows,
// ny = 32 probability columns). oxe_dot_f16_tile runs the whole length n for 4 rows at a time,
// so the ny columns (ny * n * 2 bytes, 448 KB at a 7K prefix) are re-read from L2 for every row
// group. Here a block of `pb` positions of all columns (ny * pb * 2 bytes, L1-sized) is used by
// all nx rows before moving on; the accumulators of all (row, column) pairs live in `st`
// (nx * ny * 64 bytes, caller scratch) between blocks. Every accumulator sees exactly the same
// sequence of vfmaq_f16 as in oxe_dot_f16_tile (same j, increasing i), so the result is bitwise
// the same (unit test pf_dot_f16_tile_pb_bitwise). n % 32 == 0, pb % 32 == 0, ny <= 64.
void oxe_dot_f16_tile_pb(int n, const ggml_fp16_t * x, int64_t xs, int nx, const ggml_fp16_t * y, int64_t ys, int ny,
                         float * out, int64_t ox, int64_t oy, const int * n_full, int pb, void * st_v) {
    if (n % 32 != 0 || pb <= 0 || pb % 32 != 0 || ny > OXE_DT_MAXC || st_v == nullptr) {
        oxe_dot_f16_tile(n, x, xs, nx, y, ys, ny, out, ox, oy, n_full);
        return;
    }
    float16x8_t * st = (float16x8_t *) st_v;  // [row][col][4]
    const int rs = ny * 4;
    const float16x8_t zero = vdupq_n_f16(0.0f);
    const __fp16 * yc[OXE_DT_MAXC];
    for (int c = 0; c < ny; c++) {
        yc[c] = (const __fp16 *) (y + (int64_t) c * ys);
    }
    for (int i = 0; i < nx * rs; i++) {
        st[i] = zero;
    }
    for (int b = 0; b < n; b += pb) {
        const int b1 = std::min(n, b + pb);
        for (int r0 = 0; r0 < nx; r0 += 4) {
            const int nr = std::min(4, nx - r0);
            const __fp16 * xr[4];
            for (int r = 0; r < nr; r++) {
                xr[r] = (const __fp16 *) (x + (int64_t) (r0 + r) * xs);
            }
            float16x8_t * sr = st + (int64_t) r0 * rs;
            // inner blocks of OXE_DT_BLK positions keep the 4 x rows in L1 across the j passes
            for (int bb = b; bb < b1; bb += OXE_DT_BLK) {
                const int bb1 = std::min(b1, bb + OXE_DT_BLK);
                for (int c = 0; c < ny; c += 4) {
                    const int ncc = std::min(4, ny - c);
                    switch (nr) {
                        case 1: oxe_tile_steps_c<1>(ncc, bb, bb1, xr, yc + c, sr + c * 4, rs); break;
                        case 2: oxe_tile_steps_c<2>(ncc, bb, bb1, xr, yc + c, sr + c * 4, rs); break;
                        case 3: oxe_tile_steps_c<3>(ncc, bb, bb1, xr, yc + c, sr + c * 4, rs); break;
                        default: oxe_tile_steps_c<4>(ncc, bb, bb1, xr, yc + c, sr + c * 4, rs); break;
                    }
                }
            }
        }
    }
    for (int r = 0; r < nx; r++) {
        const __fp16 * xr = (const __fp16 *) (x + (int64_t) r * xs);
        for (int c = 0; c < ny; c++) {
            float16x8_t * a = st + (int64_t) r * rs + c * 4;
            if (n_full && n_full[c] > n && oxe_has_neg_zero(a)) {
                for (int j = 0; j < 4; j++) {
                    for (int i = n + 8 * j; i < n_full[c]; i += 32) {
                        a[j] = vfmaq_f16(a[j], vld1q_f16(xr + i), zero);
                    }
                }
            }
            out[(int64_t) r * ox + (int64_t) c * oy] = oxe_f16_reduce(a[0], a[1], a[2], a[3]);
        }
    }
}

// One row of oxe_soft_max_f32 over n_kv (llama.cpp's FA-off soft_max_ext with the causal mask:
// kq[i] for i <= pos, -inf beyond), evaluated on the first n4 = round_up(pos + 1, 4) entries only:
// the remaining groups of four are exp(-inf) = +0 (ggml_v_expf returns s1*s1 = 2^-250 -> +0)
// and add +0.0 to the double sum, so max, sum and every probability are unchanged. Writes the f16
// probabilities to p16[0, n_out) (+0 beyond pos). kq is read on [0, pos]; wp/dp scratch >= n4.
void oxe_attn_softmax_row_f16(const float * kq, int pos, int n_kv, float scale, float * wp, float * dp,
                              ggml_fp16_t * p16, int n_out) {
    const float slope = 1.0f;
    const int n4 = std::min(n_kv, (pos + 4) & ~3);
    for (int i = 0; i < n4; i++) {
        wp[i] = i <= pos ? kq[i] : 0.0f;
    }
    ggml_vec_scale_f32(n4, wp, scale);
    for (int i = 0; i < n4; i++) {
        const float m = i <= pos ? 0.0f : -INFINITY;
        wp[i] += slope*m;
    }
    float max = -INFINITY;
    ggml_vec_max_f32(n4, &max, wp);
    ggml_float sum = ggml_vec_soft_max_f32(n4, dp, wp, max);
    sum = 1.0/sum;
    ggml_vec_scale_f32(n4, dp, sum);
    const int nw = std::min(n4, n_out);
    for (int i = 0; i < nw; i++) {
        p16[i] = GGML_CPU_FP32_TO_FP16(dp[i]);
    }
    for (int i = nw; i < n_out; i++) {
        p16[i] = 0;
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------------------------
// O-13: int8 (Q8_0) KV cache kernels. A Q8_0 block is 34 bytes: f16 scale d, then 32 int8 q.
// ---------------------------------------------------------------------------------------------

namespace {

static inline float oxe_q8_d(const uint8_t * b) {
    __fp16 h;
    memcpy(&h, b, 2);
    return (float) h;
}

}  // namespace

extern "C" {

// Scores of Q8_0 rows against Q8_0 columns: out[r*ox + c*oy] = ggml_vec_dot_q8_0_q8_0_generic(n,
// x + r*xs, y + c*ys) bit for bit: per block the exact int32 sum of 32 products (sdot), then
// sumf = fma(float(sumi), dx*dy, sumf) in block order (the generic loop as GCC contracts it).
// Blocking: 4 rows x 4 columns; per block the 4 columns' quants (8 q regs) and scales are loaded
// once and reused by the 4 rows; each row's 4 dots are 2 sdot each, folded to one int32x4 by
// pairwise adds and accumulated into one float32x4 (lane = column). Rows outer: a row group
// (4 x 34 B per block) stays in L1 across the column groups. Tails reuse the last row / column
// (computed, not stored). n % 32 == 0.
void oxe_dot_q8_tile(int n, const uint8_t * x, int64_t xs, int nx, const uint8_t * y, int64_t ys, int ny,
                     float * out, int64_t ox, int64_t oy) {
    const int nb = n / 32;
    const int32x4_t zi = vdupq_n_s32(0);
    for (int r0 = 0; r0 < nx; r0 += 4) {
        const int nr = std::min(4, nx - r0);
        const uint8_t * xr[4];
        for (int r = 0; r < 4; r++) {
            xr[r] = x + (int64_t) (r0 + std::min(r, nr - 1)) * xs;
        }
        for (int c0 = 0; c0 < ny; c0 += 4) {
            const int nc = std::min(4, ny - c0);
            const uint8_t * yc[4];
            for (int c = 0; c < 4; c++) {
                yc[c] = y + (int64_t) (c0 + std::min(c, nc - 1)) * ys;
            }
            float32x4_t acc[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (int ib = 0; ib < nb; ib++) {
                int8x16_t ya[4], yb[4];
                float dy[4];
                for (int c = 0; c < 4; c++) {
                    const uint8_t * b = yc[c] + ib * 34;
                    dy[c] = oxe_q8_d(b);
                    ya[c] = vld1q_s8((const int8_t *) (b + 2));
                    yb[c] = vld1q_s8((const int8_t *) (b + 18));
                }
                const float32x4_t dyv = vld1q_f32(dy);
                for (int r = 0; r < 4; r++) {
                    const uint8_t * b = xr[r] + ib * 34;
                    const float dx = oxe_q8_d(b);
                    const int8x16_t xa = vld1q_s8((const int8_t *) (b + 2));
                    const int8x16_t xb = vld1q_s8((const int8_t *) (b + 18));
                    const int32x4_t s0 = vdotq_s32(vdotq_s32(zi, xa, ya[0]), xb, yb[0]);
                    const int32x4_t s1 = vdotq_s32(vdotq_s32(zi, xa, ya[1]), xb, yb[1]);
                    const int32x4_t s2 = vdotq_s32(vdotq_s32(zi, xa, ya[2]), xb, yb[2]);
                    const int32x4_t s3 = vdotq_s32(vdotq_s32(zi, xa, ya[3]), xb, yb[3]);
                    const int32x4_t si = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
                    // dx*dy as the generic loop rounds it (multiplication commutes exactly)
                    acc[r] = vfmaq_f32(acc[r], vcvtq_f32_s32(si), vmulq_n_f32(dyv, dx));
                }
            }
            for (int r = 0; r < nr; r++) {
                float a[4];
                vst1q_f32(a, acc[r]);
                for (int c = 0; c < nc; c++) {
                    out[(int64_t) (r0 + r) * ox + (int64_t) (c0 + c) * oy] = a[c];
                }
            }
        }
    }
}

// Outputs over Q8_0 V rows (one per position): out[c*hd + d] = sum over p in [p0, p1), in
// increasing p, of fma(deq(V[p])[d], prob[c][p], acc) starting from +0, with deq = q * d in f32
// (ggml dequantize_row_q8_0, exact) and prob the f16 probability (exact in f32).
// V rows: v + p*vs; probabilities: prob + c*ps + p (f16). pscr: scratch of
// (p1 - p0) * round_up(ncol, 4) floats (the chunk's probabilities, [p][c], zero padded).
// Blocking: 16 dims (one half block: 16 int8 -> 4 float32x4) x 4 columns = 16 accumulator
// registers held over all positions of the chunk; per position the half block is loaded and
// dequantised once per column group and FMA'd into the 4 columns by lane (vfmaq_laneq_f32).
// The chunk's rows (callers use <= 256 positions, 70 KB at hd 256) stay in L1 across the passes.
void oxe_pv_q8(int hd, const uint8_t * v, int64_t vs, int p0, int p1, const uint16_t * prob, int64_t ps, int ncol,
               float * out, float * pscr) {
    const int np = p1 - p0;
    const int nc4 = (ncol + 3) & ~3;
    for (int p = 0; p < np; p++) {
        for (int c = 0; c < nc4; c++) {
            if (c < ncol) {
                __fp16 h;
                memcpy(&h, prob + (int64_t) c * ps + p0 + p, 2);
                pscr[p * nc4 + c] = (float) h;
            } else {
                pscr[p * nc4 + c] = 0.0f;
            }
        }
    }
    for (int dc = 0; dc < hd; dc += 16) {
        const int blk = dc / 32;
        const int half = (dc % 32) / 16;
        for (int c0 = 0; c0 < nc4; c0 += 4) {
            float32x4_t a0[4], a1[4], a2[4], a3[4];  // [column][dims 4k..4k+3]
            for (int k = 0; k < 4; k++) {
                a0[k] = vdupq_n_f32(0.0f);
                a1[k] = vdupq_n_f32(0.0f);
                a2[k] = vdupq_n_f32(0.0f);
                a3[k] = vdupq_n_f32(0.0f);
            }
            for (int p = 0; p < np; p++) {
                const uint8_t * b = v + (int64_t) (p0 + p) * vs + blk * 34;
                const float d = oxe_q8_d(b);
                const int8x16_t q = vld1q_s8((const int8_t *) (b + 2 + 16 * half));
                const int16x8_t lo = vmovl_s8(vget_low_s8(q));
                const int16x8_t hi = vmovl_s8(vget_high_s8(q));
                float32x4_t f[4];
                f[0] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), d);
                f[1] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), d);
                f[2] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), d);
                f[3] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), d);
                const float32x4_t pv = vld1q_f32(pscr + p * nc4 + c0);
                for (int k = 0; k < 4; k++) {
                    a0[k] = vfmaq_laneq_f32(a0[k], f[k], pv, 0);
                    a1[k] = vfmaq_laneq_f32(a1[k], f[k], pv, 1);
                    a2[k] = vfmaq_laneq_f32(a2[k], f[k], pv, 2);
                    a3[k] = vfmaq_laneq_f32(a3[k], f[k], pv, 3);
                }
            }
            float32x4_t * a[4] = {a0, a1, a2, a3};
            for (int c = 0; c < 4 && c0 + c < ncol; c++) {
                for (int k = 0; k < 4; k++) {
                    vst1q_f32(out + (int64_t) (c0 + c) * hd + dc + 4 * k, a[c][k]);
                }
            }
        }
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------------------------
// OD-4d: faster V . p over Q8_0 V rows. oxe_pv_q8 above is the reference; oxe_pv_q8_fast is bit
// identical to it; oxe_pv_q8_int is the lossy integer path (KLD gated).
// ---------------------------------------------------------------------------------------------

// the same block of the next chunk's row (a thread's chunks are consecutive), into L2
#define OXE_PV_PREFETCH(a) __builtin_prefetch((a), 0, 2)

namespace {

// 8 int8 -> 2 x float32x4, exact: tbl puts each byte in the top byte of an int32 lane (q << 24),
// scvtf with 24 fractional bits gives float(q).
static const uint8_t oxe_q8_tbl[32] = {255, 255, 255, 0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3,
                                       255, 255, 255, 4, 255, 255, 255, 5, 255, 255, 255, 6, 255, 255, 255, 7};

// One tile of the fp32 V . p: 8 dims x 8 columns, 16 accumulators (named: an accumulator array
// is stored back to the stack every position by GCC) over the chunk's positions; w = the block's
// weights [p][wst] (d * prob, exact in f32). Stores the first nst columns.
static inline void oxe_pv_q8_fp_tile8(int np, const uint8_t * b, int64_t vs, const float * w, int64_t wst, float * out,
                                      int64_t os, int nst) {
    const uint8x16_t t0 = vld1q_u8(oxe_q8_tbl), t1 = vld1q_u8(oxe_q8_tbl + 16);
    const float32x4_t z = vdupq_n_f32(0.0f);
    float32x4_t a00 = z, a01 = z, a10 = z, a11 = z, a20 = z, a21 = z, a30 = z, a31 = z;
    float32x4_t a40 = z, a41 = z, a50 = z, a51 = z, a60 = z, a61 = z, a70 = z, a71 = z;
    for (int p = 0; p < np; p++) {
        const uint8x16_t q = vcombine_u8(vld1_u8(b + (int64_t) p * vs), vdup_n_u8(0));
        const float32x4_t f0 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(q, t0)), 24);
        const float32x4_t f1 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(q, t1)), 24);
        const float32x4_t w0 = vld1q_f32(w + (int64_t) p * wst), w1 = vld1q_f32(w + (int64_t) p * wst + 4);
        a00 = vfmaq_laneq_f32(a00, f0, w0, 0);
        a01 = vfmaq_laneq_f32(a01, f1, w0, 0);
        a10 = vfmaq_laneq_f32(a10, f0, w0, 1);
        a11 = vfmaq_laneq_f32(a11, f1, w0, 1);
        a20 = vfmaq_laneq_f32(a20, f0, w0, 2);
        a21 = vfmaq_laneq_f32(a21, f1, w0, 2);
        a30 = vfmaq_laneq_f32(a30, f0, w0, 3);
        a31 = vfmaq_laneq_f32(a31, f1, w0, 3);
        a40 = vfmaq_laneq_f32(a40, f0, w1, 0);
        a41 = vfmaq_laneq_f32(a41, f1, w1, 0);
        a50 = vfmaq_laneq_f32(a50, f0, w1, 1);
        a51 = vfmaq_laneq_f32(a51, f1, w1, 1);
        a60 = vfmaq_laneq_f32(a60, f0, w1, 2);
        a61 = vfmaq_laneq_f32(a61, f1, w1, 2);
        a70 = vfmaq_laneq_f32(a70, f0, w1, 3);
        a71 = vfmaq_laneq_f32(a71, f1, w1, 3);
    }
    const float32x4_t r[16] = {a00, a01, a10, a11, a20, a21, a30, a31, a40, a41, a50, a51, a60, a61, a70, a71};
    for (int c = 0; c < nst; c++) {
        vst1q_f32(out + c * os, r[2 * c]);
        vst1q_f32(out + c * os + 4, r[2 * c + 1]);
    }
}

// The same for 16 dims x 4 columns (the last column group when round_up(ncol, 4) % 8 == 4).
static inline void oxe_pv_q8_fp_tile4(int np, const uint8_t * b, int64_t vs, const float * w, int64_t wst, float * out,
                                      int64_t os, int nst) {
    const uint8x16_t t0 = vld1q_u8(oxe_q8_tbl), t1 = vld1q_u8(oxe_q8_tbl + 16);
    const float32x4_t z = vdupq_n_f32(0.0f);
    float32x4_t a00 = z, a01 = z, a02 = z, a03 = z, a10 = z, a11 = z, a12 = z, a13 = z;
    float32x4_t a20 = z, a21 = z, a22 = z, a23 = z, a30 = z, a31 = z, a32 = z, a33 = z;
    for (int p = 0; p < np; p++) {
        const uint8x16_t q = vld1q_u8(b + (int64_t) p * vs);
        const uint8x16_t qh = vextq_u8(q, q, 8);
        const float32x4_t f0 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(q, t0)), 24);
        const float32x4_t f1 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(q, t1)), 24);
        const float32x4_t f2 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(qh, t0)), 24);
        const float32x4_t f3 = vcvtq_n_f32_s32(vreinterpretq_s32_u8(vqtbl1q_u8(qh, t1)), 24);
        const float32x4_t w0 = vld1q_f32(w + (int64_t) p * wst);
        a00 = vfmaq_laneq_f32(a00, f0, w0, 0);
        a01 = vfmaq_laneq_f32(a01, f1, w0, 0);
        a02 = vfmaq_laneq_f32(a02, f2, w0, 0);
        a03 = vfmaq_laneq_f32(a03, f3, w0, 0);
        a10 = vfmaq_laneq_f32(a10, f0, w0, 1);
        a11 = vfmaq_laneq_f32(a11, f1, w0, 1);
        a12 = vfmaq_laneq_f32(a12, f2, w0, 1);
        a13 = vfmaq_laneq_f32(a13, f3, w0, 1);
        a20 = vfmaq_laneq_f32(a20, f0, w0, 2);
        a21 = vfmaq_laneq_f32(a21, f1, w0, 2);
        a22 = vfmaq_laneq_f32(a22, f2, w0, 2);
        a23 = vfmaq_laneq_f32(a23, f3, w0, 2);
        a30 = vfmaq_laneq_f32(a30, f0, w0, 3);
        a31 = vfmaq_laneq_f32(a31, f1, w0, 3);
        a32 = vfmaq_laneq_f32(a32, f2, w0, 3);
        a33 = vfmaq_laneq_f32(a33, f3, w0, 3);
    }
    const float32x4_t r[16] = {a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33};
    for (int c = 0; c < nst; c++) {
        for (int k = 0; k < 4; k++) {
            vst1q_f32(out + c * os + 4 * k, r[4 * c + k]);
        }
    }
}
}  // namespace

extern "C" {

// oxe_pv_q8 bit for bit, faster (the production fp32 path; oxe_pv_q8 stays as the reference).
// Same value per output: q * d is exact in f32 (8 x 11 significant bits) and so is d * prob
// (11 x 11 bits, no underflow: both are f16), so fma(q * d, prob, acc) == fma(float(q),
// d * prob, acc), the same real product rounded once. Per block the weights d * prob are formed
// once ([p][col], 2 multiplies per position instead of 4 per position, half block and column
// group); tiles of 8 dims x 8 columns (16 accumulators in registers) convert each position's 8
// quants once for 8 columns (tbl + scvtf, 4 instructions per 16 FMAs).
// scr: 2 * (p1 - p0) * round_up(ncol, 4) floats.
void oxe_pv_q8_fast(int hd, const uint8_t * v, int64_t vs, int p0, int p1, const uint16_t * prob, int64_t ps, int ncol,
                    float * out, float * scr) {
    const int np = p1 - p0;
    const int nc4 = (ncol + 3) & ~3;
    float * pf = scr;
    float * w = scr + (int64_t) np * nc4;
    for (int p = 0; p < np; p++) {
        for (int c = 0; c < nc4; c++) {
            if (c < ncol) {
                __fp16 h;
                memcpy(&h, prob + (int64_t) c * ps + p0 + p, 2);
                pf[p * nc4 + c] = (float) h;
            } else {
                pf[p * nc4 + c] = 0.0f;
            }
        }
    }
    const uint8_t * v0 = v + (int64_t) p0 * vs;
    for (int blk = 0; blk < hd / 32; blk++) {
        const uint8_t * b = v0 + blk * 34;
        for (int p = 0; p < np; p++) {
            const float d = oxe_q8_d(b + (int64_t) p * vs);
            for (int c = 0; c < nc4; c += 4) {
                vst1q_f32(w + p * nc4 + c, vmulq_n_f32(vld1q_f32(pf + p * nc4 + c), d));
            }
        }
        for (int c0 = 0; c0 + 8 <= nc4; c0 += 8) {
            for (int sub = 0; sub < 4; sub++) {
                oxe_pv_q8_fp_tile8(np, b + 2 + 8 * sub, vs, w + c0, nc4, out + (int64_t) c0 * hd + blk * 32 + sub * 8, hd,
                                   std::min(8, ncol - c0));
            }
        }
        if (nc4 & 4) {
            const int c0 = nc4 - 4;
            for (int half = 0; half < 2; half++) {
                oxe_pv_q8_fp_tile4(np, b + 2 + 16 * half, vs, w + c0, nc4, out + (int64_t) c0 * hd + blk * 32 + half * 16, hd,
                                   ncol - c0);
            }
        }
    }
}

}  // extern "C"

namespace {

// max of n8 (multiple of 8) non-negative f16, as float
static inline float oxe_f16_max(const __fp16 * x, int n8) {
    float16x8_t m = vdupq_n_f16(0);
    for (int i = 0; i < n8; i += 8) {
        m = vmaxq_f16(m, vld1q_f16(x + i));
    }
    return (float) vmaxvq_f16(m);
}

// x[i] = f16(float(x[i]) * s), n8 multiple of 8
static inline void oxe_f16_scale(__fp16 * x, int n8, float s) {
    for (int i = 0; i < n8; i += 8) {
        const float16x8_t h = vld1q_f16(x + i);
        const float32x4_t a = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h)), s);
        const float32x4_t b = vmulq_n_f32(vcvt_high_f32_f16(h), s);
        vst1q_f16(x + i, vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b)));
    }
}

// 8 rows x 16 int8 (r[i]: 16 readable bytes) -> t[16 * k ..]: [dim 2k over the 8 rows | dim 2k+1
// over the 8 rows], k = 0..7 (the operand layout of usmmla: positions along the dot).
static inline void oxe_q8_tr8x16(const uint8_t * const r[8], int off, int8_t * t) {
    const int8x16_t r0 = vld1q_s8((const int8_t *) r[0] + off), r1 = vld1q_s8((const int8_t *) r[1] + off);
    const int8x16_t r2 = vld1q_s8((const int8_t *) r[2] + off), r3 = vld1q_s8((const int8_t *) r[3] + off);
    const int8x16_t r4 = vld1q_s8((const int8_t *) r[4] + off), r5 = vld1q_s8((const int8_t *) r[5] + off);
    const int8x16_t r6 = vld1q_s8((const int8_t *) r[6] + off), r7 = vld1q_s8((const int8_t *) r[7] + off);
    const int8x16_t a0 = vzip1q_s8(r0, r4), a1 = vzip1q_s8(r1, r5), a2 = vzip1q_s8(r2, r6), a3 = vzip1q_s8(r3, r7);
    const int8x16_t a4 = vzip2q_s8(r0, r4), a5 = vzip2q_s8(r1, r5), a6 = vzip2q_s8(r2, r6), a7 = vzip2q_s8(r3, r7);
    const int8x16_t b0 = vzip1q_s8(a0, a2), b1 = vzip2q_s8(a0, a2), b2 = vzip1q_s8(a1, a3), b3 = vzip2q_s8(a1, a3);
    const int8x16_t b4 = vzip1q_s8(a4, a6), b5 = vzip2q_s8(a4, a6), b6 = vzip1q_s8(a5, a7), b7 = vzip2q_s8(a5, a7);
    vst1q_s8(t, vzip1q_s8(b0, b2));
    vst1q_s8(t + 16, vzip2q_s8(b0, b2));
    vst1q_s8(t + 32, vzip1q_s8(b1, b3));
    vst1q_s8(t + 48, vzip2q_s8(b1, b3));
    vst1q_s8(t + 64, vzip1q_s8(b4, b6));
    vst1q_s8(t + 80, vzip2q_s8(b4, b6));
    vst1q_s8(t + 96, vzip1q_s8(b5, b7));
    vst1q_s8(t + 112, vzip2q_s8(b5, b7));
}

// One integer tile: 8 dims (t: 4 x 16 transposed bytes per group of 8 positions, groups 256
// bytes apart) x 4 weight operands (w: 4 x 16 bytes per group, groups gs apart) over ng groups:
// res[4 * k + j] = sum over groups of usmmla(w[j], t[k]). 16 named accumulators (an accumulator
// array is stored back to the stack every iteration by GCC; usmmla latency 4 cycles: 16 chains).
static inline void oxe_pv_q8_mm_tile(int ng, const int8_t * t, const uint8_t * w, int64_t gs, int32x4_t res[16]) {
    const int32x4_t z = vdupq_n_s32(0);
    int32x4_t a00 = z, a01 = z, a02 = z, a03 = z, a10 = z, a11 = z, a12 = z, a13 = z;
    int32x4_t a20 = z, a21 = z, a22 = z, a23 = z, a30 = z, a31 = z, a32 = z, a33 = z;
    for (int g = 0; g < ng; g++) {
        const int8x16_t t0 = vld1q_s8(t), t1 = vld1q_s8(t + 16), t2 = vld1q_s8(t + 32), t3 = vld1q_s8(t + 48);
        const uint8x16_t w0 = vld1q_u8(w), w1 = vld1q_u8(w + 16), w2 = vld1q_u8(w + 32), w3 = vld1q_u8(w + 48);
        t += 256;
        w += gs;
        a00 = vusmmlaq_s32(a00, w0, t0);
        a01 = vusmmlaq_s32(a01, w1, t0);
        a02 = vusmmlaq_s32(a02, w2, t0);
        a03 = vusmmlaq_s32(a03, w3, t0);
        a10 = vusmmlaq_s32(a10, w0, t1);
        a11 = vusmmlaq_s32(a11, w1, t1);
        a12 = vusmmlaq_s32(a12, w2, t1);
        a13 = vusmmlaq_s32(a13, w3, t1);
        a20 = vusmmlaq_s32(a20, w0, t2);
        a21 = vusmmlaq_s32(a21, w1, t2);
        a22 = vusmmlaq_s32(a22, w2, t2);
        a23 = vusmmlaq_s32(a23, w3, t2);
        a30 = vusmmlaq_s32(a30, w0, t3);
        a31 = vusmmlaq_s32(a31, w1, t3);
        a32 = vusmmlaq_s32(a32, w2, t3);
        a33 = vusmmlaq_s32(a33, w3, t3);
    }
    res[0] = a00; res[1] = a01; res[2] = a02; res[3] = a03; res[4] = a10; res[5] = a11; res[6] = a12; res[7] = a13;
    res[8] = a20; res[9] = a21; res[10] = a22; res[11] = a23; res[12] = a30; res[13] = a31; res[14] = a32; res[15] = a33;
}

#define OXE_PV8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)

// Quantised weights of 8 columns of one block: round(pn[c] * dn * r[c]) with r[c] = full /
// max_p(pn[c] * dn), the (block, column) maximum of the chunk as full scale (255, or 65280 for
// 16 bits). 8 bits: w[g][c][8]; 16 bits: w[g][c] = [high bytes of the 8 positions | low bytes].
// rr[c] = r[c] as applied (0: the column has no weight in this chunk).
template <int BITS>
static inline void oxe_pv_q8_weights(int ng, int np8, const __fp16 * pn, const __fp16 * dn, uint8_t * w, int64_t gs, float rr[8]) {
#define OXE_M(j) float16x8_t m##j = vdupq_n_f16(0);
    OXE_PV8(OXE_M)
#undef OXE_M
    for (int g = 0; g < ng; g++) {
        const float16x8_t d = vld1q_f16(dn + 8 * g);
#define OXE_M(j) m##j = vmaxq_f16(m##j, vmulq_f16(vld1q_f16(pn + (int64_t) j * np8 + 8 * g), d));
        OXE_PV8(OXE_M)
#undef OXE_M
    }
    float mx[8];
#define OXE_M(j) mx[j] = (float) vmaxvq_f16(m##j);
    OXE_PV8(OXE_M)
#undef OXE_M
    const float full = BITS == 16 ? 65280.0f : 255.0f;
    __fp16 r16[8];
    for (int j = 0; j < 8; j++) {
        r16[j] = (__fp16) (mx[j] > 0.0f ? std::min(full / mx[j], 60000.0f) : 0.0f);
        rr[j] = (float) r16[j];
    }
#define OXE_M(j) const float16x8_t r##j = vdupq_n_f16(r16[j]);
    OXE_PV8(OXE_M)
#undef OXE_M
    for (int g = 0; g < ng; g++) {
        const float16x8_t d = vld1q_f16(dn + 8 * g);
        if (BITS == 16) {
#define OXE_M(j)                                                                                                         \
    {                                                                                                                    \
        const uint16x8_t u = vcvtnq_u16_f16(vmulq_f16(vmulq_f16(vld1q_f16(pn + (int64_t) j * np8 + 8 * g), d), r##j));   \
        vst1q_u8(w + g * gs + 16 * j, vcombine_u8(vshrn_n_u16(u, 8), vmovn_u16(u)));                                     \
    }
            OXE_PV8(OXE_M)
#undef OXE_M
        } else {
#define OXE_M(j) \
    vst1_u8(w + g * gs + 8 * j, vqmovn_u16(vcvtnq_u16_f16(vmulq_f16(vmulq_f16(vld1q_f16(pn + (int64_t) j * np8 + 8 * g), d), r##j))));
            OXE_PV8(OXE_M)
#undef OXE_M
        }
    }
}

}  // namespace

extern "C" {

// Scratch bytes of oxe_pv_q8_int for np positions and ncol columns.
int64_t oxe_pv_q8_int_scratch(int hd, int np, int ncol) {
    const int64_t np8 = (np + 7) & ~7;
    const int64_t nc8 = (ncol + 7) & ~7;
    (void) hd;
    return nc8 * 8 + np8 * 2 * (nc8 + 1) + np8 * 32 + np8 * nc8 * 2 + 64;
}

// Integer V . p (LOSSY: not bit-identical to oxe_pv_q8; a reorder / requantisation class that
// needs the KLD gate). Per chunk and per (Q8_0 block, column) the weights w[p] = d[p] * prob[p]
// (>= 0) are quantised to unsigned 8 or 16 bits, sum_p w_q[p] * q[p][dim] is exact in int32
// (usmmla: 8 positions x 2 dims x 2 weight rows per instruction; 16 bits = a high and a low byte
// row), and one float multiply per output applies the scale.
// The weights are formed in f16 as (prob * 255 / pmax[col]) * (d / dmax[blk]) * r with r scaling the
// (block, column) maximum of the chunk to full scale:
//   bits 16: full scale 65280. The three f16 roundings on the way (2^-12 relative each) dominate:
//            about 3.5e-4 of the output norm, flat in the shape of the distribution (measured;
//            the int8 V rows themselves are about 6e-3 from the f32 V, f16 accumulation in the
//            f16-KV path 3e-4 to 3e-3).
//   bits 8:  full scale 255: 0.15x to 2x the int8 V error (worst when the mass is spread over
//            small weights next to a peak in the same chunk).
// Requires prob >= 0, d >= 0 (attention probabilities, Q8_0 scales), p1 - p0 <= 256 * 128
// (int32 sums), hd % 32 == 0. scr: oxe_pv_q8_int_scratch(hd, p1 - p0, ncol) bytes, 4-aligned.
void oxe_pv_q8_int(int hd, const uint8_t * v, int64_t vs, int p0, int p1, const uint16_t * prob, int64_t ps, int ncol,
                   float * out, void * scr, int bits) {
    const int np = p1 - p0;
    const int np8 = (np + 7) & ~7;
    const int ng = np8 / 8;
    const int nblk = hd / 32;
    const int nc8 = (ncol + 7) & ~7;
    float * pmax = (float *) scr;                           // [nc8]
    float * sc = pmax + nc8;                                // [nc8]: one block's output scales
    __fp16 * pn = (__fp16 *) (sc + nc8);                    // [nc8][np8]: prob * 255 / pmax
    __fp16 * dn = pn + (int64_t) nc8 * np8;                 // [np8]: one block's d / dmax
    int8_t * T = (int8_t *) (dn + np8);                     // [ng][16 x 16]: one block, transposed
    uint8_t * W = (uint8_t *) T + (int64_t) np8 * 32;       // [ng][nc8][8 or 16]
    for (int c = 0; c < nc8; c++) {
        __fp16 * x = pn + (int64_t) c * np8;
        if (c < ncol) {
            memcpy(x, prob + (int64_t) c * ps + p0, (size_t) np * 2);
            memset(x + np, 0, (size_t) (np8 - np) * 2);
            pmax[c] = oxe_f16_max(x, np8);
            oxe_f16_scale(x, np8, pmax[c] > 0.0f ? 255.0f / pmax[c] : 0.0f);
        } else {
            memset(x, 0, (size_t) np8 * 2);
            pmax[c] = 0.0f;
        }
    }
    const uint8_t * v0 = v + (int64_t) p0 * vs;
    const int64_t pfd = (int64_t) np * vs;  // the next chunk (a thread's chunks are consecutive)
    const int pfl = (int) std::min<int64_t>(nblk, (vs + 63) / 64);
    const int wst = bits == 16 ? 16 : 8;  // weight bytes per (group, column)
    const int64_t gs = (int64_t) nc8 * wst;
    for (int blk = 0; blk < nblk; blk++) {
        // transpose the block's quants and collect its scales (padding positions repeat the last
        // row; their probabilities are zero)
        for (int g = 0; g < ng; g++) {
            const uint8_t * r[8];
            const uint8_t * b = v0 + (int64_t) 8 * g * vs + blk * 34;
            if (8 * g + 8 <= np) {
                for (int i = 0; i < 8; i++) {
                    r[i] = b + i * vs;
                }
            } else {
                for (int i = 0; i < 8; i++) {
                    r[i] = b + (int64_t) std::min(i, np - 1 - 8 * g) * vs;
                }
            }
            for (int i = 0; i < 8; i++) {
                memcpy(dn + 8 * g + i, r[i], 2);
            }
            if (blk < pfl) {  // line blk of each row of the next chunk
                for (int i = 0; i < 8; i++) {
                    OXE_PV_PREFETCH(r[i] + pfd + 30 * blk);
                }
            }
            oxe_q8_tr8x16(r, 2, T + g * 256);
            oxe_q8_tr8x16(r, 18, T + g * 256 + 128);
        }
        const float dmax = oxe_f16_max(dn, np8);
        oxe_f16_scale(dn, np8, dmax > 0.0f ? 1.0f / dmax : 0.0f);
        const __fp16 * dx = dn;
        for (int c0 = 0; c0 < nc8; c0 += 8) {
            float rr[8];
            if (bits == 16) {
                oxe_pv_q8_weights<16>(ng, np8, pn + (int64_t) c0 * np8, dx, W + c0 * 16, gs, rr);
            } else {
                oxe_pv_q8_weights<8>(ng, np8, pn + (int64_t) c0 * np8, dx, W + c0 * 8, gs, rr);
            }
            for (int j = 0; j < 8; j++) {
                sc[c0 + j] = rr[j] > 0.0f ? (float) ((double) pmax[c0 + j] * (double) dmax / (255.0 * (double) rr[j])) : 0.0f;
            }
        }
        const int cstep = bits == 16 ? 4 : 8;
        for (int c0 = 0; c0 < ncol; c0 += cstep) {
            for (int sub = 0; sub < 4; sub++) {
                int32x4_t a[16];
                oxe_pv_q8_mm_tile(ng, T + sub * 64, W + (int64_t) c0 * wst, gs, a);
                float * o = out + blk * 32 + sub * 8;
                if (bits == 16) {  // a[4 * k + c]: hi*d(2k), hi*d(2k+1), lo*d(2k), lo*d(2k+1)
                    for (int c = 0; c < 4 && c0 + c < ncol; c++) {
                        for (int h = 0; h < 2; h++) {
                            const int32x4_t x = a[8 * h + c], y = a[8 * h + 4 + c];
                            const float32x4_t hi = vcvtq_f32_s32(vcombine_s32(vget_low_s32(x), vget_low_s32(y)));
                            const float32x4_t lo = vcvtq_f32_s32(vcombine_s32(vget_high_s32(x), vget_high_s32(y)));
                            vst1q_f32(o + (int64_t) (c0 + c) * hd + 4 * h, vmulq_n_f32(vfmaq_n_f32(lo, hi, 256.0f), sc[c0 + c]));
                        }
                    }
                } else {  // a[4 * k + j]: columns c0 + 2j (low half) and c0 + 2j + 1 (high half) x dims 2k, 2k+1
                    for (int c = 0; c < 8 && c0 + c < ncol; c++) {
                        for (int h = 0; h < 2; h++) {
                            const int32x4_t x = a[8 * h + c / 2], y = a[8 * h + 4 + c / 2];
                            const int32x4_t z = (c & 1) ? vcombine_s32(vget_high_s32(x), vget_high_s32(y))
                                                        : vcombine_s32(vget_low_s32(x), vget_low_s32(y));
                            vst1q_f32(o + (int64_t) (c0 + c) * hd + 4 * h, vmulq_n_f32(vcvtq_f32_s32(z), sc[c0 + c]));
                        }
                    }
                }
            }
        }
    }
}

}  // extern "C"
