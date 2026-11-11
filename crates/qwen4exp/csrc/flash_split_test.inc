// flash_split_test.inc: model-free test of chunked prefill. The chat server splits a prefill at chat-message starts;
// this asks whether a prompt prefilled as several chunks gives different per-token numbers than one chunk, and in
// which component. #included by flash_graph.cpp at file scope (declared in flash_graph.h). Test code only.
//
// Method: a prompt of T tokens (fixed-seed random inputs) is computed once as one chunk of T and once per split
// pattern (chunks in order; recurrent state and conv window carried from a chunk to the next, as the engine does).
// Each token (column) is compared bitwise with the one-chunk result; max|diff| and the worst token are printed.
//
// Expected class per component, decided from the code before running:
//   mm_ggml      ggml_mul_mat, Q6_K / Q8_0 weights: REPORTED (width-dependent). ggml_compute_forward_mul_mat feeds
//                column pairs to the 2-column vec_dot (num_rows_per_vec_dot = 2) only when the chunk's column count
//                allows it; a leftover odd column uses the 1-column kernel, which folds differently (see the
//                flash_dense.cpp header: "n_tokens odd -> the 1x1 kernels"). A column's bits depend on its chunk.
//   mm_pfon      fg_pf_mm with dense gemm on, T = 37: REPORTED (fallback). PF_MIN_TOKENS = 256, so the op returns
//                ggml_mul_mat for this width: same class as mm_ggml.
//   mm_pfoff     fg_pf_mm with dense gemm off: ggml_mul_mat by construction (flash_dense.cpp fg_pf_mm). Control.
//   mm_pfon264   fg_pf_mm with dense gemm on, T = 264 (eligible): REPORTED. The kernel choice follows the parity of the
//                whole call (even -> 2x2 mul+add fold, odd -> 1x1 fma for Q6_K, ggml fallback for Q8_0), and a chunk
//                below 256 tokens falls back to ggml, so a split can mix the two kernels.
//   gdn          ggml_gated_delta_net (K = 1): EXPECTED bitwise. The CPU kernel is a sequential per-token recurrence
//                that carries the state; each token's arithmetic depends on (S_v, token) only, not on chunk width.
//   ssm          ggml_ssm_conv with the carried conv window: EXPECTED bitwise. Each output is a fixed-order sum over
//                the d_conv taps of its own window (no column pairing).
//   rms          rms_norm * weight per row: EXPECTED bitwise (per-row reduction).
// The return value is 0 when every EXPECTED line is bitwise. REPORTED lines never fail the test.

#include <memory>

namespace {

constexpr size_t SPLIT_MEM = 256u << 20;

struct SplitRng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
    }
};

struct SplitCmp {
    bool bitwise = true;
    double maxd = 0.0;
    int worst = -1;
};

// a, b: token-major [T][W] (token t at t * W); tail-only callers pass T = 1
SplitCmp split_cmp(const float * a, const float * b, size_t W, int T) {
    SplitCmp r;
    for (int t = 0; t < T; ++t) {
        for (size_t i = 0; i < W; ++i) {
            const size_t k = (size_t) t * W + i;
            if (std::memcmp(&a[k], &b[k], sizeof(float)) != 0) {
                if (r.worst < 0) r.worst = t;
                r.bitwise = false;
            }
            const double d = std::fabs((double) a[k] - (double) b[k]);
            if (d > r.maxd) {
                r.maxd = d;
                r.worst = t;
            }
        }
    }
    return r;
}

// run-length form of a chunk list: [16,21], [2x18,1], [1x37]
std::string split_pat(const std::vector<int> & p) {
    std::string s = "[";
    for (size_t i = 0; i < p.size();) {
        size_t j = i;
        while (j < p.size() && p[j] == p[i]) ++j;
        if (i) s += ",";
        s += std::to_string(p[i]);
        if (j - i > 1) s += "x" + std::to_string(j - i);
        i = j;
    }
    return s + "]";
}

// prints one line per (component, pattern); returns 1 when an EXPECTED line is not bitwise
int split_report(const char * comp, const std::vector<int> & p, bool expected, const SplitCmp & c, bool tail) {
    char wt[32];
    if (tail) std::snprintf(wt, sizeof wt, "state");
    else if (c.worst >= 0) std::snprintf(wt, sizeof wt, "%d", c.worst);
    else std::snprintf(wt, sizeof wt, "-");
    std::fprintf(stderr, "split %s %s: bitwise %d max|diff| %.3e (worst token %s) [%s]\n", comp, split_pat(p).c_str(),
                 (int) c.bitwise, c.maxd, wt, expected ? "expected" : "reported");
    return expected && !c.bitwise;
}

// one component: full = one chunk of T; every pattern against it. tail > 0: the last `tail` floats of each result
// are a state compared as one extra line per pattern.
template <class Run>
int split_sweep(const char * comp, bool expected, int T, size_t W, size_t tail, const char * tail_name,
                const std::vector<std::vector<int>> & pats, Run run) {
    const std::vector<float> full = run(std::vector<int>{T});
    int bad = 0;
    for (const auto & p : pats) {
        const std::vector<float> part = run(p);
        bad += split_report(comp, p, expected, split_cmp(full.data(), part.data(), W, T), false);
        if (tail) {
            bad += split_report(tail_name, p, expected, split_cmp(full.data() + (size_t) T * W, part.data() + (size_t) T * W, tail, 1), true);
        }
    }
    return bad;
}

// ---- dense matmul: ggml_mul_mat (mode 0), fg_pf_mm dense off (mode 2), fg_pf_mm dense on (mode 1)
// X: token-major [T][K]; W: row-major [R][K] float, quantised to ty. Output token-major [T][R].
std::vector<float> split_mm_run(ggml_type ty, int64_t K, int64_t R, const std::vector<float> & W, const std::vector<float> & X,
                                const std::vector<int> & chunks, int mode, int nth) {
    std::unique_ptr<uint8_t[]> buf(new uint8_t[SPLIT_MEM]);
    ggml_init_params ip = {SPLIT_MEM, buf.get(), false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, ty, K, R);
    ggml_quantize_chunk(ty, W.data(), w->data, 0, R, K, nullptr);
    int T = 0;
    for (int n : chunks) T += n;
    std::vector<float> out((size_t) T * R);
    const int prev = fg_pf_get_dense();
    fg_pf_set_dense(mode == 1 ? 1 : 0);
    int64_t c0 = 0;
    for (int n : chunks) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
        std::memcpy(x->data, X.data() + (size_t) c0 * K, sizeof(float) * (size_t) K * n);
        ggml_tensor * y = mode == 0 ? ggml_mul_mat(ctx, w, x) : fg_pf_mm(ctx, w, x, nullptr);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        std::memcpy(out.data() + (size_t) c0 * R, y->data, sizeof(float) * (size_t) R * n);
        c0 += n;
    }
    fg_pf_set_dense(prev);
    ggml_free(ctx);
    return out;
}

// ---- gated delta net (K = 1): inputs token-major; state [S, S, H] carried between chunks
struct SplitGdnIn {
    int64_t S, H, T;
    std::vector<float> q, k, v;  // [T][H][S]
    std::vector<float> g, beta;  // [T][H]
    std::vector<float> st0;      // [H][S][S]
};

// output: attention rows [T][H*S], then the final state [H*S*S]
std::vector<float> split_gdn_run(const SplitGdnIn & in, const std::vector<int> & chunks, int nth) {
    const int64_t S = in.S, H = in.H;
    const size_t aw = (size_t) S * H, sw = (size_t) S * S * H;
    std::unique_ptr<uint8_t[]> buf(new uint8_t[SPLIT_MEM]);
    ggml_init_params ip = {SPLIT_MEM, buf.get(), false};
    ggml_context * ctx = ggml_init(ip);
    std::vector<float> out((size_t) in.T * aw + sw);
    std::vector<float> state(in.st0);
    int64_t c0 = 0;
    for (int n : chunks) {
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, 1);
        ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, 1);
        std::memcpy(q->data, in.q.data() + (size_t) c0 * aw, sizeof(float) * aw * n);
        std::memcpy(k->data, in.k.data() + (size_t) c0 * aw, sizeof(float) * aw * n);
        std::memcpy(v->data, in.v.data() + (size_t) c0 * aw, sizeof(float) * aw * n);
        std::memcpy(g->data, in.g.data() + (size_t) c0 * H, sizeof(float) * H * n);
        std::memcpy(b->data, in.beta.data() + (size_t) c0 * H, sizeof(float) * H * n);
        std::memcpy(st->data, state.data(), sizeof(float) * sw);
        ggml_tensor * r = ggml_gated_delta_net(ctx, q, k, v, g, b, st, 1);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        const float * rd = (const float *) r->data;
        std::memcpy(out.data() + (size_t) c0 * aw, rd, sizeof(float) * aw * n);
        std::memcpy(state.data(), rd + aw * n, sizeof(float) * sw);
        c0 += n;
    }
    std::memcpy(out.data() + (size_t) in.T * aw, state.data(), sizeof(float) * sw);
    ggml_free(ctx);
    return out;
}

// ---- ssm_conv (d_conv = NC) over C channels; the conv window (NC-1 columns per channel) carried between chunks
struct SplitSsmIn {
    int64_t C, NC, T;
    std::vector<float> x;     // [T][C]
    std::vector<float> cw;    // [C][NC]
    std::vector<float> hist0; // [C][NC-1]
};

// output: [T][C], then the final window [C][NC-1]
std::vector<float> split_ssm_run(const SplitSsmIn & in, const std::vector<int> & chunks, int nth) {
    const int64_t C = in.C, NC = in.NC;
    const size_t hw = (size_t) (NC - 1);
    std::unique_ptr<uint8_t[]> buf(new uint8_t[SPLIT_MEM]);
    ggml_init_params ip = {SPLIT_MEM, buf.get(), false};
    ggml_context * ctx = ggml_init(ip);
    std::vector<float> out((size_t) in.T * C + C * hw);
    std::vector<float> hist(in.hist0);
    int64_t c0 = 0;
    for (int n : chunks) {
        const int64_t ncs = NC - 1 + n;  // sliding window input per channel: [state (NC-1) | chunk tokens]
        ggml_tensor * sx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncs, C);
        ggml_tensor * cwt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NC, C);
        float * sd = (float *) sx->data;
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t j = 0; j < ncs; ++j) {
                sd[c * ncs + j] = j < NC - 1 ? hist[c * hw + j] : in.x[(size_t) (c0 + j - (NC - 1)) * C + c];
            }
        }
        std::memcpy(cwt->data, in.cw.data(), sizeof(float) * C * NC);
        ggml_tensor * y = ggml_ssm_conv(ctx, sx, cwt);  // [C, n]
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        std::memcpy(out.data() + (size_t) c0 * C, y->data, sizeof(float) * (size_t) C * n);
        for (int64_t c = 0; c < C; ++c) {
            for (size_t j = 0; j < hw; ++j) hist[c * hw + j] = sd[c * ncs + n + j];
        }
        c0 += n;
    }
    std::memcpy(out.data() + (size_t) in.T * C, hist.data(), sizeof(float) * C * hw);
    ggml_free(ctx);
    return out;
}

// ---- rms_norm * weight per row
std::vector<float> split_rms_run(int64_t E, const std::vector<float> & X, const std::vector<float> & wv, int64_t T,
                                 const std::vector<int> & chunks, int nth) {
    std::unique_ptr<uint8_t[]> buf(new uint8_t[SPLIT_MEM]);
    ggml_init_params ip = {SPLIT_MEM, buf.get(), false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * nw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, 1);
    std::memcpy(nw->data, wv.data(), sizeof(float) * E);
    std::vector<float> out((size_t) T * E);
    int64_t c0 = 0;
    for (int n : chunks) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, n);
        std::memcpy(x->data, X.data() + (size_t) c0 * E, sizeof(float) * (size_t) E * n);
        ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), nw);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        std::memcpy(out.data() + (size_t) c0 * E, y->data, sizeof(float) * (size_t) E * n);
        c0 += n;
    }
    ggml_free(ctx);
    return out;
}

}  // namespace

// Chunked-prefill split test (model-free). Prints one "split <component> <pattern>: ..." line per comparison.
// Returns 0 when every EXPECTED (width-independent) component is bitwise equal to the one-chunk result.
int fg_selftest_split(int nth) {
    int bad = 0;
    SplitRng rng;
    const int T = 37;
    // the server's split [16,21] and [13,24], and the decode-like widths 1 / 2 / 3 (2*18 + 1 = 37, 3*12 + 1 = 37)
    std::vector<int> w2(18, 2), w3(12, 3);
    w2.push_back(1);
    w3.push_back(1);
    const std::vector<std::vector<int>> pats = {{16, 21}, {13, 24}, std::vector<int>(37, 1), w2, w3};

    // (1) dense matmul: Flash-shaped weights K 2560 x R 1024, f32 activations with a few large / zero tokens
    const int64_t K = 2560, R = 1024;
    std::vector<float> Wf((size_t) K * R);
    for (auto & v : Wf) v = rng.next() * 0.05f;
    std::vector<float> X((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        const float mag = t % 7 == 3 ? 300.0f : 1.0f;
        for (int64_t k = 0; k < K; ++k) X[(size_t) t * K + k] = rng.next() * mag;
    }
    const struct { ggml_type ty; const char * name; } types[] = {{GGML_TYPE_Q6_K, "q6k"}, {GGML_TYPE_Q8_0, "q8_0"}};
    for (const auto & tp : types) {
        std::string c = std::string("mm_ggml_") + tp.name;
        bad += split_sweep(c.c_str(), false, T, R, 0, nullptr, pats, [&](const std::vector<int> & p) {
            return split_mm_run(tp.ty, K, R, Wf, X, p, 0, nth);
        });
        c = std::string("mm_pfon_") + tp.name;
        bad += split_sweep(c.c_str(), false, T, R, 0, nullptr, pats, [&](const std::vector<int> & p) {
            return split_mm_run(tp.ty, K, R, Wf, X, p, 1, nth);
        });
    }
    bad += split_sweep("mm_pfoff_q6k", false, T, R, 0, nullptr, pats, [&](const std::vector<int> & p) {
        return split_mm_run(GGML_TYPE_Q6_K, K, R, Wf, X, p, 2, nth);
    });
    // fg_pf_mm dense on at an eligible width (T >= 256, enough chunks): even and odd first chunks
    const int T2 = 264;
    std::vector<float> X2((size_t) T2 * K);
    for (int t = 0; t < T2; ++t) {
        const float mag = t % 7 == 3 ? 300.0f : 1.0f;
        for (int64_t k = 0; k < K; ++k) X2[(size_t) t * K + k] = rng.next() * mag;
    }
    const std::vector<std::vector<int>> pats264 = {{256, 8}, {136, 128}};
    for (const auto & tp : types) {
        std::string c = std::string("mm_pfon264_") + tp.name;
        bad += split_sweep(c.c_str(), false, T2, R, 0, nullptr, pats264, [&](const std::vector<int> & p) {
            return split_mm_run(tp.ty, K, R, Wf, X2, p, 1, nth);
        });
    }

    // (2) gated delta net: S 32, H 2 (the op is head-generic; 128 is the production size, not needed for the question)
    SplitGdnIn gin;
    gin.S = 32;
    gin.H = 2;
    gin.T = T;
    const size_t aw = (size_t) gin.S * gin.H;
    gin.q.resize((size_t) T * aw);
    gin.k.resize((size_t) T * aw);
    gin.v.resize((size_t) T * aw);
    for (auto & x : gin.q) x = rng.next();
    for (auto & x : gin.k) x = rng.next() / std::sqrt((float) gin.S);
    for (auto & x : gin.v) x = rng.next() * 0.5f;
    gin.g.resize((size_t) T * gin.H);
    gin.beta.resize((size_t) T * gin.H);
    for (auto & x : gin.g) x = -0.3f * (rng.next() + 1.0f);
    for (auto & x : gin.beta) x = 0.5f * (rng.next() + 1.0f);
    gin.st0.resize((size_t) gin.S * gin.S * gin.H);
    for (auto & x : gin.st0) x = 0.1f * rng.next();
    bad += split_sweep("gdn", true, T, aw, (size_t) gin.S * gin.S * gin.H, "gdn_state", pats, [&](const std::vector<int> & p) {
        return split_gdn_run(gin, p, nth);
    });

    // (3) ssm_conv: d_conv 4, 64 channels, conv window carried
    SplitSsmIn sin;
    sin.C = 64;
    sin.NC = 4;
    sin.T = T;
    sin.x.resize((size_t) T * sin.C);
    for (auto & x : sin.x) x = rng.next();
    sin.cw.resize((size_t) sin.C * sin.NC);
    for (auto & x : sin.cw) x = 0.5f * rng.next();
    sin.hist0.resize((size_t) sin.C * (sin.NC - 1));
    for (auto & x : sin.hist0) x = rng.next();
    bad += split_sweep("ssm_conv", true, T, sin.C, (size_t) sin.C * (sin.NC - 1), "ssm_state", pats, [&](const std::vector<int> & p) {
        return split_ssm_run(sin, p, nth);
    });

    // (4) rms_norm * weight, E 2560, sanity row
    const int64_t E = 2560;
    std::vector<float> XN((size_t) T * E), wn((size_t) E);
    for (auto & x : XN) x = rng.next();
    for (auto & x : wn) x = 1.0f + 0.1f * rng.next();
    bad += split_sweep("rms_norm_mul", true, T, E, 0, nullptr, pats, [&](const std::vector<int> & p) {
        return split_rms_run(E, XN, wn, T, p, nth);
    });

    std::fprintf(stderr, "fg_selftest_split nth %d: expected (width-independent) components bitwise: %s; reported lines are the width-dependent evidence\n",
                 nth, bad ? "NO" : "yes");
    return bad ? 1 : 0;
}
