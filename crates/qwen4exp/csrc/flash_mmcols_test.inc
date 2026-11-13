// flash_mmcols_test.inc (v0.1.3 F87): model-free evidence for the chat prefill schedule. The server splits a Flash
// prefill at every message start, so short segments reach the dense matmuls with 2-16 columns, while the decode step
// (and a /completion teacher run as one chunk) computes each token as part of another width. Question: is column c of
// a w-column ggml_mul_mat bitwise equal to the same token computed as a 1-column ggml_mul_mat?
// #included by flash_graph.cpp after flash_split_test.inc (uses its helpers). Test code only; nothing printed.
//
// Out arrays, row-major [type][w - 1] for w = 1..max_w, types {Q6_K, Q8_0} (the Flash dense weight types):
//   bitwise_out: columns of the w-column call that equal their 1-column result bit for bit (0..w)
//   maxd_out:    max |diff| over the w columns
// Returns 0 (evidence only; the caller decides what to assert).
int fg_selftest_mm_cols(int nth, int max_w, int * bitwise_out, double * maxd_out) {
    SplitRng rng;
    const int64_t K = 2560, R = 256;
    std::vector<float> Wf((size_t) K * R);
    for (auto & v : Wf) v = rng.next() * 0.05f;
    std::vector<float> X((size_t) max_w * K);
    for (int t = 0; t < max_w; ++t) {
        const float mag = t % 7 == 3 ? 300.0f : 1.0f;
        for (int64_t k = 0; k < K; ++k) X[(size_t) t * K + k] = rng.next() * mag;
    }
    const ggml_type types[] = {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
    for (int ti = 0; ti < 2; ++ti) {
        // the 1-column results of every token (computed once: a 1-column call does not depend on its neighbours)
        const std::vector<float> one = split_mm_run(types[ti], K, R, Wf, X, std::vector<int>(max_w, 1), 0, nth);
        for (int w = 1; w <= max_w; ++w) {
            const std::vector<float> many = split_mm_run(types[ti], K, R, Wf, X, std::vector<int>{w}, 0, nth);
            int same = 0;
            double maxd = 0.0;
            for (int t = 0; t < w; ++t) {
                const float * a = one.data() + (size_t) t * R;
                const float * b = many.data() + (size_t) t * R;
                if (std::memcmp(a, b, sizeof(float) * (size_t) R) == 0) ++same;
                for (int64_t i = 0; i < R; ++i) maxd = std::max(maxd, (double) std::fabs(a[i] - b[i]));
            }
            bitwise_out[ti * max_w + (w - 1)] = same;
            maxd_out[ti * max_w + (w - 1)] = maxd;
        }
    }
    return 0;
}
