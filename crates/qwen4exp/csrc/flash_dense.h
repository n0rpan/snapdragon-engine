// flash_dense: multi-column dense matmul for the prefill (see flash_dense.cpp, docs/lab-notes/PROGRESS-PF.md).
#pragma once

#ifdef __cplusplus
#include <map>
#include <utility>

struct ggml_context;
struct ggml_tensor;

// one per graph: activation pack nodes by (x, vec_dot type), shared between the matmuls of one x
typedef std::map<std::pair<ggml_tensor *, int>, ggml_tensor *> fg_pf_cache;

// drop-in for ggml_mul_mat(ctx, w, x): the multi-column op when enabled (fg_pf_set_dense) and eligible,
// bitwise equal to production's mul_mat; ggml_mul_mat otherwise
ggml_tensor * fg_pf_mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, fg_pf_cache * cache);

extern "C" {
#endif

// process-wide switch, read when a graph is built (0 = off: every matmul is ggml_mul_mat)
void fg_pf_set_dense(int mode);
int fg_pf_get_dense(void);
// matmul nodes built since the last reset: [0] multi-column, [1] ggml fallback while enabled
void fg_pf_stats(long * out2, int reset);
// model-free bitwise check vs production ggml_mul_mat at nth threads: 0 = pass
int fg_selftest_dense(int nth);
// model-free timing: out3 = ms ggml_mul_mat, ms multi-column, bitwise equal (1/0)
int fg_pf_bench_dense(int type, long K, long R, long T, int nth, int reps, double * out3);

#ifdef __cplusplus
}
#endif
