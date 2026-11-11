// flash_dense: multi-column dense matmul for the prefill (WORKSTREAM PREFILL, see PROGRESS-PF.md).
//
// fg_pf_mm(ctx, w, x, cache) is a drop-in for ggml_mul_mat(ctx, w, x). When the pair is eligible it builds
//   pack = custom(x)        "pf_pack": x quantised per token by PRODUCTION from_float (Q8_K / Q8_0), codes
//                           re-laid out as 8-token interleaved groups; shared by every matmul of the same x
//   y    = custom(w, pack)  each thread decodes its weight rows once per row block (int8 codes, L2-sized)
//                           and multiplies them with all tokens in 8x4 SMMLA tiles (pf_kernels.c)
// and otherwise returns ggml_mul_mat. The result is BITWISE what production ggml's mul_mat gives for the
// same shape (the kernels' fold contract is in pf_kernels.c; model-free check: fg_selftest_dense).
//
// Eligibility = the shapes for which ggml's kernel choice is known and thread-count independent
// (ggml_compute_forward_mul_mat: 16x16 chunks, nrc = 2 only when every chunk is even):
//   w: Q6_K or Q8_0, 2-D, rows % 16 == 0;  x: F32, contiguous, 2-D, n_tokens >= PF_MIN_TOKENS (256, so
//   that dr1 == 16);  >= 256 chunks (no re-chunk by thread up to 64 threads);  not NUMA.  n_tokens even -> production runs the 2x2 SMMLA kernels (fold: mul, add);
//   n_tokens odd -> the 1x1 kernels: Q6_K folds with fma (reproduced), Q8_0 uses 4-lane float
//   accumulators (not a per-chunk fold) -> falls back to ggml.
#include "ggml-cpu.h"
#include "ggml.h"

#include "flash_dense.h"
#include "pf_kernels.h"
#include "pf_phase.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t PF_MIN_TOKENS = 256;
constexpr size_t PF_L2_BYTES = 512u << 10;  // decoded weight codes per row block
constexpr int PF_MAX_HG = 64;               // 4-row groups per row block

std::atomic<int> g_mode{0};
std::atomic<long> g_n_dense{0}, g_n_fallback{0};

inline size_t pad64(size_t n) { return (n + 63) & ~(size_t) 63; }
inline int64_t chunk_of(ggml_type vt) { return vt == GGML_TYPE_Q8_K ? 256 : 32; }

struct Scratch {
    std::vector<uint8_t> tmp;    // one quantised activation row
    std::vector<int8_t> row;     // one decoded weight row
    std::vector<int8_t> wcodes;  // [hg][step][4 rows][8]
    std::vector<int16_t> wsc;    // [hg][chunk][sub][4]
    std::vector<float> wd;       // [hg][chunk][4]
};
Scratch & scratch() {
    static thread_local Scratch s;
    return s;
}

// userdata of both ops: bits 0-7 vec_dot type, bit 8 fused fold
inline void * pack_ud(ggml_type vt, bool fused) { return (void *) (uintptr_t) ((unsigned) vt | (fused ? 256u : 0u)); }

// dst: I8 [bytes]; src[0] = x F32 [K, T]
void pf_pack_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FgPfPhaseTimer pf_timer_(FG_PF_PHASE_DENSE_PACK, ith == 0);
    const ggml_type vt = (ggml_type) ((uintptr_t) ud & 0xff);
    const ggml_tensor * x = dst->src[0];
    const int64_t K = x->ne[0], T = x->ne[1];
    const int64_t G = (T + 7) / 8;
    const int64_t chunk = chunk_of(vt), nchunk = K / chunk;
    int8_t * codes = (int8_t *) dst->data;
    float * scales = (float *) ((char *) dst->data + pad64((size_t) G * K * 8));
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vt)->from_float;
    Scratch & S = scratch();
    S.tmp.resize(ggml_row_size(vt, K));
    const int64_t g0 = G * ith / nth, g1 = G * (ith + 1) / nth;
    for (int64_t g = g0; g < g1; ++g) {
        int8_t * gc = codes + (size_t) g * K * 8;
        float * gs = scales + (size_t) g * nchunk * 8;
        if (g * 8 + 8 > T) {  // padded group: absent tokens are zero codes with zero scales
            std::memset(gc, 0, (size_t) K * 8);
            std::memset(gs, 0, (size_t) nchunk * 8 * sizeof(float));
        }
        for (int j = 0; j < 8 && g * 8 + j < T; ++j) {
            const float * xr = (const float *) ((const char *) x->data + (size_t) (g * 8 + j) * x->nb[1]);
            from_float(xr, S.tmp.data(), K);
            const uint8_t * p = S.tmp.data();
            if (vt == GGML_TYPE_Q8_K) {
                // block_q8_K: f32 d, int8 qs[256], int16 bsums[16] = 292 bytes
                for (int64_t b = 0; b < nchunk; ++b, p += 292) {
                    float d;
                    std::memcpy(&d, p, 4);
                    gs[b * 8 + j] = d;
                    pf_interleave(gc + (size_t) b * 256 * 8, 8, j, (const int8_t *) p + 4, 256);
                }
            } else {
                // block_q8_0: f16 d, int8 qs[32] = 34 bytes
                for (int64_t b = 0; b < nchunk; ++b, p += 34) {
                    __fp16 h;
                    std::memcpy(&h, p, 2);
                    gs[b * 8 + j] = (float) h;
                    pf_interleave(gc + (size_t) b * 32 * 8, 8, j, (const int8_t *) p + 2, 32);
                }
            }
        }
    }
}

// dst: F32 [R, T]; src[0] = w (Q6_K | Q8_0) [K, R]; src[1] = pack
void pf_mm_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FgPfPhaseTimer pf_timer_(FG_PF_PHASE_DENSE_MM, ith == 0);
    const bool fused = ((uintptr_t) ud & 256u) != 0;
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * pk = dst->src[1];
    const int64_t K = w->ne[0], R = w->ne[1], T = dst->ne[1];
    const bool q6k = w->type == GGML_TYPE_Q6_K;
    const int64_t chunk = q6k ? 256 : 32, nchunk = K / chunk;
    const int64_t G = (T + 7) / 8;
    const int8_t * xcodes = (const int8_t *) pk->data;
    const float * xscales = (const float *) ((const char *) pk->data + pad64((size_t) G * K * 8));
    const int64_t H = (R + 3) / 4;  // 4-row groups
    const int64_t h0 = H * ith / nth, h1 = H * (ith + 1) / nth;
    if (h0 >= h1) return;
    int64_t bh = (int64_t) (PF_L2_BYTES / ((size_t) K * 4));
    bh = bh < 1 ? 1 : (bh > PF_MAX_HG ? PF_MAX_HG : bh);
    Scratch & S = scratch();
    S.row.resize(K);
    S.wcodes.resize((size_t) bh * K * 4);
    S.wd.resize((size_t) bh * nchunk * 4);
    if (q6k) S.wsc.resize((size_t) bh * (K / 16) * 4);
    const size_t ystride = dst->nb[1] / sizeof(float);
    float * Y = (float *) dst->data;
    for (int64_t hb = h0; hb < h1; hb += bh) {
        const int64_t he = hb + bh < h1 ? hb + bh : h1;
        // decode / re-lay the row block once
        for (int64_t h = hb; h < he; ++h) {
            const int64_t hi = h - hb;
            int8_t * wc = S.wcodes.data() + (size_t) hi * K * 4;
            float * wd = S.wd.data() + (size_t) hi * nchunk * 4;
            int16_t * ws = q6k ? S.wsc.data() + (size_t) hi * (K / 16) * 4 : nullptr;
            if (h * 4 + 4 > R) {
                std::memset(wc, 0, (size_t) K * 4);
                std::memset(wd, 0, (size_t) nchunk * 4 * sizeof(float));
                if (ws) std::memset(ws, 0, (size_t) (K / 16) * 4 * sizeof(int16_t));
            }
            for (int j = 0; j < 4 && h * 4 + j < R; ++j) {
                const uint8_t * wr = (const uint8_t *) w->data + (size_t) (h * 4 + j) * w->nb[1];
                if (q6k) {
                    pf_q6k_decode_row(wr, (size_t) nchunk, S.row.data(), ws + j, 4, wd + j, 4);
                    pf_interleave(wc, 4, j, S.row.data(), (size_t) K);
                } else {
                    for (int64_t b = 0; b < nchunk; ++b) {
                        __fp16 hd;
                        std::memcpy(&hd, wr + b * 34, 2);
                        wd[b * 4 + j] = (float) hd;
                        pf_interleave(wc + (size_t) b * 32 * 4, 4, j, (const int8_t *) wr + b * 34 + 2, 32);
                    }
                }
            }
        }
        // all tokens against the block: token group outer (L1 operand), row groups inner
        for (int64_t g = 0; g < G; ++g) {
            const int8_t * xc = xcodes + (size_t) g * K * 8;
            const float * xs = xscales + (size_t) g * nchunk * 8;
            const bool tfull = g * 8 + 8 <= T;
            for (int64_t h = hb; h < he; ++h) {
                const int64_t hi = h - hb;
                const int8_t * wc = S.wcodes.data() + (size_t) hi * K * 4;
                const float * wd = S.wd.data() + (size_t) hi * nchunk * 4;
                const bool full = tfull && h * 4 + 4 <= R;
                float tile[32];
                float * y = full ? Y + (size_t) (g * 8) * ystride + h * 4 : tile;
                const ptrdiff_t ys = full ? (ptrdiff_t) ystride : 4;
                if (q6k) pf_tile8x4_two(xc, wc, S.wsc.data() + (size_t) hi * (K / 16) * 4, (size_t) nchunk, 16, 2, xs, wd, y, ys, fused);
                else     pf_tile8x4_one(xc, wc, (size_t) nchunk, 4, xs, wd, y, ys, fused);
                if (!full) {
                    for (int t = 0; t < 8 && g * 8 + t < T; ++t)
                        for (int r = 0; r < 4 && h * 4 + r < R; ++r) Y[(size_t) (g * 8 + t) * ystride + h * 4 + r] = tile[t * 4 + r];
                }
            }
        }
    }
}

// 0 = not eligible, 1 = fold with mul+add (production 2x2 kernels), 2 = fold with fma (production 1x1 Q6_K)
int eligible(const ggml_tensor * w, const ggml_tensor * x) {
    if (w->type != GGML_TYPE_Q6_K && w->type != GGML_TYPE_Q8_0) return 0;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w)) return 0;
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) return 0;
    const int64_t K = w->ne[0], R = w->ne[1], T = x->ne[1];
    if (x->ne[0] != K || K % 256 != 0 || R % 16 != 0 || T < PF_MIN_TOKENS) return 0;
    // ggml re-chunks by thread (thread-count dependent kernel choice) when nchunk0 * nchunk1 < 4 * nth: stay clear of it up to 64 threads
    if ((R / 16) * ((T + 15) / 16) < 256) return 0;
    // Q8_0 needs one float fold per 32 weights (the contract), which caps the gain: measured x2.0 at K 10240 (hc down),
    // x0.97 at K 320 / 640 (hc up, shexp down: output-bound). Short rows stay on ggml.
    if (w->type == GGML_TYPE_Q8_0 && K < 1024) return 0;
    if (ggml_is_numa()) return 0;
    if (T % 2 == 0) return 1;
    return w->type == GGML_TYPE_Q6_K ? 2 : 0;
}

}  // namespace

ggml_tensor * fg_pf_mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, fg_pf_cache * cache) {
    const int el = g_mode.load(std::memory_order_relaxed) ? eligible(w, x) : 0;
    if (!el) {
        if (g_mode.load(std::memory_order_relaxed)) g_n_fallback.fetch_add(1, std::memory_order_relaxed);
        return ggml_mul_mat(ctx, w, x);
    }
    g_n_dense.fetch_add(1, std::memory_order_relaxed);
    const ggml_type vt = w->type == GGML_TYPE_Q6_K ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
    const int64_t K = x->ne[0], T = x->ne[1], G = (T + 7) / 8;
    ggml_tensor * pack = nullptr;
    const std::pair<ggml_tensor *, int> key{x, (int) vt};
    if (cache) {
        auto it = cache->find(key);
        if (it != cache->end()) pack = it->second;
    }
    if (!pack) {
        const int64_t bytes = (int64_t) (pad64((size_t) G * K * 8) + (size_t) G * (K / chunk_of(vt)) * 8 * sizeof(float));
        ggml_tensor * a[1] = {x};
        pack = ggml_custom_4d(ctx, GGML_TYPE_I8, bytes, 1, 1, 1, a, 1, pf_pack_op, GGML_N_TASKS_MAX, pack_ud(vt, false));
        ggml_set_name(pack, "pf_pack");
        if (cache) (*cache)[key] = pack;
    }
    ggml_tensor * a[2] = {w, pack};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], T, 1, 1, a, 2, pf_mm_op, GGML_N_TASKS_MAX, pack_ud(vt, el == 2));
}

extern "C" {

void fg_pf_set_dense(int mode) { g_mode.store(mode, std::memory_order_relaxed); }
int fg_pf_get_dense(void) { return g_mode.load(std::memory_order_relaxed); }
void fg_pf_stats(long * out2, int reset) {
    out2[0] = g_n_dense.load();
    out2[1] = g_n_fallback.load();
    if (reset) {
        g_n_dense.store(0);
        g_n_fallback.store(0);
    }
}

// Model-free check: (1) the SMMLA tiles against their scalar definitions on random codes, (2) fg_pf_mm
// against production ggml_mul_mat on random quantised weights, bitwise, at `nth` threads. 0 = pass.
int fg_selftest_dense(int nth) {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    auto rf = [&]() { return (float) ((rnd() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    int bad = 0;
    // (1) tiles vs scalar definitions
    for (int it = 0; it < 24; ++it) {
        const bool two = it % 2 == 0;
        const size_t nchunk = two ? 1 + it % 3 : 4 + it % 5;
        const size_t nsub = two ? (it % 4 == 0 ? 8 : 16) : 1, steps = two ? (nsub == 8 ? 4 : 2) : 4;
        const size_t K = nchunk * nsub * steps * 8;
        std::vector<int8_t> x(K * 8), w(K * 4);
        std::vector<int16_t> sc(nchunk * nsub * 4);
        std::vector<float> dx(nchunk * 8), dw(nchunk * 4);
        const bool extreme = it >= 16;
        for (auto & v : x) v = extreme ? (int8_t) (rnd() & 1 ? 127 : -127) : (int8_t) ((int) (rnd() % 255) - 127);
        for (auto & v : w) v = two ? (extreme ? (int8_t) -32 : (int8_t) ((int) (rnd() % 64) - 32)) : (int8_t) ((int) (rnd() % 255) - 127);
        for (auto & v : sc) v = extreme ? (int16_t) (rnd() & 1 ? 127 : -128) : (int16_t) ((int) (rnd() % 256) - 128);
        for (auto & v : dx) v = rf() * 0.01f;
        for (auto & v : dw) v = rf() * 0.001f;
        for (int fused = 0; fused < 2; ++fused) {
            float a[5 * 8] = {0}, b[5 * 8] = {0};
            if (two) {
                pf_tile8x4_two(x.data(), w.data(), sc.data(), nchunk, nsub, steps, dx.data(), dw.data(), a, 5, fused);
                pf_tile8x4_two_ref(x.data(), w.data(), sc.data(), nchunk, nsub, steps, dx.data(), dw.data(), b, 5, fused);
            } else {
                pf_tile8x4_one(x.data(), w.data(), nchunk, steps, dx.data(), dw.data(), a, 5, fused);
                pf_tile8x4_one_ref(x.data(), w.data(), nchunk, steps, dx.data(), dw.data(), b, 5, fused);
            }
            if (std::memcmp(a, b, sizeof a) != 0) {
                ++bad;
                std::fprintf(stderr, "fg_selftest_dense: tile %s it %d fused %d differs from its scalar definition\n", two ? "two" : "one", it, fused);
            }
        }
    }
    // (2) the op vs production mul_mat
    struct Case { ggml_type ty; int64_t K, R, T; int expect; };
    const Case cases[] = {
        {GGML_TYPE_Q6_K, 2560, 640, 256, 1}, {GGML_TYPE_Q6_K, 512, 320, 258, 1}, {GGML_TYPE_Q6_K, 768, 224, 301, 2},
        {GGML_TYPE_Q6_K, 2560, 64, 1030, 1}, {GGML_TYPE_Q8_0, 2560, 320, 256, 1}, {GGML_TYPE_Q8_0, 1024, 1024, 262, 1},
        {GGML_TYPE_Q8_0, 10240, 320, 264, 1}, {GGML_TYPE_Q8_0, 2560, 320, 257, 0}, {GGML_TYPE_Q6_K, 512, 64, 64, 0}, {GGML_TYPE_Q8_0, 512, 1024, 262, 0},
    };
    const int prev = g_mode.exchange(1);
    for (const Case & cs : cases) {
        const size_t mem = 512u << 20;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context * ctx = ggml_init(ip);
        std::vector<float> wf((size_t) cs.K * cs.R);
        for (auto & v : wf) v = rf() * 0.05f;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, cs.ty, cs.K, cs.R);
        ggml_quantize_chunk(cs.ty, wf.data(), w->data, 0, cs.R, cs.K, nullptr);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cs.K, cs.T);
        float * xd = (float *) x->data;
        for (int64_t t = 0; t < cs.T; ++t) {
            const float mag = t % 7 == 3 ? 300.0f : (t % 11 == 5 ? 1e-4f : 1.0f);
            for (int64_t k = 0; k < cs.K; ++k) xd[t * cs.K + k] = rf() * mag;
        }
        for (int64_t k = 0; k < cs.K; ++k) xd[2 * cs.K + k] = 0.0f;                    // an all-zero token
        for (int64_t k = 0; k < 256 && k < cs.K; ++k) xd[5 * cs.K + k] = 0.0f;         // a zero block
        fg_pf_cache cache;
        long st0[2];
        fg_pf_stats(st0, 0);
        ggml_tensor * ref = ggml_mul_mat(ctx, w, x);
        ggml_tensor * got = fg_pf_mm(ctx, w, x, &cache);
        ggml_tensor * got2 = fg_pf_mm(ctx, w, x, &cache);  // shares the pack node
        long st1[2];
        fg_pf_stats(st1, 0);
        const bool took = st1[0] - st0[0] == 2;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, got);
        ggml_build_forward_expand(gf, got2);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        const size_t nb = sizeof(float) * (size_t) cs.R * cs.T;
        const bool same = std::memcmp(ref->data, got->data, nb) == 0 && std::memcmp(ref->data, got2->data, nb) == 0;
        const bool path_ok = took == (cs.expect != 0);
        if (!same || !path_ok) {
            ++bad;
            size_t first = 0;
            while (first < (size_t) cs.R * cs.T && ((uint32_t *) ref->data)[first] == ((uint32_t *) got->data)[first]) ++first;
            std::fprintf(stderr, "fg_selftest_dense: %s K %lld R %lld T %lld nth %d: bitwise %d, dense path taken %d (expected %d), first diff at row %zu token %zu\n",
                         ggml_type_name(cs.ty), (long long) cs.K, (long long) cs.R, (long long) cs.T, nth, (int) same, (int) took, cs.expect != 0,
                         first % (size_t) cs.R, first / (size_t) cs.R);
        }
        ggml_free(ctx);
    }
    g_mode.store(prev);
    std::fprintf(stderr, "fg_selftest_dense nth %d: %s\n", nth, bad ? "FAIL" : "ok (tiles == scalar definitions; op bitwise == ggml_mul_mat, 10 shapes)");
    return bad;
}

// Model-free timing of one matmul shape: ms per call for ggml_mul_mat (out[0]) and fg_pf_mm (out[1], incl. the
// pack node), best of `reps`; out[2] = 1 if the two results are bitwise equal. DEV windows only (CPU-heavy).
int fg_pf_bench_dense(int type, long K, long R, long T, int nth, int reps, double * out3) {
    const size_t mem = ((size_t) 256u << 20) + (size_t) K * T * 16 + (size_t) R * T * 12 + (size_t) K * R * 5;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t s = 0x2545F4914F6CDD1Dull;
    auto rf = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    std::vector<float> wf((size_t) K * R);
    for (auto & v : wf) v = rf() * 0.05f;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, (ggml_type) type, K, R);
    ggml_quantize_chunk((ggml_type) type, wf.data(), w->data, 0, R, K, nullptr);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    for (int64_t i = 0; i < (int64_t) K * T; ++i) ((float *) x->data)[i] = rf();
    const int prev = g_mode.exchange(1);
    ggml_tensor * ref = ggml_mul_mat(ctx, w, x);
    ggml_tensor * got = fg_pf_mm(ctx, w, x, nullptr);
    g_mode.store(prev);
    ggml_cgraph * ga = ggml_new_graph(ctx);
    ggml_build_forward_expand(ga, ref);
    ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(gb, got);
    double best[2] = {1e30, 1e30};
    for (int r = 0; r < reps; ++r) {
        for (int k = 0; k < 2; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_graph_compute_with_ctx(ctx, k ? gb : ga, nth);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms < best[k]) best[k] = ms;
        }
    }
    out3[0] = best[0];
    out3[1] = best[1];
    out3[2] = std::memcmp(ref->data, got->data, sizeof(float) * (size_t) R * T) == 0 ? 1.0 : 0.0;
    ggml_free(ctx);
    return 0;
}

}  // extern "C"
