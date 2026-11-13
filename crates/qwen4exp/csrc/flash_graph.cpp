// flash_graph: the qwen4exp (Qwen3.8-Flash-Next) forward pass as a ggml graph, op for op the graph
// production builds (llama.cpp 04ff7df src/models/qwen4exp.cpp + the llama-graph.cpp helpers it
// calls, BigMoeOnEdge 930b2c9 context settings), computed by the PRODUCTION ggml-base/ggml-cpu
// objects (linked, not recompiled). Same ops, same node order (so the CPU backend fuses the same
// RMS_NORM+MUL pairs), same op parameters => bitwise the same tensors as production.
//
// What production configures and this file hard-wires (from the fixed353 run's log and source):
//   flash attention ON, fused GDN (AR + chunked) ON, fused DSV4 HC pre/post ON, fused top-k mask
//   ON, f16 K/V cache (no rotation), one sequence, unified cache, n_rs_seq 0, no MTP, outputs of
//   the last token only, cell index == position (one sequence from position 0 in a cleared cache).
// Engine-owned state replaces llama's memory objects: KV cache, indexer cache (raw | pooled keys),
// GDN conv/ssm states, the PLE conv history, and the token history the N-gram hash reads.
//
// Experts: ffn_{gate,up,down}_exps are ordinary [.., .., 512] tensors (the caller maps the shards)
// unless an expert callback is set, in which case the three mul_mat_id + swiglu are replaced by one
// custom node that produces the [n_embd, n_used, n_tokens] down outputs (the engine's streaming
// cache path). Everything around it (router, softmax, top-k, weights, the weighted sum) stays.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "flash_graph.h"
#include "flash_dense.h"
#include "pf_attn.h"
#include "flash_fuse_common.h"

// production ggml-cpu's swiglu row kernel (vec.cpp, C linkage; what GGML_OP_GLU/SWIGLU calls per row)
extern "C" void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

// ---- S-b executor support: production ggml-cpu's own op functions (global symbols of libggml-cpu.a) ----
// struct layout = ggml-cpu-impl.h at the production commit (eae95db)
struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
    bool use_ref;
};
#define FG_OPS(X) X(dup) X(add) X(add1) X(sub) X(mul) X(div) X(sqr) X(sqrt) X(log) X(sum) X(sum_rows) X(mean) X(argmax) \
    X(repeat) X(concat) X(norm) X(rms_norm) X(l2_norm) X(mul_mat) X(scale) X(set) X(cpy) X(cont) X(get_rows) X(set_rows) \
    X(soft_max) X(rope) X(clamp) X(argsort) X(top_k) X(fill) X(flash_attn_ext) X(ssm_conv) X(unary) X(glu) \
    X(gated_delta_net) X(lightning_indexer) X(dsv4_hc_comb) X(dsv4_hc_pre) X(dsv4_hc_post) X(top_k_mask) X(custom)
#define FG_DECL(name) extern "C" void ggml_compute_forward_##name(const ggml_compute_params *, ggml_tensor *);
FG_OPS(FG_DECL)
#undef FG_DECL
extern "C" void ggml_compute_forward_rms_norm_mul_fused(const ggml_compute_params *, ggml_tensor *, ggml_tensor *);
extern "C" bool ggml_cpu_extra_compute_forward(ggml_compute_params *, ggml_tensor *);

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cctype>
#include <functional>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <array>
#include <deque>
#include <omp.h>
#include <unordered_map>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

constexpr int64_t N_EMBD = 2560, HC = 4, HC_LR = 320, N_LAYER = 48, N_EXPERT = 512, N_USED = 10;
constexpr int64_t FF_EXP = 640; // routed expert FFN width
constexpr int64_t N_VOCAB = 248320;
constexpr int64_t N_HEAD = 24, N_HEAD_KV = 2, HEAD_D = 256, N_ROT = 64;
constexpr int64_t IDX_H = 4, IDX_D = 128, KPOOL = 4, IDX_TOP_K = 2048;
constexpr int64_t SSM_D = 128, SSM_KH = 16, SSM_VH = 48, SSM_CONV = 4;
constexpr int64_t CONV_CH = SSM_D * SSM_KH * 2 + SSM_D * SSM_VH; // 10240
constexpr int64_t N_EMBD_S = SSM_D * SSM_D * SSM_VH;              // 786432
constexpr int64_t N_EMBD_R = (SSM_CONV - 1) * CONV_CH;            // 30720
constexpr int64_t PLE_HEADS = 16, PLE_DIM = 160, PLE_NGRAM = 3, PLE_PER_GRAM = 8, PLE_KERN = 4;
constexpr int64_t PLE_HIST = (PLE_KERN - 1) * PLE_NGRAM; // 9
constexpr float EPS = 1e-6f;
constexpr int PLE_LAYER = 1;
constexpr int ROPE_TYPE = GGML_ROPE_TYPE_IMROPE;
constexpr float FREQ_BASE = 10000000.0f, FREQ_SCALE = 1.0f;
constexpr int N_CTX_ORIG = 262144;
constexpr float EXT_FACTOR = 0.0f, ATTN_FACTOR = 1.0f, BETA_FAST = 32.0f, BETA_SLOW = 1.0f;

bool is_recr(int il) { return (il + 1) % 4 != 0; }

struct Layer {
    ggml_tensor *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
    ggml_tensor *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
    // gdn
    ggml_tensor *wqkv, *wqkv_gate, *ssm_conv1d, *ssm_dt, *ssm_a, *ssm_beta, *ssm_alpha, *ssm_norm, *ssm_out;
    // attention
    ggml_tensor *wq, *wk, *wv, *wo, *attn_q_norm, *attn_k_norm, *index_q_proj, *index_k_proj, *index_q_norm, *index_k_norm;
    // ple
    ggml_tensor *ple_key, *ple_value, *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_conv1d;
    // moe
    ggml_tensor *ffn_gate_inp, *ffn_up_exps, *ffn_gate_exps, *ffn_down_exps;
    ggml_tensor *ffn_gate_inp_shexp, *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp;
};

} // namespace

struct fg_model {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> by_name;
    Layer layers[N_LAYER + 1] = {};  // [N_LAYER] = the MTP head block (blk.48), set by fg_mtp_attach (T2)
    struct fg_mtp_model * mtp = nullptr;
    ggml_tensor *tok_embd = nullptr, *output = nullptr, *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;
    ggml_tensor * per_layer_tok_embd = nullptr;
    uint64_t ple_mult[PLE_NGRAM] = {};
    uint32_t ple_off[PLE_HEADS] = {}, ple_voc[PLE_HEADS] = {};
    int32_t ple_eos = 0;
};

// cached graphs: one slot per ubatch width 1..FG_GRAPH_SLOTS (decode = 1, MTP verify = 2..4), each with its own
// gallocr, so decode, verify and prefill graphs never invalidate one another
constexpr int FG_GRAPH_SLOTS = 4;

struct fg_ctx {
    fg_model * m = nullptr;
    uint32_t n_ctx = 0;
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    // index N_LAYER = the MTP head block's cells (own buffer, fg_mtp_ctx_init; not part of the sequence state)
    ggml_tensor * k_l[N_LAYER + 1] = {};   // [HEAD_D*N_HEAD_KV, n_ctx] f16
    ggml_tensor * v_l[N_LAYER + 1] = {};
    ggml_tensor * ik_l[N_LAYER + 1] = {};  // indexer cache [2*IDX_D, n_ctx] f16 (raw | pooled)
    ggml_tensor * r_l[N_LAYER] = {};   // conv state [N_EMBD_R, 1]
    ggml_tensor * s_l[N_LAYER] = {};   // ssm state [N_EMBD_S, 1]
    ggml_tensor * p_l = nullptr;       // PLE conv history [PLE_HIST*HC*N_EMBD, 1]
    std::vector<int32_t> toks;         // token at each position (cell)
    bool fresh = true;                 // the recurrent states are zeroed by the next ubatch (rs_z = 0)
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    fg_expert_fn expert_fn = nullptr;
    fg_options opts = {};
    std::map<std::string, std::pair<uint64_t, double>> prof; // kind -> (calls, ms)
    double tim[5] = {};  // ms since the last timing reset: build, alloc, inputs, compute, calls
    // decode-graph reuse (S-a): the last one-token graph, valid while no other graph has been allocated
    struct DecCache {
        std::vector<uint8_t> meta;
        ggml_context * ctx0 = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_tensor * out = nullptr;
        void * g = nullptr;
        void (*del)(void *) = nullptr;
        uint64_t key[6] = {};
        bool valid = false;
        // S-b executor schedule for this graph (built on first use): node index, fused partner (-1), barrier before
        std::vector<int> ex_node, ex_fused;
        std::vector<uint8_t> ex_barrier, ex_work;
        int ex_nth = 0, ex_barriers = 0;
        bool ex_built = false, ex_ok = false;
        void reset() {
            if (g && del) del(g);
            if (ctx0) ggml_free(ctx0);
            g = nullptr;
            ctx0 = nullptr;
            gf = nullptr;
            out = nullptr;
            valid = false;
            std::vector<uint8_t>().swap(meta);
            ex_node.clear();
            ex_fused.clear();
            ex_barrier.clear();
            std::vector<uint8_t>().swap(ex_work);
            ex_built = ex_ok = false;
            bars.clear();
        }
        // per-slot allocator (kept across resets: its buffer is reused) and the slot graph's fused-node barriers
        ggml_gallocr_t galloc = nullptr;
        std::deque<FuseBarrier> bars;
    } dcs[2 * FG_GRAPH_SLOTS]; // [0, SLOTS) plain widths, [SLOTS, 2 SLOTS) T2 verify widths (g_mtp_verify_cache)
    DecCache * cur_dc = &dcs[0]; // the slot of the graph being built / run (the executor schedules it)
    int reuse_graph = 1;  // fg_ctx_set_graph_reuse
    // S-c: one barrier per fused-phase node of the current graph (op userdata); cleared when a new graph is built
    std::deque<FuseBarrier> fuse_bars;              // barriers of an uncached graph's fused nodes
    std::deque<FuseBarrier> * cur_bars = &fuse_bars; // where the graph being built takes its barriers
    FuseBarrier * fuse_barrier() { return &cur_bars->emplace_back(); }
    ggml_threadpool * ex_tp = nullptr; // S-b executor's threadpool (its n_graph / current_chunk serve ggml's ops)
    int ex_tp_nth = 0;
    double ex_stats[3] = {};           // graphs scheduled, nodes, barriers (last schedule)
    // serving abort (fg_ctx_set_abort): checked by ggml between graph nodes; an aborted decode leaves the recurrent
    // state half-updated, so the context refuses decodes (rs_dirty) until fg_ctx_restore_recurrent / fg_ctx_clear
    const volatile unsigned char * abort_flag = nullptr; // a Rust AtomicBool (one byte)
    // tokens the per-token expert op may take (FG_EXP_CAP query at each graph build; wider -> batched op)
    int per_tok_cap = 1 << 30;
    bool rs_dirty = false;
    // FD-1 trace (fg_ctx_set_trace; executor runs only): per node kind {calls, work ms, graph-barrier wait ms, nodes}
    // summed per token over threads / nth; fused-phase counters of the last graph build
    int trace = 0;
    std::map<std::string, std::array<double, 4>> tr_kinds;
    double tr_tokens = 0, tr_fuse_spin_ms = 0, tr_fuse_waits = 0, tr_region_ms = 0;
    uint64_t fuse_built[6] = {}, fuse_fallback[6] = {}; // P1 hc-mix, P2 hc-combine, P3/4 GDN, P5 router, P6 combine, P7 attn
    void * expert_ud = nullptr;
    struct Call { fg_ctx * c; int il; } calls[N_LAYER + 1] = {};
    // T2 (flash_mtp.inc): a verify's provisional recurrent results, applied by fg_commit; the head's state
    struct Pend {
        bool active = false;
        int n = 0;
        uint32_t p0 = 0;
        struct Gdn { float * state; const float * snaps; };            // snaps: n slots of N_EMBD_S floats, slot s = s tokens back
        struct Conv { float * state; const float * input; int64_t cols, ch; }; // input [cols + n, ch]
        std::vector<Gdn> gdn;
        std::vector<Conv> conv;
    } pend;
    struct fg_mtp_ctx * mtp = nullptr;
    double mtp_stats[4] = {};  // verify calls, verify rows, per-column matmul fallbacks, head passes
};

// S-c fused decode phases (model-free bitwise tests in flash-m3 selftest; wired behind fg_options.fuse)
#include "flash_fuse_hc.inc"
#include "flash_fuse_ffn.inc"
#include "flash_fuse_comb.inc"
#include "flash_fuse_gdn.inc"
#include "flash_fuse_attn.inc"
#include "flash_split_test.inc"  // chunked-prefill split test (model-free, fg_selftest_split)
#include "flash_mmcols_test.inc"  // F87: mul_mat w columns vs 1 column (model-free, fg_selftest_mm_cols)

namespace {

template <int KIND>
void expert_trampoline(ggml_tensor * dst, int ith, int nth, void * ud) {
    auto * call = (fg_ctx::Call *) ud;
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    fg_expert_call e;
    e.kind = KIND;
    e.il = call->il;
    e.ith = ith;
    e.nth = nth;
    e.n_tokens = (int) ids->ne[1];
    e.ids = (const int32_t *) ids->data;
    e.ids_nb1 = ids->nb[1];
    e.src = (const float *) src->data;
    e.src_nb1 = src->nb[1];
    e.src_nb2 = src->nb[2];
    e.dst = (float *) dst->data;
    e.dst_nb1 = dst->nb[1];
    e.dst_nb2 = dst->nb[2];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    call->c->expert_fn(&e, call->c->expert_ud);
}


// ---- fixed-chunk attention (fg_options.fixed_attn): one query row's softmax(QK^T)V over the KV
// cells in FIXED blocks of FG_ATTN_BLOCK cells, merged in block order. Independent of the thread
// count and of how many rows are evaluated together (decode == MTP verify per row, by construction).
// Per cell: production's K vec_dot (Q as f16, f16 K) and scale/mask as ggml's flash-attention; the
// V accumulator is f32 (production's decode path accumulates in f16): within noise, not bitwise.
constexpr int64_t FG_ATTN_BLOCK = 256;

// acc[0..DV) = sum_j p[j] * V[cells[j]][0..DV) (V f16, row stride vnb1 bytes, already offset to the kv head).
// Cell order is fixed, so the result depends only on the block's own inputs. NEON: 64-float chunks held
// in 16 float32x4 accumulators, V converted from f16 on the fly. Scalar path for other DV or no NEON.
// mode 1 online update for one cell, the exact expressions of the original loop (so results are bitwise those of
// tickets 002-010): vf = the V row in f32, acc rescaled when the max grows
static inline void attn_online_step(float s, const ggml_fp16_t * vd, float * vf, float * acc, int64_t DV, float & M, float & S) {
    const float Mold = M;
    float ms = 1.0f, vs = 1.0f;
    if (s > M) {
        M = s;
        ms = expf(Mold - M);
    } else {
        vs = expf(s - M);
    }
    ggml_fp16_to_fp32_row(vd, vf, DV);
    if (ms != 1.0f) for (int64_t i = 0; i < DV; ++i) acc[i] *= ms;
    for (int64_t i = 0; i < DV; ++i) acc[i] += vf[i] * vs;
    S = S * ms + vs;
}
static std::vector<float> & attn_vf(int64_t n) {
    thread_local std::vector<float> v;
    if ((int64_t) v.size() < n) v.resize(n);
    return v;
}

static void attn_pv_accum(float * acc, const float * p, const int64_t * cells, int nc, const char * vbase, size_t vnb1, int64_t DV) {
#if defined(__ARM_NEON)
    if (DV % 64 == 0) {
        for (int64_t i0 = 0; i0 < DV; i0 += 64) {
            float32x4_t a[16];
            for (int q = 0; q < 16; ++q) a[q] = vdupq_n_f32(0.0f);
            for (int j = 0; j < nc; ++j) {
                const ggml_fp16_t * vr = (const ggml_fp16_t *) (vbase + (size_t) cells[j] * vnb1) + i0;
                const float16_t * vh = reinterpret_cast<const float16_t *>(vr);
                const float pj = p[j];
                for (int q = 0; q < 16; ++q) {
                    const float32x4_t vf4 = vcvt_f32_f16(vld1_f16(vh + 4 * q));
                    a[q] = vfmaq_n_f32(a[q], vf4, pj);
                }
            }
            for (int q = 0; q < 16; ++q) vst1q_f32(acc + i0 + 4 * q, a[q]);
        }
        return;
    }
#endif
    for (int64_t i = 0; i < DV; ++i) acc[i] = 0.0f;
    for (int j = 0; j < nc; ++j) {
        const ggml_fp16_t * vr = (const ggml_fp16_t *) (vbase + (size_t) cells[j] * vnb1);
        const float pj = p[j];
        for (int64_t i = 0; i < DV; ++i) acc[i] += pj * ggml_fp16_to_fp32(vr[i]);
    }
}

// Row/head-tiled mode-1 path for many-row ubatches (prefill): unit = (tile of AT_ROWS query rows, kv head, block); the
// rk query heads sharing the kv head and the tile's rows use each K row and each converted V chunk once. Per query the
// operations are exactly the untiled path's (same kq_dot score, cell-order max / expf / sum, one FMA per (cell, dim)
// in cell order starting from 0), so every output equals the untiled unit's bit for bit: thread-count, batch and
// tiling independent (decode == verify).
constexpr int64_t AT_ROWS = 8;
static void attn_part_tiled(ggml_tensor * dst, int ith, int nth) {
    if (fg_pf_attn_part(dst, ith, nth)) return; // PREFILL FP-1 (pf_attn.cpp, --attn-tile): mode 1 in register tiles, same bits
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t DK = k->ne[0], DV = v->ne[0];
    const int64_t n_rows = q->ne[1], n_head = q->ne[2];
    const int64_t n_kv = k->ne[1];
    const int64_t nb = dst->ne[1];
    const int64_t rk = n_head / k->ne[2];
    const int64_t n_kvh = k->ne[2];
    const float sc = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1];
    const auto * tf16 = ggml_get_type_traits_cpu(GGML_TYPE_F16);
    const ggml_vec_dot_t kq_dot = ggml_get_type_traits_cpu(k->type)->vec_dot;
    const int64_t n_rt = (n_rows + AT_ROWS - 1) / AT_ROWS;
    const int64_t QM = AT_ROWS * rk; // queries per unit
    thread_local std::vector<ggml_fp16_t> qh;
    // [cell in block][query] (aliasing sweep: the hot passes walk cells outer, queries inner -> contiguous)
    thread_local std::vector<float> pv;     // score, then p
    thread_local std::vector<uint8_t> on;   // unmasked
    thread_local std::vector<float> mk;     // [tile row][cell in block]: the tile's mask rows, converted once per unit
    mk.resize((size_t) AT_ROWS * FG_ATTN_BLOCK);
    qh.resize((size_t) QM * DK);
    pv.resize((size_t) QM * FG_ATTN_BLOCK);
    on.resize((size_t) QM * FG_ATTN_BLOCK);
    const int64_t units = n_rt * n_kvh * nb;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t b = u % nb, g = (u / nb) % n_kvh, t = u / (nb * n_kvh);
        const int64_t r0 = t * AT_ROWS, r1 = std::min(n_rows, r0 + AT_ROWS);
        const int64_t c0 = b * FG_ATTN_BLOCK, c1 = std::min(n_kv, (b + 1) * FG_ATTN_BLOCK), ncell = c1 - c0;
        const int64_t nq = (r1 - r0) * rk;
        auto qrow = [&](int64_t qi) { return r0 + qi / rk; };
        auto qhead = [&](int64_t qi) { return g * rk + qi % rk; };
        auto outp = [&](int64_t qi) {
            return (float *) ((char *) dst->data + b * dst->nb[1] + qhead(qi) * dst->nb[2] + qrow(qi) * dst->nb[3]);
        };
        // the tile's mask rows once (power-of-two n_kv rows alias in L1 when re-read per query); masks, lazy Q conversion
        for (int64_t r = r0; r < r1; ++r) {
            const ggml_fp16_t * mp = (const ggml_fp16_t *) ((const char *) mask->data + r * mask->nb[1]);
            for (int64_t j = 0; j < ncell; ++j) mk[(r - r0) * FG_ATTN_BLOCK + j] = ggml_fp16_to_fp32(mp[c0 + j]);
        }
        std::vector<int> cnt(nq, 0);
        for (int64_t qi = 0; qi < nq; ++qi) {
            const float * mr = mk.data() + (qi / rk) * FG_ATTN_BLOCK;
            for (int64_t j = 0; j < ncell; ++j) {
                const bool o = mr[j] != -INFINITY;
                on[j * QM + qi] = o;
                cnt[qi] += o;
            }
            if (cnt[qi]) {
                const float * pq = (const float *) ((const char *) q->data + qrow(qi) * q->nb[1] + qhead(qi) * q->nb[2]);
                tf16->from_float(pq, qh.data() + qi * DK, DK);
            }
        }
        // pass 1: scores, K row loaded once for all queries
        for (int64_t j = 0; j < ncell; ++j) {
            const char * kr = (const char *) k->data + (c0 + j) * k->nb[1] + g * k->nb[2];
            for (int64_t qi = 0; qi < nq; ++qi) {
                if (!on[j * QM + qi]) continue;
                float s;
                kq_dot((int) DK, &s, 0, kr, 0, qh.data() + qi * DK, 0, 1);
                pv[j * QM + qi] = s * sc + mk[(qi / rk) * FG_ATTN_BLOCK + j];
            }
        }
        if (((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] != 3.0f) {
            // mode 1: the original online update per query in cell order (bitwise = the untiled mode-1 unit); each V
            // row converted once per cell for all queries of the tile
            std::vector<float> & vf = attn_vf(DV);
            std::vector<float> Mq(nq, -INFINITY), Sq(nq, 0.0f);
            for (int64_t qi = 0; qi < nq; ++qi) std::fill(outp(qi) + 2, outp(qi) + 2 + DV, 0.0f);
            for (int64_t j = 0; j < ncell; ++j) {
                bool conv = false;
                for (int64_t qi = 0; qi < nq; ++qi) {
                    if (!on[j * QM + qi]) continue;
                    if (!conv) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) ((const char *) v->data + (c0 + j) * v->nb[1] + g * v->nb[2]), vf.data(), DV);
                        conv = true;
                    }
                    const float s = pv[j * QM + qi];
                    float * acc = outp(qi) + 2;
                    float & M = Mq[qi];
                    float & S = Sq[qi];
                    const float Mold = M;
                    float ms = 1.0f, vs = 1.0f;
                    if (s > M) {
                        M = s;
                        ms = expf(Mold - M);
                    } else {
                        vs = expf(s - M);
                    }
                    if (ms != 1.0f) for (int64_t i = 0; i < DV; ++i) acc[i] *= ms;
                    for (int64_t i = 0; i < DV; ++i) acc[i] += vf[i] * vs;
                    S = S * ms + vs;
                }
            }
            for (int64_t qi = 0; qi < nq; ++qi) {
                outp(qi)[0] = Mq[qi];
                outp(qi)[1] = Sq[qi];
            }
            continue;
        }
        // mode 3 (--attn twopass): pass 2 per query: max, p, sum in cell order
        for (int64_t qi = 0; qi < nq; ++qi) {
            float * out = outp(qi);
            std::fill(out + 2, out + 2 + DV, 0.0f);
            if (!cnt[qi]) {
                out[0] = -INFINITY;
                out[1] = 0.0f;
                continue;
            }
            auto ps = [&](int64_t j) -> float & { return pv[j * QM + qi]; };
            auto os = [&](int64_t j) { return on[j * QM + qi]; };
            float Mb = -INFINITY;
            bool firstm = true;
            for (int64_t j = 0; j < ncell; ++j)
                if (os(j)) {
                    Mb = firstm ? ps(j) : std::max(Mb, ps(j));
                    firstm = false;
                }
            float Sb = 0.0f;
            for (int64_t j = 0; j < ncell; ++j)
                if (os(j)) {
                    ps(j) = expf(ps(j) - Mb);
                    Sb += ps(j);
                }
            out[0] = Mb;
            out[1] = Sb;
        }
        // pass 3: V chunk converted once per cell, FMA into every query's accumulator (cell order per query)
        for (int64_t i0 = 0; i0 < DV; i0 += 64) {
            for (int64_t j = 0; j < ncell; ++j) {
                const ggml_fp16_t * vr = (const ggml_fp16_t *) ((const char *) v->data + (c0 + j) * v->nb[1] + g * v->nb[2]) + i0;
#if defined(__ARM_NEON)
                float32x4_t vf[16];
                const float16_t * vh = reinterpret_cast<const float16_t *>(vr);
                for (int qq = 0; qq < 16; ++qq) vf[qq] = vcvt_f32_f16(vld1_f16(vh + 4 * qq));
#else
                float vf[64];
                for (int qq = 0; qq < 64; ++qq) vf[qq] = ggml_fp16_to_fp32(vr[qq]);
#endif
                for (int64_t qi = 0; qi < nq; ++qi) {
                    if (!on[j * QM + qi]) continue;
                    const float pj = pv[j * QM + qi];
                    float * acc = outp(qi) + 2 + i0;
#if defined(__ARM_NEON)
                    for (int qq = 0; qq < 16; ++qq) vst1q_f32(acc + 4 * qq, vfmaq_n_f32(vld1q_f32(acc + 4 * qq), vf[qq], pj));
#else
                    for (int qq = 0; qq < 64; ++qq) acc[qq] += pj * vf[qq];
#endif
                }
            }
        }
    }
}

void attn_part_op(ggml_tensor * dst, int ith, int nth, void *) {
    {
        // many rows (prefill, but not the precise reference mode): the tiled path, same bits per query
        const ggml_tensor * q0 = dst->src[0];
        const bool precise = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] == 2.0f;
        if (!precise && q0->ne[1] >= 2 * AT_ROWS && dst->src[2]->ne[0] % 64 == 0 && q0->ne[2] / dst->src[1]->ne[2] == q0->ne[2] / dst->src[2]->ne[2]) {
            attn_part_tiled(dst, ith, nth);
            return;
        }
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t DK = k->ne[0], DV = v->ne[0];
    const int64_t n_rows = q->ne[1], n_head = q->ne[2];
    const int64_t n_kv = k->ne[1];
    const int64_t nb = dst->ne[1];
    const int64_t rk = n_head / k->ne[2], rv = n_head / v->ne[2];
    const auto * tf16 = ggml_get_type_traits_cpu(GGML_TYPE_F16);
    const ggml_vec_dot_t kq_dot = ggml_get_type_traits_cpu(k->type)->vec_dot;
    std::vector<ggml_fp16_t> qh(DK);
    const int64_t units = n_rows * n_head * nb;
    for (int64_t u = ith; u < units; u += nth) {
        const int64_t b = u % nb, h = (u / nb) % n_head, r = u / (nb * n_head);
        float * out = (float *) ((char *) dst->data + b * dst->nb[1] + h * dst->nb[2] + r * dst->nb[3]);
        float * acc = out + 2;
        std::fill(acc, acc + DV, 0.0f);
        const ggml_fp16_t * mp = (const ggml_fp16_t *) ((const char *) mask->data + r * mask->nb[1]);
        const float * pq = (const float *) ((const char *) q->data + r * q->nb[1] + h * q->nb[2]);
        bool q_ready = false;
        // the KQ scale rides in the last op_params slot (the custom-op params use the first 24 bytes),
        // the mode in the one before: 2 = precise reference (f32 Q, f64 KQ dot, f64 accumulators)
        const float sc = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1];
        if (((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] == 2.0f) {
            std::vector<double> accd(DV);
            double Md = -INFINITY, Sd = 0.0;
            for (int64_t ic = b * FG_ATTN_BLOCK; ic < std::min(n_kv, (b + 1) * FG_ATTN_BLOCK); ++ic) {
                const float mv = ggml_fp16_to_fp32(mp[ic]);
                if (mv == -INFINITY) continue;
                GGML_ASSERT(k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
                const ggml_fp16_t * kd = (const ggml_fp16_t *) ((const char *) k->data + ic * k->nb[1] + (h / rk) * k->nb[2]);
                double sd = 0.0;
                for (int64_t i = 0; i < DK; ++i) sd += (double) pq[i] * (double) ggml_fp16_to_fp32(kd[i]);
                sd = sd * (double) sc + (double) mv;
                const double Mold = Md;
                double ms = 1.0, vs = 1.0;
                if (sd > Md) {
                    Md = sd;
                    ms = std::exp(Mold - Md);
                } else {
                    vs = std::exp(sd - Md);
                }
                const ggml_fp16_t * vd = (const ggml_fp16_t *) ((const char *) v->data + ic * v->nb[1] + (h / rv) * v->nb[2]);
                for (int64_t i = 0; i < DV; ++i) accd[i] = accd[i] * ms + (double) ggml_fp16_to_fp32(vd[i]) * vs;
                Sd = Sd * ms + vs;
            }
            // partials stored as f32 relative to the block max (rel. error ~1e-7, far below the f16 effects measured)
            out[0] = (float) Md;
            out[1] = (float) Sd;
            for (int64_t i = 0; i < DV; ++i) acc[i] = (float) accd[i];
            continue;
        }
        const float amode = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2];
        if (amode != 3.0f) {
            // mode 1: the original online update (f32), bitwise as at af4cda7 / tickets 002-010
            std::vector<float> & vf = attn_vf(DV);
            float M = -INFINITY, S = 0.0f;
            for (int64_t ic = b * FG_ATTN_BLOCK; ic < std::min(n_kv, (b + 1) * FG_ATTN_BLOCK); ++ic) {
                const float mv = ggml_fp16_to_fp32(mp[ic]);
                if (mv == -INFINITY) continue;
                if (!q_ready) {
                    tf16->from_float(pq, qh.data(), DK);
                    q_ready = true;
                }
                float s;
                kq_dot((int) DK, &s, 0, (const char *) k->data + ic * k->nb[1] + (h / rk) * k->nb[2], 0, qh.data(), 0, 1);
                s = s * sc + mv;
                attn_online_step(s, (const ggml_fp16_t *) ((const char *) v->data + ic * v->nb[1] + (h / rv) * v->nb[2]), vf.data(), acc, DV, M, S);
            }
            out[0] = M;
            out[1] = S;
            continue;
        }
        // mode 3 (--attn twopass): pass 1: scores of the unmasked cells of this block (same kq_dot and f16 Q)
        const int64_t c0 = b * FG_ATTN_BLOCK, c1 = std::min(n_kv, (b + 1) * FG_ATTN_BLOCK);
        float sco[FG_ATTN_BLOCK];
        int64_t cell[FG_ATTN_BLOCK];
        int nc = 0;
        for (int64_t ic = c0; ic < c1; ++ic) {
            const float mv = ggml_fp16_to_fp32(mp[ic]);
            if (mv == -INFINITY) continue;
            if (!q_ready) {
                tf16->from_float(pq, qh.data(), DK);
                q_ready = true;
            }
            float s;
            kq_dot((int) DK, &s, 0, (const char *) k->data + ic * k->nb[1] + (h / rk) * k->nb[2], 0, qh.data(), 0, 1);
            sco[nc] = s * sc + mv;
            cell[nc] = ic;
            ++nc;
        }
        if (nc == 0) {
            // fully masked block: M = -inf, S = 0, acc = 0 (attn_merge_op skips it)
            out[0] = -INFINITY;
            out[1] = 0.0f;
            continue;
        }
        // pass 2: block max, p = exp(s - M), S = sum p (cell order)
        float Mb = sco[0];
        for (int j = 1; j < nc; ++j) Mb = std::max(Mb, sco[j]);
        float Sb = 0.0f;
        for (int j = 0; j < nc; ++j) {
            sco[j] = expf(sco[j] - Mb);
            Sb += sco[j];
        }
        // pass 3: acc = sum p * V (f16 V converted on the fly)
        attn_pv_accum(acc, sco, cell, nc, (const char *) v->data + (h / rv) * v->nb[2], v->nb[1], DV);
        out[0] = Mb;
        out[1] = Sb;
    }
}

void attn_merge_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * p = dst->src[0];
    const int64_t DV = dst->ne[0], n_head = dst->ne[1], n_rows = dst->ne[2], nb = p->ne[1];
    for (int64_t u = ith; u < n_rows * n_head; u += nth) {
        const int64_t h = u % n_head, r = u / n_head;
        float * out = (float *) ((char *) dst->data + h * dst->nb[1] + r * dst->nb[2]);
        float M = -INFINITY;
        for (int64_t b = 0; b < nb; ++b) M = std::max(M, ((const float *) ((const char *) p->data + b * p->nb[1] + h * p->nb[2] + r * p->nb[3]))[0]);
        std::fill(out, out + DV, 0.0f);
        float S = 0.0f;
        for (int64_t b = 0; b < nb; ++b) {
            const float * pb = (const float *) ((const char *) p->data + b * p->nb[1] + h * p->nb[2] + r * p->nb[3]);
            if (pb[1] == 0.0f) continue; // fully masked block
            const float w = expf(pb[0] - M);
            S += pb[1] * w;
            for (int64_t i = 0; i < DV; ++i) out[i] += pb[2 + i] * w;
        }
        const float inv = S == 0.0f ? 0.0f : 1.0f / S;
        for (int64_t i = 0; i < DV; ++i) out[i] *= inv;
    }
}

// T2 (flash_mtp.inc): W x for every column of x with the one-token kernels (from_float + vec_dot nrc = 1 per row)
void mm_cols_op(ggml_tensor * dst, int ith, int nth, void *);
// T2 (fg_mtp_set_verify_cache): verify graphs of 1..FG_GRAPH_SLOTS rows are kept per width like the decode graph
// (their own slots, so a one-row verify never replaces the plain decode graph). Process-wide, default off.
int g_mtp_verify_cache = 0;

// one ubatch's graph: the llm_graph_context pieces qwen4exp uses, nothing else
struct G {
    fg_model & m;
    fg_ctx & c;
    ggml_context * ctx0;
    ggml_cgraph * gf;
    int64_t n_tokens;
    int64_t n_outputs;
    uint32_t n_kv;        // attention cells visible to this ubatch (padded to 256)
    uint32_t n_pool;      // kpool_pad(n_pool_real)
    uint32_t n_new_g;     // new pools to (re)pool, padded
    uint32_t n_sel;
    // inputs
    ggml_tensor *inp_tokens, *inp_pos, *inp_out_ids, *ple_rows, *k_idxs, *v_idxs, *idx_k_idxs, *s_copy;
    ggml_tensor *pool_cells, *pool_idxs, *pool_mask, *tail_idxs, *new_pool_idxs, *new_pool_flat, *new_pool_rep, *new_pool_pos;
    std::map<ggml_tensor *, ggml_tensor *> rs_rows;
    ggml_tensor * shexp_pre = nullptr; // shared expert built ahead of the routed experts (opts.shexp_first)
    ggml_tensor * fused_ffn_out = nullptr; // S-c P6: ffn_out produced by the fused combine node
    int32_t rs_z;
    // T2: vmode 1 = verify (all rows out, recurrent state provisional, per-column matmuls), 2 = prefill that also
    // returns h of every row (the crop to the output row moves behind layer 47; per-column matmuls from there on)
    int vmode = 0;
    bool cols_exact = false;
    int cols_fallback = 0;
    ggml_tensor * h_src = nullptr; // [N_EMBD, HC, rows] pre-head residual (output-flagged)
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> pend_gdn; // (state tensor, GDN result)
    struct PConv { ggml_tensor * state; ggml_tensor * input; int64_t cols, ch; };
    std::vector<PConv> pend_conv;

    void cb(ggml_tensor * t, const char * name, int il) {
        if (il >= 0) ggml_format_name(t, "%s-%d", name, il);
        else ggml_set_name(t, name);
    }
    fg_pf_cache pf_packs; // PREFILL: activation packs of the multi-column matmul (flash_dense.cpp); off => plain ggml_mul_mat
    ggml_tensor * mm(ggml_tensor * w, ggml_tensor * x) {
        // T2 verify (cols_exact, 2-16 columns): per-column vec_dot so every row equals its one-token decode
        if (cols_exact && x->ne[1] > 1) {
            if (x->ne[2] == 1 && x->ne[3] == 1 && w->ne[2] == 1 && w->ne[3] == 1 && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x)) {
                ggml_tensor * a[2] = {w, x};
                return ggml_custom_4d(ctx0, GGML_TYPE_F32, w->ne[1], x->ne[1], 1, 1, a, 2, mm_cols_op, GGML_N_TASKS_MAX, nullptr);
            }
            ++cols_fallback;
        }
        return fg_pf_mm(ctx0, w, x, &pf_packs); // PREFILL dispatch (plain ggml_mul_mat when its switch is off)
    }
    ggml_tensor * build_norm(ggml_tensor * cur, ggml_tensor * mw, int il) {
        cur = ggml_rms_norm(ctx0, cur, EPS);
        if (mw) {
            cb(cur, "norm", il);
            cur = ggml_mul(ctx0, cur, mw);
        }
        return cur;
    }

    struct hc_mix { ggml_tensor * cur; ggml_tensor * inject; };

    hc_mix build_hc_mix(ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down, ggml_tensor * w_up, ggml_tensor * w_inject, int il) {
        const int64_t hc_dim = HC * N_EMBD;
        const int64_t nt = x->ne[2];
        if (il >= 0 && (c.opts.fuse & FG_FUSE_HC) && !fuse_hc_mix_supported(nt, w_down, w_up, w_inject)) c.fuse_fallback[0]++;
        if (il >= 0 && (c.opts.fuse & FG_FUSE_HC) && fuse_hc_mix_supported(nt, w_down, w_up, w_inject)) {
            c.fuse_built[0]++;
            // S-c P1: one fused node for norm, down, silu, up and inject (bitwise = the nodes below); hc_pre stays
            ggml_tensor * f = fuse_build_hc_mix(ctx0, c.fuse_barrier(), x, w_norm, w_down, w_up, w_inject);
            cb(f, "hc_mix_fused", il);
            ggml_tensor * xn3 = ggml_view_3d(ctx0, f, N_EMBD, HC, nt, N_EMBD * sizeof(float), f->nb[1], 0);
            ggml_tensor * gate3 = ggml_view_3d(ctx0, f, N_EMBD, HC, nt, N_EMBD * sizeof(float), f->nb[1], hc_dim * sizeof(float));
            ggml_tensor * mixed = ggml_dsv4_hc_pre_gated(ctx0, xn3, gate3, 1.0f / (float) HC);
            cb(mixed, "hc_mixed", il);
            ggml_tensor * inject = nullptr;
            if (w_inject) inject = ggml_view_2d(ctx0, f, w_inject->ne[1], nt, f->nb[1], 2 * hc_dim * sizeof(float));
            return {mixed, inject};
        }
        ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, EPS), w_norm);
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
        cb(xn, "hc_norm", il);
        ggml_tensor * lo = ggml_scale(ctx0, mm(w_down, xn), 1.0f / (float) HC);
        lo = ggml_silu(ctx0, lo);
        ggml_tensor * gate = mm(w_up, lo);
        cb(gate, "hc_gate", il);
        ggml_tensor * mixed = nullptr;
        if (il >= 0) {
            mixed = ggml_dsv4_hc_pre_gated(ctx0, ggml_reshape_3d(ctx0, xn, N_EMBD, HC, nt),
                                           ggml_reshape_3d(ctx0, gate, N_EMBD, HC, nt), 1.0f / (float) HC);
        } else {
            ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
            gated = ggml_reshape_3d(ctx0, gated, N_EMBD, HC, nt);
            mixed = ggml_view_2d(ctx0, gated, N_EMBD, nt, ggml_row_size(gated->type, N_EMBD) * HC, 0);
            mixed = ggml_cont(ctx0, mixed);
            for (int64_t s = 1; s < HC; ++s) {
                ggml_tensor * v = ggml_view_2d(ctx0, gated, N_EMBD, nt, ggml_row_size(gated->type, N_EMBD) * HC,
                                               ggml_row_size(gated->type, N_EMBD) * s);
                mixed = ggml_add(ctx0, mixed, v);
            }
            mixed = ggml_scale(ctx0, mixed, 1.0f / (float) HC);
        }
        cb(mixed, "hc_mixed", il);
        ggml_tensor * inject = nullptr;
        if (w_inject) {
            inject = ggml_scale(ctx0, mm(w_inject, xn), 1.0f / (float) HC);
            cb(inject, "hc_inject", il);
        }
        return {mixed, inject};
    }

    ggml_tensor * build_hc_combine(ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject, int il) {
        if (c.opts.fuse & FG_FUSE_HCPOST) {
            // S-c P2: sigmoid, scale 2 and hc_post in one node (bitwise = the nodes below; no FMA, as production)
            if (ggml_tensor * f = fuse_build_hc_combine(ctx0, c.fuse_barrier(), residual, block_out, inject)) {
                cb(f, "hc_combine", il);
                c.fuse_built[1]++;
                return f;
            }
            c.fuse_fallback[1]++;
        }
        ggml_tensor * w = ggml_sigmoid(ctx0, inject);
        w = ggml_scale(ctx0, w, 2.0f);
        ggml_tensor * cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        cb(cur, "hc_combine", il);
        return cur;
    }

    // llm_graph_context::build_rs, one sequence: s_copy = [0], head 0, size 1
    ggml_tensor * build_rs(ggml_tensor * s, int32_t state_size) {
        ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);
        if (rs_z < 0 && c.opts.rs_view) {
            // one sequence, not fresh: s_copy = [0], so get_rows(states, s_copy) is a plain f32 copy of row 0. ggml
            // splits get_rows by rows, i.e. ONE thread copied each 3 MB GDN state (36 layers, ~9 ms/token); a view of
            // row 0 gives the same bits. Readers run before the state write-back (it depends on their output).
            return ggml_view_2d(ctx0, states, state_size, 1, states->nb[1], 0);
        }
        ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size * (rs_z >= 0), rs_z * states->nb[1] * (rs_z >= 0));
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
        const int64_t n_rs = 1, n_seqs = 1;
        const int64_t i0 = n_rs - s_copy->ne[0];
        ggml_tensor * states_all = ggml_get_rows(ctx0, states, s_copy);
        ggml_tensor * output_states = ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
        ggml_build_forward_expand(gf, output_states);
        ggml_tensor * states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1], (n_seqs - i0) * states_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, states_extra, ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (0 + n_seqs) * s->nb[1])));
        return output_states;
    }

    ggml_tensor * build_conv_state_at(ggml_tensor * conv_states_all, ggml_tensor * x, int64_t state_cols, int64_t channels, int il) {
        const int64_t n_seqs = 1;
        const int64_t row_total = conv_states_all->ne[0];
        GGML_ASSERT(state_cols * channels == row_total);
        auto it = rs_rows.find(conv_states_all);
        if (it == rs_rows.end()) it = rs_rows.emplace(conv_states_all, build_rs(conv_states_all, (int32_t) row_total)).first;
        ggml_tensor * rows = it->second;
        ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
        cb(state, "conv_state_at", il);
        ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);
        if (vmode == 1) {
            // verify: no state write; fg_commit(m) copies the window [m, m + state_cols) of conv_input per channel
            ggml_set_output(conv_input);
            ggml_build_forward_expand(gf, conv_input);
            pend_conv.push_back({conv_states_all, conv_input, state_cols, channels});
            return conv_input;
        }
        const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
        const uint32_t mem_size = 1, kv_head = 0;
        const int64_t n_slots = 1;
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);
            ggml_tensor * tail = ggml_view_3d(ctx0, conv_input, state_cols, channels, n_seqs, conv_input->nb[1], conv_input->nb[2],
                                              ggml_row_size(conv_input->type, s_idx));
            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all, state_cols * channels, n_seqs, conv_states_all->nb[1],
                                             (slot * mem_size + kv_head) * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
        }
        return conv_input;
    }

    ggml_tensor * build_ple(ggml_tensor * emb, ggml_tensor * hidden, int il) {
        const int64_t hc_dim = HC * N_EMBD;
        const Layer & L = m.layers[il];
        ggml_tensor * key = mm(L.ple_key, emb);
        ggml_tensor * value = mm(L.ple_value, emb);
        auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
            ggml_tensor * t = ggml_reshape_3d(ctx0, x, N_EMBD, HC, n_tokens);
            return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, EPS), w);
        };
        key = grouped_norm(key, L.ple_norm_key);
        ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);
        ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
        s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) N_EMBD));
        ggml_tensor * mag = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
        cb(gate, "ple_gate", il);
        ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, N_EMBD, 1, n_tokens);
        v3 = ggml_repeat_4d(ctx0, v3, N_EMBD, HC, n_tokens, 1);
        ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
        cb(gated, "ple_gated_value", il);
        ggml_tensor * normalized = grouped_norm(ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens), L.ple_norm_conv);
        normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);
        const int64_t kern = PLE_KERN, dil = PLE_NGRAM, hist = (kern - 1) * dil;
        const int64_t n_seqs = 1, n_seq_tokens = n_tokens;
        ggml_tensor * padded = build_conv_state_at(c.p_l, ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs), hist, hc_dim, il);
        ggml_tensor * conv_out = nullptr;
        for (int64_t k = 0; k < kern; ++k) {
            const int64_t start = hist - (kern - 1 - k) * dil;
            ggml_tensor * shifted = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs, padded->nb[1],
                                                                                      padded->nb[2], ggml_row_size(padded->type, start))));
            ggml_tensor * wk = ggml_cont(ctx0, ggml_view_2d(ctx0, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1], k * L.ple_conv1d->nb[0]));
            wk = ggml_reshape_1d(ctx0, wk, hc_dim);
            if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
            ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
            conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
        }
        conv_out = ggml_silu(ctx0, conv_out);
        conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), N_EMBD, HC, n_tokens);
        cb(conv_out, "ple_conv_out", il);
        return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
    }

    ggml_tensor * build_qsa_sel(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        ggml_tensor * k_raw = mm(L.index_k_proj, cur);
        cb(k_raw, "indexer_k_raw", il);
        ggml_tensor * pzero = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, IDX_D, n_tokens), 0.0f);
        ggml_tensor * packed = ggml_reshape_3d(ctx0, ggml_concat(ctx0, k_raw, pzero, 0), 2 * IDX_D, 1, n_tokens);
        // cpy_k of the indexer cache: [2*IDX_D, 1 head, n_tokens] -> set_rows into [2*IDX_D, n_ctx]
        {
            ggml_tensor * kc = ggml_view_2d(ctx0, packed, 2 * IDX_D, n_tokens, packed->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, c.ik_l[il], kc, idx_k_idxs));
        }
        ggml_tensor * kst = c.ik_l[il];
        const int64_t n_cells = kst->ne[1] * kst->ne[2];
        ggml_tensor * key_gate = ggml_view_2d(ctx0, kst, IDX_D, n_cells, kst->nb[1], 0);
        ggml_tensor * pooled_st = ggml_view_2d(ctx0, kst, IDX_D, n_cells, kst->nb[1], ggml_row_size(kst->type, IDX_D));

        ggml_tensor * rows = ggml_get_rows(ctx0, key_gate, new_pool_flat);
        rows = ggml_reshape_3d(ctx0, rows, IDX_D, KPOOL, n_new_g);
        ggml_tensor * pooled_new = nullptr;
        for (int64_t i = 0; i < KPOOL; ++i) {
            ggml_tensor * slice = ggml_view_2d(ctx0, rows, IDX_D, n_new_g, rows->nb[2], i * rows->nb[1]);
            pooled_new = pooled_new ? ggml_add(ctx0, pooled_new, slice) : ggml_cont(ctx0, slice);
        }
        pooled_new = ggml_scale(ctx0, pooled_new, 1.0f / (float) KPOOL);
        pooled_new = build_norm(pooled_new, L.index_k_norm, il);
        int sections[4] = {11, 11, 10, 0};
        pooled_new = ggml_reshape_3d(ctx0, pooled_new, IDX_D, 1, n_new_g);
        pooled_new = ggml_rope_multi(ctx0, pooled_new, new_pool_pos, nullptr, N_ROT, sections, ROPE_TYPE, N_CTX_ORIG, FREQ_BASE, FREQ_SCALE,
                                     EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW);
        pooled_new = ggml_reshape_2d(ctx0, pooled_new, IDX_D, n_new_g);
        cb(pooled_new, "indexer_pool_k_new", il);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, pooled_st, pooled_new, new_pool_rep));
        ggml_tensor * pooled = ggml_get_rows(ctx0, pooled_st, pool_cells);
        pooled = ggml_reshape_3d(ctx0, pooled, IDX_D, 1, n_pool);
        cb(pooled, "indexer_k", il);

        ggml_tensor * q = mm(L.index_q_proj, cur);
        q = ggml_reshape_3d(ctx0, q, IDX_D, IDX_H, n_tokens);
        q = build_norm(q, L.index_q_norm, il);
        q = ggml_rope_multi(ctx0, q, inp_pos, nullptr, N_ROT, sections, ROPE_TYPE, N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
                            ATTN_FACTOR, BETA_FAST, BETA_SLOW);
        cb(q, "indexer_q", il);
        ggml_tensor * weights = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, IDX_H, n_tokens), 1.0f / sqrtf((float) IDX_D));
        ggml_tensor * score = ggml_lightning_indexer(ctx0, q, pooled, weights, pool_mask);
        cb(score, "indexer_score", il);
        const int64_t n_top_pool = std::min<int64_t>(n_pool, IDX_TOP_K / KPOOL);
        ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool);
        cb(top_k, "indexer_top_k", il);
        GGML_ASSERT(KPOOL * n_top_pool + tail_idxs->ne[0] == n_sel);
        ggml_tensor * keep = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F16, n_kv), 0.0f);
        ggml_tensor * sel = ggml_top_k_mask(ctx0, keep, top_k, score, pool_idxs, tail_idxs);
        cb(sel, "indexer_sel", il);
        return sel;
    }

    ggml_tensor * build_attn_qsa(ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur, ggml_tensor * sel, float kq_scale, int il) {
        ggml_build_forward_expand(gf, q_cur);
        ggml_build_forward_expand(gf, v_cur);
        ggml_build_forward_expand(gf, k_cur);
        {
            ggml_tensor * kc = ggml_view_2d(ctx0, k_cur, HEAD_D * N_HEAD_KV, n_tokens, k_cur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, c.k_l[il], kc, k_idxs));
            ggml_tensor * vc = ggml_view_2d(ctx0, v_cur, HEAD_D * N_HEAD_KV, n_tokens, v_cur->nb[2], 0);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, c.v_l[il], vc, v_idxs));
        }
        ggml_tensor * q = q_cur;
        ggml_tensor * kk = c.k_l[il];
        ggml_tensor * vv = c.v_l[il];
        const int64_t gqa = HEAD_D * N_HEAD_KV;
        ggml_tensor * k = ggml_view_4d(ctx0, kk, HEAD_D, N_HEAD_KV, n_kv, 1, ggml_row_size(kk->type, HEAD_D), ggml_row_size(kk->type, gqa),
                                       ggml_row_size(kk->type, gqa * c.n_ctx), 0);
        ggml_tensor * v = ggml_view_4d(ctx0, vv, HEAD_D, N_HEAD_KV, n_kv, 1, ggml_row_size(vv->type, HEAD_D), ggml_row_size(vv->type, gqa),
                                       ggml_row_size(vv->type, gqa * c.n_ctx), 0);
        const int64_t n_stream = k->ne[3];
        GGML_ASSERT(sel->ne[0] == k->ne[2] && sel->ne[1] % n_stream == 0);
        sel = ggml_reshape_4d(ctx0, sel, sel->ne[0], sel->ne[1] / n_stream, 1, n_stream);
        cb(sel, "kq_mask_qsa", il);
        // build_attn_mha, flash-attention branch
        q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream, q->nb[1], q->nb[2], q->nb[2] * (q->ne[2] / n_stream), 0);
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);
        ggml_tensor * cur;
        if (c.opts.fixed_attn) {
            const int64_t nb = (n_kv + FG_ATTN_BLOCK - 1) / FG_ATTN_BLOCK;
            ggml_tensor * pa[4] = {q, k, v, sel};
            if (c.opts.fixed_attn == 1 && fg_pf_attn_fused_ok(q, k, v, sel)) {
                // PREFILL W4-I (pf_attn.cpp, --attn-tile 2): part + merge as ONE node, no [2 + D, blocks, heads, rows] tensor
                // (6.5 GB at 8,192 rows); same bits as the two nodes
                cur = ggml_custom_4d(ctx0, GGML_TYPE_F32, HEAD_D, N_HEAD, n_tokens, 1, pa, 4, fg_pf_attn_fused_op, GGML_N_TASKS_MAX, nullptr);
                ((float *) cur->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1] = kq_scale;
                ((float *) cur->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] = 1.0f;
                cb(cur, "kqv_fused", il);
            } else {
            ggml_tensor * part = ggml_custom_4d(ctx0, GGML_TYPE_F32, 2 + HEAD_D, nb, N_HEAD, n_tokens, pa, 4, attn_part_op, GGML_N_TASKS_MAX, nullptr);
            ((float *) part->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1] = kq_scale;
            ((float *) part->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] = (float) c.opts.fixed_attn;
            cb(part, "kqv_part", il);
            ggml_tensor * ma[1] = {part};
            cur = ggml_custom_4d(ctx0, GGML_TYPE_F32, HEAD_D, N_HEAD, n_tokens, 1, ma, 1, attn_merge_op, GGML_N_TASKS_MAX, nullptr);
            }
        } else {
            cur = ggml_flash_attn_ext(ctx0, q, k, v, sel, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, nullptr);
            ggml_flash_attn_ext_set_n_kv_max(cur, (int32_t) n_sel);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
        }
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        ggml_build_forward_expand(gf, cur);
        cb(cur, "kqv_out", il);
        return cur;
    }

    ggml_tensor * build_layer_attn(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        ggml_tensor * sel = build_qsa_sel(cur, il);
        // S-c P7 (fuse bit 32): Q/K/V projections, q/k norms and the gate split in one node; rope, KV writes, the
        // indexer and the fixed-chunk attention stay; POST = sigmoid(gate) * attention, wo (bitwise, model-free tested)
        ggml_tensor * fpre = (c.opts.fuse & FG_FUSE_ATTN)
                                 ? fuse_build_attn_pre(ctx0, c.fuse_barrier(), cur, L.wq, L.wk, L.wv, L.attn_q_norm, L.attn_k_norm)
                                 : nullptr;
        ggml_tensor * Qcur;
        ggml_tensor * Kcur;
        ggml_tensor * Vcur;
        ggml_tensor * gate;
        if ((c.opts.fuse & FG_FUSE_ATTN) && !fpre) c.fuse_fallback[5]++;
        if (fpre) {
            c.fuse_built[5]++;
            cb(fpre, "attn_pre_fused", il);
            Qcur = fuse_attn_pre_q(ctx0, fpre, n_tokens);
            Kcur = fuse_attn_pre_k(ctx0, fpre, n_tokens);
            Vcur = fuse_attn_pre_v(ctx0, fpre, n_tokens);
            gate = fuse_attn_pre_gate(ctx0, fpre, n_tokens);
        } else {
        ggml_tensor * Qcur_full = mm(L.wq, cur);
        cb(Qcur_full, "Qcur_full", il);
        Qcur = ggml_view_3d(ctx0, Qcur_full, HEAD_D, N_HEAD, n_tokens, ggml_element_size(Qcur_full) * HEAD_D * 2,
                                          ggml_element_size(Qcur_full) * HEAD_D * 2 * N_HEAD, 0);
        cb(Qcur, "Qcur_reshaped", il);
        Qcur = build_norm(Qcur, L.attn_q_norm, il);
        cb(Qcur, "Qcur_normed", il);
        Kcur = mm(L.wk, cur);
        cb(Kcur, "Kcur", il);
        Vcur = mm(L.wv, cur);
        cb(Vcur, "Vcur", il);
        Kcur = ggml_reshape_3d(ctx0, Kcur, HEAD_D, N_HEAD_KV, n_tokens);
        Kcur = build_norm(Kcur, L.attn_k_norm, il);
        cb(Kcur, "Kcur_normed", il);
        gate = ggml_view_3d(ctx0, Qcur_full, HEAD_D, N_HEAD, n_tokens, ggml_element_size(Qcur_full) * HEAD_D * 2,
                                          ggml_element_size(Qcur_full) * HEAD_D * 2 * N_HEAD, ggml_element_size(Qcur_full) * HEAD_D);
        gate = ggml_cont_2d(ctx0, gate, HEAD_D * N_HEAD, n_tokens);
        cb(gate, "gate_reshaped", il);
        Vcur = ggml_reshape_3d(ctx0, Vcur, HEAD_D, N_HEAD_KV, n_tokens);
        }
        int sections[4] = {11, 11, 10, 0};
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr, N_ROT, sections, ROPE_TYPE, N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
                               ATTN_FACTOR, BETA_FAST, BETA_SLOW);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr, N_ROT, sections, ROPE_TYPE, N_CTX_ORIG, FREQ_BASE, FREQ_SCALE, EXT_FACTOR,
                               ATTN_FACTOR, BETA_FAST, BETA_SLOW);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        const float kq_scale = 1.0f / sqrtf(float(HEAD_D));
        cur = build_attn_qsa(Qcur, Kcur, Vcur, sel, kq_scale, il);
        cb(cur, "attn_pregate", il);
        if (fpre) {
            if (ggml_tensor * o = fuse_build_attn_post(ctx0, c.fuse_barrier(), cur, gate, L.wo)) {
                cb(o, "attn_output", il);
                return o;
            }
        }
        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
        cb(gate_sigmoid, "gate_sigmoid", il);
        cur = ggml_mul(ctx0, cur, gate_sigmoid);
        cb(cur, "attn_gated", il);
        cur = mm(L.wo, cur);
        cb(cur, "attn_output", il);
        return cur;
    }

    ggml_tensor * build_layer_attn_linear(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        const int64_t n_seqs = 1, n_seq_tokens = n_tokens;
        const int64_t head_k_dim = SSM_D, num_k_heads = SSM_KH, num_v_heads = SSM_VH, head_v_dim = SSM_D;
        if ((c.opts.fuse & FG_FUSE_GDN) && c.opts.rs_view && rs_z < 0 && vmode != 1) { // T2: P3/P4 write the states in place; a verify must not
            // S-c P3 + P4 (bitwise = the nodes below, model-free tested): P3 = projections, gates, conv (+ the conv
            // state write, in place), silu, q/k norms; the GDN op stays; P4 = the ssm state write, gated norm, ssm_out
            ggml_tensor * crow = ggml_view_2d(ctx0, c.r_l[il], N_EMBD_R, 1, c.r_l[il]->nb[1], 0);
            if (fuse_gdn_pre_supported(cur, crow, L.wqkv, L.wqkv_gate, L.ssm_beta, L.ssm_alpha, L.ssm_conv1d, L.ssm_dt, L.ssm_a)) {
                ggml_tensor * f = fuse_build_gdn_pre(ctx0, c.fuse_barrier(), cur, crow, L.wqkv, L.wqkv_gate, L.ssm_beta,
                                                     L.ssm_alpha, L.ssm_conv1d, L.ssm_dt, L.ssm_a);
                if (f) {
                    cb(f, "gdn_pre_fused", il);
                    const int64_t nt = n_tokens;
                    ggml_tensor * zf = fuse_gdn_pre_z(ctx0, f, nt);
                    ggml_tensor * srow = ggml_view_2d(ctx0, c.s_l[il], N_EMBD_S, 1, c.s_l[il]->nb[1], 0);
                    ggml_tensor * state = ggml_reshape_4d(ctx0, srow, head_v_dim, head_v_dim, num_v_heads, n_seqs);
                    ggml_tensor * result = ggml_gated_delta_net(ctx0, fuse_gdn_pre_q(ctx0, f, nt), fuse_gdn_pre_k(ctx0, f, nt),
                                                                fuse_gdn_pre_v(ctx0, f, nt), fuse_gdn_pre_gate(ctx0, f, nt),
                                                                fuse_gdn_pre_beta(ctx0, f, nt), state, 1);
                    ggml_tensor * att = fuse_gdn_attn_view(ctx0, result, nt);
                    ggml_tensor * ns = fuse_gdn_state_view(ctx0, result, nt);
                    ggml_tensor * tgt = fuse_gdn_state_target(ctx0, c.s_l[il]);
                    if (fuse_gdn_post_supported(att, ns, zf, L.ssm_norm, L.ssm_out, tgt)) {
                        if (ggml_tensor * lin = fuse_build_gdn_post(ctx0, c.fuse_barrier(), att, ns, zf, L.ssm_norm, L.ssm_out, tgt)) {
                            cb(lin, "linear_attn_out", il);
                            c.fuse_built[2]++;
                            return lin;
                        }
                    }
                    // P4 refused: the original tail on P3's views (same nodes as below)
                    c.fuse_fallback[2]++;
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ns, ggml_view_2d(ctx0, c.s_l[il], N_EMBD_S, n_seqs, c.s_l[il]->nb[1], 0)));
                    ggml_tensor * z_2d = ggml_view_4d(ctx0, zf, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, ggml_row_size(zf->type, head_v_dim),
                                                      zf->nb[1], zf->nb[1] * n_seq_tokens, 0);
                    ggml_tensor * normalized = build_norm(att, L.ssm_norm, il);
                    ggml_tensor * attn_out_norm = ggml_mul(ctx0, normalized, ggml_sigmoid(ctx0, z_2d));
                    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
                    ggml_tensor * lin = mm(L.ssm_out, final_output);
                    cb(lin, "linear_attn_out", il);
                    return lin;
                }
            }
        }
        ggml_tensor * qkv_mixed = mm(L.wqkv, cur);
        qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);
        ggml_tensor * z = mm(L.wqkv_gate, cur);
        cb(z, "z", il);
        ggml_tensor * beta = mm(L.ssm_beta, cur);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * alpha = mm(L.ssm_alpha, cur);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        cb(beta, "beta", il);
        cb(alpha, "alpha", il);
        beta = ggml_sigmoid(ctx0, beta);
        cb(beta, "beta_sigmoid", il);
        ggml_tensor * alpha_biased = ggml_add(ctx0, alpha, L.ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);
        ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, L.ssm_a);
        cb(gate, "gate", il);
        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * conv_states_all = c.r_l[il];
        ggml_tensor * ssm_states_all = c.s_l[il];
        ggml_tensor * conv_kernel = L.ssm_conv1d;
        const int64_t conv_kernel_size = conv_kernel->ne[0];
        const int64_t conv_channels = CONV_CH;
        ggml_tensor * conv_input = build_conv_state_at(conv_states_all, qkv_mixed, conv_kernel_size - 1, conv_channels, il);
        ggml_tensor * state = build_rs(ssm_states_all, (int32_t) N_EMBD_S);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);
        ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);
        ggml_tensor * conv_qkv_mix = conv_output_silu;
        int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);
        ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                                            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
        ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
                                            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
                                            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));
        ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                                            ggml_row_size(conv_qkv_mix->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
                                            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));
        cb(q_conv, "q_conv", il);
        cb(k_conv, "k_conv", il);
        cb(v_conv, "v_conv", il);
        // build_gdn_l2_norm
        auto l2 = [&](ggml_tensor * x) {
            const float n = x->ne[0];
            return ggml_scale(ctx0, ggml_rms_norm(ctx0, x, EPS / n), 1.0f / sqrtf(n));
        };
        q_conv = l2(q_conv);
        k_conv = l2(k_conv);
        cb(q_conv, "q_conv_predelta", il);
        cb(k_conv, "k_conv_predelta", il);
        cb(v_conv, "v_conv_predelta", il);
        // build_recurrent_attn (n_rs_seq 0) -> build_delta_net -> fused
        // verify: one state snapshot per token (slot s = s tokens back; same arithmetic as K = 1), kept for fg_commit
        ggml_tensor * result = ggml_gated_delta_net(ctx0, q_conv, k_conv, v_conv, gate, beta, state, vmode == 1 ? n_seq_tokens : 1);
        if (vmode == 1) {
            ggml_set_output(result);
            pend_gdn.push_back({ssm_states_all, result});
        }
        const int64_t S_v = SSM_D, H_v = SSM_VH;
        ggml_tensor * output = ggml_view_4d(ctx0, result, S_v, H_v, n_seq_tokens, n_seqs, ggml_row_size(result->type, S_v),
                                            ggml_row_size(result->type, S_v * H_v), ggml_row_size(result->type, S_v * H_v * n_seq_tokens), 0);
        ggml_tensor * new_state = ggml_view_4d(ctx0, result, S_v, S_v, H_v, n_seqs, ggml_row_size(result->type, S_v),
                                               ggml_row_size(result->type, S_v * S_v), ggml_row_size(result->type, S_v * S_v * H_v),
                                               ggml_row_size(result->type, S_v * H_v * n_seq_tokens * n_seqs));
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);
        if (vmode != 1) ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ggml_view_2d(ctx0, ssm_states_all, N_EMBD_S, n_seqs, ssm_states_all->nb[1], 0)));
        ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs, ggml_row_size(z->type, head_v_dim), z->nb[1],
                                          z->nb[1] * n_seq_tokens, 0);
        // build_norm_gated: sigmoid output gate
        ggml_tensor * normalized = build_norm(output, L.ssm_norm, il);
        ggml_tensor * gated = ggml_sigmoid(ctx0, z_2d);
        ggml_tensor * attn_out_norm = ggml_mul(ctx0, normalized, gated);
        ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
        cb(final_output, "final_output", il);
        cur = mm(L.ssm_out, final_output);
        cb(cur, "linear_attn_out", il);
        return cur;
    }

    ggml_tensor * build_moe_ffn(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        ggml_tensor * cur_in = cur;
        fused_ffn_out = nullptr;
        ggml_tensor * fr = (c.opts.fuse & FG_FUSE_ROUTER) ? fuse_build_router(ctx0, c.fuse_barrier(), cur, L.ffn_gate_inp) : nullptr;
        ggml_tensor * selected_experts;
        ggml_tensor * weights;
        if ((c.opts.fuse & FG_FUSE_ROUTER) && !fr) c.fuse_fallback[3]++;
        if (fr) {
            // S-c P5: logits, softmax, top-k, weights and their normalisation in one node (bitwise = the nodes below)
            c.fuse_built[3]++;
            cb(fr, "router_fused", il);
            selected_experts = fuse_router_selected(ctx0, fr, n_tokens);
            weights = fuse_router_weights(ctx0, fr, n_tokens);
            ggml_build_forward_expand(gf, weights);
        } else {
        ggml_tensor * logits = mm(L.ffn_gate_inp, cur);
        cb(logits, "ffn_moe_logits", il);
        ggml_tensor * probs = ggml_soft_max(ctx0, logits);
        cb(probs, "ffn_moe_probs", il);
        selected_experts = ggml_argsort_top_k(ctx0, probs, N_USED);
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
        cb(selected_experts, "ffn_moe_topk", il);
        probs = ggml_reshape_3d(ctx0, probs, 1, N_EXPERT, n_tokens);
        weights = ggml_get_rows(ctx0, probs, selected_experts);
        cb(weights, "ffn_moe_weights", il);
        weights = ggml_reshape_2d(ctx0, weights, N_USED, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        cb(weights_sum, "ffn_moe_weights_sum", il);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
        weights = ggml_div(ctx0, weights, weights_sum);
        cb(weights, "ffn_moe_weights_norm", il);
        weights = ggml_reshape_3d(ctx0, weights, 1, N_USED, n_tokens);
        ggml_build_forward_expand(gf, weights);
        }
        cur = ggml_reshape_3d(ctx0, cur, N_EMBD, 1, n_tokens);
        ggml_tensor * experts = nullptr;
        if (c.expert_fn && ((c.opts.batch_min_tokens > 0 && n_tokens >= c.opts.batch_min_tokens) || n_tokens > c.per_tok_cap)) {
            // (F38: also any ubatch wider than the per-token op's cap, e.g. 2-16 token chunks on small pools)
            // engine path, large ubatch (prefill): one node streams the ubatch's distinct experts through
            // the cache in waves (same row-wise kernels, same production swiglu => same bits)
            c.calls[il] = {&c, il};
            ggml_tensor * ba[2] = {cur, selected_experts};
            experts = ggml_custom_4d(ctx0, GGML_TYPE_F32, N_EMBD, N_USED, n_tokens, 1, ba, 2, expert_trampoline<FG_EXP_BATCH>,
                                     GGML_N_TASKS_MAX, &c.calls[il]);
            cb(experts, "ffn_moe_down", il);
        } else if (c.expert_fn) {
            // engine path: the experts come from the streaming cache; swiglu stays production's op
            c.calls[il] = {&c, il};
            ggml_tensor * fa[2] = {cur, selected_experts};
            ggml_tensor * fetch = ggml_custom_4d(ctx0, GGML_TYPE_I32, 1, 1, 1, 1, fa, 2, expert_trampoline<FG_EXP_FETCH>, 1, &c.calls[il]);
            cb(fetch, "ffn_moe_fetch", il);
            if (c.opts.shexp_first && !(c.opts.fuse & FG_FUSE_COMB)) {
                // the shared expert (dense, resident) runs while the misses are read
                ggml_build_forward_expand(gf, fetch);
                shexp_pre = build_shexp(cur_in, il);
                ggml_build_forward_expand(gf, shexp_pre);
            }
            ggml_tensor * ga[3] = {cur, selected_experts, fetch};
            ggml_tensor * up = ggml_custom_4d(ctx0, GGML_TYPE_F32, FF_EXP, N_USED, n_tokens, 1, ga, 3, expert_trampoline<FG_EXP_UP>,
                                              GGML_N_TASKS_MAX, &c.calls[il]);
            cb(up, "ffn_moe_up", il);
            ggml_tensor * g = ggml_custom_4d(ctx0, GGML_TYPE_F32, FF_EXP, N_USED, n_tokens, 1, ga, 3, expert_trampoline<FG_EXP_GATE>,
                                             GGML_N_TASKS_MAX, &c.calls[il]);
            cb(g, "ffn_moe_gate", il);
            ggml_tensor * act = ggml_swiglu_split(ctx0, g, up);
            cb(act, "ffn_moe_swiglu", il);
            ggml_tensor * da[3] = {act, selected_experts, fetch};
            experts = ggml_custom_4d(ctx0, GGML_TYPE_F32, N_EMBD, N_USED, n_tokens, 1, da, 3, expert_trampoline<FG_EXP_DOWN>,
                                     GGML_N_TASKS_MAX, &c.calls[il]);
            cb(experts, "ffn_moe_down", il);
        } else {
            ggml_tensor * up = ggml_mul_mat_id(ctx0, L.ffn_up_exps, cur, selected_experts);
            cb(up, "ffn_moe_up", il);
            ggml_tensor * g = ggml_mul_mat_id(ctx0, L.ffn_gate_exps, cur, selected_experts);
            cb(g, "ffn_moe_gate", il);
            ggml_tensor * act = ggml_swiglu_split(ctx0, g, up);
            cb(act, "ffn_moe_swiglu", il);
            experts = ggml_mul_mat_id(ctx0, L.ffn_down_exps, act, selected_experts);
            cb(experts, "ffn_moe_down", il);
        }
        if (c.opts.fuse & FG_FUSE_COMB) {
            // S-c P6: shared expert + the weighted sum of the routed experts + the final add in one node
            // (bitwise = the nodes below and build_shexp + build_layer_ffn's add); build_layer_ffn returns it as ffn_out
            ggml_tensor * fc = fuse_build_shexp_combine(ctx0, c.fuse_barrier(), cur_in, experts, weights, L.ffn_gate_shexp,
                                                        L.ffn_up_shexp, L.ffn_down_shexp, L.ffn_gate_inp_shexp);
            if (!fc) c.fuse_fallback[4]++;
            if (fc) {
                c.fuse_built[4]++;
                cb(fc, "shexp_combine_fused", il);
                fused_ffn_out = fuse_shexp_combine_out(ctx0, fc, n_tokens);
                return fused_ffn_out;
            }
        }
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
        ggml_build_forward_expand(gf, experts);
        ggml_tensor * cur_experts[N_USED];
        for (int64_t i = 0; i < N_USED; ++i) {
            cur_experts[i] = ggml_view_2d(ctx0, experts, N_EMBD, n_tokens, experts->nb[2], i * experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[i]);
        }
        ggml_tensor * moe_out = cur_experts[0];
        for (int64_t i = 1; i < N_USED; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        cb(moe_out, "ffn_moe_out", il);
        return moe_out;
    }

    ggml_tensor * build_layer_ffn(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        shexp_pre = nullptr;
        ggml_tensor * moe_out = build_moe_ffn(cur, il);
        if (fused_ffn_out) {
            cb(fused_ffn_out, "ffn_out", il);
            return fused_ffn_out;
        }
        cb(moe_out, "ffn_moe_out", il);
        ggml_tensor * ffn_shexp = shexp_pre ? shexp_pre : build_shexp(cur, il);
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        return cur;
    }

    ggml_tensor * build_shexp(ggml_tensor * cur, int il) {
        const Layer & L = m.layers[il];
        // shared expert: build_ffn(SILU, PAR)
        ggml_tensor * tmp = mm(L.ffn_up_shexp, cur);
        cb(tmp, "ffn_up", il);
        ggml_tensor * g = mm(L.ffn_gate_shexp, cur);
        cb(g, "ffn_gate", il);
        g = ggml_swiglu_split(ctx0, g, tmp);
        cb(g, "ffn_swiglu", il);
        ggml_tensor * ffn_shexp = mm(L.ffn_down_shexp, g);
        cb(ffn_shexp, "ffn_shexp", il);
        ggml_tensor * shared_gate = mm(L.ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);
        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);
        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        return ffn_shexp;
    }

    ggml_tensor * build() {
        // inputs (created in the order llama creates them; inputs carry no arithmetic)
        // PLE input (build_inp_ple)
        ple_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, PLE_HEADS * n_tokens);
        ggml_set_input(ple_rows);
        ggml_tensor * ple_emb = ggml_get_rows(ctx0, m.per_layer_tok_embd, ple_rows);
        ple_emb = ggml_reshape_2d(ctx0, ple_emb, PLE_DIM * PLE_HEADS, n_tokens);
        cb(ple_emb, "ple_embd", -1);
        ggml_build_forward_expand(gf, ple_emb);
        // token embeddings (build_inp_embd; the token path)
        inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(inp_tokens, "inp_tokens");
        ggml_set_input(inp_tokens);
        ggml_tensor * inpL = ggml_get_rows(ctx0, m.tok_embd, inp_tokens);
        ggml_build_forward_order(gf, inpL);
        cb(inpL, "model.input_embed", -1);
        ggml_build_forward_expand(gf, inpL);
        // k-pool inputs
        idx_k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
        ggml_set_input(idx_k_idxs);
        pool_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool);
        pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL, n_pool);
        pool_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n_tokens);
        tail_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL - 1, n_tokens);
        for (ggml_tensor * t : {pool_cells, pool_idxs, pool_mask, tail_idxs}) {
            ggml_set_input(t);
            ggml_build_forward_expand(gf, t);
        }
        new_pool_idxs = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL, n_new_g);
        ggml_set_input(new_pool_idxs);
        new_pool_flat = ggml_reshape_1d(ctx0, new_pool_idxs, KPOOL * n_new_g);
        new_pool_rep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new_g);
        ggml_set_input(new_pool_rep);
        new_pool_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_new_g);
        ggml_set_input(new_pool_pos);
        inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * 4);
        ggml_set_input(inp_pos);
        inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
        ggml_set_input(inp_out_ids);
        k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
        ggml_set_input(k_idxs);
        v_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens);
        ggml_set_input(v_idxs);
        s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(s_copy);

        ggml_tensor * res_hc = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inpL, N_EMBD, 1, n_tokens), N_EMBD, HC, n_tokens, 1);
        cb(res_hc, "hc_init", -1);
        ggml_build_forward_expand(gf, res_hc);

        for (int il = 0; il < N_LAYER; ++il) {
            const Layer & L = m.layers[il];
            if (il == PLE_LAYER) res_hc = build_ple(ple_emb, res_hc, il);
            hc_mix attn_mix = build_hc_mix(res_hc, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, il);
            ggml_tensor * cur = attn_mix.cur;
            ggml_build_forward_expand(gf, cur);
            cur = is_recr(il) ? build_layer_attn_linear(cur, il) : build_layer_attn(cur, il);
            if (il == N_LAYER - 1 && vmode == 2) {
                cols_exact = true; // the rows stay; from here on every column takes the one-token kernels the cropped row takes
            } else if (il == N_LAYER - 1) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                attn_mix.inject = ggml_get_rows(ctx0, attn_mix.inject, inp_out_ids);
                res_hc = ggml_reshape_2d(ctx0, res_hc, N_EMBD * HC, res_hc->ne[2]);
                res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
                res_hc = ggml_reshape_3d(ctx0, res_hc, N_EMBD, HC, res_hc->ne[1]);
                n_tokens = n_outputs; // everything below is per output row
            }
            res_hc = build_hc_combine(res_hc, cur, attn_mix.inject, il);
            hc_mix ffn_mix = build_hc_mix(res_hc, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, il);
            cur = ffn_mix.cur;
            cur = build_layer_ffn(cur, il);
            cb(cur, "ffn_out", il);
            res_hc = build_hc_combine(res_hc, cur, ffn_mix.inject, il);
            cb(res_hc, "l_last", il);
        }
        ggml_tensor * h = ggml_reshape_2d(ctx0, res_hc, N_EMBD * HC, res_hc->ne[2]);
        cb(h, "h_nextn", -1);
        ggml_build_forward_expand(gf, h);
        if (vmode) {
            // res_hc is the last hc_combine's own tensor: flag it so its rows survive the graph
            GGML_ASSERT(res_hc->view_src == nullptr);
            ggml_set_output(res_hc);
            h_src = res_hc;
        }
        if (vmode == 2) {
            res_hc = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, h, inp_out_ids), N_EMBD, HC, n_outputs);
            n_tokens = n_outputs;
        }
        hc_mix head_mix = build_hc_mix(res_hc, m.hc_head_norm, m.hc_head_down, m.hc_head_up, nullptr, -1);
        ggml_tensor * cur = head_mix.cur;
        cb(cur, "result_norm", -1);
        cur = mm(m.output, cur);
        cb(cur, "result_output", -1);
        ggml_build_forward_expand(gf, cur);
        return cur;
    }
};

uint32_t kpool_pad(uint32_t n_pool) { return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u)); }

void write_dump(const std::string & path, const ggml_tensor * t) {
    const size_t es = ggml_type_size(t->type);
    std::vector<char> out((size_t) ggml_nelements(t) * es);
    char * o = out.data();
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                const char * row = (const char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, o += es) std::memcpy(o, row + i0 * t->nb[0], es);
            }
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const int32_t ty = (int32_t) t->type;
    std::fwrite("FGD1", 1, 4, f);
    std::fwrite(&ty, 4, 1, f);
    std::fwrite(t->ne, 8, 4, f);
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
}

bool name_match(const char * name, const std::vector<std::string> & pats) {
    for (const auto & p : pats) {
        if (p.size() > 1 && p.front() == '*') {
            const size_t n = std::strlen(name), k = p.size() - 1;
            if (n >= k && std::strcmp(name + n - k, p.c_str() + 1) == 0) return true;
        } else if (!p.empty() && p.back() == '*') {
            if (std::strncmp(name, p.c_str(), p.size() - 1) == 0) return true;
        } else if (p == name) {
            return true;
        }
    }
    return false;
}

} // namespace

static void fg_mtp_model_free(fg_model * m); // flash_mtp.inc
static void fg_mtp_ctx_free(fg_ctx * c);
// PREFILL FP-1: the part op, for the model-free off-vs-on check in pf_attn.cpp
ggml_custom_op_t fg_attn_part_op_ptr() { return attn_part_op; }
ggml_custom_op_t fg_attn_merge_op_ptr() { return attn_merge_op; }


extern "C" {

fg_model * fg_model_new(const fg_tensor_ref * refs, int n, const fg_region * regions, int n_regions, const fg_ple_params * ple) {
    auto * m = new fg_model();
    ggml_init_params ip = {(size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true};
    m->ctx = ggml_init(ip);
    for (int i = 0; i < n_regions; ++i) {
        m->bufs.push_back(ggml_backend_cpu_buffer_from_ptr(regions[i].base, regions[i].size));
    }
    for (int i = 0; i < n; ++i) {
        const fg_tensor_ref & r = refs[i];
        int nd = 4;
        while (nd > 1 && r.ne[nd - 1] == 1) --nd;
        ggml_tensor * t = ggml_new_tensor(m->ctx, (ggml_type) r.type, nd, r.ne);
        ggml_set_name(t, r.name);
        t->data = r.data;
        if (r.region >= 0) t->buffer = m->bufs[r.region];
        m->by_name[r.name] = t;
    }
    auto get = [&](const std::string & name, bool required = true) -> ggml_tensor * {
        auto it = m->by_name.find(name);
        if (it == m->by_name.end()) {
            if (required) {
                std::fprintf(stderr, "flash_graph: missing tensor %s\n", name.c_str());
                std::abort();
            }
            return nullptr;
        }
        return it->second;
    };
    // [n_embd*hc] gammas load as [n_embd, hc] (TENSOR_ALLOW_RESHAPE)
    auto norm2 = [&](const std::string & name) {
        ggml_tensor * t = get(name);
        GGML_ASSERT(ggml_nelements(t) == N_EMBD * HC);
        t->ne[0] = N_EMBD;
        t->ne[1] = HC;
        t->nb[1] = t->nb[0] * N_EMBD;
        t->nb[2] = t->nb[3] = t->nb[1] * HC;
        return t;
    };
    m->tok_embd = get("token_embd.weight");
    m->output = get("output.weight");
    m->hc_head_norm = norm2("output_hc_norm.weight");
    m->hc_head_down = get("output_hc_down.weight");
    m->hc_head_up = get("output_hc_up.weight");
    m->per_layer_tok_embd = get("per_layer_token_embd.weight");
    for (int il = 0; il < N_LAYER; ++il) {
        Layer & L = m->layers[il];
        auto n = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        L.hc_attn_norm = norm2(n("hc_attn_norm.weight"));
        L.hc_attn_down = get(n("hc_attn_down.weight"));
        L.hc_attn_up = get(n("hc_attn_up.weight"));
        L.hc_attn_inject = get(n("hc_attn_inject.weight"));
        L.hc_ffn_norm = norm2(n("hc_ffn_norm.weight"));
        L.hc_ffn_down = get(n("hc_ffn_down.weight"));
        L.hc_ffn_up = get(n("hc_ffn_up.weight"));
        L.hc_ffn_inject = get(n("hc_ffn_inject.weight"));
        if (is_recr(il)) {
            L.wqkv = get(n("attn_qkv.weight"));
            L.wqkv_gate = get(n("attn_gate.weight"));
            L.ssm_conv1d = get(n("ssm_conv1d.weight"));
            L.ssm_dt = get(n("ssm_dt.bias"));
            L.ssm_a = get(n("ssm_a"));
            L.ssm_beta = get(n("ssm_beta.weight"));
            L.ssm_alpha = get(n("ssm_alpha.weight"));
            L.ssm_norm = get(n("ssm_norm.weight"));
            L.ssm_out = get(n("ssm_out.weight"));
        } else {
            L.wq = get(n("attn_q.weight"));
            L.wk = get(n("attn_k.weight"));
            L.wv = get(n("attn_v.weight"));
            L.wo = get(n("attn_output.weight"));
            L.attn_q_norm = get(n("attn_q_norm.weight"));
            L.attn_k_norm = get(n("attn_k_norm.weight"));
            L.index_q_proj = get(n("indexer.q_proj.weight"));
            L.index_k_proj = get(n("indexer.k_proj.weight"));
            L.index_q_norm = get(n("indexer.q_norm.weight"));
            L.index_k_norm = get(n("indexer.k_norm.weight"));
        }
        if (il == PLE_LAYER) {
            L.ple_key = get(n("ple_key.weight"));
            L.ple_value = get(n("ple_value.weight"));
            L.ple_norm_key = norm2(n("ple_norm_key.weight"));
            L.ple_norm_query = norm2(n("ple_norm_query.weight"));
            L.ple_norm_conv = norm2(n("ple_norm_conv.weight"));
            L.ple_conv1d = get(n("ple_conv1d.weight"));
        }
        L.ffn_gate_inp = get(n("ffn_gate_inp.weight"));
        L.ffn_up_exps = get(n("ffn_up_exps.weight"), false);
        L.ffn_gate_exps = get(n("ffn_gate_exps.weight"), false);
        L.ffn_down_exps = get(n("ffn_down_exps.weight"), false);
        L.ffn_gate_inp_shexp = get(n("ffn_gate_inp_shexp.weight"));
        L.ffn_gate_shexp = get(n("ffn_gate_shexp.weight"));
        L.ffn_up_shexp = get(n("ffn_up_shexp.weight"));
        L.ffn_down_shexp = get(n("ffn_down_shexp.weight"));
    }
    for (int i = 0; i < PLE_NGRAM; ++i) m->ple_mult[i] = ple->mult[i];
    for (int i = 0; i < PLE_HEADS; ++i) {
        m->ple_off[i] = ple->head_offsets[i];
        m->ple_voc[i] = ple->head_vocab[i];
    }
    m->ple_eos = ple->eos;
    return m;
}

void fg_model_free(fg_model * m) {
    if (!m) return;
    fg_mtp_model_free(m);
    for (auto b : m->bufs) ggml_backend_buffer_free(b);
    ggml_free(m->ctx);
    delete m;
}

fg_ctx * fg_ctx_new(fg_model * m, uint32_t n_ctx) {
    auto * c = new fg_ctx();
    c->m = m;
    c->n_ctx = n_ctx;
    ggml_init_params ip = {(size_t) 4 * N_LAYER * ggml_tensor_overhead() + 64 * ggml_tensor_overhead(), nullptr, true};
    c->sctx = ggml_init(ip);
    for (int il = 0; il < N_LAYER; ++il) {
        if (is_recr(il)) {
            c->r_l[il] = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F32, N_EMBD_R, 1);
            c->s_l[il] = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F32, N_EMBD_S, 1);
            ggml_format_name(c->r_l[il], "cache_r_l%d", il);
            ggml_format_name(c->s_l[il], "cache_s_l%d", il);
        } else {
            c->k_l[il] = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F16, HEAD_D * N_HEAD_KV, n_ctx);
            c->v_l[il] = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F16, HEAD_D * N_HEAD_KV, n_ctx);
            c->ik_l[il] = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F16, 2 * IDX_D, n_ctx);
            ggml_format_name(c->k_l[il], "cache_k_l%d", il);
            ggml_format_name(c->v_l[il], "cache_v_l%d", il);
            ggml_format_name(c->ik_l[il], "cache_idx_k_l%d", il);
        }
    }
    c->p_l = ggml_new_tensor_2d(c->sctx, GGML_TYPE_F32, PLE_HIST * HC * N_EMBD, 1);
    ggml_set_name(c->p_l, "cache_p_l1");
    c->backend = ggml_backend_cpu_init();
    c->sbuf = ggml_backend_alloc_ctx_tensors(c->sctx, c->backend);
    ggml_backend_buffer_clear(c->sbuf, 0);
    c->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    return c;
}

void fg_ctx_free(fg_ctx * c) {
    if (!c) return;
    fg_mtp_ctx_free(c);
    for (auto & d : c->dcs) {
        d.reset();
        if (d.galloc) ggml_gallocr_free(d.galloc);
        d.galloc = nullptr;
    }
    if (c->ex_tp) ggml_threadpool_free(c->ex_tp);
    ggml_gallocr_free(c->galloc);
    ggml_backend_buffer_free(c->sbuf);
    ggml_free(c->sctx);
    ggml_backend_free(c->backend);
    delete c;
}

void fg_ctx_clear(fg_ctx * c) {
    ggml_backend_buffer_clear(c->sbuf, 0);
    c->toks.clear();
    c->fresh = true;
    c->rs_dirty = false;
    c->pend.active = false; // W4-K (FLASH-MTP): a cleared context has no verify left to commit
}

static bool fg_abort_cb(void * d) { return *(const volatile unsigned char *) d != 0; }

// serving cancel: ggml checks *flag between graph nodes (a routed-expert node can run for a while in prefill, so the
// latency is up to one node, ~1 layer). nullptr = off.
void fg_ctx_set_abort(fg_ctx * c, const volatile unsigned char * flag) {
    c->abort_flag = flag;
    ggml_backend_cpu_set_abort_callback(c->backend, flag ? fg_abort_cb : nullptr, (void *) flag);
}

// model-free: ggml's CPU backend stops a multi-node graph when the flag is set, and runs it fully when it is not
int fg_selftest_abort(void) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip = {64u << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1 << 16);
    for (int i = 0; i < (1 << 16); ++i) ((float *) x->data)[i] = 1.0f;
    ggml_tensor * y = x;
    for (int i = 0; i < 64; ++i) y = ggml_scale(ctx, y, 1.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    volatile unsigned char flag = 0;
    ggml_backend_cpu_set_abort_callback(be, fg_abort_cb, (void *) &flag);
    const ggml_status s0 = ggml_backend_graph_compute(be, gf);
    flag = 1;
    const ggml_status s1 = ggml_backend_graph_compute(be, gf);
    ggml_free(ctx);
    ggml_backend_free(be);
    std::fprintf(stderr, "fg_selftest_abort: flag 0 -> %d, flag 1 -> %d\n", (int) s0, (int) s1);
    return s0 == GGML_STATUS_SUCCESS && s1 == GGML_STATUS_ABORTED ? 0 : 1;
}

void fg_ctx_set_options(fg_ctx * c, const fg_options * o) { c->opts = *o; }
void fg_ctx_profile_reset(fg_ctx * c) { c->prof.clear(); }
void fg_ctx_timing(fg_ctx * c, double * out5, int reset) {
    for (int i = 0; i < 5; ++i) out5[i] = c->tim[i];
    if (reset) std::fill(c->tim, c->tim + 5, 0.0);
}
size_t fg_ctx_profile_text(const fg_ctx * c, char * buf, size_t cap) {
    std::vector<std::pair<std::string, std::pair<uint64_t, double>>> v(c->prof.begin(), c->prof.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.second > b.second.second; });
    std::string out;
    char line[256];
    for (const auto & e : v) {
        std::snprintf(line, sizeof line, "%s\t%llu\t%.3f\n", e.first.c_str(), (unsigned long long) e.second.first, e.second.second);
        out += line;
    }
    if (buf && cap) {
        std::memcpy(buf, out.c_str(), std::min(cap, out.size() + 1));
        buf[cap - 1] = 0;
    }
    return out.size() + 1;
}
void fg_vec_swiglu(int n, float * y, const float * g, const float * u) { ggml_vec_swiglu_f32(n, y, g, u); }

void fg_ctx_set_expert_fn(fg_ctx * c, fg_expert_fn fn, void * ud) {
    c->expert_fn = fn;
    c->expert_ud = ud;
}

size_t fg_ctx_state_bytes(const fg_ctx * c) { return ggml_backend_buffer_get_size(c->sbuf); }

int fg_ctx_copy_state(fg_ctx * dst, const fg_ctx * src) {
    if (dst->m != src->m || dst->n_ctx != src->n_ctx) return -1;
    const size_t n = ggml_backend_buffer_get_size(src->sbuf);
    if (ggml_backend_buffer_get_size(dst->sbuf) != n) return -2;
    std::memcpy(ggml_backend_buffer_get_base(dst->sbuf), ggml_backend_buffer_get_base(src->sbuf), n);
    dst->toks = src->toks;
    dst->fresh = src->fresh;
    return 0;
}

uint32_t fg_ctx_pos(const fg_ctx * c) { return (uint32_t) c->toks.size(); }
void * fg_ctx_state_data(fg_ctx * c) { return ggml_backend_buffer_get_base(c->sbuf); }
void fg_ctx_get_tokens(const fg_ctx * c, int32_t * out) { std::copy(c->toks.begin(), c->toks.end(), out); }
void fg_ctx_set_tokens(fg_ctx * c, const int32_t * toks, uint32_t n) {
    c->toks.assign(toks, toks + n);
    c->fresh = n == 0;
}

// the recurrent part of the state: GDN conv + ssm states and the PLE conv history (KV and the
// indexer cache stay in place: cells are positions, later writes only touch cells >= pos)
static std::vector<ggml_tensor *> recurrent_tensors(const fg_ctx * c) {
    std::vector<ggml_tensor *> v;
    for (int il = 0; il < N_LAYER; ++il) {
        if (c->r_l[il]) v.push_back(c->r_l[il]);
        if (c->s_l[il]) v.push_back(c->s_l[il]);
    }
    v.push_back(c->p_l);
    return v;
}
size_t fg_ctx_recurrent_bytes(const fg_ctx * c) {
    size_t n = 0;
    for (auto * t : recurrent_tensors(c)) n += ggml_nbytes(t);
    return n;
}
void fg_ctx_save_recurrent(const fg_ctx * c, void * buf) {
    char * p = (char *) buf;
    for (auto * t : recurrent_tensors(c)) { std::memcpy(p, t->data, ggml_nbytes(t)); p += ggml_nbytes(t); }
}
int fg_ctx_restore_recurrent(fg_ctx * c, const void * buf, uint32_t pos) {
    if (pos > c->toks.size()) return -1;
    const char * p = (const char *) buf;
    for (auto * t : recurrent_tensors(c)) { std::memcpy(t->data, p, ggml_nbytes(t)); p += ggml_nbytes(t); }
    c->toks.resize(pos);
    c->fresh = pos == 0;
    c->rs_dirty = false;
    return 0;
}


// Self-test (no model): fixed-chunk attention vs production flash attention on random tensors.
// Returns 0 if (a) the fixed-chunk result is bitwise the same at 1/3/8 threads, (b) a row computed in a
// batch of 4 rows equals the same row computed alone (verify == decode), (c) |fixed - production| is
// below 2e-3 relative to the output scale. Prints the numbers to stderr.
int fg_selftest_attn(int n_kv, int n_rows) {
    const int64_t DK = HEAD_D, NH = N_HEAD, NKV = N_HEAD_KV;
    // K + V (f16) + mask + the graphs' outputs/partials of ~9 runs; 64 MB covered n_kv <= 2304, the ctx-32768 fixture
    // needs ~180 MB
    const size_t kv_bytes = (size_t) n_kv * DK * NKV * 2;
    std::vector<uint8_t> meta(64u * 1024 * 1024 + 2 * kv_bytes + (size_t) n_kv * n_rows * 2 +
                              16 * (size_t) (2 + DK) * ((n_kv + 255) / 256) * NH * n_rows * 4);
    ggml_init_params ip = {meta.size(), meta.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (float) ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_rows, NH);        // already "permuted" layout
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, DK, n_kv, NKV);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, DK, n_kv, NKV);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_rows);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = rnd() * 3.0f;
    for (int64_t i = 0; i < ggml_nelements(k); ++i) ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(rnd());
    for (int64_t i = 0; i < ggml_nelements(v); ++i) ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(rnd());
    const int64_t valid = n_kv - 37; // causal: row r sees cells <= valid - (n_rows - 1 - r); a few holes
    for (int64_t r = 0; r < mask->ne[1]; ++r)
        for (int64_t c2 = 0; c2 < n_kv; ++c2) {
            const bool ok = r < n_rows && c2 <= valid - (n_rows - 1 - r) && (c2 % 97) != 5;
            ((ggml_fp16_t *) mask->data)[r * n_kv + c2] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
        }
    const float scale = 1.0f / sqrtf((float) DK);
    auto run = [&](bool fixed, int nth, int r0, int nr, float mode = 1.0f) {
        ggml_tensor * qv = ggml_view_3d(ctx, q, DK, nr, NH, q->nb[1], q->nb[2], r0 * q->nb[1]);
        ggml_tensor * mv = ggml_view_2d(ctx, mask, n_kv, nr, mask->nb[1], r0 * mask->nb[1]);
        ggml_tensor * out;
        if (fixed) {
            const int64_t nb = (n_kv + FG_ATTN_BLOCK - 1) / FG_ATTN_BLOCK;
            ggml_tensor * pa[4] = {qv, k, v, mv};
            ggml_tensor * part = ggml_custom_4d(ctx, GGML_TYPE_F32, 2 + DK, nb, NH, nr, pa, 4, attn_part_op, GGML_N_TASKS_MAX, nullptr);
            ((float *) part->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1] = scale;
            ((float *) part->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] = mode;
            ggml_tensor * ma[1] = {part};
            out = ggml_custom_4d(ctx, GGML_TYPE_F32, DK, NH, nr, 1, ma, 1, attn_merge_op, GGML_N_TASKS_MAX, nullptr);
        } else {
            out = ggml_flash_attn_ext(ctx, qv, k, v, mv, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(out, nullptr);
            ggml_prec_set_acc(out, GGML_PREC_F32);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        return std::vector<float>((float *) out->data, (float *) out->data + ggml_nelements(out));
    };
    const auto f1 = run(true, 1, 0, n_rows), f3 = run(true, 3, 0, n_rows), f8 = run(true, 8, 0, n_rows);
    const auto p4 = run(false, 4, 0, n_rows);
    const auto x8 = run(true, 8, 0, n_rows, 2.0f); // precise reference mode
    // two-pass mode: thread and batch (tiled vs untiled) invariance as for mode 1
    const auto t1 = run(true, 1, 0, n_rows, 3.0f), t8 = run(true, 8, 0, n_rows, 3.0f), tl = run(true, 4, n_rows - 1, 1, 3.0f);
    const bool two_ok = std::memcmp(t1.data(), t8.data(), t1.size() * 4) == 0 &&
                        std::memcmp(tl.data(), t1.data() + (size_t) (n_rows - 1) * DK * NH, tl.size() * 4) == 0;
    bool thr_same = std::memcmp(f1.data(), f3.data(), f1.size() * 4) == 0 && std::memcmp(f1.data(), f8.data(), f1.size() * 4) == 0;
    // last row alone vs in the batch
    const auto last = run(true, 4, n_rows - 1, 1);
    bool row_same = std::memcmp(last.data(), f1.data() + (size_t) (n_rows - 1) * DK * NH, last.size() * 4) == 0;
    double maxd = 0, maxv = 0;
    for (size_t i = 0; i < f1.size(); ++i) {
        maxd = std::max(maxd, (double) std::fabs(f1[i] - p4[i]));
        maxv = std::max(maxv, (double) std::fabs(p4[i]));
    }
    // f64 reference (Q rounded to f16 as both kernels do): which side carries the difference
    double err_fixed = 0, err_prod = 0, err_precise = 0;
    {
        std::vector<double> sc(n_kv), o(DK);
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t h = 0; h < NH; ++h) {
                const int64_t hk = h / (NH / NKV);
                const float * pq = (const float *) q->data + h * DK * n_rows + r * DK;
                double M = -INFINITY;
                for (int64_t c2 = 0; c2 < n_kv; ++c2) {
                    if (ggml_fp16_to_fp32(((ggml_fp16_t *) mask->data)[r * n_kv + c2]) == -INFINITY) { sc[c2] = -INFINITY; continue; }
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) k->data + hk * DK * n_kv + c2 * DK;
                    double d = 0;
                    for (int64_t i = 0; i < DK; ++i) d += (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(pq[i])) * ggml_fp16_to_fp32(kr[i]);
                    sc[c2] = d * scale;
                    M = std::max(M, sc[c2]);
                }
                double S = 0;
                std::fill(o.begin(), o.end(), 0.0);
                for (int64_t c2 = 0; c2 < n_kv; ++c2) {
                    if (sc[c2] == -INFINITY) continue;
                    const double w = std::exp(sc[c2] - M);
                    S += w;
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) v->data + hk * DK * n_kv + c2 * DK;
                    for (int64_t i = 0; i < DK; ++i) o[i] += w * ggml_fp16_to_fp32(vr[i]);
                }
                const size_t base = (size_t) r * DK * NH + (size_t) h * DK;
                for (int64_t i = 0; i < DK; ++i) {
                    err_fixed = std::max(err_fixed, std::fabs(f1[base + i] - o[i] / S));
                    err_prod = std::max(err_prod, std::fabs(p4[base + i] - o[i] / S));
                    // the precise mode uses f32 Q (not f16): compare it with an f32-Q reference below
                    (void) 0;
                }
            }
    }
    {
        std::vector<double> sc(n_kv), o(DK);
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t h = 0; h < NH; ++h) {
                const int64_t hk = h / (NH / NKV);
                const float * pq = (const float *) q->data + h * DK * n_rows + r * DK;
                double M = -INFINITY, S = 0;
                for (int64_t c2 = 0; c2 < n_kv; ++c2) {
                    if (ggml_fp16_to_fp32(((ggml_fp16_t *) mask->data)[r * n_kv + c2]) == -INFINITY) { sc[c2] = -INFINITY; continue; }
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) k->data + hk * DK * n_kv + c2 * DK;
                    double d = 0;
                    for (int64_t i = 0; i < DK; ++i) d += (double) pq[i] * ggml_fp16_to_fp32(kr[i]);
                    sc[c2] = d * scale;
                    M = std::max(M, sc[c2]);
                }
                std::fill(o.begin(), o.end(), 0.0);
                for (int64_t c2 = 0; c2 < n_kv; ++c2) {
                    if (sc[c2] == -INFINITY) continue;
                    const double w = std::exp(sc[c2] - M);
                    S += w;
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) v->data + hk * DK * n_kv + c2 * DK;
                    for (int64_t i = 0; i < DK; ++i) o[i] += w * ggml_fp16_to_fp32(vr[i]);
                }
                const size_t base = (size_t) r * DK * NH + (size_t) h * DK;
                for (int64_t i = 0; i < DK; ++i) err_precise = std::max(err_precise, std::fabs(x8[base + i] - o[i] / S));
            }
    }
    std::fprintf(stderr, "fg_selftest_attn precise mode vs f64 (f32 Q): %.3e\n", err_precise);
    std::fprintf(stderr, "fg_selftest_attn n_kv %d rows %d: threads 1/3/8 bitwise %d, row alone == in batch %d, max|fixed-prod| %.3e (max|out| %.3e); vs f64: fixed %.3e, prod %.3e\n",
                 n_kv, n_rows, (int) thr_same, (int) row_same, maxd, maxv, err_fixed, err_prod);
    ggml_free(ctx);
    std::fprintf(stderr, "fg_selftest_attn twopass mode: threads + row alone == in batch %d\n", (int) two_ok);
    return thr_same && row_same && two_ok && maxd <= 2e-3 * std::max(1.0, maxv) && err_precise <= 1e-5 ? 0 : 1;
}


// compact state for persistence: KV / indexer rows of the cells < pos (cells are positions) + the
// recurrent tensors whole. Token history is the caller's (fg_ctx_get_tokens / fg_ctx_set_tokens).
static void compact_walk(fg_ctx * c, uint32_t pos, const std::function<void(char *, size_t)> & f) {
    for (int il = 0; il < N_LAYER; ++il) {
        for (ggml_tensor * t : {c->k_l[il], c->v_l[il], c->ik_l[il]}) {
            if (t) f((char *) t->data, (size_t) pos * t->nb[1]);
        }
    }
    for (auto * t : recurrent_tensors(c)) f((char *) t->data, ggml_nbytes(t));
}
size_t fg_ctx_compact_bytes(fg_ctx * c, uint32_t pos) {
    size_t n = 0;
    compact_walk(c, pos, [&](char *, size_t b) { n += b; });
    return n;
}
void fg_ctx_save_compact(fg_ctx * c, void * buf) {
    char * p = (char *) buf;
    compact_walk(c, (uint32_t) c->toks.size(), [&](char * d, size_t b) { std::memcpy(p, d, b); p += b; });
}
// FP-2 (v0.1.3): the bytes of fg_ctx_save_compact region by region through `cb` (no buffer of the whole state);
// stops at the first non-zero return of `cb` and returns it.
int fg_ctx_save_compact_cb(fg_ctx * c, int (*cb)(void * user, const char * data, size_t n), void * user) {
    int rc = 0;
    compact_walk(c, (uint32_t) c->toks.size(), [&](char * d, size_t b) {
        if (rc == 0) rc = cb(user, d, b);
    });
    return rc;
}
int fg_ctx_load_compact(fg_ctx * c, const void * buf, uint32_t pos) {
    if (pos > c->n_ctx) return -1;
    const char * p = (const char *) buf;
    compact_walk(c, pos, [&](char * d, size_t b) { std::memcpy(d, p, b); p += b; });
    return 0;
}

int32_t fg_vec_dot_type(int32_t type) { return (int32_t) ggml_get_type_traits_cpu((ggml_type) type)->vec_dot_type; }
size_t fg_row_size(int32_t type, int64_t n) { return ggml_row_size((ggml_type) type, n); }
void fg_from_float(int32_t vdt, const float * x, void * y, int64_t n) { ggml_get_type_traits_cpu((ggml_type) vdt)->from_float(x, y, n); }
void fg_vec_dot(int32_t type, int n, float * s, const void * x, const void * y) {
    ggml_get_type_traits_cpu((ggml_type) type)->vec_dot(n, s, 0, x, 0, y, 0, 1);
}


// ---- S-b executor (fg_options.executor): the cached one-token graph run by production ggml's own op functions in one
// OpenMP region, with a barrier only where needed. Every thread runs every scheduled node in graph order with ggml's
// (ith, nth) partition, so each op computes exactly what it computes under ggml_graph_compute: same bits. A barrier
// goes before node j iff j reads a byte range written since the last barrier, writes a range read or written since
// then, or j touches shared state (the work buffer / the threadpool's chunk counter) after another such node did.
// Fusion follows ggml_cpu_try_fuse_ops (RMS_NORM + MUL) exactly. Unknown ops => the graph falls back to ggml.
static bool ex_supported(ggml_op op) {
    switch (op) {
#define FG_CASE(name) case GGML_OP_##name:
        FG_CASE(DUP) FG_CASE(ADD) FG_CASE(ADD1) FG_CASE(SUB) FG_CASE(MUL) FG_CASE(DIV) FG_CASE(SQR) FG_CASE(SQRT) FG_CASE(LOG)
        FG_CASE(SUM) FG_CASE(SUM_ROWS) FG_CASE(MEAN) FG_CASE(ARGMAX) FG_CASE(REPEAT) FG_CASE(CONCAT) FG_CASE(NORM)
        FG_CASE(RMS_NORM) FG_CASE(L2_NORM) FG_CASE(MUL_MAT) FG_CASE(SCALE) FG_CASE(SET) FG_CASE(CPY) FG_CASE(CONT)
        FG_CASE(GET_ROWS) FG_CASE(SET_ROWS) FG_CASE(SOFT_MAX) FG_CASE(ROPE) FG_CASE(CLAMP) FG_CASE(ARGSORT) FG_CASE(TOP_K)
        FG_CASE(FILL) FG_CASE(FLASH_ATTN_EXT) FG_CASE(SSM_CONV) FG_CASE(UNARY) FG_CASE(GLU) FG_CASE(GATED_DELTA_NET)
        FG_CASE(LIGHTNING_INDEXER) FG_CASE(DSV4_HC_COMB) FG_CASE(DSV4_HC_PRE) FG_CASE(DSV4_HC_POST) FG_CASE(TOP_K_MASK)
        FG_CASE(CUSTOM)
#undef FG_CASE
            return true;
        default:
            return false;
    }
}

static void ex_forward(ggml_compute_params * p, ggml_tensor * t) {
    if (ggml_cpu_extra_compute_forward(p, t)) return;
    switch (t->op) {
#define FG_CALL(NAME, name) case GGML_OP_##NAME: ggml_compute_forward_##name(p, t); break;
        FG_CALL(DUP, dup) FG_CALL(ADD, add) FG_CALL(ADD1, add1) FG_CALL(SUB, sub) FG_CALL(MUL, mul) FG_CALL(DIV, div)
        FG_CALL(SQR, sqr) FG_CALL(SQRT, sqrt) FG_CALL(LOG, log) FG_CALL(SUM, sum) FG_CALL(SUM_ROWS, sum_rows) FG_CALL(MEAN, mean)
        FG_CALL(ARGMAX, argmax) FG_CALL(REPEAT, repeat) FG_CALL(CONCAT, concat) FG_CALL(NORM, norm) FG_CALL(RMS_NORM, rms_norm)
        FG_CALL(L2_NORM, l2_norm) FG_CALL(MUL_MAT, mul_mat) FG_CALL(SCALE, scale) FG_CALL(SET, set) FG_CALL(CPY, cpy)
        FG_CALL(CONT, cont) FG_CALL(GET_ROWS, get_rows) FG_CALL(SET_ROWS, set_rows) FG_CALL(SOFT_MAX, soft_max)
        FG_CALL(ROPE, rope) FG_CALL(CLAMP, clamp) FG_CALL(ARGSORT, argsort) FG_CALL(TOP_K, top_k) FG_CALL(FILL, fill)
        FG_CALL(FLASH_ATTN_EXT, flash_attn_ext) FG_CALL(SSM_CONV, ssm_conv) FG_CALL(UNARY, unary) FG_CALL(GLU, glu)
        FG_CALL(GATED_DELTA_NET, gated_delta_net) FG_CALL(LIGHTNING_INDEXER, lightning_indexer)
        FG_CALL(DSV4_HC_COMB, dsv4_hc_comb) FG_CALL(DSV4_HC_PRE, dsv4_hc_pre) FG_CALL(DSV4_HC_POST, dsv4_hc_post)
        FG_CALL(TOP_K_MASK, top_k_mask) FG_CALL(CUSTOM, custom)
#undef FG_CALL
        default: GGML_ABORT("fg executor: unsupported op %s", ggml_op_desc(t));
    }
}

// ops that never touch the shared work buffer or the threadpool (checked against ops.cpp at eae95db: no wdata /
// threadpool / ggml_barrier use in these paths); everything else is "shared" and serialised against each other
static bool ex_pure(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_ADD: case GGML_OP_ADD1: return !ggml_is_quantized(t->src[0]->type);
        case GGML_OP_DUP: case GGML_OP_CPY: case GGML_OP_CONT: return !ggml_is_quantized(t->type);
        case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV: case GGML_OP_SQR: case GGML_OP_SQRT: case GGML_OP_LOG:
        case GGML_OP_SUM_ROWS: case GGML_OP_REPEAT: case GGML_OP_CONCAT: case GGML_OP_NORM: case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM: case GGML_OP_SCALE: case GGML_OP_GET_ROWS: case GGML_OP_CLAMP: case GGML_OP_ARGSORT:
        case GGML_OP_FILL: case GGML_OP_SSM_CONV: case GGML_OP_UNARY: case GGML_OP_GLU: case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE: case GGML_OP_DSV4_HC_POST: case GGML_OP_TOP_K_MASK:
        case GGML_OP_CUSTOM: // the engine's ops: own state, ordered through their tensor edges (fetch -> up/gate)
            return true;
        default:
            return false;
    }
}

static bool ex_is_noop(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static void ex_build(fg_ctx * c, ggml_cgraph * gf, int nth) {
    auto & d = *c->cur_dc;
    d.ex_built = true;
    d.ex_ok = false;
    d.ex_node.clear();
    d.ex_fused.clear();
    d.ex_barrier.clear();
    const int nn = ggml_graph_n_nodes(gf);
    // use counts as ggml_visit_parents counts them: one per src slot of every node
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < nn; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        for (int k = 0; k < GGML_MAX_SRC; ++k)
            if (t->src[k]) uses[t->src[k]]++;
    }
    static const bool no_fusion = [] { const char * e = getenv("GGML_CPU_DISABLE_FUSION"); return e && atoi(e) == 1; }();
    struct R { uintptr_t a, b; };
    std::vector<R> rd, wr;
    bool shared_since = false, first = true;
    auto overlap = [](const std::vector<R> & v, uintptr_t a, uintptr_t b) {
        for (const R & r : v)
            if (a < r.b && r.a < b) return true;
        return false;
    };
    auto rng = [](const ggml_tensor * t) { const uintptr_t a = (uintptr_t) t->data; return R{a, a + ggml_nbytes(t)}; };
    int barriers = 0;
    for (int i = 0; i < nn; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (ex_is_noop(t->op) || (t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(t)) continue;
        if (!ex_supported(t->op)) return; // fall back to ggml for this graph
        int fused = -1;
        if (!no_fusion && t->op == GGML_OP_RMS_NORM && i + 1 < nn) {
            ggml_tensor * m = ggml_graph_node(gf, i + 1);
            auto it = uses.find(t);
            const bool one_use = it != uses.end() && it->second == 1 && !t->view_src && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
            if (m->op == GGML_OP_MUL && (m->flags & GGML_TENSOR_FLAG_COMPUTE) && one_use && (m->src[0] == t || m->src[1] == t) &&
                ggml_are_same_shape(m, t)) {
                const ggml_tensor * w = m->src[0] == t ? m->src[1] : m->src[0];
                if (t->src[0]->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && w->ne[0] == t->ne[0] &&
                    w->nb[0] == sizeof(float))
                    fused = i + 1;
            }
        }
        std::vector<R> r_in, w_out;
        for (int k = 0; k < GGML_MAX_SRC; ++k)
            if (t->src[k] && t->src[k]->data) r_in.push_back(rng(t->src[k]));
        w_out.push_back(rng(t));
        bool shared = !ex_pure(t);
        if (fused >= 0) {
            ggml_tensor * m = ggml_graph_node(gf, fused);
            for (int k = 0; k < GGML_MAX_SRC; ++k)
                if (m->src[k] && m->src[k] != t && m->src[k]->data) r_in.push_back(rng(m->src[k]));
            w_out.push_back(rng(m));
        }
        bool need = first; // the inputs were written by the host before the region: no barrier needed, but keep node 0 simple
        need = false;
        for (const R & r : r_in) need |= overlap(wr, r.a, r.b);
        for (const R & r : w_out) need |= overlap(wr, r.a, r.b) || overlap(rd, r.a, r.b);
        need |= shared && shared_since;
        if (need) {
            rd.clear();
            wr.clear();
            shared_since = false;
            ++barriers;
        }
        first = false;
        rd.insert(rd.end(), r_in.begin(), r_in.end());
        wr.insert(wr.end(), w_out.begin(), w_out.end());
        shared_since |= shared;
        d.ex_node.push_back(i);
        d.ex_fused.push_back(fused);
        d.ex_barrier.push_back(need);
        if (fused >= 0) ++i;
    }
    // work buffer and the threadpool whose n_graph (= nth) ggml_barrier and mul_mat's chunk counter read
    if (!c->ex_tp || c->ex_tp_nth != nth) {
        if (c->ex_tp) ggml_threadpool_free(c->ex_tp);
        ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
        c->ex_tp = ggml_threadpool_new(&tpp);
        c->ex_tp_nth = nth;
        // one small compute sets tp->n_graph = nth (ggml_graph_compute does it under OpenMP). It must be a node ggml
        // plans with n_tasks = n_threads (ADD), else the plan runs 1 thread, n_graph becomes 1 and ggml_barrier turns
        // into a no-op inside ggml's ops (selftest: mul_mat read its quantised src1 before it was written)
        std::vector<uint8_t> cm(64 * 1024);
        ggml_init_params cp = {cm.size(), cm.data(), false};
        ggml_context * cc = ggml_init(cp);
        ggml_tensor * x = ggml_new_tensor_1d(cc, GGML_TYPE_F32, 4096);
        std::memset(x->data, 0, ggml_nbytes(x));
        ggml_cgraph * g1 = ggml_new_graph_custom(cc, 1, false);
        ggml_graph_add_node(g1, ggml_add(cc, x, x));
        ggml_cplan cpl = ggml_graph_plan(g1, nth, c->ex_tp);
        GGML_ASSERT(cpl.n_threads == nth && "executor threadpool init must run nth threads");
        std::vector<uint8_t> w(cpl.work_size + 64);
        cpl.work_data = w.data();
        ggml_graph_compute(g1, &cpl);
        ggml_free(cc);
    }
    ggml_cplan cplan = ggml_graph_plan(gf, nth, c->ex_tp);
    d.ex_work.assign(cplan.work_size + 64, 0);
    d.ex_nth = nth;
    d.ex_barriers = barriers;
    d.ex_ok = true;
    c->ex_stats[0] += 1;
    c->ex_stats[1] = (double) d.ex_node.size();
    c->ex_stats[2] = barriers;
}

static std::string node_kind(const ggml_tensor * t) {
    std::string kind = t->name;
    const size_t dash = kind.rfind('-');
    if (dash != std::string::npos && dash + 1 < kind.size() && std::isdigit((unsigned char) kind[dash + 1])) kind.resize(dash);
    if (kind.empty() || kind.rfind("node_", 0) == 0 || kind.find("(view)") != std::string::npos ||
        kind.find("(reshaped)") != std::string::npos || kind.find("(permuted)") != std::string::npos ||
        kind.find("(copy") != std::string::npos)
        kind = std::string("op:") + ggml_op_desc(t);
    return kind;
}

static void ex_run(fg_ctx * c, ggml_cgraph * gf) {
    auto & d = *c->cur_dc;
    const int nth = d.ex_nth;
    const int n = (int) d.ex_node.size();
    void * wdata = d.ex_work.data();
    const size_t wsize = d.ex_work.size() - 64;
    const bool tr = c->trace != 0;
    // FD-1: per (thread, scheduled node) work and graph-barrier wait in ns; a timestamp pair per node and thread
    thread_local std::vector<uint64_t> dummy;
    static std::vector<uint64_t> work, waitv;
    if (tr) {
        work.assign((size_t) nth * n, 0);
        waitv.assign((size_t) nth * n, 0);
    }
    const auto r0 = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(nth)
    {
        const int ith = omp_get_thread_num();
        ggml_compute_params p = {ith, nth, wsize, wdata, c->ex_tp, false};
        for (int k = 0; k < n; ++k) {
            if (d.ex_barrier[k]) {
                if (tr) {
                    const auto b0 = std::chrono::steady_clock::now();
#pragma omp barrier
                    waitv[(size_t) ith * n + k] += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - b0).count();
                } else {
#pragma omp barrier
                }
            }
            ggml_tensor * t = ggml_graph_node(gf, d.ex_node[k]);
            const auto w0 = tr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (d.ex_fused[k] >= 0) ggml_compute_forward_rms_norm_mul_fused(&p, t, ggml_graph_node(gf, d.ex_fused[k]));
            else ex_forward(&p, t);
            if (tr) work[(size_t) ith * n + k] += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
        }
    }
    if (tr) {
        c->tr_region_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
        for (int k = 0; k < n; ++k) {
            double wk = 0, wt = 0;
            for (int i = 0; i < nth; ++i) {
                wk += work[(size_t) i * n + k];
                wt += waitv[(size_t) i * n + k];
            }
            auto & e = c->tr_kinds[node_kind(ggml_graph_node(gf, d.ex_node[k]))];
            e[0] += 1;
            e[1] += wk / nth / 1e6; // mean over threads, ms
            e[2] += wt / nth / 1e6;
        }
        c->tr_tokens += 1;
    }
    (void) dummy;
}


// model-free check of the S-b executor: a decode-shaped synthetic graph (quantised matvecs, RMS_NORM+MUL fusion,
// independent branches, softmax / argsort / get_rows / cpy / concat / unary / glu) computed by ggml_graph_compute and by
// the executor, compared bitwise; 0 = pass. Also reports nodes / barriers of the schedule.
int fg_selftest_exec(int nth) {
    const size_t mem = 256u << 20;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0x2545F4914F6CDD1Dull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (float) ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    auto f32 = [&](int64_t a, int64_t b) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b); for (int64_t i = 0; i < a * b; ++i) ((float *) t->data)[i] = rnd(); return t; };
    auto qmat = [&](ggml_type ty, int64_t k, int64_t n) {
        std::vector<float> w(k * n);
        for (auto & v : w) v = rnd() * 0.05f;
        ggml_tensor * t = ggml_new_tensor_2d(ctx, ty, k, n);
        ggml_quantize_chunk(ty, w.data(), t->data, 0, n, k, nullptr);
        return t;
    };
    const int64_t E = 2560;
    ggml_tensor * inp = f32(E, 1), * nw = f32(E, 1);
    ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, inp, 1e-6f), nw);        // fused pair
    ggml_tensor * a = ggml_mul_mat(ctx, qmat(GGML_TYPE_Q8_0, E, 640), xn);
    ggml_tensor * b = ggml_mul_mat(ctx, qmat(GGML_TYPE_Q6_K, E, 640), xn);       // independent of a
    ggml_tensor * g = ggml_swiglu_split(ctx, a, b);
    ggml_tensor * d = ggml_mul_mat(ctx, qmat(GGML_TYPE_Q8_0, 640, E), g);
    ggml_tensor * r = ggml_add(ctx, inp, ggml_scale(ctx, d, 0.5f));
    ggml_tensor * logits = ggml_mul_mat(ctx, qmat(GGML_TYPE_Q8_0, E, 512), ggml_mul(ctx, ggml_rms_norm(ctx, r, 1e-6f), nw));
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_tensor * top = ggml_argsort_top_k(ctx, probs, 10);
    ggml_tensor * rows = ggml_get_rows(ctx, ggml_reshape_2d(ctx, f32(E, 512), E, 512), top);
    ggml_tensor * s1 = ggml_sigmoid(ctx, ggml_sum_rows(ctx, rows));
    ggml_tensor * c1 = ggml_concat(ctx, ggml_silu(ctx, r), ggml_cont(ctx, ggml_view_2d(ctx, rows, E, 1, rows->nb[1], 0)), 1);
    ggml_tensor * out = ggml_add(ctx, ggml_sum_rows(ctx, c1), ggml_sum_rows(ctx, ggml_reshape_2d(ctx, s1, 10, 1)));
    ggml_tensor * keep = ggml_cpy(ctx, r, f32(E, 1));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, keep);
    ggml_graph_compute_with_ctx(ctx, gf, nth);
    // every node's bytes after the reference compute, to name the first node the executor gets wrong
    std::vector<std::vector<uint8_t>> ref_nodes(ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->data) ref_nodes[i].assign((uint8_t *) t->data, (uint8_t *) t->data + ggml_nbytes(t));
    }
    std::vector<float> ref_out((float *) out->data, (float *) out->data + ggml_nelements(out));
    std::vector<float> ref_keep((float *) keep->data, (float *) keep->data + ggml_nelements(keep));
    std::vector<float> ref_logits((float *) logits->data, (float *) logits->data + ggml_nelements(logits));
    fg_ctx tmp;
    ex_build(&tmp, gf, nth);
    bool same = tmp.dcs[0].ex_ok;
    if (same) {
        for (int rep = 0; rep < 3; ++rep) {
            ex_run(&tmp, gf);
            same &= std::memcmp(ref_out.data(), out->data, ref_out.size() * 4) == 0 &&
                    std::memcmp(ref_keep.data(), keep->data, ref_keep.size() * 4) == 0 &&
                    std::memcmp(ref_logits.data(), logits->data, ref_logits.size() * 4) == 0;
        }
    }
    if (!same) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->data && !ref_nodes[i].empty() && std::memcmp(ref_nodes[i].data(), t->data, ref_nodes[i].size()) != 0) {
                int k = -1;
                for (size_t j = 0; j < tmp.dcs[0].ex_node.size(); ++j) if (tmp.dcs[0].ex_node[j] == i || tmp.dcs[0].ex_fused[j] == i) k = (int) j;
                std::fprintf(stderr, "fg_selftest_exec: first differing node %d %s (%s), sched idx %d barrier %d\n", i, t->name,
                             ggml_op_desc(t), k, k >= 0 ? (int) tmp.dcs[0].ex_barrier[k] : -1);
                break;
            }
        }
    }
    std::fprintf(stderr, "fg_selftest_exec nth %d: schedule ok %d, %zu nodes, %d barriers, bitwise == ggml_graph_compute %d\n", nth,
                 (int) tmp.dcs[0].ex_ok, tmp.dcs[0].ex_node.size(), tmp.dcs[0].ex_barriers, (int) same);
    if (tmp.ex_tp) ggml_threadpool_free(tmp.ex_tp);
    tmp.ex_tp = nullptr;
    ggml_free(ctx);
    return same ? 0 : 1;
}

// FD-1: tracing on/off (forces nothing else: the caller runs with fg_options.executor = 1); text = per kind
// "kind\tcalls\twork_ms\twait_ms" summed over the traced decodes (divide by tokens), then "#" lines with totals
void fg_ctx_set_trace(fg_ctx * c, int on) {
    c->trace = on;
    fuse_trace_on().store(on != 0);
    if (on) {
        c->tr_kinds.clear();
        c->tr_tokens = c->tr_region_ms = 0;
        fuse_spin_ns().store(0);
        fuse_waits().store(0);
    }
}
size_t fg_ctx_trace_text(const fg_ctx * c, char * buf, size_t cap) {
    std::string out;
    char line[512];
    for (const auto & kv : c->tr_kinds) {
        std::snprintf(line, sizeof line, "%s\t%.0f\t%.4f\t%.4f\n", kv.first.c_str(), kv.second[0], kv.second[1], kv.second[2]);
        out += line;
    }
    std::snprintf(line, sizeof line, "#tokens\t%.0f\n#region_ms\t%.4f\n#fuse_spin_ms_all_threads\t%.4f\n#fuse_waits_all_threads\t%llu\n",
                  c->tr_tokens, c->tr_region_ms, fuse_spin_ns().load() / 1e6, (unsigned long long) fuse_waits().load());
    out += line;
    static const char * names[6] = {"hc_mix", "hc_combine", "gdn", "router", "shexp_combine", "attn"};
    for (int i = 0; i < 6; ++i) {
        std::snprintf(line, sizeof line, "#fused\t%s\t%llu\t%llu\n", names[i], (unsigned long long) c->fuse_built[i],
                      (unsigned long long) c->fuse_fallback[i]);
        out += line;
    }
    if (buf && cap) {
        std::memcpy(buf, out.c_str(), std::min(cap, out.size() + 1));
        buf[cap - 1] = 0;
    }
    return out.size() + 1;
}

void fg_ctx_exec_stats(const fg_ctx * c, double * out3) {
    for (int i = 0; i < 3; ++i) out3[i] = c->ex_stats[i];
}


// Model-free dense matvec bandwidth bench at Flash's real shapes/types (036 post-mortem: where do the ~52 ms of dense
// matvecs go vs the 42 ms bandwidth floor?). One GDN-layer and one attention-layer matrix set (types from the GGUF
// header), replicated `copies` times so the working set exceeds the caches. Measures, per pass over all matrices:
//  A  ggml one-token mul_mat graph (one node per matrix, ggml_graph_compute with nth threads: the engine's path)
//  B  one static row partition over all matrices, no barriers between matrices (the upper bound for fused phases)
//  C  raw read bandwidth of the same weight bytes (nth threads, 64-bit loads summed)
// Prints GB/s and the implied ms for the real 3.93 GB of dense weights per token.
void fg_bench_dense(int nth, int copies, int reps) {
    struct M { ggml_type t; int64_t k, n; };
    const M gdn[] = {{GGML_TYPE_Q6_K, 2560, 6144}, {GGML_TYPE_Q6_K, 2560, 10240}, {GGML_TYPE_Q6_K, 6144, 2560},
                     {GGML_TYPE_Q8_0, 640, 2560}, {GGML_TYPE_F32, 2560, 512}, {GGML_TYPE_Q6_K, 2560, 640}, {GGML_TYPE_Q6_K, 2560, 640},
                     {GGML_TYPE_Q8_0, 10240, 320}, {GGML_TYPE_Q8_0, 320, 10240}, {GGML_TYPE_Q8_0, 10240, 320}, {GGML_TYPE_Q8_0, 320, 10240},
                     {GGML_TYPE_F32, 2560, 48}, {GGML_TYPE_F32, 2560, 48}};
    const M att[] = {{GGML_TYPE_Q6_K, 2560, 12288}, {GGML_TYPE_Q6_K, 2560, 512}, {GGML_TYPE_Q6_K, 2560, 512}, {GGML_TYPE_Q6_K, 6144, 2560},
                     {GGML_TYPE_Q8_0, 640, 2560}, {GGML_TYPE_F32, 2560, 512}, {GGML_TYPE_Q6_K, 2560, 640}, {GGML_TYPE_Q6_K, 2560, 640},
                     {GGML_TYPE_Q8_0, 10240, 320}, {GGML_TYPE_Q8_0, 320, 10240}, {GGML_TYPE_Q8_0, 10240, 320}, {GGML_TYPE_Q8_0, 320, 10240}};
    std::vector<M> all;
    for (int c = 0; c < copies; ++c) {
        for (int r = 0; r < 3; ++r) for (const M & m : gdn) all.push_back(m);  // 3 GDN : 1 attention, as the model
        for (const M & m : att) all.push_back(m);
    }
    size_t wbytes = 0, maxk = 0;
    for (const M & m : all) { wbytes += ggml_row_size(m.t, m.k) * m.n; maxk = std::max<size_t>(maxk, m.k); }
    const size_t mem = wbytes + all.size() * (maxk * 4 + 4 * 16384 + 4096) + (64u << 20);
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (float) ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    std::vector<float> tmp;
    std::vector<ggml_tensor *> W, X, Y;
    std::map<int64_t, ggml_tensor *> xin;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (const M & m : all) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, m.t, m.k, m.n);
        tmp.resize((size_t) m.k * m.n);
        // only a few distinct rows of random data (quantising 1 GB is slow): fill, then repeat the first 64 rows
        const int64_t base = std::min<int64_t>(m.n, 64);
        for (int64_t i = 0; i < m.k * base; ++i) tmp[i] = rnd() * 0.05f;
        if (m.t == GGML_TYPE_F32) {
            for (int64_t r = 0; r < m.n; ++r) std::memcpy((float *) w->data + r * m.k, tmp.data() + (r % base) * m.k, m.k * 4);
        } else {
            ggml_quantize_chunk(m.t, tmp.data(), w->data, 0, base, m.k, nullptr);
            const size_t rb = ggml_row_size(m.t, m.k);
            for (int64_t r = base; r < m.n; ++r) std::memcpy((char *) w->data + r * rb, (char *) w->data + (r % base) * rb, rb);
        }
        if (!xin.count(m.k)) {
            ggml_tensor * xv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m.k);
            for (int64_t i = 0; i < m.k; ++i) ((float *) xv->data)[i] = rnd();
            xin[m.k] = xv;
        }
        ggml_tensor * y = ggml_mul_mat(ctx, w, xin[m.k]);
        ggml_build_forward_expand(gf, y);
        W.push_back(w);
        Y.push_back(y);
    }
    const double gb = wbytes / 1e9;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    // A: ggml graph
    ggml_graph_compute_with_ctx(ctx, gf, nth); // warm
    double tA = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto a = now();
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        tA = std::min(tA, ms(a, now()));
    }
    // B: static rows over all matrices, one parallel region, no barriers between matrices
    std::vector<std::vector<uint8_t>> xq(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
        const auto * tr = ggml_get_type_traits_cpu(all[i].t);
        const ggml_type vt = tr->vec_dot_type;
        xq[i].resize(ggml_row_size(vt, all[i].k));
        if (vt == GGML_TYPE_F32) std::memcpy(xq[i].data(), xin[all[i].k]->data, all[i].k * 4);
        else ggml_get_type_traits_cpu(vt)->from_float((float *) xin[all[i].k]->data, xq[i].data(), all[i].k);
    }
    std::vector<int64_t> rowoff(all.size() + 1, 0);
    for (size_t i = 0; i < all.size(); ++i) rowoff[i + 1] = rowoff[i] + all[i].n;
    const int64_t R = rowoff.back();
    std::vector<float> out(R);
    auto runB = [&]() {
#pragma omp parallel num_threads(nth)
        {
            const int ith = omp_get_thread_num();
            const int64_t r0 = R * ith / nth, r1 = R * (ith + 1) / nth;
            size_t mi = std::upper_bound(rowoff.begin(), rowoff.end(), r0) - rowoff.begin() - 1;
            for (int64_t r = r0; r < r1; ++r) {
                while (r >= rowoff[mi + 1]) ++mi;
                const ggml_tensor * w = W[mi];
                const auto * tr = ggml_get_type_traits_cpu(w->type);
                tr->vec_dot((int) all[mi].k, &out[r], 0, (const char *) w->data + (r - rowoff[mi]) * w->nb[1], 0, xq[mi].data(), 0, 1);
            }
        }
    };
    runB();
    double tB = 1e30;
    for (int r = 0; r < reps; ++r) { const auto a = now(); runB(); tB = std::min(tB, ms(a, now())); }
    // C: raw read bandwidth over the same bytes
    std::vector<std::pair<const uint64_t *, size_t>> spans;
    for (ggml_tensor * w : W) spans.push_back({(const uint64_t *) w->data, ggml_nbytes(w) / 8});
    size_t total_w = 0;
    for (auto & sp : spans) total_w += sp.second;
    volatile uint64_t sink = 0;
    auto runC = [&]() {
        uint64_t acc_all = 0;
#pragma omp parallel num_threads(nth) reduction(+ : acc_all)
        {
            const int ith = omp_get_thread_num();
            const size_t w0 = total_w * ith / nth, w1 = total_w * (ith + 1) / nth;
            size_t pos = 0;
            uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for (auto & sp : spans) {
                const size_t a = std::max(w0, pos), b = std::min(w1, pos + sp.second);
                for (size_t i = a; i + 3 < b; i += 4) {
                    acc0 += sp.first[i - pos]; acc1 += sp.first[i - pos + 1]; acc2 += sp.first[i - pos + 2]; acc3 += sp.first[i - pos + 3];
                }
                pos += sp.second;
            }
            acc_all += acc0 + acc1 + acc2 + acc3;
        }
        sink = acc_all;
    };
    runC();
    double tC = 1e30;
    for (int r = 0; r < reps; ++r) { const auto a = now(); runC(); tC = std::min(tC, ms(a, now())); }
    const double per_tok = 3.93; // GB of dense weights per token (GGUF header: 36 GDN + 12 attention layers + output)
    std::fprintf(stderr, "fg_bench_dense nth %d: %zu matrices, %.3f GB/pass | A ggml graph %.2f ms = %.1f GB/s (-> %.1f ms/token) | "
                 "B static rows %.2f ms = %.1f GB/s (-> %.1f ms) | C raw read %.2f ms = %.1f GB/s (-> %.1f ms)\n",
                 nth, all.size(), gb, tA, gb / tA * 1e3, per_tok / (gb / tA * 1e3) * 1e3, tB, gb / tB * 1e3, per_tok / (gb / tB * 1e3) * 1e3,
                 tC, gb / tC * 1e3, per_tok / (gb / tC * 1e3) * 1e3);
    (void) sink;
    ggml_free(ctx);
}

// ext (T2): vmode 1 = verify (logits_all n x N_VOCAB, h_all n x N_EMBD*HC, recurrent state left to fg_commit),
// vmode 2 = prefill with h of every row (logits = last row). nullptr = the plain path, untouched.
struct fg_ext { int vmode; float * logits_all; float * h_all; };
static int fg_decode_impl(fg_ctx * c, const int32_t * tokens, int n_tokens, int n_threads, float * logits, const fg_dump * dump, const fg_ext * ext) {
    fg_model & m = *c->m;
    const uint32_t p0 = (uint32_t) c->toks.size();
    if (n_tokens <= 0 || p0 + n_tokens > c->n_ctx) return -1;
    if (c->rs_dirty) return -7;                          // an aborted decode was not rolled back
    if (c->abort_flag && *c->abort_flag) return -6;      // cancelled before it started: nothing touched
    if (c->pend.active) return -4; // a verify must be committed first
    const int vmode = ext ? ext->vmode : 0;
    if (vmode == 1 && (c->fresh || n_tokens > 16)) return -5;
    for (int i = 0; i < n_tokens; ++i) c->toks.push_back(tokens[i]);
    const uint32_t n_cells = p0 + n_tokens;

    // sizes the graph depends on (llama_kv_cache::get_n_kv, kpool_build_state)
    const uint32_t n_kv = std::min<uint32_t>(c->n_ctx, std::max<uint32_t>(256u, GGML_PAD(n_cells, 256u)));
    const uint32_t n_pool_real = n_cells / KPOOL;
    const uint32_t n_pool = kpool_pad(n_pool_real);
    // pools this ubatch touches that are complete (in order mode: the pool of each token's rank)
    std::vector<uint32_t> new_pools;
    for (uint32_t p = p0; p < n_cells; ++p) {
        const uint32_t k = p / KPOOL;
        if (k < n_pool_real && (new_pools.empty() || new_pools.back() != k)) new_pools.push_back(k);
    }
    const uint32_t n_new = (uint32_t) new_pools.size();
    const uint32_t n_pool_max = c->n_ctx / KPOOL;
    const uint32_t bound = (uint32_t) n_tokens / KPOOL + 1;
    const uint32_t n_new_g = std::max({n_new, 1u, std::min({bound, n_pool - 1, n_pool_max})});
    const int64_t n_top_pool = std::min<int64_t>(n_pool, IDX_TOP_K / KPOOL);
    const uint32_t n_sel = (uint32_t) (KPOOL * n_top_pool + KPOOL - 1);

    // graph reuse: a ubatch of 1..FG_GRAPH_SLOTS tokens with the same shape as the last one of its width reuses that
    // slot's graph and allocation (build() depends only on these shapes and the options, never on token values or
    // positions; the inputs below are rewritten every call). Each slot has its own gallocr, so no other graph
    // (prefill, another width) invalidates it.
    const bool cacheable = c->reuse_graph && n_tokens >= 1 && n_tokens <= FG_GRAPH_SLOTS && !(dump && dump->dir) &&
                           !c->opts.profile && !c->fresh && (vmode == 0 || (vmode == 1 && g_mtp_verify_cache));
    // T2: a verify graph differs from the plain graph of its width (all-row outputs, provisional recurrent state),
    // so it has its own slot; build() of a verify graph depends on the same shapes and options (reviewed: the
    // pend_gdn / pend_conv tensors and the output flags are fixed at build time, inp_out_ids is rewritten per call)
    fg_ctx::DecCache * dslot = cacheable ? &c->dcs[n_tokens - 1 + (vmode == 1 ? FG_GRAPH_SLOTS : 0)] : nullptr;
    // F38: the per-token op's cap (the pools may change size: governor); part of the graph identity
    c->per_tok_cap = 1 << 30;
    if (c->expert_fn) {
        float capf = 0.0f;
        fg_expert_call q{};
        q.kind = FG_EXP_CAP;
        q.nth = 1;
        q.dst = &capf;
        c->expert_fn(&q, c->expert_ud);
        if (capf >= 1.0f) c->per_tok_cap = (int) capf;
    }
    const uint64_t key[6] = {n_kv, n_pool, n_new_g, (uint64_t) n_sel | (uint64_t) std::min(c->per_tok_cap, 0xffff) << 32,
                             (uint64_t) (uintptr_t) c->expert_fn,
                             (uint64_t) c->opts.fixed_attn | (uint64_t) c->opts.shexp_first << 8 | (uint64_t) c->opts.batch_min_tokens << 16 |
                             (uint64_t) c->opts.rs_view << 40 | (uint64_t) c->opts.fuse << 48};
    const bool reuse = cacheable && dslot->valid && std::memcmp(key, dslot->key, sizeof key) == 0;
    if (dslot) c->cur_dc = dslot;
    const auto tt0 = std::chrono::steady_clock::now();
    auto lap = [&](int k, std::chrono::steady_clock::time_point & t) {
        const auto n = std::chrono::steady_clock::now();
        c->tim[k] += std::chrono::duration<double, std::milli>(n - t).count();
        t = n;
    };
    auto tt = tt0;
    const size_t graph_size = 16384;
    std::vector<uint8_t> meta;
    ggml_context * ctx0;
    ggml_cgraph * gf;
    G * gp;
    ggml_tensor * out;
    if (reuse) {
        ctx0 = dslot->ctx0;
        gf = dslot->gf;
        gp = (G *) dslot->g;
        out = dslot->out;
    } else {
        if (dslot) {
            dslot->reset(); // this width's previous graph is replaced
            c->cur_bars = &dslot->bars;
        } else {
            c->fuse_bars.clear(); // the previous uncached graph's fused nodes are gone with it
            c->cur_bars = &c->fuse_bars;
        }
        meta.resize(ggml_tensor_overhead() * graph_size * 2 + ggml_graph_overhead_custom(graph_size, false));
        ggml_init_params ip = {meta.size(), meta.data(), true};
        ctx0 = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx0, graph_size, false);
        std::fill(c->fuse_built, c->fuse_built + 6, 0);
        std::fill(c->fuse_fallback, c->fuse_fallback + 6, 0);
        gp = new G{m, *c, ctx0, gf, n_tokens, vmode == 1 ? n_tokens : 1, n_kv, n_pool, n_new_g, n_sel};
        gp->rs_z = c->fresh ? 0 : -1;
        gp->vmode = vmode;
        gp->cols_exact = vmode == 1;
        out = gp->build();
    }
    G & g = *gp;
    // on every exit: keep a cacheable graph for the next call, free anything else
    auto finish = [&]() {
        if (reuse) return;
        if (cacheable) {
            dslot->meta.swap(meta);
            dslot->ctx0 = ctx0;
            dslot->gf = gf;
            dslot->out = out;
            dslot->g = gp;
            dslot->del = [](void * p) { delete (G *) p; };
            std::memcpy(dslot->key, key, sizeof key);
            dslot->valid = true;
        } else {
            delete gp;
            ggml_free(ctx0);
        }
    };
    std::vector<std::string> pats;
    if (dump && dump->dir) {
        for (int i = 0; i < dump->n_names; ++i) pats.push_back(dump->names[i]);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_RMS_NORM && name_match(t->name, pats)) ggml_set_output(t);
        }
    }
    ggml_set_output(out);
    lap(0, tt);
    ggml_gallocr_t ga = c->galloc;
    if (dslot) {
        if (!dslot->galloc) dslot->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
        ga = dslot->galloc;
    }
    if (!reuse && !ggml_gallocr_alloc_graph(ga, gf)) {
        delete gp;
        ggml_free(ctx0);
        return -2;
    }
    lap(1, tt);

    // ---- inputs ----
    {
        std::memcpy(g.inp_tokens->data, tokens, sizeof(int32_t) * n_tokens);
        int32_t * pos = (int32_t *) g.inp_pos->data;
        for (int i = 0; i < n_tokens; ++i) {
            pos[i] = pos[n_tokens + i] = pos[2 * n_tokens + i] = (int32_t) (p0 + i);
            pos[3 * n_tokens + i] = 0;
        }
        if (vmode == 1) for (int i = 0; i < n_tokens; ++i) ((int32_t *) g.inp_out_ids->data)[i] = i;
        else ((int32_t *) g.inp_out_ids->data)[0] = n_tokens - 1;
        int64_t * ki = (int64_t *) g.k_idxs->data;
        int64_t * vi = (int64_t *) g.v_idxs->data;
        int64_t * ii = (int64_t *) g.idx_k_idxs->data;
        for (int i = 0; i < n_tokens; ++i) ki[i] = vi[i] = ii[i] = p0 + i;
        // with opts.rs_view no node reads s_copy in a non-fresh ubatch, so gallocr leaves it unallocated (data NULL;
        // ticket 010 segfault): write it only when it is part of the graph
        if (g.s_copy->data) ((int32_t *) g.s_copy->data)[0] = 0;
        // PLE n-gram rows
        int32_t * rows = (int32_t *) g.ple_rows->data;
        for (int i = 0; i < n_tokens; ++i) {
            const int64_t p = p0 + i;
            int64_t ctxw[PLE_NGRAM];
            ctxw[0] = tokens[i];
            bool cut = false;
            for (int64_t s = 1; s < PLE_NGRAM; ++s) {
                const int64_t t = (cut || p - s < 0) ? -1 : c->toks[p - s];
                cut = cut || t < 0 || t == m.ple_eos;
                ctxw[s] = cut ? m.ple_eos : t;
            }
            for (int64_t ng = 2; ng <= PLE_NGRAM; ++ng) {
                uint64_t mixed = (uint64_t) ctxw[0] * m.ple_mult[0];
                for (int64_t j = 1; j < ng; ++j) mixed ^= (uint64_t) ctxw[j] * m.ple_mult[j];
                const int64_t base = (ng - 2) * PLE_PER_GRAM;
                for (int64_t gi = 0; gi < PLE_PER_GRAM; ++gi) {
                    const int64_t h = base + gi;
                    rows[i * PLE_HEADS + h] = (int32_t) (mixed % m.ple_voc[h] + m.ple_off[h]);
                }
            }
        }
        // k-pool (llama_memory_hybrid_idx_context::set_input_kpool, one sequence, cell == position)
        const int32_t sentinel = (int32_t) n_kv;
        int32_t * pcell = (int32_t *) g.pool_cells->data;
        int32_t * pidx = (int32_t *) g.pool_idxs->data;
        int32_t * nidx = (int32_t *) g.new_pool_idxs->data;
        int64_t * nrep = (int64_t *) g.new_pool_rep->data;
        int32_t * npos = (int32_t *) g.new_pool_pos->data;
        std::fill(npos, npos + 4 * n_new_g, 0);
        uint32_t i_new = 0;
        for (uint32_t ip = 0; ip < n_pool_real; ++ip) {
            const uint32_t j = ip * KPOOL;
            const uint32_t rep = j + KPOOL - 1;
            pcell[ip] = (int32_t) rep;
            for (uint32_t k = 0; k < KPOOL; ++k) pidx[ip * KPOOL + k] = (int32_t) (j + k);
            if (i_new < n_new && new_pools[i_new] == ip) {
                for (uint32_t k = 0; k < KPOOL; ++k) nidx[i_new * KPOOL + k] = (int32_t) (j + k);
                nrep[i_new] = rep;
                npos[0 * n_new_g + i_new] = (int32_t) j;
                npos[1 * n_new_g + i_new] = (int32_t) j; // ext.y = the token's y position = p for text
                npos[2 * n_new_g + i_new] = (int32_t) j; // ext.x
                npos[3 * n_new_g + i_new] = (int32_t) j;
                ++i_new;
            }
        }
        GGML_ASSERT(i_new == n_new);
        if (n_new_g > n_new) {
            std::vector<int64_t> reps(pcell, pcell + n_pool_real);
            std::sort(reps.begin(), reps.end());
            int64_t pad_cell = 0;
            for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
                while (std::binary_search(reps.begin(), reps.end(), pad_cell)) ++pad_cell;
                for (uint32_t k = 0; k < KPOOL; ++k) nidx[i * KPOOL + k] = (int32_t) pad_cell;
                nrep[i] = pad_cell;
            }
        }
        for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
            pcell[ip] = (int32_t) p0; // dummy_cell: the first cell of this ubatch
            for (uint32_t k = 0; k < KPOOL; ++k) pidx[ip * KPOOL + k] = sentinel;
        }
        ggml_fp16_t * pm = (ggml_fp16_t *) g.pool_mask->data;
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
        for (int i = 0; i < n_tokens; ++i) {
            const int64_t rank = p0 + i;
            ggml_fp16_t * row = pm + (size_t) i * n_pool;
            std::fill(row, row + n_pool, drop);
            const uint32_t nv = std::min<uint32_t>(n_pool_real, (uint32_t) ((rank + 1) / KPOOL));
            std::fill(row, row + nv, keep);
        }
        int32_t * tidx = (int32_t *) g.tail_idxs->data;
        for (int i = 0; i < n_tokens; ++i) {
            const int64_t rank = p0 + i;
            const uint32_t n_tail = (uint32_t) ((rank + 1) % KPOOL);
            for (uint32_t k = 0; k < KPOOL - 1; ++k) tidx[i * (KPOOL - 1) + k] = k < n_tail ? (int32_t) (rank - k) : sentinel;
        }
    }

    lap(2, tt);
    ggml_backend_cpu_set_n_threads(c->backend, n_threads);
    if (c->opts.profile) {
        // node by node: each node in a one-node graph (its sources are already computed)
        const int nn = ggml_graph_n_nodes(gf);
        std::vector<uint8_t> pm(ggml_graph_overhead_custom(1, false) + 1024);
        for (int i = 0; i < nn; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            ggml_init_params pp = {pm.size(), pm.data(), true};
            ggml_context * pc = ggml_init(pp);
            ggml_cgraph * g1 = ggml_new_graph_custom(pc, 1, false);
            ggml_graph_add_node(g1, t);
            const auto t0 = std::chrono::steady_clock::now();
            const ggml_status st = ggml_backend_graph_compute(c->backend, g1);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ggml_free(pc);
            if (st != GGML_STATUS_SUCCESS) {
                finish();
                if (dslot) dslot->reset();
                return -3;
            }
            std::string kind = t->name;
            const size_t dash = kind.rfind('-');
            if (dash != std::string::npos && dash + 1 < kind.size() && std::isdigit((unsigned char) kind[dash + 1])) kind.resize(dash);
            if (kind.rfind("node_", 0) == 0) kind = std::string("op:") + ggml_op_desc(t);
            if (kind.empty() || kind.find("(view)") != std::string::npos || kind.find("(reshaped)") != std::string::npos ||
                kind.find("(permuted)") != std::string::npos || kind.find("(copy") != std::string::npos)
                kind = std::string("op:") + ggml_op_desc(t);
            auto & e = c->prof[kind];
            e.first += 1;
            e.second += ms;
        }
        // calibration: the same one-node dispatch with a trivial node (a 1-float scale on all threads),
        // 64 times per profiled step; "_empty_node" ms / calls = the dispatch cost inside every row above
        {
            alignas(64) static uint8_t cm[16 * 1024];
            ggml_init_params cp = {sizeof cm, cm, false};
            ggml_context * cc = ggml_init(cp);
            ggml_tensor * x = ggml_new_tensor_1d(cc, GGML_TYPE_F32, 1);
            ((float *) x->data)[0] = 1.0f;
            ggml_tensor * y = ggml_scale(cc, x, 1.0f);
            ggml_cgraph * g1 = ggml_new_graph_custom(cc, 1, false);
            ggml_graph_add_node(g1, y);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; ++i) ggml_backend_graph_compute(c->backend, g1);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ggml_free(cc);
            auto & e = c->prof["_empty_node"];
            e.first += 64;
            e.second += ms;
        }
    } else if (c->opts.executor && dslot && (dslot->ex_built ? dslot->ex_ok && dslot->ex_nth == n_threads : true) &&
               (dslot->ex_built || (ex_build(c, gf, n_threads), dslot->ex_ok))) {
        ex_run(c, gf);
    } else {
        const ggml_status st = ggml_backend_graph_compute(c->backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            // positions back to before the call; attention cells past p0 are dead (overwritten later); the recurrent
            // state ran part of the ubatch, so the caller must restore its snapshot (or clear) before the next decode
            c->toks.resize(p0);
            c->rs_dirty = true;
            finish();
            if (dslot) dslot->reset();
            return st == GGML_STATUS_ABORTED ? -6 : -3;
        }
    }
    lap(3, tt);
    c->tim[4] += 1;
    c->fresh = false;
    if (vmode) {
        const int n_h = n_tokens;
        if (ext->h_all) std::memcpy(ext->h_all, g.h_src->data, sizeof(float) * N_EMBD * HC * n_h);
        c->mtp_stats[2] += g.cols_fallback;
    }
    if (vmode == 1) {
        std::memcpy(ext->logits_all, out->data, sizeof(float) * N_VOCAB * n_tokens);
        // the provisional recurrent results stay in the allocator's buffer (output-flagged) until fg_commit
        c->pend.gdn.clear();
        c->pend.conv.clear();
        for (auto & e : g.pend_gdn)
            c->pend.gdn.push_back({(float *) e.first->data, (const float *) e.second->data + SSM_D * SSM_VH * n_tokens});
        for (auto & e : g.pend_conv) c->pend.conv.push_back({(float *) e.state->data, (const float *) e.input->data, e.cols, e.ch});
        c->pend.n = n_tokens;
        c->pend.p0 = p0;
        c->pend.active = true;
        c->mtp_stats[0] += 1;
        c->mtp_stats[1] += n_tokens;
    } else if (logits) std::memcpy(logits, out->data, sizeof(float) * N_VOCAB);
    if (dump && dump->dir) {
        const std::string d = std::string(dump->dir);
        ::mkdir(d.c_str(), 0700);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op != GGML_OP_RMS_NORM && name_match(t->name, pats) && t->data) write_dump(d + "/" + t->name + ".bin", t);
        }
    }
    finish();
    return 0;
}

int fg_decode(fg_ctx * c, const int32_t * tokens, int n_tokens, int n_threads, float * logits, const fg_dump * dump) {
    return fg_decode_impl(c, tokens, n_tokens, n_threads, logits, dump, nullptr);
}

void fg_ctx_set_graph_reuse(fg_ctx * c, int on) {
    c->reuse_graph = on;
    if (!on)
        for (auto & d : c->dcs) d.reset();
}

} // extern "C"

// T2: MTP speculative decoding (verify / commit, the head graph, restricted draft output, self-test)
#include "flash_mtp.inc"
