// flash_fuse_ffn.inc: fused decode phases P5 (MoE router) and P6 (shared expert + routed-expert combine).
// #included by flash_graph.cpp after struct G and fg_ctx (N_EMBD, N_EXPERT, N_USED, FuseBarrier, fuse_split and
// ggml_vec_swiglu_f32 are visible there). Plain C++, no includes of its own.
//
// Replaces, per layer (results/m3/sc-op-inventory.md, sections P5 and P6):
//   P5: FFN.01-FFN.12 of build_moe_ffn (logits mm, soft_max, argsort_top_k, get_rows, sum_rows, clamp, div)
//   P6: FFN.S1-FFN.S7 (build_shexp), FFN.18 (weighted MUL), FFN.19-37 (10 VIEW + 9 ADD), FFN.38 (ffn_out ADD)
// The expert kernels FETCH/UP/GATE/DOWN stay ggml custom nodes; they consume the router's selected_experts view and
// the combine consumes the DOWN output (moe_down).
//
// Numerics: each value is bit-identical to the ggml node chain it replaces, for the production build (ggml-prod
// libggml-cpu.a). Measured there: ggml_cpu_has_llamafile() returns 0 and ggml-cpu.c.o has no reference to
// llamafile_sgemm, so ggml_compute_forward_mul_mat never takes the sgemm path; every matvec is the weight type's
// vec_dot with nrc = 1 per (output row, column). The type traits give nrows == 2 for Q8_0 and Q6_K under
// __ARM_FEATURE_MATMUL_INT8 (the production flags include +i8mm); production then calls vec_dot with nrc = 2 when the
// column count is even. fuse_matvec_ok() refuses that case (builders return nullptr, the caller keeps the ggml nodes),
// so the kernels here only run with nrc = 1, the same as production for odd column counts (decode nt = 1).
//
// Barrier structure (every thread of the graph calls each custom node; FuseBarrier::wait(nth) per step):
//   P5 router: S0 quantise cur (block-split) | barrier | S1 logit rows split | barrier | S2 one token per thread
//   P6 combine: S1 quantise cur per input type (block-split) | barrier | S2 gate/up rows split, gate_inp by thread 0
//               | barrier | S3 swiglu over nt*F (chunks of 4) + sigmoid per token | barrier
//               | S4 quantise swiglu output (block-split) | barrier | S5 down rows split by element e: per-element
//               down, *sigmoid, routed-expert sum in graph order, + shared expert (no barrier needed, own elements)
// Each node is followed by ggml's own barrier, so consumers see complete outputs.
//
// Output layouts. Custom node dst = F32 with ne0 = total_bytes / 4 (scratch lives in the tail of dst; the caller views
// only the head). All offsets are bytes from dst->data.
//
// P5 router, fuse_build_router(ctx, bar, cur [N_EMBD, nt], w_gate_inp): srcs {cur, w_gate_inp}, userdata = bar.
//   [0, 40 nt)                 ids: I32 [N_USED, nt], contiguous (view: fuse_router_selected, nb1 = 40).
//                              The view is an F32 custom output re-typed to I32 (both 4 bytes).
//   [40 nt, 80 nt)             weights: F32 [1, N_USED, nt], nb = {4, 4, 40} (view: fuse_router_weights).
//                              Values = ffn_moe_weights_norm: probs[sel] / clamp(sum(probs[sel]), 6.1035e-5).
//   [80 nt, 2128 nt)           logits scratch F32 [N_EXPERT, nt]
//   [2128 nt, 4176 nt)         probs scratch F32 [N_EXPERT, nt]
//   [4176 nt, +nt*row_size)    quantised cur scratch
//
// P6 combine, fuse_build_shexp_combine(ctx, bar, cur, moe_down, weights, w_gate, w_up, w_down, w_gate_inp):
//   srcs {cur, moe_down, weights, w_gate, w_up, w_down, w_gate_inp}, userdata = bar.
//   moe_down: F32 [N_EMBD, N_USED, nt] (strides read from the tensor); weights: F32 [1, N_USED, nt] (strides read).
//   [0, 4 N_EMBD nt)           ffn_out: F32 [N_EMBD, nt], contiguous (view: fuse_shexp_combine_out)
//   then scratch: D (nt x N_EMBD), sigmoid(gate) (nt), G (nt x F), U (nt x F), A = swiglu (nt x F),
//   quantised cur per distinct input type (up to 3), quantised A for w_down.
//   F = w_gate->ne[1] (shared-expert width; must be a multiple of 4 and of the w_down quant block size).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" double ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max); // vec.cpp, exported

// ---------------------------------------------------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------------------------------------------------

// Element-wise helpers, marked noinline so no multiply and add of different graph nodes can be contracted into one FMA
// (production stores each product to memory as its own F32 node). Each helper does one IEEE operation per element.
__attribute__((noinline)) static void fuse_mul_store(float * p, const float * x, float w, int64_t n) {
    for (int64_t i = 0; i < n; ++i) p[i] = x[i] * w;
}
__attribute__((noinline)) static void fuse_add_inplace(float * a, const float * b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) a[i] = a[i] + b[i];
}
__attribute__((noinline)) static void fuse_scale_inplace(float * a, float s, int64_t n) {
    for (int64_t i = 0; i < n; ++i) a[i] = a[i] * s;
}

// A type can take this path when production would call its vec_dot with nrc = 1 for the given column count.
static bool fuse_matvec_ok(const ggml_tensor * w, int64_t nt) {
    const auto * tr = ggml_get_type_traits_cpu(w->type);
    if (!tr || !tr->vec_dot || !tr->from_float) return false;
    if (w->nb[0] != ggml_type_size(w->type)) return false;
    return tr->nrows == 1 || (nt % 2) == 1;
}

// src[t] (row stride src_stride bytes, K floats per row) quantised to vt into q (row stride q_stride bytes).
// Split by whole quant blocks over the threads; per block the result is the same as production's from_float.
static void fuse_quant_rows(ggml_type vt, const char * src, size_t src_stride, int64_t nt, int64_t K,
                            char * q, size_t q_stride, int ith, int nth) {
    const auto * tr = ggml_get_type_traits_cpu(vt);
    int64_t lo = 0, hi = 0;
    fuse_split(K, ith, nth, ggml_blck_size(vt), lo, hi);
    if (hi <= lo) return;
    for (int64_t t = 0; t < nt; ++t) {
        const float * x = (const float *) (src + t * src_stride) + lo;
        tr->from_float(x, q + t * q_stride + ggml_row_size(vt, lo), hi - lo);
    }
}

// out[t * ldo + r] = vec_dot(K, row r of w, quantised column t) for r in [r0, r1): production's nrc = 1 path
// (ggml_compute_forward_mul_mat_one_chunk: vec_dot(ne00, &tmp, 0, src0_row, 0, src1_col, 0, 1)).
static void fuse_matvec_rows(const ggml_tensor * w, const char * qin, size_t qin_stride, int64_t nt,
                             float * out, int64_t ldo, int64_t r0, int64_t r1) {
    const auto * tr = ggml_get_type_traits_cpu(w->type);
    const int K = (int) w->ne[0];
    for (int64_t r = r0; r < r1; ++r) {
        const char * row = (const char *) w->data + r * w->nb[1];
        for (int64_t t = 0; t < nt; ++t) {
            float s = 0.0f;
            tr->vec_dot(K, &s, 0, row, 0, qin + t * qin_stride, 0, 1);
            out[t * ldo + r] = s;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// P5: router (build_moe_ffn FFN.01-FFN.12)
// ---------------------------------------------------------------------------------------------------------------------

struct FuseRouterLayout {
    size_t o_w, o_logit, o_prob, o_q, total;
};

static FuseRouterLayout fuse_router_layout(int64_t nt, ggml_type vdt) {
    FuseRouterLayout L;
    L.o_w     = (size_t) nt * N_USED * 4;            // after ids [0, 40 nt)
    L.o_logit = L.o_w + (size_t) nt * N_USED * 4;
    L.o_prob  = L.o_logit + (size_t) nt * N_EXPERT * 4;
    L.o_q     = L.o_prob + (size_t) nt * N_EXPERT * 4;
    L.total   = L.o_q + (size_t) nt * ggml_row_size(vdt, N_EMBD);
    return L;
}

// One token: soft_max over the 512 logits, argsort DESC (full std::sort, as production's argsort; the top-k view is
// its first N_USED entries), gather, sum_rows (sequential double), clamp, div.
static void fuse_router_token(const float * lg, float * pr, int32_t * sel, float * wn) {
    // ggml_compute_forward_soft_max_f32: max = sequential MAX, sum = ggml_vec_soft_max_f32, then
    // sum = 1.0/sum (double) and the scale by (float) of it (ggml_vec_scale_f32 takes a float)
    float mx = -INFINITY;
    for (int64_t i = 0; i < N_EXPERT; ++i) mx = mx > lg[i] ? mx : lg[i];
    const double s0 = ggml_vec_soft_max_f32((int) N_EXPERT, pr, lg, mx);
    const double inv = 1.0 / s0;
    const float v = (float) inv;
    for (int64_t i = 0; i < N_EXPERT; ++i) pr[i] = pr[i] * v;

    // ggml_compute_forward_argsort_f32: identity array, std::sort with data[a] > data[b]
    int32_t idx[N_EXPERT];
    for (int64_t i = 0; i < N_EXPERT; ++i) idx[i] = (int32_t) i;
    std::sort(idx, idx + N_EXPERT, [pr](int32_t a, int32_t b) { return pr[a] > pr[b]; });

    // get_rows (exact copies), sum_rows (ggml_vec_sum_f32: double accumulation, then float)
    float w[N_USED];
    for (int64_t j = 0; j < N_USED; ++j) {
        sel[j] = idx[j];
        w[j] = pr[idx[j]];
    }
    double s = 0.0;
    for (int64_t j = 0; j < N_USED; ++j) s += (double) w[j];
    const float fs = (float) s;

    // ggml_clamp(min = 6.103515625e-5, max = +inf): MAX(MIN(x, max), min) with the ggml macros
    const float lo = 6.103515625e-5f;
    float c = fs < INFINITY ? fs : INFINITY;
    c = c > lo ? c : lo;

    // ggml_div (broadcast over the 10 weights)
    for (int64_t j = 0; j < N_USED; ++j) wn[j] = w[j] / c;
}

// dst: F32 custom node. dst->src = {cur [N_EMBD, nt], w_gate_inp}. ud = FuseBarrier *.
void fuse_router_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const int64_t nt = cur->ne[1];
    const ggml_type vdt = ggml_get_type_traits_cpu(w->type)->vec_dot_type;
    const FuseRouterLayout L = fuse_router_layout(nt, vdt);
    char * base = (char *) dst->data;
    int32_t * ids = (int32_t *) base;
    float * wout = (float *) (base + L.o_w);
    float * logit = (float *) (base + L.o_logit);
    float * prob = (float *) (base + L.o_prob);
    char * q = base + L.o_q;
    const size_t qs = ggml_row_size(vdt, N_EMBD);

    // S0: quantise the input (vec_dot_type of the router weight; F32 is a copy)
    fuse_quant_rows(vdt, (const char *) cur->data, cur->nb[1], nt, N_EMBD, q, qs, ith, nth);
    bar->wait(nth);

    // S1: logits rows (512) split over threads
    int64_t r0 = 0, r1 = 0;
    fuse_split(N_EXPERT, ith, nth, 1, r0, r1);
    fuse_matvec_rows(w, q, qs, nt, logit, N_EXPERT, r0, r1);
    bar->wait(nth);

    // S2: one token per thread (reductions must keep production's order)
    for (int64_t t = ith; t < nt; t += nth) {
        fuse_router_token(logit + t * N_EXPERT, prob + t * N_EXPERT, ids + t * N_USED, wout + t * N_USED);
    }
}

// Views of the router output (caller side). nt = number of tokens.
ggml_tensor * fuse_router_selected(ggml_context * ctx, ggml_tensor * node, int64_t nt) {
    ggml_tensor * v = ggml_view_2d(ctx, node, N_USED, nt, N_USED * sizeof(int32_t), 0);
    v->type = GGML_TYPE_I32; // same 4-byte element as the F32 node
    return v;
}

ggml_tensor * fuse_router_weights(ggml_context * ctx, ggml_tensor * node, int64_t nt) {
    return ggml_view_3d(ctx, node, 1, N_USED, nt, sizeof(float), N_USED * sizeof(float),
                        (size_t) nt * N_USED * sizeof(float));
}

// Returns the custom node, or nullptr when the shapes or types are not covered (caller keeps the ggml nodes).
ggml_tensor * fuse_build_router(ggml_context * ctx, FuseBarrier * bar, ggml_tensor * cur, ggml_tensor * w_gate_inp) {
    if (cur->type != GGML_TYPE_F32 || cur->nb[0] != sizeof(float) || cur->ne[0] != N_EMBD || cur->ne[2] != 1 ||
        cur->ne[3] != 1) {
        return nullptr;
    }
    const int64_t nt = cur->ne[1];
    if (nt < 1 || w_gate_inp->ne[0] != N_EMBD || w_gate_inp->ne[1] != N_EXPERT || !fuse_matvec_ok(w_gate_inp, nt)) {
        return nullptr;
    }
    const ggml_type vdt = ggml_get_type_traits_cpu(w_gate_inp->type)->vec_dot_type;
    const FuseRouterLayout L = fuse_router_layout(nt, vdt);
    ggml_tensor * args[2] = {cur, w_gate_inp};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) (L.total / 4), 1, 1, 1, args, 2, fuse_router_op,
                          GGML_N_TASKS_MAX, bar);
}

// ---------------------------------------------------------------------------------------------------------------------
// P6: shared expert + routed combine (build_shexp FFN.S1-S7, FFN.18, FFN.19-38)
// ---------------------------------------------------------------------------------------------------------------------

struct FuseCombLayout {
    int64_t F;                  // shared-expert width
    ggml_type vt[3];            // distinct quantised-input types among gate / up / gate_inp
    int nslot;
    int slot_g, slot_u, slot_sg;
    ggml_type vd;               // quantised-input type of w_down
    size_t o_out, o_D, o_sg, o_G, o_U, o_A, o_q[3], o_qa, total;
};

static FuseCombLayout fuse_comb_layout(int64_t nt, const ggml_tensor * wg, const ggml_tensor * wu,
                                       const ggml_tensor * wd, const ggml_tensor * wsg) {
    FuseCombLayout L{};
    L.F = wg->ne[1];
    const ggml_type want[3] = {ggml_get_type_traits_cpu(wg->type)->vec_dot_type,
                               ggml_get_type_traits_cpu(wu->type)->vec_dot_type,
                               ggml_get_type_traits_cpu(wsg->type)->vec_dot_type};
    L.vd = ggml_get_type_traits_cpu(wd->type)->vec_dot_type;
    int * slot[3] = {&L.slot_g, &L.slot_u, &L.slot_sg};
    L.nslot = 0;
    for (int i = 0; i < 3; ++i) {
        int s = -1;
        for (int k = 0; k < L.nslot; ++k) {
            if (L.vt[k] == want[i]) s = k;
        }
        if (s < 0) {
            s = L.nslot;
            L.vt[L.nslot++] = want[i];
        }
        *slot[i] = s;
    }
    size_t o = 0;
    L.o_out = o; o += (size_t) nt * N_EMBD * 4;
    L.o_D   = o; o += (size_t) nt * N_EMBD * 4;
    L.o_sg  = o; o += (size_t) nt * 4;
    L.o_G   = o; o += (size_t) nt * L.F * 4;
    L.o_U   = o; o += (size_t) nt * L.F * 4;
    L.o_A   = o; o += (size_t) nt * L.F * 4;
    for (int k = 0; k < L.nslot; ++k) {
        L.o_q[k] = o;
        o += (size_t) nt * ggml_row_size(L.vt[k], N_EMBD);
    }
    L.o_qa = o; o += (size_t) nt * ggml_row_size(L.vd, L.F);
    L.total = o;
    return L;
}

// dst: F32 custom node. dst->src = {cur, moe_down, weights, w_gate, w_up, w_down, w_gate_inp}. ud = FuseBarrier *.
void fuse_shexp_combine_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * md = dst->src[1];
    const ggml_tensor * wt = dst->src[2];
    const ggml_tensor * wg = dst->src[3];
    const ggml_tensor * wu = dst->src[4];
    const ggml_tensor * wd = dst->src[5];
    const ggml_tensor * wsg = dst->src[6];
    const int64_t nt = cur->ne[1];
    const FuseCombLayout L = fuse_comb_layout(nt, wg, wu, wd, wsg);
    const int64_t F = L.F;

    char * base = (char *) dst->data;
    float * out = (float *) (base + L.o_out);
    float * D = (float *) (base + L.o_D);
    float * sg = (float *) (base + L.o_sg);
    float * G = (float *) (base + L.o_G);
    float * U = (float *) (base + L.o_U);
    float * A = (float *) (base + L.o_A);
    char * qa = base + L.o_qa;
    const size_t qa_stride = ggml_row_size(L.vd, F);

    // S1: quantise cur once per distinct input type
    for (int s = 0; s < L.nslot; ++s) {
        fuse_quant_rows(L.vt[s], (const char *) cur->data, cur->nb[1], nt, N_EMBD, base + L.o_q[s],
                        ggml_row_size(L.vt[s], N_EMBD), ith, nth);
    }
    bar->wait(nth);

    // S2: gate and up (F rows each, split), gate_inp (one row, thread 0)
    int64_t r0 = 0, r1 = 0;
    fuse_split(F, ith, nth, 1, r0, r1);
    const char * qg = base + L.o_q[L.slot_g];
    const char * qu = base + L.o_q[L.slot_u];
    const size_t sg_stride_g = ggml_row_size(L.vt[L.slot_g], N_EMBD);
    const size_t sg_stride_u = ggml_row_size(L.vt[L.slot_u], N_EMBD);
    fuse_matvec_rows(wg, qg, sg_stride_g, nt, G, F, r0, r1);
    fuse_matvec_rows(wu, qu, sg_stride_u, nt, U, F, r0, r1);
    if (ith == 0) {
        fuse_matvec_rows(wsg, base + L.o_q[L.slot_sg], ggml_row_size(L.vt[L.slot_sg], N_EMBD), nt, sg, 1, 0, 1);
    }
    bar->wait(nth);

    // S3: ffn_swiglu = silu(gate) * up over nt*F elements (chunks on multiples of 4, as ggml's NEON groups);
    //     shared_expert_gate_sigmoid = 1/(1+expf(-x)) per token (owner thread)
    int64_t e0 = 0, e1 = 0;
    fuse_split(nt * F, ith, nth, 4, e0, e1);
    if (e1 > e0) ggml_vec_swiglu_f32((int) (e1 - e0), A + e0, G + e0, U + e0);
    for (int64_t t = ith; t < nt; t += nth) {
        sg[t] = 1.0f / (1.0f + expf(-sg[t]));
    }
    bar->wait(nth);

    // S4: quantise the swiglu output for w_down
    fuse_quant_rows(L.vd, (const char *) A, F * sizeof(float), nt, F, qa, qa_stride, ith, nth);
    bar->wait(nth);

    // S5: element range [e0, e1) of N_EMBD owned by this thread: down rows, gating, then the routed sum in graph order
    //     (acc = ev0; acc = acc + ev_j for j = 1..9; ffn_out = acc + shexp_gated), all on the owned elements only.
    fuse_split(N_EMBD, ith, nth, 1, e0, e1);
    const int64_t Ln = e1 - e0;
    if (Ln <= 0) return;
    fuse_matvec_rows(wd, qa, qa_stride, nt, D, N_EMBD, e0, e1);
    float p[N_EMBD];
    for (int64_t t = 0; t < nt; ++t) {
        float * d = D + t * N_EMBD + e0;
        fuse_scale_inplace(d, sg[t], Ln); // ffn_shexp_gated = ffn_shexp * sigmoid(gate)
        float * o = out + t * N_EMBD + e0;
        const char * mdt = (const char *) md->data + t * md->nb[2];
        const char * wtt = (const char *) wt->data + t * wt->nb[2];
        fuse_mul_store(o, (const float *) (mdt + 0 * md->nb[1]) + e0, *(const float *) (wtt + 0 * wt->nb[1]), Ln);
        for (int64_t j = 1; j < N_USED; ++j) {
            fuse_mul_store(p, (const float *) (mdt + j * md->nb[1]) + e0, *(const float *) (wtt + j * wt->nb[1]), Ln);
            fuse_add_inplace(o, p, Ln);
        }
        fuse_add_inplace(o, d, Ln); // ffn_out = moe_out + ffn_shexp_gated
    }
}

// Returns the custom node, or nullptr when shapes / types are not covered (caller keeps the ggml nodes).
ggml_tensor * fuse_build_shexp_combine(ggml_context * ctx, FuseBarrier * bar, ggml_tensor * cur, ggml_tensor * moe_down,
                                       ggml_tensor * weights, ggml_tensor * w_gate, ggml_tensor * w_up,
                                       ggml_tensor * w_down, ggml_tensor * w_gate_inp) {
    if (cur->type != GGML_TYPE_F32 || cur->nb[0] != sizeof(float) || cur->ne[0] != N_EMBD || cur->ne[2] != 1 ||
        cur->ne[3] != 1) {
        return nullptr;
    }
    const int64_t nt = cur->ne[1];
    const int64_t F = w_gate->ne[1];
    if (nt < 1 || F < 4 || (F % 4) != 0) return nullptr;
    if (w_gate->ne[0] != N_EMBD || w_up->ne[0] != N_EMBD || w_up->ne[1] != F) return nullptr;
    if (w_down->ne[0] != F || w_down->ne[1] != N_EMBD) return nullptr;
    if (w_gate_inp->ne[0] != N_EMBD || w_gate_inp->ne[1] != 1) return nullptr;
    if (F % ggml_blck_size(ggml_get_type_traits_cpu(w_down->type)->vec_dot_type) != 0) return nullptr;
    if (moe_down->type != GGML_TYPE_F32 || moe_down->nb[0] != sizeof(float) || moe_down->ne[0] != N_EMBD ||
        moe_down->ne[1] != N_USED || moe_down->ne[2] != nt || moe_down->ne[3] != 1) {
        return nullptr;
    }
    if (weights->type != GGML_TYPE_F32 || weights->nb[0] != sizeof(float) || weights->ne[0] != 1 ||
        weights->ne[1] != N_USED || weights->ne[2] != nt) {
        return nullptr;
    }
    if (!fuse_matvec_ok(w_gate, nt) || !fuse_matvec_ok(w_up, nt) || !fuse_matvec_ok(w_down, nt) ||
        !fuse_matvec_ok(w_gate_inp, nt)) {
        return nullptr;
    }
    const FuseCombLayout L = fuse_comb_layout(nt, w_gate, w_up, w_down, w_gate_inp);
    ggml_tensor * args[7] = {cur, moe_down, weights, w_gate, w_up, w_down, w_gate_inp};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t) (L.total / 4), 1, 1, 1, args, 7, fuse_shexp_combine_op,
                          GGML_N_TASKS_MAX, bar);
}

// ffn_out view [N_EMBD, nt] of the combine node (caller side).
ggml_tensor * fuse_shexp_combine_out(ggml_context * ctx, ggml_tensor * node, int64_t nt) {
    return ggml_view_2d(ctx, node, N_EMBD, nt, N_EMBD * sizeof(float), 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// model-free test: fused P5 + P6 against the ggml node chain of build_moe_ffn / build_shexp / the combine
// ---------------------------------------------------------------------------------------------------------------------

// Returns the number of mismatching checks (0 = all bitwise equal).
static int fuse_ffn_case(int nth, int64_t nt, ggml_type rty) {
    const int64_t E = N_EMBD, F = 512;
    const size_t mem = 96u << 20;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0x2545F4914F6CDD1Dull ^ (uint64_t) (nt * 7919 + (int) rty * 104729);
    auto rnd = [&]() {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return (float) ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
    };
    auto fill = [&](ggml_tensor * t, float sc) {
        float * p = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = rnd() * sc;
    };
    auto wmat = [&](ggml_type ty, int64_t k, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, ty, k, n);
        if (ty == GGML_TYPE_F32) {
            fill(t, 0.05f);
        } else {
            std::vector<float> w(k * n);
            for (auto & v : w) v = rnd() * 0.05f;
            ggml_quantize_chunk(ty, w.data(), t->data, 0, n, k, nullptr);
        }
        return t;
    };

    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, nt);
    fill(cur, 1.0f);
    ggml_tensor * md = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, N_USED, nt); // stands in for FFN.17 (DOWN)
    fill(md, 1.0f);
    ggml_tensor * wgi = wmat(rty, E, N_EXPERT);
    // identical router rows: exact logit ties, so the argsort order is exercised
    memcpy((char *) wgi->data + 9 * wgi->nb[1], (char *) wgi->data + 5 * wgi->nb[1], wgi->nb[1]);
    memcpy((char *) wgi->data + 300 * wgi->nb[1], (char *) wgi->data + 5 * wgi->nb[1], wgi->nb[1]);
    ggml_tensor * wg = wmat(GGML_TYPE_Q6_K, E, F);
    ggml_tensor * wu = wmat(GGML_TYPE_Q8_0, E, F);
    ggml_tensor * wd = wmat(GGML_TYPE_Q6_K, F, E);
    ggml_tensor * wsg = wmat(GGML_TYPE_F32, E, 1);

    // reference: build_moe_ffn up to the expert fetch (FFN.01-12), build_shexp, and the combine (FFN.18-38)
    ggml_tensor * logits = ggml_mul_mat(ctx, wgi, cur);
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_tensor * top = ggml_argsort_top_k(ctx, probs, N_USED);
    ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, nt);
    ggml_tensor * rw = ggml_get_rows(ctx, probs3, top);
    rw = ggml_reshape_2d(ctx, rw, N_USED, nt);
    ggml_tensor * ws = ggml_sum_rows(ctx, rw);
    ws = ggml_clamp(ctx, ws, 6.103515625e-5, INFINITY);
    ggml_tensor * rwn = ggml_div(ctx, rw, ws);
    ggml_tensor * rwn3 = ggml_reshape_3d(ctx, rwn, 1, N_USED, nt);

    ggml_tensor * up = ggml_mul_mat(ctx, wu, cur);
    ggml_tensor * gt = ggml_mul_mat(ctx, wg, cur);
    gt = ggml_swiglu_split(ctx, gt, up);
    ggml_tensor * dn = ggml_mul_mat(ctx, wd, gt);
    ggml_tensor * sgt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, wsg, cur));
    dn = ggml_mul(ctx, dn, sgt);

    ggml_tensor * ex = ggml_mul(ctx, md, rwn3);
    ggml_tensor * acc = ggml_view_2d(ctx, ex, E, nt, ex->nb[2], 0);
    for (int64_t i = 1; i < N_USED; ++i) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, ex, E, nt, ex->nb[2], i * ex->nb[1]));
    }
    ggml_tensor * ref_out = ggml_add(ctx, acc, dn);

    ggml_cgraph * gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, ref_out);
    ggml_graph_compute_with_ctx(ctx, gr, nth);

    std::vector<int32_t> ref_ids(nt * N_USED);
    std::vector<float> ref_w(nt * N_USED), ref_o(E * nt);
    for (int64_t t = 0; t < nt; ++t) {
        for (int64_t j = 0; j < N_USED; ++j) {
            ref_ids[t * N_USED + j] = *(const int32_t *) ((const char *) top->data + j * top->nb[0] + t * top->nb[1]);
            ref_w[t * N_USED + j] = *(const float *) ((const char *) rwn3->data + j * rwn3->nb[1] + t * rwn3->nb[2]);
        }
    }
    memcpy(ref_o.data(), ref_out->data, ref_o.size() * sizeof(float));

    // fused: P5 router node + views, P6 combine node + output view
    FuseBarrier bar_r, bar_c;
    ggml_tensor * rn = fuse_build_router(ctx, &bar_r, cur, wgi);
    ggml_tensor * sel = rn ? fuse_router_selected(ctx, rn, nt) : nullptr;
    ggml_tensor * wsel = rn ? fuse_router_weights(ctx, rn, nt) : nullptr;
    ggml_tensor * cn = rn ? fuse_build_shexp_combine(ctx, &bar_c, cur, md, wsel, wg, wu, wd, wsg) : nullptr;
    ggml_tensor * fo = cn ? fuse_shexp_combine_out(ctx, cn, nt) : nullptr;
    if (!fo) {
        std::fprintf(stderr, "fg_selftest_fuse_ffn nth %d nt %lld router %d: builders refused the shapes\n", nth,
                     (long long) nt, (int) rty);
        ggml_free(ctx);
        return 1;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fo);
    ggml_build_forward_expand(gf, sel);

    int bad_ids = 0, bad_w = 0, bad_o = 0;
    const int reps = 3;
    for (int rep = 0; rep < reps; ++rep) {
        ggml_graph_compute_with_ctx(ctx, gf, nth);
        for (int64_t t = 0; t < nt; ++t) {
            for (int64_t j = 0; j < N_USED; ++j) {
                const int32_t a = *(const int32_t *) ((const char *) sel->data + j * sel->nb[0] + t * sel->nb[1]);
                const float b = *(const float *) ((const char *) wsel->data + j * wsel->nb[1] + t * wsel->nb[2]);
                const float rb = ref_w[t * N_USED + j];
                if (a != ref_ids[t * N_USED + j]) ++bad_ids;
                if (std::memcmp(&b, &rb, sizeof(float)) != 0) ++bad_w;
            }
        }
        if (std::memcmp(fo->data, ref_o.data(), ref_o.size() * sizeof(float)) != 0) {
            for (size_t i = 0; i < ref_o.size(); ++i) {
                if (std::memcmp((const float *) fo->data + i, &ref_o[i], sizeof(float)) != 0) ++bad_o;
            }
        }
    }
    const bool ok = bad_ids == 0 && bad_w == 0 && bad_o == 0;
    std::fprintf(stderr,
                 "fg_selftest_fuse_ffn nth %d nt %lld router %s: selected %s, weights %s, ffn_out %s (mismatches %d %d %d, %d reps)\n",
                 nth, (long long) nt, rty == GGML_TYPE_F32 ? "F32" : "Q8_0", bad_ids ? "FAIL" : "ok",
                 bad_w ? "FAIL" : "ok", bad_o ? "FAIL" : "ok", bad_ids, bad_w, bad_o, reps);
    ggml_free(ctx);
    return ok ? 0 : 1;
}

// 0 = pass: for router types F32 and Q8_0, decode (1 token) and 3 tokens, with the given thread count.
int fg_selftest_fuse_ffn(int nth) {
    int fails = 0;
    for (ggml_type rty : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        for (int64_t nt : {1, 3}) {
            fails += fuse_ffn_case(nth, nt, rty);
        }
    }
    return fails ? 1 : 0;
}
