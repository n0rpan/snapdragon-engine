// flash_fuse_attn.inc: fused decode phase P7 (attention pre / post) for the Flash decode graph, one ggml custom node
// each. #included by flash_graph.cpp at file scope, after flash_fuse_gdn.inc (uses FuseBarrier, fuse_split,
// fuse_quant_rows, fuse_matvec_rows, fuse_matvec_ok, N_EMBD, N_HEAD, N_HEAD_KV, HEAD_D, EPS).
// Replaces (results/m3/sc-op-inventory.md, sections 2.8 and P7; build_layer_attn in flash_graph.cpp):
//   PRE  = AT.01-AT.12 and AT.14's input side: wq / wk / wv matvecs (one quantised input, cur), the gate split out of
//          the interleaved wq output (AT.10-AT.11), the q / k rms norms (AT.02-AT.04, AT.07-AT.09), the v reshape.
//          Outputs are the pre-rope q / k and the raw v, plus the gate (consumed by POST).
//   POST = AT.29-AT.31: sigmoid(gate) * attention output, then the wo matvec.
// Kept as ggml nodes (not fused, see the scope note below): AT.13 / AT.14 rope_multi (q and k), AT.16 / AT.18 KV
// set_rows, AT.19-AT.25 cache views and permutes, the fixed-chunk attention customs AT.26 / AT.27 (attn_part_op,
// attn_merge_op), the indexer QSA.01-QSA.33 (its projections, norms and rope included) and QSA.34-QSA.37
// (lightning, top-k, top-k mask). The indexer projections take cur too, but none of their outputs is an input of PRE
// or POST, so build_qsa_sel is unchanged and may run before or after the PRE node.
//
// Scope decision: RoPE stays a ggml node. results/m3/sc-ggml-numerics.md section 18 leaves the rope products'
// fused / unfused operand mapping UNVERIFIED and requires the running theta product, so bit-exact replication was not
// established from source. PRE therefore ends at the normed q / k (pre-rope), which is exactly what rope_multi reads.
//
// Numerics (results/m3/sc-ggml-numerics.md):
//   matvec   = weight type's vec_dot with nrc = 1 per (row, column), on cur quantised once per distinct vec_dot_type
//              (fuse_quant_rows / fuse_matvec_rows of flash_fuse_ffn.inc). Routing: fuse_matvec_ok (nrows == 1 types,
//              or odd nt for nrows == 2 types), and nt <= ATN_MAX_NT (= 7) as in flash_fuse_gdn.inc.
//   rms_norm = ggml_compute_forward_rms_norm_f32: one thread per (token, head) row, double sum in element order,
//              scale = 1 / sqrtf((float)(sum / 256) + EPS), then (x * scale) * w as the fused rms_norm_mul does.
//   sigmoid  = libm 1.f / (1.f + expf(-x)) per element (unary-ops.cpp).
//   mul      = one IEEE product per element (attn * sigmoid(gate)).
// Contraction: no product in this phase feeds an add (the rms sum adds a double widened from a float product, which
// GCC does not contract; the scale add is mean + eps with no product), so the helpers carry no asm barrier. This is
// argued from the source, not from a disassembly of the build the owner will compile.
//
// Output layout. Custom node dst = F32 with ne0 = total floats (scratch is process-wide static, not in dst).
// PRE, fuse_build_attn_pre(ctx, bar, cur [N_EMBD, nt], wq [N_EMBD, 12288], wk [N_EMBD, 512], wv [N_EMBD, 512],
//                          q_norm [HEAD_D], k_norm [HEAD_D]):
//   srcs {cur, wq, wk, wv, q_norm, k_norm}, userdata = FuseBarrier *, dst F32 [13312 * nt, 1].
//   Regions in floats (L = attn_pre_layout(nt)); each holds nt tokens back to back:
//     q    L.q = 0,        [HEAD_D, N_HEAD, nt]   nb = {4, 1024, 24576}  (normed, pre-rope; = AT.04 contiguous)
//     k    L.k = 6144 nt,  [HEAD_D, N_HEAD_KV, nt] nb = {4, 1024, 2048}  (normed, pre-rope; = AT.09)
//     v    L.v = 6656 nt,  [HEAD_D, N_HEAD_KV, nt] nb = {4, 1024, 2048}  (raw; = AT.12, the reshape of AT.06)
//     gate L.g = 7168 nt,  [N_HEAD * HEAD_D, nt]  nb = {4, 24576}       (= AT.11, the cont of AT.10)
//   Views (fuse_attn_pre_q / _k / _v / _gate) reproduce these ne / nb exactly.
//
// POST, fuse_build_attn_post(ctx, bar, att [6144, nt] (kqv_out after the reshape_2d), gate [6144, nt] (view of PRE),
//                            wo [6144, N_EMBD]):
//   srcs {att, gate, wo}, userdata = FuseBarrier *, dst F32 [N_EMBD, nt] contiguous (= AT.31 attn_output).
//
// Steps (PRE, 2 barriers; S2 is the last step):
//   S0  quantise cur per distinct vec_dot_type of wq / wk / wv (block-split)                         | barrier 1
//   S1  matvec rows over [0, 13312): wq rows -> qraw scratch (12288 wide), wk rows -> kraw scratch (512 wide),
//       wv rows -> v region directly; each (row, token) = vec_dot over K = N_EMBD                    | barrier 2
//   S2  q and k rows (token, head): rms_norm(EPS) * q_norm / k_norm, one thread per row; gate element copy from qraw
//       (gate of head h, dim d = qraw row h * 512 + 256 + d)
//
// Steps (POST, 2 barriers; S2 is the last step):
//   S0  gated[t][j] = att[t][j] * (1 / (1 + expf(-gate[t][j])))  (element split)                       | barrier 1
//   S1  quantise gated per token to vec_dot_type(wo) (block-split over K = 6144)                       | barrier 2
//   S2  wo rows (N_EMBD): vec_dot over K = 6144 per (row, token) into dst
//
// Caller side (per QSA layer, cur = the layer's attn-mix output [N_EMBD, nt]; c = the graph builder):
//   ggml_tensor * sel = build_qsa_sel(cur, il);                               // unchanged, kept nodes
//   ggml_tensor * f = fuse_build_attn_pre(ctx0, c.fuse_barrier(), cur, L.wq, L.wk, L.wv, L.attn_q_norm, L.attn_k_norm);
//   if (!f) { ...the existing Qfull / Kcur / Vcur / norm / gate nodes... }                      // nullptr: keep nodes
//   else {
//     ggml_tensor * q = fuse_attn_pre_q(ctx0, f, nt);      ggml_tensor * k = fuse_attn_pre_k(ctx0, f, nt);
//     ggml_tensor * v = fuse_attn_pre_v(ctx0, f, nt);      ggml_tensor * g = fuse_attn_pre_gate(ctx0, f, nt);
//     q = ggml_rope_multi(...q...);  k = ggml_rope_multi(...k...);                 // kept (AT.13, AT.14)
//     ggml_build_forward_expand / set_rows of k and of v = ggml_view_2d(ctx0, v, HEAD_D * N_HEAD_KV, nt, v->nb[2], 0)
//     cur = build_attn_qsa(q, k, v, sel, kq_scale, il);                            // kept, unchanged
//     ggml_tensor * o = fuse_build_attn_post(ctx0, c.fuse_barrier(), cur, g, L.wo);
//     if (!o) { ...the existing sigmoid / mul / mm(wo) nodes on cur and g... }
//   }
// Contract:
//   - Use a builder's result only when it is non-null; both builders check every ne / nb and the matvec routing.
//   - The PRE node reads cur and the weights only. It does not read the KV or indexer caches, so no n_kv dependence.
//   - The POST att input must be the contiguous [6144, nt] kqv_out (AT.28); the gate input must be the PRE gate view.
//   - Scratch is process-wide static sized for ATN_MAX_TOK tokens; one fused attention node runs at a time (as for
//     P1 / P3 / P4 / P5 / P6). Each node gets its own FuseBarrier (c.fuse_barrier()); no barrier is shared.
//   - The views are plain ggml views of the custom output; graph order comes from the data dependencies.
// ---------------------------------------------------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t ATN_NH = N_HEAD;                   // 24 query heads
constexpr int64_t ATN_NKV = N_HEAD_KV;               // 2 kv heads
constexpr int64_t ATN_D = HEAD_D;                    // 256 head dim
constexpr int64_t ATN_QW = 2 * ATN_D * ATN_NH;       // 12288: wq rows (per head: q 256, then gate 256)
constexpr int64_t ATN_KW = ATN_D * ATN_NKV;          // 512: wk / wv rows
constexpr int64_t ATN_QD = ATN_D * ATN_NH;           // 6144: attention output / gate width
constexpr int64_t ATN_MAX_TOK = 16;                  // scratch sized for this many tokens
constexpr int64_t ATN_MAX_NT = 7;                    // matvec routing (see the header comment)

static_assert(ATN_QD == 6144 && ATN_KW == 512 && ATN_QW == 12288, "production attention shapes");

// PRE output layout in floats for nt tokens.
struct AttnPreLayout {
    int64_t q, k, v, g, total;
};

AttnPreLayout attn_pre_layout(int64_t nt) {
    AttnPreLayout L;
    L.q = 0;
    L.k = L.q + ATN_QD * nt;
    L.v = L.k + ATN_KW * nt;
    L.g = L.v + ATN_KW * nt;
    L.total = L.g + ATN_QD * nt;
    return L;
}

// Scratch (process-wide, see the header comment).
alignas(64) float   g_attn_qraw[ATN_MAX_TOK * ATN_QW];               // wq matvec output [ATN_QW, nt]
alignas(64) float   g_attn_kraw[ATN_MAX_TOK * ATN_KW];               // wk matvec output [ATN_KW, nt]
alignas(64) uint8_t g_attn_q8[3][ATN_MAX_TOK * N_EMBD * sizeof(float)]; // cur quantised per matvec weight slot
alignas(64) float   g_attn_gated[ATN_MAX_TOK * ATN_QD];              // POST: att * sigmoid(gate) [ATN_QD, nt]
alignas(64) uint8_t g_attn_qo[ATN_MAX_TOK * ATN_QD * sizeof(float)]; // POST: gated quantised for wo

// ggml_compute_forward_rms_norm_f32 for one row (same as gdn_rms_scale, own copy so this file stands alone).
inline float attn_rms_scale(const float * x, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (double) (x[i] * x[i]);
    }
    const float mean = (float) (sum / (double) n);
    return 1.0f / sqrtf(mean + eps);
}

// PRE node. Srcs: see the header comment.
void fuse_attn_pre_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * w[3] = {dst->src[1], dst->src[2], dst->src[3]}; // wq, wk, wv
    const ggml_tensor * qnw_t = dst->src[4];
    const ggml_tensor * knw_t = dst->src[5];

    const int64_t nt = cur->ne[1];
    const AttnPreLayout L = attn_pre_layout(nt);
    float *       out = (float *) dst->data;
    const float * qnw = (const float *) qnw_t->data;
    const float * knw = (const float *) knw_t->data;
    int64_t lo = 0, hi = 0;

    // matvec weights: quantised operand per distinct vec_dot_type (qs[i] = slot holding weight i's operand)
    ggml_type vd[3];
    int qs[3];
    for (int i = 0; i < 3; ++i) {
        vd[i] = ggml_get_type_traits_cpu(w[i]->type)->vec_dot_type;
        qs[i] = i;
        for (int j = 0; j < i; ++j) {
            if (vd[j] == vd[i]) {
                qs[i] = qs[j];
                break;
            }
        }
    }

    // S0: quantise cur (block-split) once per distinct vec_dot_type
    for (int i = 0; i < 3; ++i) {
        if (qs[i] == i && vd[i] != GGML_TYPE_F32) {
            fuse_quant_rows(vd[i], (const char *) cur->data, cur->nb[1], nt, N_EMBD, (char *) g_attn_q8[i],
                            ggml_row_size(vd[i], N_EMBD), ith, nth);
        }
    }
    bar->wait(nth); // barrier 1

    // S1: matvec rows over the combined row space: wq (ATN_QW) | wk (ATN_KW) | wv (ATN_KW)
    const int64_t gs[4] = {0, ATN_QW, ATN_QW + ATN_KW, ATN_QW + 2 * ATN_KW};
    fuse_split(gs[3], ith, nth, 1, lo, hi);
    for (int g = 0; g < 3; ++g) {
        const int64_t a0 = std::max(lo, gs[g]);
        const int64_t a1 = std::min(hi, gs[g + 1]);
        if (a0 >= a1) continue;
        const bool f32 = vd[g] == GGML_TYPE_F32;
        const char * qin = f32 ? (const char *) cur->data : (const char *) g_attn_q8[qs[g]];
        const size_t qst = f32 ? cur->nb[1] : ggml_row_size(vd[g], N_EMBD);
        float * o = nullptr;
        int64_t ldo = 0;
        switch (g) {
            case 0:  o = g_attn_qraw; ldo = ATN_QW; break;
            case 1:  o = g_attn_kraw; ldo = ATN_KW; break;
            default: o = out + L.v;   ldo = ATN_KW; break;
        }
        fuse_matvec_rows(w[g], qin, qst, nt, o, ldo, a0 - gs[g], a1 - gs[g]);
    }
    bar->wait(nth); // barrier 2

    // S2a: q and k rows, rms_norm over HEAD_D per (token, head) row, one thread per row. Rows [0, NQ) are q rows
    // (t * N_HEAD + h), rows [NQ, NR) are k rows (t * N_HEAD_KV + h). Output row index = region row (rr * HEAD_D).
    const int64_t NQ = ATN_NH * nt;
    const int64_t NR = NQ + ATN_NKV * nt;
    fuse_split(NR, ith, nth, 1, lo, hi);
    for (int64_t r = lo; r < hi; ++r) {
        const bool isq = r < NQ;
        const int64_t rr = isq ? r : r - NQ;
        const int64_t nh = isq ? ATN_NH : ATN_NKV;
        const int64_t t = rr / nh;
        const int64_t h = rr % nh;
        const float * x = isq ? g_attn_qraw + t * ATN_QW + h * 2 * ATN_D : g_attn_kraw + t * ATN_KW + h * ATN_D;
        float *       y = out + (isq ? L.q : L.k) + rr * ATN_D;
        const float * wn = isq ? qnw : knw;
        const float scale = attn_rms_scale(x, ATN_D, EPS);
        for (int64_t i = 0; i < ATN_D; ++i) {
            y[i] = (x[i] * scale) * wn[i];
        }
    }

    // S2b: gate = the second half of each wq head row (AT.10 view at offset HEAD_D, then AT.11 cont), element split.
    fuse_split(nt * ATN_QD, ith, nth, 1, lo, hi);
    for (int64_t e = lo; e < hi; ++e) {
        const int64_t t = e / ATN_QD;
        const int64_t j = e % ATN_QD;
        const int64_t h = j / ATN_D;
        const int64_t d = j % ATN_D;
        out[L.g + e] = g_attn_qraw[t * ATN_QW + h * 2 * ATN_D + ATN_D + d];
    }
}

// POST node. Srcs: see the header comment.
void fuse_attn_post_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * att = dst->src[0];
    const ggml_tensor * gt  = dst->src[1];
    const ggml_tensor * wo  = dst->src[2];

    const int64_t nt = att->ne[1];
    float *       out = (float *) dst->data;
    const ggml_type vd = ggml_get_type_traits_cpu(wo->type)->vec_dot_type;
    int64_t lo = 0, hi = 0;

    // S0: gated[t][j] = att[t][j] * sigmoid(gate[t][j]) (graph order: cur * gate_sigmoid), element split
    fuse_split(nt * ATN_QD, ith, nth, 1, lo, hi);
    for (int64_t e = lo; e < hi; ++e) {
        const int64_t t = e / ATN_QD;
        const int64_t j = e % ATN_QD;
        const float x  = *(const float *) ((const char *) att->data + t * att->nb[1] + j * att->nb[0]);
        const float gv = *(const float *) ((const char *) gt->data + t * gt->nb[1] + j * gt->nb[0]);
        const float sig = 1.f / (1.f + expf(-gv));
        g_attn_gated[e] = x * sig;
    }
    bar->wait(nth); // barrier 1

    // S1: quantise gated (block-split) to vec_dot_type(wo)
    if (vd != GGML_TYPE_F32) {
        fuse_quant_rows(vd, (const char *) g_attn_gated, ATN_QD * sizeof(float), nt, ATN_QD, (char *) g_attn_qo,
                        ggml_row_size(vd, ATN_QD), ith, nth);
    }
    bar->wait(nth); // barrier 2

    // S2: wo rows (N_EMBD) per token into dst [N_EMBD, nt]
    const bool f32 = vd == GGML_TYPE_F32;
    fuse_split(N_EMBD, ith, nth, 1, lo, hi);
    fuse_matvec_rows(wo, f32 ? (const char *) g_attn_gated : (const char *) g_attn_qo,
                     f32 ? ATN_QD * sizeof(float) : ggml_row_size(vd, ATN_QD), nt, out, N_EMBD, lo, hi);
}

// Gate for PRE. Caller contract: use the fused node only when this returns true; otherwise keep the ggml nodes.
bool fuse_attn_pre_supported(const ggml_tensor * cur, const ggml_tensor * w_q, const ggml_tensor * w_k,
                             const ggml_tensor * w_v, const ggml_tensor * q_norm, const ggml_tensor * k_norm) {
    const int64_t nt = cur->ne[1];
    if (nt < 1 || nt > ATN_MAX_NT) return false;
    if (cur->type != GGML_TYPE_F32 || cur->ne[0] != N_EMBD || cur->ne[2] != 1 || cur->ne[3] != 1 ||
        cur->nb[0] != sizeof(float) || cur->nb[1] != N_EMBD * sizeof(float)) {
        return false;
    }
    if (!w_q || !fuse_matvec_ok(w_q, nt) || w_q->ne[0] != N_EMBD || w_q->ne[1] != ATN_QW) return false;
    if (!w_k || !fuse_matvec_ok(w_k, nt) || w_k->ne[0] != N_EMBD || w_k->ne[1] != ATN_KW) return false;
    if (!w_v || !fuse_matvec_ok(w_v, nt) || w_v->ne[0] != N_EMBD || w_v->ne[1] != ATN_KW) return false;
    for (const ggml_tensor * n : {q_norm, k_norm}) {
        if (n->type != GGML_TYPE_F32 || n->ne[0] != ATN_D || !ggml_is_contiguous(n)) return false;
    }
    return true;
}

// Gate for POST. Same contract as fuse_attn_pre_supported.
bool fuse_attn_post_supported(const ggml_tensor * att, const ggml_tensor * gate, const ggml_tensor * w_o) {
    const int64_t nt = att->ne[1];
    if (nt < 1 || nt > ATN_MAX_NT) return false;
    if (att->type != GGML_TYPE_F32 || att->ne[0] != ATN_QD || att->ne[2] != 1 || att->ne[3] != 1 ||
        att->nb[0] != sizeof(float) || att->nb[1] != ATN_QD * sizeof(float)) {
        return false;
    }
    if (gate->type != GGML_TYPE_F32 || gate->ne[0] != ATN_QD || gate->ne[1] != nt || gate->ne[2] != 1 ||
        gate->nb[0] != sizeof(float) || gate->nb[1] != ATN_QD * sizeof(float)) {
        return false;
    }
    if (!w_o || !fuse_matvec_ok(w_o, nt) || w_o->ne[0] != ATN_QD || w_o->ne[1] != N_EMBD) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Public builders and views (same translation unit as flash_graph.cpp).
// ---------------------------------------------------------------------------------------------------------------------

// PRE node. cur [N_EMBD, nt] F32, wq / wk / wv quantised [N_EMBD, 12288 / 512 / 512], q_norm / k_norm [HEAD_D] F32.
// Returns nullptr if unsupported (the caller keeps the ggml nodes).
ggml_tensor * fuse_build_attn_pre(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * cur, ggml_tensor * w_q,
                                  ggml_tensor * w_k, ggml_tensor * w_v, ggml_tensor * q_norm, ggml_tensor * k_norm) {
    if (!fuse_attn_pre_supported(cur, w_q, w_k, w_v, q_norm, k_norm)) {
        return nullptr;
    }
    const int64_t nt = cur->ne[1];
    const AttnPreLayout L = attn_pre_layout(nt);
    ggml_tensor * args[6] = {cur, w_q, w_k, w_v, q_norm, k_norm};
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, L.total, 1, 1, 1, args, 6, fuse_attn_pre_op, GGML_N_TASKS_MAX, bar);
}

// PRE views. Each reproduces the ne / nb of the production tensor it replaces.
// q: AT.04 [HEAD_D, N_HEAD, nt, 1], contiguous.
ggml_tensor * fuse_attn_pre_q(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const AttnPreLayout L = attn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, ATN_D, ATN_NH, nt, 1, ATN_D * sizeof(float), ATN_QD * sizeof(float),
                        ATN_QD * sizeof(float) * nt, L.q * sizeof(float));
}

// k: AT.09 [HEAD_D, N_HEAD_KV, nt, 1], contiguous.
ggml_tensor * fuse_attn_pre_k(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const AttnPreLayout L = attn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, ATN_D, ATN_NKV, nt, 1, ATN_D * sizeof(float), ATN_KW * sizeof(float),
                        ATN_KW * sizeof(float) * nt, L.k * sizeof(float));
}

// v: AT.12 reshape [HEAD_D, N_HEAD_KV, nt, 1], contiguous. The kept KV write reads view_2d(v, HEAD_D * N_HEAD_KV, nt,
// v->nb[2], 0), exactly as build_attn_qsa does with the production reshape.
ggml_tensor * fuse_attn_pre_v(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const AttnPreLayout L = attn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, ATN_D, ATN_NKV, nt, 1, ATN_D * sizeof(float), ATN_KW * sizeof(float),
                        ATN_KW * sizeof(float) * nt, L.v * sizeof(float));
}

// gate: AT.11 [N_HEAD * HEAD_D, nt], contiguous (the cont_2d of the interleaved wq output's gate half).
ggml_tensor * fuse_attn_pre_gate(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const AttnPreLayout L = attn_pre_layout(nt);
    return ggml_view_2d(ctx0, f, ATN_QD, nt, ATN_QD * sizeof(float), L.g * sizeof(float));
}

// POST node. att: kqv_out [ATN_QD, nt] contiguous (AT.28), gate: fuse_attn_pre_gate, wo: attn_output weight
// [ATN_QD, N_EMBD]. Returns nullptr if unsupported.
ggml_tensor * fuse_build_attn_post(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * att, ggml_tensor * gate,
                                   ggml_tensor * w_o) {
    if (!fuse_attn_post_supported(att, gate, w_o)) {
        return nullptr;
    }
    const int64_t nt = att->ne[1];
    ggml_tensor * args[3] = {att, gate, w_o};
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, N_EMBD, nt, 1, 1, args, 3, fuse_attn_post_op, GGML_N_TASKS_MAX, bar);
}

// ---------------------------------------------------------------------------------------------------------------------
// Model-free check.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// Bitwise equality of two F32 tensors element by element (strides read from each tensor).
bool attn_same_bits(const ggml_tensor * a, const ggml_tensor * b) {
    for (int d = 0; d < 4; ++d) {
        if (a->ne[d] != b->ne[d]) return false;
    }
    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < a->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
        const char * pa = (const char *) a->data + i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3];
        const char * pb = (const char *) b->data + i0 * b->nb[0] + i1 * b->nb[1] + i2 * b->nb[2] + i3 * b->nb[3];
        uint32_t xa = 0, xb = 0;
        std::memcpy(&xa, pa, sizeof(float));
        std::memcpy(&xb, pb, sizeof(float));
        if (xa != xb) return false;
    }
    return true;
}

// build_layer_attn's rope_multi call, as written there (applied to both reference and fused views).
ggml_tensor * attn_ref_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos) {
    int sections[4] = {11, 11, 10, 0};
    return ggml_rope_multi(ctx, x, pos, nullptr, N_ROT, sections, ROPE_TYPE, N_CTX_ORIG, FREQ_BASE, FREQ_SCALE,
                           EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW);
}

} // namespace

// Compares PRE + POST against the node sequence of build_layer_attn (Qfull / Kcur / Vcur matvecs, the q / k rms norms and
// muls, the gate view + cont, sigmoid, mul, the wo matvec), with the rope and the attention output kept as ggml nodes
// (the attention output is a shared F32 input here, standing in for AT.28). Cases: 1 and 3 tokens; Q8_0 and Q6_K
// weights at production shapes (N_EMBD 2560, 24 query heads, 2 kv heads, head dim 256). Bitwise on the replaced outputs
// q (pre-rope), k (pre-rope), v, gate and attn_output, plus the rope'd q / k and the v view read by the kept KV write,
// which checks the views. One line per case on stderr. Returns 0 on pass.
int fg_selftest_fuse_attn(int nth) {
    struct Case { int64_t nt; ggml_type t_q, t_k, t_v, t_o; };
    const Case cases[] = {
        {1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {1, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K},
        {3, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {3, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0},
    };
    int fails = 0;
    for (const Case & cs : cases) {
        const int64_t nt = cs.nt;
        // weights total < 60 MB at production shapes; activations are a few MB
        const size_t mem = (size_t) 128 << 20;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context * ctx = ggml_init(ip);
        uint64_t s = 0x2545F4914F6CDD1Dull ^ (uint64_t) nt;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
        auto fill = [&](ggml_tensor * t, float amp, float bias) {
            for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float *) t->data)[i] = rnd() * amp + bias;
        };
        // n rows of k values (ne0 = k, ne1 = n) of type ty, quantised row by row (keeps the temporary small)
        auto mat = [&](ggml_type ty, int64_t k, int64_t n, float amp) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, ty, k, n);
            std::vector<float> row((size_t) k);
            for (int64_t r = 0; r < n; ++r) {
                for (auto & v : row) v = rnd() * amp;
                ggml_quantize_chunk(ty, row.data(), (char *) t->data + r * t->nb[1], 0, 1, k, nullptr);
            }
            return t;
        };

        ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, nt);
        fill(cur, 1.0f, 0.0f);
        ggml_tensor * att = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ATN_QD, nt);   // stands in for AT.28
        fill(att, 1.0f, 0.0f);
        ggml_tensor * wq = mat(cs.t_q, N_EMBD, ATN_QW, 0.05f);
        ggml_tensor * wk = mat(cs.t_k, N_EMBD, ATN_KW, 0.05f);
        ggml_tensor * wv = mat(cs.t_v, N_EMBD, ATN_KW, 0.05f);
        ggml_tensor * wo = mat(cs.t_o, ATN_QD, N_EMBD, 0.05f);
        ggml_tensor * qnw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_D);
        fill(qnw, 0.1f, 1.0f);
        ggml_tensor * knw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_D);
        fill(knw, 0.1f, 1.0f);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * nt);   // 4 mrope positions per token
        for (int64_t i = 0; i < 4 * nt; ++i) ((int32_t *) pos->data)[i] = (int32_t) (100 + i / 4);

        // reference: build_layer_attn, node for node (AT.01-AT.04, AT.05, AT.06-AT.09, AT.10-AT.12, AT.13-AT.14, AT.29-31)
        ggml_cgraph * gr = ggml_new_graph(ctx);
        ggml_tensor * qf_r = ggml_mul_mat(ctx, wq, cur);                                       // AT.01 Qcur_full
        ggml_tensor * qv_r = ggml_view_3d(ctx, qf_r, HEAD_D, N_HEAD, nt, ATN_D * 2 * sizeof(float),
                                          ATN_D * 2 * N_HEAD * sizeof(float), 0);              // AT.02
        ggml_tensor * qn_r = ggml_mul(ctx, ggml_rms_norm(ctx, qv_r, EPS), qnw);                // AT.03-AT.04
        ggml_tensor * kr_raw = ggml_mul_mat(ctx, wk, cur);                                     // AT.05 Kcur
        ggml_tensor * k3_r = ggml_reshape_3d(ctx, kr_raw, HEAD_D, N_HEAD_KV, nt);             // AT.07
        ggml_tensor * kn_r = ggml_mul(ctx, ggml_rms_norm(ctx, k3_r, EPS), knw);                // AT.08-AT.09
        ggml_tensor * vr_raw = ggml_mul_mat(ctx, wv, cur);                                     // AT.06 Vcur
        ggml_tensor * v3_r = ggml_reshape_3d(ctx, vr_raw, HEAD_D, N_HEAD_KV, nt);             // AT.12
        ggml_tensor * gv_r = ggml_view_3d(ctx, qf_r, HEAD_D, N_HEAD, nt, ATN_D * 2 * sizeof(float),
                                          ATN_D * 2 * N_HEAD * sizeof(float), ATN_D * sizeof(float)); // AT.10
        ggml_tensor * g_r = ggml_cont_2d(ctx, gv_r, ATN_QD, nt);                               // AT.11
        ggml_tensor * qr_r = attn_ref_rope(ctx, qn_r, pos);                                    // AT.13 (kept)
        ggml_tensor * kc_r = attn_ref_rope(ctx, kn_r, pos);                                    // AT.14 (kept)
        ggml_tensor * vc_r = ggml_view_2d(ctx, v3_r, ATN_KW, nt, v3_r->nb[2], 0);              // AT.17 (KV write input)
        ggml_tensor * sg_r = ggml_sigmoid(ctx, g_r);                                           // AT.29
        ggml_tensor * gm_r = ggml_mul(ctx, att, sg_r);                                         // AT.30
        ggml_tensor * out_r = ggml_mul_mat(ctx, wo, gm_r);                                     // AT.31
        ggml_build_forward_expand(gr, qn_r);
        ggml_build_forward_expand(gr, kn_r);
        ggml_build_forward_expand(gr, v3_r);
        ggml_build_forward_expand(gr, g_r);
        ggml_build_forward_expand(gr, qr_r);
        ggml_build_forward_expand(gr, kc_r);
        ggml_build_forward_expand(gr, vc_r);
        ggml_build_forward_expand(gr, out_r);

        // fused: PRE, the kept views and rope, POST
        FuseBarrier bar_pre, bar_post;
        ggml_tensor * f = fuse_build_attn_pre(ctx, &bar_pre, cur, wq, wk, wv, qnw, knw);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * qn_f = nullptr, * kn_f = nullptr, * v3_f = nullptr, * g_f = nullptr;
        ggml_tensor * qr_f = nullptr, * kc_f = nullptr, * vc_f = nullptr, * out_f = nullptr;
        if (f) {
            qn_f = fuse_attn_pre_q(ctx, f, nt);
            kn_f = fuse_attn_pre_k(ctx, f, nt);
            v3_f = fuse_attn_pre_v(ctx, f, nt);
            g_f  = fuse_attn_pre_gate(ctx, f, nt);
            qr_f = attn_ref_rope(ctx, qn_f, pos);
            kc_f = attn_ref_rope(ctx, kn_f, pos);
            vc_f = ggml_view_2d(ctx, v3_f, ATN_KW, nt, v3_f->nb[2], 0);
            out_f = fuse_build_attn_post(ctx, &bar_post, att, g_f, wo);
        }
        if (!f || !out_f) {
            std::fprintf(stderr, "fg_selftest_fuse_attn nth %d tokens %d (wq %s, wk %s, wv %s, wo %s): unsupported "
                         "(matvec routing or shapes), FAIL\n", nth, (int) nt, ggml_type_name(cs.t_q),
                         ggml_type_name(cs.t_k), ggml_type_name(cs.t_v), ggml_type_name(cs.t_o));
            ++fails;
            ggml_free(ctx);
            continue;
        }
        ggml_build_forward_expand(gf, qn_f);
        ggml_build_forward_expand(gf, kn_f);
        ggml_build_forward_expand(gf, v3_f);
        ggml_build_forward_expand(gf, g_f);
        ggml_build_forward_expand(gf, qr_f);
        ggml_build_forward_expand(gf, kc_f);
        ggml_build_forward_expand(gf, vc_f);
        ggml_build_forward_expand(gf, out_f);

        ggml_graph_compute_with_ctx(ctx, gr, nth);
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        const bool ok_q  = attn_same_bits(qn_r, qn_f);
        const bool ok_k  = attn_same_bits(kn_r, kn_f);
        const bool ok_v  = attn_same_bits(v3_r, v3_f);
        const bool ok_g  = attn_same_bits(g_r, g_f);
        const bool ok_rq = attn_same_bits(qr_r, qr_f);
        const bool ok_rk = attn_same_bits(kc_r, kc_f);
        const bool ok_vc = attn_same_bits(vc_r, vc_f);
        const bool ok_o  = attn_same_bits(out_r, out_f);
        const bool ok = ok_q && ok_k && ok_v && ok_g && ok_rq && ok_rk && ok_vc && ok_o;
        std::fprintf(stderr,
                     "fg_selftest_fuse_attn nth %d tokens %d (wq %s, wk %s, wv %s, wo %s): bitwise %d "
                     "(q %d k %d v %d gate %d rope_q %d rope_k %d v_view %d attn_out %d)\n",
                     nth, (int) nt, ggml_type_name(cs.t_q), ggml_type_name(cs.t_k), ggml_type_name(cs.t_v),
                     ggml_type_name(cs.t_o), (int) ok, (int) ok_q, (int) ok_k, (int) ok_v, (int) ok_g, (int) ok_rq,
                     (int) ok_rk, (int) ok_vc, (int) ok_o);
        if (!ok) ++fails;
        ggml_free(ctx);
    }
    return fails == 0 ? 0 : 1;
}
