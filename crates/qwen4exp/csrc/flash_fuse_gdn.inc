// flash_fuse_gdn.inc: fused decode phases P3 (GDN pre) and P4 (GDN post) for the Flash decode graph, one ggml custom
// node each. #included by flash_graph.cpp at file scope, after flash_fuse_hc.inc and flash_fuse_ffn.inc (uses
// FuseBarrier, fuse_split, fuse_quant_rows, fuse_matvec_rows, fuse_matvec_ok, ggml_vec_silu_f32, N_EMBD_R, N_EMBD_S).
// Replaces (results/m3/sc-op-inventory.md, section 2.5 and P3 / P4):
//   P3 = GDN.01-GDN.32 of build_layer_attn_linear, plus the conv state write GDN.20:
//        qkv / z / beta / alpha matvecs, beta = sigmoid(beta), gate = softplus(alpha + ssm_dt) * ssm_a,
//        conv input = concat(stored conv state, qkv row), ssm_conv, silu, q / k: rms_norm(EPS/128) * (1/sqrt(128)),
//        conv state <- last SSM_CONV-1 columns of the conv input.
//        The GDN op GDN.33 (ggml_gated_delta_net) stays a ggml node and reads P3's views.
//   P4 = GDN.34-GDN.44: gated norm ((rms_norm(attn) * ssm_norm) * sigmoid(z)) and the ssm_out matvec, plus the ssm
//        state write GDN.37 (new_state -> ssm state rows).
//
// Numerics (results/m3/sc-ggml-numerics.md): matvec = weight type's vec_dot with nrc = 1 per (row, column) on the
// activation quantised once per distinct vec_dot_type (fuse_quant_rows / fuse_matvec_rows of flash_fuse_ffn.inc);
// elementwise ops use the op's own formula (libm expf / log1pf; ggml_vec_silu_f32 with chunk bounds on multiples of 4);
// the rms / l2 reductions run one thread per row, a sequential double sum in element order. ssm_conv is the
// ggml_compute_forward_ssm_conv_f32 loop with fmaf (sumf = fma(s, c, sumf) from 0.0f, tap order). No add of two
// products is written, so no FMA contraction question arises outside ssm_conv.
//
// Routing predicate (fuse_matvec_ok, flash_fuse_ffn.inc): nrows == 1 types, or nt odd for nrows == 2 types. Additionally
// nt <= GDN_MAX_NT (= 7): the tiled kernel of production starts at nt >= 8 for K-quant weights (flash_fuse_hc.inc notes).
//
// Threads and scratch: all nth threads of a custom node run the op with (ith, nth); every step is split over threads
// and dependent steps are separated by FuseBarrier::wait(nth). Every thread reaches every barrier. Scratch is
// process-wide static storage sized for GDN_MAX_TOK tokens (one fused GDN node runs at a time, as for P1 / P5 / P6).
//
// ---------------------------------------------------------------------------------------------------------------------
// P3 (fuse_build_gdn_pre): custom node, dst = F32 [total, 1], total = gdn_pre_layout(nt).total floats.
//   srcs: {cur [N_EMBD, nt], conv_rows [N_EMBD_R, 1] (state row 0, read in place and overwritten), w_qkv, w_z, w_beta,
//          w_alpha, ssm_conv1d [SSM_CONV, CONV_CH], ssm_dt [SSM_VH], ssm_a [SSM_VH]}, userdata = FuseBarrier *.
//   Regions (F32, each holds nt tokens back to back; byte offset = 4 * float offset). L = gdn_pre_layout(nt):
//     conv  [0, CONV_CH nt)        silu(ssm_conv) output, [CONV_CH, nt] (= production conv_output_silu)
//                                  v view = offset 2*GDN_QK floats in each token row, nb1 = 512, nb2 = CONV_CH*4
//     q     L.q, [SSM_D, SSM_KH, nt]  nb = {4, 512, 8192}   (q_conv_predelta, contiguous)
//     k     L.k, [SSM_D, SSM_KH, nt]  nb = {4, 512, 8192}   (k_conv_predelta, contiguous)
//     z     L.z, [GDN_V, nt]          nb = {4, 24576}       (z, flat; P4 views it as [SSM_D, SSM_VH, nt])
//     gate  L.g, [1, SSM_VH, nt]      nb = {4, 4, 192}      (contiguous, as the GDN op requires)
//     beta  L.b, [1, SSM_VH, nt]      nb = {4, 4, 192}
//   Views (fuse_gdn_pre_q/k/v/z/gate/beta) give exactly the ne / nb of the production tensors they replace.
//
//   Steps (5 barriers):
//     S0  quantise cur per distinct vec_dot_type of the 4 matvec weights (block-split)                 | barrier 1
//     S1  matvec rows over the combined row space [0, 16480): qkv 10240 (-> g_gdn_qkv), z 6144 (-> z region),
//         beta 48 (-> g_gdn_braw), alpha 48 (-> g_gdn_araw); each (row, token) = vec_dot over K = N_EMBD    | barrier 2
//     S2  (a) conv input per channel: cin[ch][0..2] = conv state (old values), cin[ch][3+t] = qkv[t][ch]
//         (b) per (token, head): beta = sigmoid(braw), gate = softplus(araw + dt[h]) * a[h]                | barrier 3
//         The old conv state is read only in S2.
//     S3  ssm_conv per channel (fmaf chain) -> g_gdn_craw; conv state write: state[ch][j] = cin[ch][nt + j]  | barrier 4
//         (the write is after barrier 3, i.e. after every thread's read of the old state)
//     S4  silu over the flat nt*CONV_CH array, chunk bounds multiples of 4 -> conv region                    | barrier 5
//     S5  q / k rows (2 * SSM_KH * nt): rms_norm with eps EPS/128, then * (1/sqrt(128)), one thread per row
//
// P4 (fuse_build_gdn_post): custom node, dst = F32 [N_EMBD, nt] contiguous = linear_attn_out.
//   srcs: {attn [SSM_D, SSM_VH, nt] view of the GDN result (nb 512, 24576), new_state [SSM_D, SSM_D, SSM_VH, 1] view
//          of the GDN result (nb 512, 65536), z [GDN_V, nt] (P3 z region), ssm_norm [SSM_D], ssm_out [GDN_V, N_EMBD],
//          state target [N_EMBD_S, 1] (view of the stored ssm state, written here)}, userdata = FuseBarrier *.
//   Steps (3 barriers):
//     S0  rms scale per (token, head) row (sequential double sum, eps EPS) ; ssm state write: new_state rows
//         (head, j) -> state rows 128 floats each (GDN.37, bytes as ggml_cpy)                              | barrier 1
//     S1  gated[t][h][d] = ((attn * rs) * ssm_norm[d]) * sigmoid(z[t][h][d])   (graph order)               | barrier 2
//     S2  quantise gated per token to vec_dot_type(ssm_out) (block-split)                                   | barrier 3
//     S3  ssm_out rows (N_EMBD): vec_dot over K = GDN_V per (row, token) into dst
//
// Caller side (per GDN layer, nt = cur->ne[1]; conv_rows = the build_rs view of r_l, as build_conv_state_at takes it;
// only valid under opts.rs_view with rs_z < 0, where build_rs returns exactly that view):
//   ggml_tensor * f = fuse_build_gdn_pre(ctx0, bar3, cur, conv_rows, wqkv, wqkv_gate, ssm_beta, ssm_alpha,
//                                        ssm_conv1d, ssm_dt, ssm_a);                                      // nullptr: keep nodes
//   q = fuse_gdn_pre_q(ctx0, f, nt); k = fuse_gdn_pre_k(...); v = fuse_gdn_pre_v(...);
//   gate = fuse_gdn_pre_gate(...); beta = fuse_gdn_pre_beta(...); z = fuse_gdn_pre_z(...);
//   state = reshape_4d(build_rs(ssm_states_all, N_EMBD_S), SSM_D, SSM_D, SSM_VH, 1);
//   result = ggml_gated_delta_net(ctx0, q, k, v, gate, beta, state, 1);                                    // unchanged
//   attn = fuse_gdn_attn_view(ctx0, result, nt); ns = fuse_gdn_state_view(ctx0, result, nt);
//   lin = fuse_build_gdn_post(ctx0, bar4, attn, ns, z, L.ssm_norm, L.ssm_out,
//                             fuse_gdn_state_target(ctx0, ssm_states_all));                              // nullptr: keep nodes
// ---------------------------------------------------------------------------------------------------------------------

namespace {

constexpr int64_t GDN_MAX_TOK = 16;         // scratch sized for this many tokens
constexpr int64_t GDN_MAX_NT = 7;           // matvec routing (see the header comment)
constexpr int64_t GDN_QK = SSM_D * SSM_KH;  // 2048: q (and k) width inside the conv output
constexpr int64_t GDN_V = SSM_D * SSM_VH;   // 6144: v width, z width, gated-output width
constexpr int64_t GDN_NS = SSM_CONV - 1;    // 3: conv state columns per channel

static_assert(GDN_NS == 3, "conv state layout");
static_assert(CONV_CH % 4 == 0, "silu chunks rely on CONV_CH % 4 == 0 (every row takes the NEON groups)");

// P3 output layout in floats (region offsets for nt tokens).
struct GdnPreLayout {
    int64_t conv, q, k, z, g, b, total;
};

GdnPreLayout gdn_pre_layout(int64_t nt) {
    GdnPreLayout L;
    L.conv = 0;
    L.q = L.conv + CONV_CH * nt;
    L.k = L.q + GDN_QK * nt;
    L.z = L.k + GDN_QK * nt;
    L.g = L.z + GDN_V * nt;
    L.b = L.g + SSM_VH * nt;
    L.total = L.b + SSM_VH * nt;
    return L;
}

// Scratch (process-wide, see the header comment).
alignas(64) float   g_gdn_qkv[GDN_MAX_TOK * CONV_CH];                   // qkv matvec output [CONV_CH, nt]
alignas(64) float   g_gdn_braw[GDN_MAX_TOK * SSM_VH];                   // beta matvec output [SSM_VH, nt]
alignas(64) float   g_gdn_araw[GDN_MAX_TOK * SSM_VH];                   // alpha matvec output [SSM_VH, nt]
alignas(64) float   g_gdn_cin[CONV_CH * (GDN_NS + GDN_MAX_TOK)];        // conv input, [GDN_NS + nt] per channel
alignas(64) float   g_gdn_craw[GDN_MAX_TOK * CONV_CH];                  // ssm_conv output before silu [CONV_CH, nt]
alignas(64) uint8_t g_gdn_q[4][GDN_MAX_TOK * N_EMBD * sizeof(float)];   // cur quantised per matvec weight slot
alignas(64) float   g_gdn_rs[GDN_MAX_TOK * SSM_VH];                     // P4 rms scale per (token, head)
alignas(64) float   g_gdn_gated[GDN_MAX_TOK * GDN_V];                   // P4 gated output [GDN_V, nt]
alignas(64) uint8_t g_gdn_qo[GDN_MAX_TOK * GDN_V * sizeof(float)];      // P4 gated output quantised for ssm_out

// ggml_compute_forward_rms_norm_f32 for one row: scale = 1 / sqrtf(mean + eps), mean = (float)(sum / n) with the
// sum in double, sequential in element order (one thread per row).
inline float gdn_rms_scale(const float * x, int64_t n, float eps) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (double) (x[i] * x[i]);
    }
    const float mean = (float) (sum / (double) n);
    return 1.0f / sqrtf(mean + eps);
}

// Matvec output rows are computed with fuse_matvec_rows (flash_fuse_ffn.inc): out[t * ldo + r] for r in [r0, r1).

// The P3 custom op. src: see the header comment.
void fuse_gdn_pre_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * cur  = dst->src[0];
    const ggml_tensor * crow = dst->src[1];
    const ggml_tensor * w[4] = {dst->src[2], dst->src[3], dst->src[4], dst->src[5]}; // qkv, z, beta, alpha
    const ggml_tensor * cw  = dst->src[6];
    const ggml_tensor * dtw = dst->src[7];
    const ggml_tensor * aw  = dst->src[8];

    const int64_t nt = cur->ne[1];
    const GdnPreLayout L = gdn_pre_layout(nt);
    float *       out  = (float *) dst->data;
    float *       st   = (float *) crow->data;   // conv state rows, state(j, ch) = st[ch * GDN_NS + j]
    const float * cwf  = (const float *) cw->data;
    const float * dtv  = (const float *) dtw->data;
    const float * av   = (const float *) aw->data;
    const int64_t ncs  = GDN_NS + nt;
    int64_t lo = 0, hi = 0;

    // matvec weights: quantised operand per distinct vec_dot_type (qs[i] = slot holding weight i's operand)
    ggml_type vd[4];
    int qs[4];
    for (int i = 0; i < 4; ++i) {
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
    for (int i = 0; i < 4; ++i) {
        if (qs[i] == i && vd[i] != GGML_TYPE_F32) {
            fuse_quant_rows(vd[i], (const char *) cur->data, cur->nb[1], nt, N_EMBD, (char *) g_gdn_q[i],
                            ggml_row_size(vd[i], N_EMBD), ith, nth);
        }
    }
    bar->wait(nth); // barrier 1

    // S1: matvec rows over the combined row space: qkv | z | beta | alpha
    const int64_t gs[5] = {0, CONV_CH, CONV_CH + GDN_V, CONV_CH + GDN_V + SSM_VH, CONV_CH + GDN_V + 2 * SSM_VH};
    fuse_split(gs[4], ith, nth, 1, lo, hi);
    for (int g = 0; g < 4; ++g) {
        const int64_t a0 = std::max(lo, gs[g]);
        const int64_t a1 = std::min(hi, gs[g + 1]);
        if (a0 >= a1) continue;
        const bool f32 = vd[g] == GGML_TYPE_F32;
        const char * qin = f32 ? (const char *) cur->data : (const char *) g_gdn_q[qs[g]];
        const size_t qst = f32 ? cur->nb[1] : ggml_row_size(vd[g], N_EMBD);
        float * o = nullptr;
        int64_t ldo = 0;
        switch (g) {
            case 0:  o = g_gdn_qkv;  ldo = CONV_CH; break;
            case 1:  o = out + L.z;  ldo = GDN_V;   break;
            case 2:  o = g_gdn_braw; ldo = SSM_VH;  break;
            default: o = g_gdn_araw; ldo = SSM_VH;  break;
        }
        fuse_matvec_rows(w[g], qin, qst, nt, o, ldo, a0 - gs[g], a1 - gs[g]);
    }
    bar->wait(nth); // barrier 2

    // S2a: conv input. Reads the OLD conv state (rows st), which S3 overwrites after barrier 3.
    fuse_split(CONV_CH, ith, nth, 1, lo, hi);
    for (int64_t ch = lo; ch < hi; ++ch) {
        float * c = g_gdn_cin + ch * ncs;
        for (int64_t k = 0; k < GDN_NS; ++k) c[k] = st[ch * GDN_NS + k];
        for (int64_t t = 0; t < nt; ++t) c[GDN_NS + t] = g_gdn_qkv[t * CONV_CH + ch];
    }
    // S2b: beta = sigmoid(beta_raw); gate = softplus(alpha_raw + dt) * a, per (token, head); element e = t * SSM_VH + h
    fuse_split(nt * SSM_VH, ith, nth, 1, lo, hi);
    for (int64_t e = lo; e < hi; ++e) {
        const int64_t h = e % SSM_VH;
        const float braw = g_gdn_braw[e];
        out[L.b + e] = 1.f / (1.f + expf(-braw));                          // ggml sigmoid
        const float ab = g_gdn_araw[e] + dtv[h];                          // ggml add (alpha + ssm_dt)
        const float sp = (ab > 20.0f) ? ab : log1pf(expf(ab));           // ggml softplus
        out[L.g + e] = sp * av[h];                                        // ggml mul (* ssm_a)
    }
    bar->wait(nth); // barrier 3

    // S3: ssm_conv (ggml_compute_forward_ssm_conv_f32: sumf = fma(s, c, sumf) from 0.0f, taps in order), and the
    // conv state write (GDN.20): state[ch][j] = cin[ch][nt + j], the last GDN_NS columns of the conv input.
    fuse_split(CONV_CH, ith, nth, 1, lo, hi);
    for (int64_t ch = lo; ch < hi; ++ch) {
        const float * c  = g_gdn_cin + ch * ncs;
        const float * wt = cwf + ch * SSM_CONV;
        for (int64_t t = 0; t < nt; ++t) {
            float sumf = 0.0f;
            for (int64_t i0 = 0; i0 < SSM_CONV; ++i0) {
                sumf = fmaf(c[t + i0], wt[i0], sumf);
            }
            g_gdn_craw[t * CONV_CH + ch] = sumf;
        }
        for (int64_t j = 0; j < GDN_NS; ++j) {
            st[ch * GDN_NS + j] = c[nt + j];
        }
    }
    bar->wait(nth); // barrier 4

    // S4: silu over the flat [nt, CONV_CH] array (ggml_vec_silu_f32 per row in production; CONV_CH % 4 == 0, so the
    // chunk bounds on multiples of 4 give the same NEON groups and no scalar tail)
    fuse_split(nt * CONV_CH, ith, nth, 4, lo, hi);
    if (hi > lo) {
        ggml_vec_silu_f32((int) (hi - lo), out + L.conv + lo, g_gdn_craw + lo);
    }
    bar->wait(nth); // barrier 5

    // S5: q and k. build_gdn_l2_norm: rms_norm(x, EPS / n) then ggml_scale by 1 / sqrtf(n) (two float products).
    // Row r: kind = r / (SSM_KH * nt) (0 = q, 1 = k), then token and head.
    const int64_t NRW = 2 * SSM_KH * nt;
    const float eps = EPS / (float) SSM_D;                 // EPS / n, float division as in the graph
    const float inv = 1.0f / sqrtf((float) SSM_D);         // 1.0f / sqrtf(n)
    fuse_split(NRW, ith, nth, 1, lo, hi);
    for (int64_t r = lo; r < hi; ++r) {
        const int64_t kind = r / (SSM_KH * nt);
        const int64_t rem = r % (SSM_KH * nt);
        const int64_t t = rem / SSM_KH;
        const int64_t h = rem % SSM_KH;
        const float * x = out + L.conv + t * CONV_CH + kind * GDN_QK + h * SSM_D;
        float *       y = out + (kind == 0 ? L.q : L.k) + t * GDN_QK + h * SSM_D;
        const float scale = gdn_rms_scale(x, SSM_D, eps);
        for (int64_t i = 0; i < SSM_D; ++i) {
            y[i] = (x[i] * scale) * inv;
        }
    }
}

// The P4 custom op. src: see the header comment.
void fuse_gdn_post_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * att = dst->src[0];
    const ggml_tensor * ns  = dst->src[1];
    const ggml_tensor * z   = dst->src[2];
    const ggml_tensor * wn  = dst->src[3];
    const ggml_tensor * wo  = dst->src[4];
    ggml_tensor *       sd  = dst->src[5];

    const int64_t nt = att->ne[2];
    float *       out = (float *) dst->data;
    const float * wnf = (const float *) wn->data;
    const ggml_type vd = ggml_get_type_traits_cpu(wo->type)->vec_dot_type;
    int64_t lo = 0, hi = 0;

    // S0: rms scale per (token, head) row; ssm state write (GDN.37): new_state (head h, row j) -> state row h*SSM_D + j
    fuse_split(nt * SSM_VH, ith, nth, 1, lo, hi);
    for (int64_t r = lo; r < hi; ++r) {
        const int64_t t = r / SSM_VH;
        const int64_t h = r % SSM_VH;
        const float * x = (const float *) ((const char *) att->data + t * att->nb[2] + h * att->nb[1]);
        g_gdn_rs[r] = gdn_rms_scale(x, SSM_D, EPS);
    }
    fuse_split(SSM_VH * SSM_D, ith, nth, 1, lo, hi);
    float * sdst = (float *) sd->data;
    for (int64_t q = lo; q < hi; ++q) {
        const int64_t h = q / SSM_D;
        const int64_t j = q % SSM_D;
        std::memcpy(sdst + q * SSM_D, (const char *) ns->data + h * ns->nb[2] + j * ns->nb[1], SSM_D * sizeof(float));
    }
    bar->wait(nth); // barrier 1

    // S1: gated[t][h][d] = ((attn * rs) * ssm_norm[d]) * sigmoid(z[t][h][d]), graph order
    fuse_split(nt * GDN_V, ith, nth, 1, lo, hi);
    for (int64_t e = lo; e < hi; ++e) {
        const int64_t t = e / GDN_V;
        const int64_t rem = e % GDN_V;
        const int64_t h = rem / SSM_D;
        const int64_t d = rem % SSM_D;
        const float x  = *(const float *) ((const char *) att->data + t * att->nb[2] + h * att->nb[1] + d * att->nb[0]);
        const float zv = *(const float *) ((const char *) z->data + t * z->nb[1] + rem * z->nb[0]);
        const float normalized = (x * g_gdn_rs[t * SSM_VH + h]) * wnf[d];
        const float gate = 1.f / (1.f + expf(-zv));
        g_gdn_gated[e] = normalized * gate;
    }
    bar->wait(nth); // barrier 2

    // S2: quantise gated (block-split) to vec_dot_type(ssm_out)
    if (vd != GGML_TYPE_F32) {
        fuse_quant_rows(vd, (const char *) g_gdn_gated, GDN_V * sizeof(float), nt, GDN_V, (char *) g_gdn_qo,
                        ggml_row_size(vd, GDN_V), ith, nth);
    }
    bar->wait(nth); // barrier 3

    // S3: ssm_out rows (N_EMBD) per token into dst [N_EMBD, nt]
    const bool f32 = vd == GGML_TYPE_F32;
    fuse_split(N_EMBD, ith, nth, 1, lo, hi);
    fuse_matvec_rows(wo, f32 ? (const char *) g_gdn_gated : (const char *) g_gdn_qo,
                     f32 ? GDN_V * sizeof(float) : ggml_row_size(vd, GDN_V), nt, out, N_EMBD, lo, hi);
}

// Gate for P3. Caller contract: use the fused node only when this returns true; otherwise keep the ggml nodes.
bool fuse_gdn_pre_supported(const ggml_tensor * cur, const ggml_tensor * crow, const ggml_tensor * w_qkv,
                            const ggml_tensor * w_z, const ggml_tensor * w_beta, const ggml_tensor * w_alpha,
                            const ggml_tensor * conv1d, const ggml_tensor * dt, const ggml_tensor * a) {
    const int64_t nt = cur->ne[1];
    if (nt < 1 || nt > GDN_MAX_NT) return false;
    if (cur->type != GGML_TYPE_F32 || cur->ne[0] != N_EMBD || cur->nb[0] != sizeof(float) ||
        cur->nb[1] != N_EMBD * sizeof(float)) {
        return false;
    }
    if (crow->type != GGML_TYPE_F32 || crow->ne[0] != N_EMBD_R || crow->nb[0] != sizeof(float) || !ggml_is_contiguous(crow)) {
        return false;
    }
    if (!w_qkv || !fuse_matvec_ok(w_qkv, nt) || w_qkv->ne[0] != N_EMBD || w_qkv->ne[1] != CONV_CH) return false;
    if (!w_z || !fuse_matvec_ok(w_z, nt) || w_z->ne[0] != N_EMBD || w_z->ne[1] != GDN_V) return false;
    if (!w_beta || !fuse_matvec_ok(w_beta, nt) || w_beta->ne[0] != N_EMBD || w_beta->ne[1] != SSM_VH) return false;
    if (!w_alpha || !fuse_matvec_ok(w_alpha, nt) || w_alpha->ne[0] != N_EMBD || w_alpha->ne[1] != SSM_VH) return false;
    if (conv1d->type != GGML_TYPE_F32 || conv1d->ne[0] != SSM_CONV || conv1d->ne[1] != CONV_CH ||
        conv1d->nb[0] != sizeof(float) || conv1d->nb[1] != SSM_CONV * sizeof(float)) {
        return false;
    }
    for (const ggml_tensor * v : {dt, a}) {
        if (v->type != GGML_TYPE_F32 || v->ne[0] != SSM_VH || !ggml_is_contiguous(v)) return false;
    }
    return true;
}

// Gate for P4. Same contract as fuse_gdn_pre_supported.
bool fuse_gdn_post_supported(const ggml_tensor * att, const ggml_tensor * ns, const ggml_tensor * z,
                             const ggml_tensor * w_norm, const ggml_tensor * w_out, const ggml_tensor * st) {
    const int64_t nt = att->ne[2];
    if (nt < 1 || nt > GDN_MAX_NT) return false;
    if (att->type != GGML_TYPE_F32 || att->ne[0] != SSM_D || att->ne[1] != SSM_VH || att->nb[0] != sizeof(float) ||
        att->nb[1] != SSM_D * sizeof(float)) {
        return false;
    }
    if (ns->type != GGML_TYPE_F32 || ns->ne[0] != SSM_D || ns->ne[1] != SSM_D || ns->ne[2] != SSM_VH ||
        ns->nb[0] != sizeof(float) || ns->nb[1] != SSM_D * sizeof(float) || ns->nb[2] != SSM_D * SSM_D * sizeof(float)) {
        return false;
    }
    if (z->type != GGML_TYPE_F32 || z->ne[0] != GDN_V || z->ne[1] != nt || z->nb[0] != sizeof(float)) return false;
    if (w_norm->type != GGML_TYPE_F32 || w_norm->ne[0] != SSM_D || !ggml_is_contiguous(w_norm)) return false;
    if (!w_out || !fuse_matvec_ok(w_out, nt) || w_out->ne[0] != GDN_V || w_out->ne[1] != N_EMBD) return false;
    if (st->type != GGML_TYPE_F32 || st->ne[0] != N_EMBD_S || !ggml_is_contiguous(st)) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Public builders and views (same translation unit as flash_graph.cpp).
// ---------------------------------------------------------------------------------------------------------------------

// P3 node. cur [N_EMBD, nt] F32, conv_rows [N_EMBD_R, 1] F32 (the state row, from build_rs), the four matvec weights
// (qkv, z, beta, alpha), ssm_conv1d [SSM_CONV, CONV_CH], ssm_dt [SSM_VH], ssm_a [SSM_VH]. Returns nullptr if unsupported.
ggml_tensor * fuse_build_gdn_pre(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * cur, ggml_tensor * conv_rows,
                                 ggml_tensor * w_qkv, ggml_tensor * w_z, ggml_tensor * w_beta, ggml_tensor * w_alpha,
                                 ggml_tensor * conv1d, ggml_tensor * dt, ggml_tensor * a) {
    if (!fuse_gdn_pre_supported(cur, conv_rows, w_qkv, w_z, w_beta, w_alpha, conv1d, dt, a)) {
        return nullptr;
    }
    const int64_t nt = cur->ne[1];
    const GdnPreLayout L = gdn_pre_layout(nt);
    ggml_tensor * args[9] = {cur, conv_rows, w_qkv, w_z, w_beta, w_alpha, conv1d, dt, a};
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, L.total, 1, 1, 1, args, 9, fuse_gdn_pre_op, GGML_N_TASKS_MAX, bar);
}

// P3 views. Each one reproduces the ne / nb of the production tensor it replaces (build_layer_attn_linear).
// q_conv_predelta [SSM_D, SSM_KH, nt, 1] contiguous rows; k likewise.
ggml_tensor * fuse_gdn_pre_q(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const GdnPreLayout L = gdn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, SSM_D, SSM_KH, nt, 1, SSM_D * sizeof(float), GDN_QK * sizeof(float),
                        GDN_QK * sizeof(float) * nt, L.q * sizeof(float));
}

ggml_tensor * fuse_gdn_pre_k(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const GdnPreLayout L = gdn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, SSM_D, SSM_KH, nt, 1, SSM_D * sizeof(float), GDN_QK * sizeof(float),
                        GDN_QK * sizeof(float) * nt, L.k * sizeof(float));
}

// v_conv: [SSM_D, SSM_VH, nt, 1] with the production strides of the conv output view (nb1 = 512, nb2 = CONV_CH * 4).
ggml_tensor * fuse_gdn_pre_v(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    return ggml_view_4d(ctx0, f, SSM_D, SSM_VH, nt, 1, SSM_D * sizeof(float), CONV_CH * sizeof(float),
                        CONV_CH * sizeof(float) * nt, 2 * GDN_QK * sizeof(float));
}

// z [GDN_V, nt] flat (P4 reads it as [SSM_D, SSM_VH, nt] with nb1 = 512, nb2 = 24576 bytes, the production z_2d).
ggml_tensor * fuse_gdn_pre_z(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const GdnPreLayout L = gdn_pre_layout(nt);
    return ggml_view_2d(ctx0, f, GDN_V, nt, GDN_V * sizeof(float), L.z * sizeof(float));
}

// g [1, SSM_VH, nt, 1] contiguous, as the GDN op requires (production: reshape_4d of gate).
ggml_tensor * fuse_gdn_pre_gate(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const GdnPreLayout L = gdn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, 1, SSM_VH, nt, 1, sizeof(float), SSM_VH * sizeof(float), SSM_VH * sizeof(float) * nt,
                        L.g * sizeof(float));
}

// beta [1, SSM_VH, nt, 1] contiguous (production: sigmoid of reshape_4d of beta).
ggml_tensor * fuse_gdn_pre_beta(ggml_context * ctx0, ggml_tensor * f, int64_t nt) {
    const GdnPreLayout L = gdn_pre_layout(nt);
    return ggml_view_4d(ctx0, f, 1, SSM_VH, nt, 1, sizeof(float), SSM_VH * sizeof(float), SSM_VH * sizeof(float) * nt,
                        L.b * sizeof(float));
}

// GDN result views, as build_layer_attn_linear takes them (output = attn [SSM_D, SSM_VH, nt]; new_state).
ggml_tensor * fuse_gdn_attn_view(ggml_context * ctx0, ggml_tensor * res, int64_t nt) {
    const ggml_type t = res->type;
    return ggml_view_4d(ctx0, res, SSM_D, SSM_VH, nt, 1, ggml_row_size(t, SSM_D), ggml_row_size(t, SSM_D * SSM_VH),
                        ggml_row_size(t, SSM_D * SSM_VH * nt), 0);
}

ggml_tensor * fuse_gdn_state_view(ggml_context * ctx0, ggml_tensor * res, int64_t nt) {
    const ggml_type t = res->type;
    return ggml_view_4d(ctx0, res, SSM_D, SSM_D, SSM_VH, 1, ggml_row_size(t, SSM_D), ggml_row_size(t, SSM_D * SSM_D),
                        ggml_row_size(t, SSM_D * SSM_D * SSM_VH), ggml_row_size(t, SSM_D * SSM_VH * nt));
}

// The state rows the GDN state write targets: view_2d(ssm_states_all, N_EMBD_S, 1, nb1, 0) (production GDN.36).
ggml_tensor * fuse_gdn_state_target(ggml_context * ctx0, ggml_tensor * s_all) {
    return ggml_view_2d(ctx0, s_all, N_EMBD_S, 1, s_all->nb[1], 0);
}

// P4 node. att / ns: the GDN result views (fuse_gdn_attn_view / fuse_gdn_state_view), z: fuse_gdn_pre_z,
// w_norm: ssm_norm [SSM_D], w_out: ssm_out [GDN_V, N_EMBD], st: fuse_gdn_state_target. Returns nullptr if unsupported.
ggml_tensor * fuse_build_gdn_post(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * att, ggml_tensor * ns,
                                  ggml_tensor * z, ggml_tensor * w_norm, ggml_tensor * w_out, ggml_tensor * st) {
    if (!fuse_gdn_post_supported(att, ns, z, w_norm, w_out, st)) {
        return nullptr;
    }
    const int64_t nt = att->ne[2];
    ggml_tensor * args[6] = {att, ns, z, w_norm, w_out, st};
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, N_EMBD, nt, 1, 1, args, 6, fuse_gdn_post_op, GGML_N_TASKS_MAX, bar);
}

// ---------------------------------------------------------------------------------------------------------------------
// Model-free check.
// ---------------------------------------------------------------------------------------------------------------------

namespace {

// Bitwise equality of two F32 tensors element by element (strides read from each tensor).
bool gdn_same_bits(const ggml_tensor * a, const ggml_tensor * b) {
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

// The graph's build_gdn_l2_norm, as written there.
ggml_tensor * gdn_ref_l2(ggml_context * ctx, ggml_tensor * x) {
    const float n = (float) x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, EPS / n), 1.0f / sqrtf(n));
}

} // namespace

// Compares P3 + ggml_gated_delta_net + P4 against the node sequence of build_layer_attn_linear (with the same
// ggml_gated_delta_net), for 1 and 3 tokens and synthetic Q8_0 / Q6_K / F32 weights. Bitwise on q, k, v, gate, beta, z,
// the conv state and the ssm state after the writes, the GDN outputs and linear_attn_out. One line per case on stderr.
// Returns 0 on pass.
int fg_selftest_fuse_gdn(int nth) {
    struct Case { int64_t nt; ggml_type t_qkv, t_z, t_ba, t_out; };
    const Case cases[] = {
        {1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32,  GGML_TYPE_Q8_0},
        {1, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K},
        {3, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {3, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F32,  GGML_TYPE_Q6_K},
    };
    const int64_t C = CONV_CH, NV = SSM_VH, D = SSM_D;
    int fails = 0;
    for (const Case & cs : cases) {
        const int64_t nt = cs.nt;
        const size_t mem = (size_t) 256 << 20;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context * ctx = ggml_init(ip);
        uint64_t s = 0x2545F4914F6CDD1Dull;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
        auto fill = [&](ggml_tensor * t, float amp, float bias) {
            for (int64_t i = 0; i < ggml_nelements(t); ++i) ((float *) t->data)[i] = rnd() * amp + bias;
        };
        // k x n matrix (ne0 = k, ne1 = n) of type ty, quantised like the other selftests
        auto mat = [&](ggml_type ty, int64_t k, int64_t n, float amp) {
            std::vector<float> w((size_t) (k * n));
            for (auto & v : w) v = rnd() * amp;
            ggml_tensor * t = ggml_new_tensor_2d(ctx, ty, k, n);
            if (ty == GGML_TYPE_F32) {
                std::memcpy(t->data, w.data(), w.size() * sizeof(float));
            } else {
                ggml_quantize_chunk(ty, w.data(), t->data, 0, n, k, nullptr);
            }
            return t;
        };

        ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, nt);
        fill(cur, 1.0f, 0.0f);
        ggml_tensor * conv_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD_R, 1);
        ggml_tensor * conv_fus = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD_R, 1);
        fill(conv_ref, 0.5f, 0.0f);
        std::memcpy(conv_fus->data, conv_ref->data, ggml_nbytes(conv_ref));
        ggml_tensor * ssm_ref = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD_S, 1);
        ggml_tensor * ssm_fus = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD_S, 1);
        fill(ssm_ref, 0.05f, 0.0f);
        std::memcpy(ssm_fus->data, ssm_ref->data, ggml_nbytes(ssm_ref));

        ggml_tensor * wqkv = mat(cs.t_qkv, N_EMBD, C, 0.05f);
        ggml_tensor * wz   = mat(cs.t_z, N_EMBD, GDN_V, 0.05f);
        ggml_tensor * wba  = mat(cs.t_ba, N_EMBD, NV, 0.05f);
        ggml_tensor * wal  = mat(cs.t_ba, N_EMBD, NV, 0.05f);
        ggml_tensor * conv1d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SSM_CONV, C);
        fill(conv1d, 0.5f, 0.0f);
        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NV);
        fill(dt, 0.5f, 0.0f);
        ggml_tensor * aa = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NV);
        fill(aa, 0.3f, -0.3f);
        ggml_tensor * wnorm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        fill(wnorm, 0.1f, 1.0f);
        ggml_tensor * wout = mat(cs.t_out, GDN_V, N_EMBD, 0.05f);

        // reference: build_layer_attn_linear (rs_view state rows), node for node
        ggml_cgraph * gr = ggml_new_graph(ctx);
        ggml_tensor * crow_r = ggml_view_2d(ctx, conv_ref, N_EMBD_R, 1, conv_ref->nb[1], 0);
        ggml_tensor * srow_r = ggml_view_2d(ctx, ssm_ref, N_EMBD_S, 1, ssm_ref->nb[1], 0);
        ggml_tensor * state_r = ggml_reshape_4d(ctx, srow_r, D, D, NV, 1);
        ggml_tensor * qkv = ggml_mul_mat(ctx, wqkv, cur);
        ggml_tensor * qkv3 = ggml_reshape_3d(ctx, qkv, C, nt, 1);
        ggml_tensor * z_r = ggml_mul_mat(ctx, wz, cur);
        ggml_tensor * beta_r = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, wba, cur), 1, NV, nt, 1);
        ggml_tensor * alpha_r = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wal, cur), NV, nt, 1);
        ggml_tensor * beta_s = ggml_sigmoid(ctx, beta_r);
        ggml_tensor * alpha_b = ggml_add(ctx, alpha_r, dt);
        ggml_tensor * gate_r = ggml_mul(ctx, ggml_softplus(ctx, alpha_b), aa);
        gate_r = ggml_reshape_4d(ctx, gate_r, 1, NV, nt, 1);
        // build_conv_state_at
        ggml_tensor * cst = ggml_reshape_3d(ctx, crow_r, GDN_NS, C, 1);
        ggml_tensor * cin = ggml_concat(ctx, cst, ggml_transpose(ctx, qkv3), 0);
        ggml_tensor * ctail = ggml_view_3d(ctx, cin, GDN_NS, C, 1, cin->nb[1], cin->nb[2], ggml_row_size(cin->type, nt));
        ggml_tensor * cdst = ggml_view_2d(ctx, conv_ref, GDN_NS * C, 1, conv_ref->nb[1], 0);
        ggml_build_forward_expand(gr, ggml_cpy(ctx, ctail, cdst));
        // conv output and the q / k / v views
        ggml_tensor * silu_r = ggml_silu(ctx, ggml_ssm_conv(ctx, cin, conv1d));
        const size_t rowC = ggml_row_size(GGML_TYPE_F32, C);
        ggml_tensor * q_r = gdn_ref_l2(ctx, ggml_view_4d(ctx, silu_r, D, SSM_KH, nt, 1, ggml_row_size(GGML_TYPE_F32, D),
                                                         rowC, rowC * nt, 0));
        ggml_tensor * k_r = gdn_ref_l2(ctx, ggml_view_4d(ctx, silu_r, D, SSM_KH, nt, 1, ggml_row_size(GGML_TYPE_F32, D),
                                                         rowC, rowC * nt, ggml_row_size(GGML_TYPE_F32, GDN_QK)));
        ggml_tensor * v_r = ggml_view_4d(ctx, silu_r, D, NV, nt, 1, ggml_row_size(GGML_TYPE_F32, D), rowC, rowC * nt,
                                         ggml_row_size(GGML_TYPE_F32, 2 * GDN_QK));
        ggml_tensor * gdn_r = ggml_gated_delta_net(ctx, q_r, k_r, v_r, gate_r, beta_s, state_r, 1);
        const size_t rD = ggml_row_size(GGML_TYPE_F32, D);
        ggml_tensor * out_r = ggml_view_4d(ctx, gdn_r, D, NV, nt, 1, rD, ggml_row_size(GGML_TYPE_F32, D * NV),
                                           ggml_row_size(GGML_TYPE_F32, D * NV * nt), 0);
        ggml_tensor * ns_r = ggml_view_4d(ctx, gdn_r, D, D, NV, 1, rD, ggml_row_size(GGML_TYPE_F32, D * D),
                                          ggml_row_size(GGML_TYPE_F32, D * D * NV), ggml_row_size(GGML_TYPE_F32, D * NV * nt));
        ggml_build_forward_expand(gr, ggml_cpy(ctx, ns_r, ggml_view_2d(ctx, ssm_ref, N_EMBD_S, 1, ssm_ref->nb[1], 0)));
        ggml_tensor * z2_r = ggml_view_4d(ctx, z_r, D, NV, nt, 1, rD, z_r->nb[1], z_r->nb[1] * nt, 0);
        ggml_tensor * norm_r = ggml_mul(ctx, ggml_rms_norm(ctx, out_r, EPS), wnorm);
        ggml_tensor * attn_r = ggml_mul(ctx, norm_r, ggml_sigmoid(ctx, z2_r));
        ggml_tensor * lin_r = ggml_mul_mat(ctx, wout, ggml_reshape_2d(ctx, attn_r, GDN_V, nt));
        ggml_build_forward_expand(gr, lin_r);

        // fused: P3, the same GDN op on P3's views, P4
        FuseBarrier bar3, bar4;
        ggml_tensor * crow_f = ggml_view_2d(ctx, conv_fus, N_EMBD_R, 1, conv_fus->nb[1], 0);
        ggml_tensor * f = fuse_build_gdn_pre(ctx, &bar3, cur, crow_f, wqkv, wz, wba, wal, conv1d, dt, aa);
        ggml_tensor * lin_f = nullptr;
        ggml_tensor * q_f = nullptr, * k_f = nullptr, * v_f = nullptr, * g_f = nullptr, * b_f = nullptr, * z_f = nullptr;
        ggml_tensor * out_f = nullptr, * ns_f = nullptr;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        if (f) {
            q_f = fuse_gdn_pre_q(ctx, f, nt);
            k_f = fuse_gdn_pre_k(ctx, f, nt);
            v_f = fuse_gdn_pre_v(ctx, f, nt);
            g_f = fuse_gdn_pre_gate(ctx, f, nt);
            b_f = fuse_gdn_pre_beta(ctx, f, nt);
            z_f = fuse_gdn_pre_z(ctx, f, nt);
            ggml_tensor * srow_f = ggml_view_2d(ctx, ssm_fus, N_EMBD_S, 1, ssm_fus->nb[1], 0);
            ggml_tensor * state_f = ggml_reshape_4d(ctx, srow_f, D, D, NV, 1);
            ggml_tensor * gdn_f = ggml_gated_delta_net(ctx, q_f, k_f, v_f, g_f, b_f, state_f, 1);
            out_f = fuse_gdn_attn_view(ctx, gdn_f, nt);
            ns_f = fuse_gdn_state_view(ctx, gdn_f, nt);
            lin_f = fuse_build_gdn_post(ctx, &bar4, out_f, ns_f, z_f, wnorm, wout, fuse_gdn_state_target(ctx, ssm_fus));
        }
        if (!f || !lin_f) {
            std::fprintf(stderr, "fg_selftest_fuse_gdn nth %d tokens %d: unsupported (matvec routing), skipped\n", nth,
                         (int) nt);
            ggml_free(ctx);
            continue;
        }
        ggml_build_forward_expand(gf, lin_f);

        ggml_graph_compute_with_ctx(ctx, gr, nth);
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        const bool ok_q = gdn_same_bits(q_r, q_f);
        const bool ok_k = gdn_same_bits(k_r, k_f);
        const bool ok_v = gdn_same_bits(v_r, v_f);
        const bool ok_g = gdn_same_bits(gate_r, g_f);
        const bool ok_b = gdn_same_bits(beta_s, b_f);
        const bool ok_z = gdn_same_bits(z_r, z_f);
        const bool ok_c = gdn_same_bits(conv_ref, conv_fus);
        const bool ok_s = gdn_same_bits(ssm_ref, ssm_fus);
        const bool ok_o = gdn_same_bits(out_r, out_f);
        const bool ok_n = gdn_same_bits(ns_r, ns_f);
        const bool ok_l = gdn_same_bits(lin_r, lin_f);
        const bool ok = ok_q && ok_k && ok_v && ok_g && ok_b && ok_z && ok_c && ok_s && ok_o && ok_n && ok_l;
        std::fprintf(stderr,
                     "fg_selftest_fuse_gdn nth %d tokens %d (qkv %s, z %s, beta/alpha %s, ssm_out %s): bitwise %d "
                     "(q %d k %d v %d gate %d beta %d z %d conv %d ssm %d gdn %d ns %d lin %d)\n",
                     nth, (int) nt, ggml_type_name(cs.t_qkv), ggml_type_name(cs.t_z), ggml_type_name(cs.t_ba),
                     ggml_type_name(cs.t_out), (int) ok, (int) ok_q, (int) ok_k, (int) ok_v, (int) ok_g, (int) ok_b,
                     (int) ok_z, (int) ok_c, (int) ok_s, (int) ok_o, (int) ok_n, (int) ok_l);
        if (!ok) ++fails;
        ggml_free(ctx);
    }
    return fails == 0 ? 0 : 1;
}
