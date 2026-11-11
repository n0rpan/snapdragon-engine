// flash_fuse_hc.inc: fused phase P1 (hc-mix) for the Flash decode graph, one ggml custom node.
// Included by flash_graph.cpp at file scope, after build_hc_mix's owner (struct G) and fg_ctx are defined.
// Replaces the first half of build_hc_mix for il >= 0 (results/m3/sc-op-inventory.md, P1):
//   xn     = rms_norm(x, EPS) * w_norm                       [N_EMBD, HC, nt]
//   lo     = silu(scale(mm(w_down, xn), 1/HC))               [HC_LR, nt]
//   gate   = mm(w_up, lo)                                    [N_EMBD, HC, nt]
//   inject = scale(mm(w_inject, xn), 1/HC)                   [HC, nt]   (only if w_inject)
// DSV4_HC_PRE (ggml_dsv4_hc_pre_gated over xn and gate) stays a ggml node.
//
// Numerics: every product is ggml's production symbol or the same expression ggml evaluates: vec_dot / from_float from
// the CPU type traits (ggml-cpu.c type_traits_cpu), ggml_vec_silu_f32 (vec.cpp), rms sum in double over float products in
// element order (ops.cpp ggml_compute_forward_rms_norm_f32), y = (x*scale)*w, y = x*(1/HC). No add-of-products is written
// here, so no FMA contraction question arises. The vec_dot path equals ggml's mul_mat only where ggml does not divert to
// llamafile_sgemm or the tiled kernel: see fuse_hc_mix_supported().
//
// Threads and scratch: all nth threads of the custom node run fuse_hc_mix_op(ith, nth). Steps are split over threads
// (FuseBarrier between dependent steps). Scratch is process-wide static storage shared by the threads of one call,
// sized for FUSE_MAX_TOK tokens. Assumes one fused hc-mix node runs at a time (ggml runs graph nodes one after another).
//
// Step / barrier structure (6 barriers):
//   a1  rms scale per (token, hc row): one thread per row, ggml's sequential double sum            -> barrier 1
//   a2  xn = (x * scale) * w_norm written into dst columns [0, HC*N_EMBD)                          -> barrier 2
//   b   quantise xn per token to vec_dot_type(w_down) (and w_inject if different), blocks split  -> barrier 3
//   c   down rows (HC_LR) and inject rows: vec_dot(nrc 1) per (token, row), scaled by 1/HC        -> barrier 4
//   d   silu over the nt*HC_LR scaled lo values, chunk bounds multiples of 4                      -> barrier 5
//   e   quantise lo per token to vec_dot_type(w_up), blocks split                                  -> barrier 6
//   f   up rows (HC*N_EMBD) vec_dot(nrc 1) into dst columns [HC*N_EMBD, 2*HC*N_EMBD)
// F32 operands (vec_dot_type F32) are used in place, without quantisation.

namespace {

constexpr int64_t FUSE_MAX_TOK = 16;
constexpr int64_t FUSE_D = HC * N_EMBD;                    // 10240: xn / gate width = w_down in-dim = w_up out-dim
constexpr int64_t FUSE_QROW_MAX = FUSE_D * (int64_t) sizeof(float); // largest vec_dot_type row used here (F32 worst case)

alignas(64) float   g_fuse_rs[FUSE_MAX_TOK * HC];            // rms scale per (token, hc row)
alignas(64) float   g_fuse_pre[FUSE_MAX_TOK * HC_LR];        // lo before silu, already scaled by 1/HC
alignas(64) float   g_fuse_lo[FUSE_MAX_TOK * HC_LR];         // silu(lo)
alignas(64) uint8_t g_fuse_qd[FUSE_MAX_TOK * FUSE_QROW_MAX]; // xn quantised for vec_dot_type(w_down)
alignas(64) uint8_t g_fuse_qi[FUSE_MAX_TOK * FUSE_QROW_MAX]; // xn quantised for vec_dot_type(w_inject), if different
alignas(64) uint8_t g_fuse_qu[FUSE_MAX_TOK * FUSE_QROW_MAX]; // lo quantised for vec_dot_type(w_up)

// Quantise `rows` rows of `n` floats (row r starts at src + r*src_stride) into dst (row r at dst + r*ggml_row_size(type, n)),
// splitting whole blocks over the threads. from_float is the CPU traits' from_float of `type` (production code); each block
// is quantised independently, so the bytes equal ggml's per-row packing (ggml-cpu.c mul_mat, block-split loop).
void fuse_quant_part(ggml_type type, ggml_from_float_t from_float, const float * src, int64_t src_stride, uint8_t * dst,
                     int64_t n, int64_t rows, int ith, int nth) {
    const int64_t bs = ggml_blck_size(type);
    const int64_t nblk = n / bs;
    const size_t row_bytes = ggml_row_size(type, n);
    const int64_t b0 = nblk * ith / nth;
    const int64_t b1 = nblk * (ith + 1) / nth;
    if (b1 <= b0) return;
    for (int64_t r = 0; r < rows; ++r) {
        from_float(src + r * src_stride + b0 * bs, dst + r * row_bytes + ggml_row_size(type, b0 * bs), (b1 - b0) * bs);
    }
}

// The custom op. ud = FuseBarrier *. dst = F32 [OUT_W, nt], OUT_W = 2*HC*N_EMBD + n_inj, per token column:
// [xn (HC*N_EMBD) | gate (HC*N_EMBD) | inject (n_inj)]. src = {x, w_norm, w_down, w_up, w_inject} (w_inject optional).
void fuse_hc_mix_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * wn = dst->src[1];
    const ggml_tensor * wd = dst->src[2];
    const ggml_tensor * wu = dst->src[3];
    const ggml_tensor * wi = dst->src[4]; // nullptr without w_inject
    const int64_t nt = dst->ne[1];
    const int64_t n_inj = wi ? wi->ne[1] : 0;
    const int64_t OUT_W = 2 * FUSE_D + n_inj;

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == OUT_W && dst->nb[0] == sizeof(float));
    GGML_ASSERT(nt >= 1 && nt <= FUSE_MAX_TOK);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == N_EMBD && x->ne[1] == HC && x->ne[2] == nt && x->nb[0] == sizeof(float));
    GGML_ASSERT(wn->type == GGML_TYPE_F32 && wn->ne[0] == N_EMBD && wn->ne[1] == HC && wn->nb[0] == sizeof(float) &&
                wn->nb[1] == N_EMBD * sizeof(float));
    GGML_ASSERT(wd->ne[0] == FUSE_D && wd->ne[1] == HC_LR);
    GGML_ASSERT(wu->ne[0] == HC_LR && wu->ne[1] == FUSE_D);
    GGML_ASSERT(!wi || (wi->ne[0] == FUSE_D && n_inj >= 1));

    const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(wd->type);
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(wu->type);
    const ggml_type_traits_cpu * ti = wi ? ggml_get_type_traits_cpu(wi->type) : nullptr;
    const ggml_type vd = td->vec_dot_type;
    const ggml_type vu = tu->vec_dot_type;
    const ggml_type vi = ti ? ti->vec_dot_type : GGML_TYPE_F32;
    GGML_ASSERT(td->vec_dot && (vd == GGML_TYPE_F32 || td->from_float));
    GGML_ASSERT(tu->vec_dot && (vu == GGML_TYPE_F32 || tu->from_float));
    GGML_ASSERT(!ti || (ti->vec_dot && (vi == GGML_TYPE_F32 || ti->from_float)));
    GGML_ASSERT(FUSE_D % ggml_blck_size(vd) == 0);
    GGML_ASSERT(HC_LR % ggml_blck_size(vu) == 0);
    GGML_ASSERT(ggml_row_size(vd, FUSE_D) <= FUSE_QROW_MAX && ggml_row_size(vu, HC_LR) <= FUSE_QROW_MAX);
    if (ti) GGML_ASSERT(FUSE_D % ggml_blck_size(vi) == 0 && ggml_row_size(vi, FUSE_D) <= FUSE_QROW_MAX);

    // inject shares the xn quantisation of w_down when the vec_dot_types agree
    const bool inj_shares_d = ti && vi == vd;
    const uint8_t * qbuf_i = inj_shares_d ? g_fuse_qd : g_fuse_qi;

    float * out = (float *) dst->data;
    const float sc = 1.0f / (float) HC; // ggml_scale factor (production: 1.0f / (float) HC)

    // row pointer of xn for token t as vec_dot's second operand: F32 -> dst column in place, else packed bytes
    auto xq_row = [&](ggml_type ty, const uint8_t * qbuf, int64_t t) -> const void * {
        return ty == GGML_TYPE_F32 ? (const void *) (out + t * OUT_W)
                                   : (const void *) (qbuf + t * ggml_row_size(ty, FUSE_D));
    };
    auto lo_row = [&](int64_t t) -> const void * {
        return vu == GGML_TYPE_F32 ? (const void *) (g_fuse_lo + t * HC_LR)
                                   : (const void *) (g_fuse_qu + t * ggml_row_size(vu, HC_LR));
    };

    int64_t lo = 0, hi = 0;

    // (a1) rms scale per (token, hc row): ggml_compute_forward_rms_norm_f32, one thread per row
    fuse_split(nt * HC, ith, nth, 1, lo, hi);
    for (int64_t r = lo; r < hi; ++r) {
        const int64_t t = r / HC;
        const int64_t hc = r % HC;
        const float * xr = (const float *) ((const char *) x->data + t * x->nb[2] + hc * x->nb[1]);
        double sum = 0.0;
        for (int64_t i0 = 0; i0 < N_EMBD; ++i0) {
            sum += (double) (xr[i0] * xr[i0]);
        }
        const float mean = sum / N_EMBD;
        g_fuse_rs[r] = 1.0f / sqrtf(mean + EPS);
    }
    bar->wait(nth); // barrier 1

    // (a2) xn = (x * scale) * w_norm, element-wise; w_norm[hc * N_EMBD + i0] (ggml_mul broadcast over tokens)
    const float * wnf = (const float *) wn->data;
    fuse_split(nt * FUSE_D, ith, nth, 1, lo, hi);
    for (int64_t e = lo; e < hi; ++e) {
        const int64_t t = e / FUSE_D;
        const int64_t k = e % FUSE_D;
        const int64_t hc = k / N_EMBD;
        const int64_t i0 = k % N_EMBD;
        const float xv = *(const float *) ((const char *) x->data + t * x->nb[2] + hc * x->nb[1] + i0 * x->nb[0]);
        out[t * OUT_W + k] = xv * g_fuse_rs[t * HC + hc] * wnf[k];
    }
    bar->wait(nth); // barrier 2

    // (b) quantise xn for the matvec operands that are not F32
    if (vd != GGML_TYPE_F32) {
        fuse_quant_part(vd, td->from_float, out, OUT_W, g_fuse_qd, FUSE_D, nt, ith, nth);
    }
    if (ti && !inj_shares_d && vi != GGML_TYPE_F32) {
        fuse_quant_part(vi, ti->from_float, out, OUT_W, g_fuse_qi, FUSE_D, nt, ith, nth);
    }
    bar->wait(nth); // barrier 3

    // (c) down rows (HC_LR, scaled by 1/HC into g_fuse_pre) and inject rows (HC, scaled into dst), one (token, row) per item
    const int64_t W = HC_LR + n_inj;
    fuse_split(nt * W, ith, nth, 1, lo, hi);
    for (int64_t k = lo; k < hi; ++k) {
        const int64_t t = k / W;
        const int64_t j = k % W;
        float s = 0.0f;
        if (j < HC_LR) {
            td->vec_dot((int) FUSE_D, &s, 0, (const char *) wd->data + j * wd->nb[1], 0,
                        xq_row(vd, g_fuse_qd, t), 0, 1);
            g_fuse_pre[t * HC_LR + j] = s * sc;
        } else {
            const int64_t jj = j - HC_LR;
            ti->vec_dot((int) FUSE_D, &s, 0, (const char *) wi->data + jj * wi->nb[1], 0,
                        xq_row(vi, qbuf_i, t), 0, 1);
            out[t * OUT_W + 2 * FUSE_D + jj] = s * sc;
        }
    }
    bar->wait(nth); // barrier 4

    // (d) lo = silu(pre): ggml_vec_silu_f32 over the flat nt*HC_LR array. Rows are multiples of 4 long, so chunk bounds
    // on multiples of 4 give the same NEON groups as ggml's per-row call (no scalar tail inside a chunk).
    fuse_split(nt * HC_LR, ith, nth, 4, lo, hi);
    if (hi > lo) {
        ggml_vec_silu_f32((int) (hi - lo), g_fuse_lo + lo, g_fuse_pre + lo);
    }
    bar->wait(nth); // barrier 5

    // (e) quantise lo for w_up
    if (vu != GGML_TYPE_F32) {
        fuse_quant_part(vu, tu->from_float, g_fuse_lo, HC_LR, g_fuse_qu, HC_LR, nt, ith, nth);
    }
    bar->wait(nth); // barrier 6

    // (f) gate rows: gate[t, j] = vec_dot(w_up row j, lo_t), no scale (as build_hc_mix's mm(w_up, lo))
    fuse_split(nt * FUSE_D, ith, nth, 1, lo, hi);
    for (int64_t k = lo; k < hi; ++k) {
        const int64_t t = k / FUSE_D;
        const int64_t j = k % FUSE_D;
        float s = 0.0f;
        tu->vec_dot((int) HC_LR, &s, 0, (const char *) wu->data + j * wu->nb[1], 0, lo_row(t), 0, 1);
        out[t * OUT_W + FUSE_D + j] = s;
    }
}

// True when the vec_dot path of fuse_hc_mix_op is what ggml's mul_mat computes for this weight at nt tokens.
// Routes in ggml-cpu.c / llamafile/sgemm.cpp / tiled/tiled.cpp for this build (GGML_LLAMAFILE=ON, aarch64 NEON + dotprod,
// no MMA, GGML_CPU_TILED_MM default on):
//   llamafile_sgemm: never for n = nt < 2 (sgemm.cpp: "if (n < 2) return false" without __MMA__);
//     F32 weights at n >= 4 (NEON branch, "if (n < 4) return false"); Q8_0 and Q4_0 at n >= 2 (dotprod branch).
//   tiled kernel: K-quant / IQ weights at nt >= 8, or at any nt when GGML_CPU_TILED_MM_FORCE=1.
//   Q4_0 / Q8_0 have nrows = 2 (vec_dot with nrc 2 for even nt): those are only supported at nt == 1 here.
// Any other weight type is supported at nt == 1 only (conservative).
bool fuse_w_ok(int64_t nt, const ggml_tensor * w) {
    if (!w) return true;
    switch (w->type) {
        case GGML_TYPE_F32:
            return nt < 4;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
            return nt == 1;
        default: {
            const char * forced = std::getenv("GGML_CPU_TILED_MM_FORCE");
            return nt == 1 && !(forced && std::atoi(forced) == 1);
        }
    }
}

// Caller contract: use the fused node only when this returns true; otherwise keep build_hc_mix.
bool fuse_hc_mix_supported(int64_t nt, const ggml_tensor * w_down, const ggml_tensor * w_up, const ggml_tensor * w_inject) {
    return nt >= 1 && nt <= FUSE_MAX_TOK && fuse_w_ok(nt, w_down) && fuse_w_ok(nt, w_up) && fuse_w_ok(nt, w_inject);
}

// Builds the custom node. bar: a FuseBarrier that outlives the graph's compute (e.g. fg_ctx::fuse_barrier()).
// Returns dst F32 [2*HC*N_EMBD + n_inj, nt]; the caller takes views (see the comment below).
ggml_tensor * fuse_build_hc_mix(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * x, ggml_tensor * w_norm,
                                ggml_tensor * w_down, ggml_tensor * w_up, ggml_tensor * w_inject) {
    const int64_t nt = x->ne[2];
    GGML_ASSERT(fuse_hc_mix_supported(nt, w_down, w_up, w_inject));
    const int64_t n_inj = w_inject ? w_inject->ne[1] : 0;
    ggml_tensor * args[5] = {x, w_norm, w_down, w_up, w_inject};
    const int n_args = w_inject ? 5 : 4;
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, 2 * FUSE_D + n_inj, nt, 1, 1, args, n_args, fuse_hc_mix_op,
                          GGML_N_TASKS_MAX, bar);
}

// Caller side, replacing build_hc_mix's first half for il >= 0 (nt = res_hc->ne[2]):
//   ggml_tensor * f = fuse_build_hc_mix(ctx0, bar, res_hc, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject);
//   const size_t rs = N_EMBD * sizeof(float);
//   ggml_tensor * xn     = ggml_view_3d(ctx0, f, N_EMBD, HC, nt, rs, f->nb[1], 0);       // [N_EMBD, HC, nt]
//   ggml_tensor * gate   = ggml_view_3d(ctx0, f, N_EMBD, HC, nt, rs, f->nb[1], HC * rs); // [N_EMBD, HC, nt]
//   ggml_tensor * inject = ggml_view_2d(ctx0, f, HC, nt, f->nb[1], 2 * HC * rs);         // [HC, nt], as build_hc_mix's inject
//   ggml_tensor * mixed  = ggml_dsv4_hc_pre_gated(ctx0, xn, gate, 1.0f / (float) HC);   // unchanged ggml node
// The views are strided (nb[2] = OUT_W * 4); ggml_compute_forward_dsv4_hc_pre reads through nb, so this is valid.

} // namespace

// Model-free check: fused node vs the ggml subgraph build_hc_mix builds (rms_norm, mul, reshape, mul_mat, scale, silu),
// compared bitwise on xn, gate and inject, read back through the same views the caller uses. Returns 0 on pass.
int fg_selftest_fuse_hc(int nth) {
    struct Case { int64_t nt; ggml_type t_down, t_up, t_inj; };
    const Case cases[] = {
        {1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32},
        {1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {3, GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32},
        {3, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}, // not supported at 3 tokens (llamafile route): must be skipped
    };
    const int64_t D = FUSE_D, R = HC_LR, NI = HC;
    int fails = 0;
    for (const Case & cs : cases) {
        const int64_t nt = cs.nt;
        const size_t mem = 128u << 20;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context * ctx = ggml_init(ip);
        uint64_t s = 0x2545F4914F6CDD1Dull;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
        // k x n matrix (ne0 = k, ne1 = n) of type ty, quantised like fg_selftest_exec does
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
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, HC, nt);
        for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = rnd();
        ggml_tensor * wn = mat(GGML_TYPE_F32, N_EMBD, HC, 1.0f);
        ggml_tensor * wd = mat(cs.t_down, D, R, 0.05f);
        ggml_tensor * wu = mat(cs.t_up, R, D, 0.05f);
        ggml_tensor * wi = mat(cs.t_inj, D, NI, 0.05f);

        if (!fuse_hc_mix_supported(nt, wd, wu, wi)) {
            std::fprintf(stderr, "fg_selftest_fuse_hc nth %d tokens %d: unsupported (ggml takes the llamafile/tiled route), skipped\n",
                         nth, (int) nt);
            ggml_free(ctx);
            continue;
        }

        // reference: the subgraph build_hc_mix builds for il >= 0, without the DSV4_HC_PRE that consumes it
        ggml_tensor * xr  = ggml_mul(ctx, ggml_rms_norm(ctx, x, EPS), wn);
        ggml_tensor * xr2 = ggml_reshape_2d(ctx, xr, D, nt);
        ggml_tensor * lo  = ggml_silu(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, wd, xr2), 1.0f / (float) HC));
        ggml_tensor * gate_r = ggml_mul_mat(ctx, wu, lo);
        ggml_tensor * inj_r  = ggml_scale(ctx, ggml_mul_mat(ctx, wi, xr2), 1.0f / (float) HC);
        ggml_cgraph * gr = ggml_new_graph(ctx);
        ggml_build_forward_expand(gr, xr2);
        ggml_build_forward_expand(gr, gate_r);
        ggml_build_forward_expand(gr, inj_r);
        ggml_graph_compute_with_ctx(ctx, gr, nth);
        std::vector<float> r_xn((size_t) (D * nt)), r_gate((size_t) (D * nt)), r_inj((size_t) (NI * nt));
        std::memcpy(r_xn.data(), xr2->data, r_xn.size() * sizeof(float));
        std::memcpy(r_gate.data(), gate_r->data, r_gate.size() * sizeof(float));
        std::memcpy(r_inj.data(), inj_r->data, r_inj.size() * sizeof(float));

        // fused node and the views a caller takes from it
        FuseBarrier bar;
        ggml_tensor * f = fuse_build_hc_mix(ctx, &bar, x, wn, wd, wu, wi);
        const size_t rs = N_EMBD * sizeof(float);
        ggml_tensor * xn_v   = ggml_view_3d(ctx, f, N_EMBD, HC, nt, rs, f->nb[1], 0);
        ggml_tensor * gate_v = ggml_view_3d(ctx, f, N_EMBD, HC, nt, rs, f->nb[1], HC * rs);
        ggml_tensor * inj_v  = ggml_view_2d(ctx, f, NI, nt, f->nb[1], 2 * HC * rs);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, f);
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        bool ok_xn = true, ok_gate = true, ok_inj = true;
        for (int64_t t = 0; t < nt; ++t) {
            ok_xn   &= std::memcmp((const char *) xn_v->data + t * xn_v->nb[2], r_xn.data() + t * D, D * sizeof(float)) == 0;
            ok_gate &= std::memcmp((const char *) gate_v->data + t * gate_v->nb[2], r_gate.data() + t * D, D * sizeof(float)) == 0;
            ok_inj  &= std::memcmp((const char *) inj_v->data + t * inj_v->nb[1], r_inj.data() + t * NI, NI * sizeof(float)) == 0;
        }
        const bool ok = ok_xn && ok_gate && ok_inj;
        std::fprintf(stderr, "fg_selftest_fuse_hc nth %d tokens %d: bitwise %d (w_inject %s; xn %d gate %d inject %d)\n", nth,
                     (int) nt, (int) ok, ggml_type_name(cs.t_inj), (int) ok_xn, (int) ok_gate, (int) ok_inj);
        if (!ok) ++fails;
        ggml_free(ctx);
    }
    return fails == 0 ? 0 : 1;
}
