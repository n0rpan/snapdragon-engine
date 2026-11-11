// flash_fuse_comb.inc: fused phase P2 (hc-combine) for the Flash decode graph, one ggml custom node.
// #included by flash_graph.cpp at file scope, after flash_fuse_common.h (FuseBarrier, fuse_split) and the ggml API.
// Replaces build_hc_combine (flash_graph.cpp) for one call (results/m3/sc-op-inventory.md, P2, nodes HCC.01-03):
//   w[t, hc]       = scale(sigmoid(inject[hc, t]), 2.0)           [HC, nt]
//   out[i0, hc, t] = dsv4_hc_post(block_out, residual, w, NULL)  [N_EMBD, HC, nt]
//     i.e. out = block_out[i0, t] * w[t, hc] + residual[i0, hc, t], rounded exactly as ggml does (see Numerics).
//
// Numerics (results/m3/sc-ggml-numerics.md, sections 4, 5b, 16, and the disassembly noted below):
//   sigmoid  : libm, 1.0f / (1.0f + expf(-x)) (unary-ops.cpp), per element, no contraction (no multiply in the expression).
//   scale 2  : 2.0f * s is exact (power of two, s in (0, 1)); equals ggml_vec_scale_f32 bit for bit.
//   hc_post  : comb == NULL branch (ops.cpp ggml_compute_forward_dsv4_hc_post_f32, lines ~11615-11640):
//                float sum = xv * pv;               // pv = w[t, hc], xv = block_out[i0, t]
//                sum += residual[i0, idst, t];      // identity mixing, one add
//              Production object (libggml-cpu.a, ops.cpp.o, ggml_compute_forward_dsv4_hc_post), comb == NULL path:
//                fmul s30, s1, s30   ; xv * pv, rounded to float
//                fadd s30, s30, s31  ; + rv, plain add
//              (the only fmadd/fmla in the function is the comb != NULL loop, `fmadd s30, s29, s0, s30`). So the
//              production result is round(round(xv*pv) + rv): NO fused multiply-add. The compiler building this file
//              may contract `p = xv*pv; p + rv` into fmadd (GCC: -ffp-contract=fast by default), so the product is
//              passed through an empty asm that forces it into a register (fuse_comb_round) before the add.
//   Element-wise over (i0, hc, t): re-partition safe (each output is a two-operand chain in one thread).
//
// Threads and scratch: all nth threads of the custom node run fuse_hc_combine_op(ith, nth). Step A computes the
// nt*HC scaled sigmoids (split over threads) into process-wide scratch g_comb_w; FuseBarrier; step B writes the
// output elements (split over threads). One barrier per call. Scratch is sized for COMB_MAX_TOK tokens. Assumes one
// fused hc-combine node runs at a time (ggml runs graph nodes one after another).
//
// Shapes (as ggml_dsv4_hc_post asserts): residual [N_EMBD, HC, nt], block_out [N_EMBD, nt] (ne[2] == 1),
// inject [HC, nt]. All F32. Operands are read through their strides (views are fine); dst is a fresh contiguous tensor.

namespace {

constexpr int64_t COMB_MAX_TOK = 16;
alignas(64) float g_comb_w[COMB_MAX_TOK * HC]; // scaled sigmoid(inject) per (token, hc): index t * HC + hc

// Forces v to be a rounded float value before the next operation: the empty asm consumes and produces v in an FP
// register, so the compiler cannot fuse the multiply that produced v into a following add.
inline float fuse_comb_round(float v) {
#if defined(__aarch64__)
    __asm__ __volatile__("" : "+w"(v));
#elif defined(__x86_64__)
    __asm__ __volatile__("" : "+x"(v));
#else
    __asm__ __volatile__("" : "+m"(v));
#endif
    return v;
}

// The custom op. src = {residual, block_out, inject}; dst = F32 [N_EMBD, HC, nt].
void fuse_hc_combine_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    FuseBarrier * bar = (FuseBarrier *) ud;
    const ggml_tensor * res = dst->src[0]; // residual  [N_EMBD, HC, nt]
    const ggml_tensor * blk = dst->src[1]; // block_out [N_EMBD, nt]
    const ggml_tensor * inj = dst->src[2]; // inject    [HC, nt]
    const int64_t nt = dst->ne[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == N_EMBD && dst->ne[1] == HC && dst->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float) && dst->nb[1] == N_EMBD * sizeof(float) &&
                dst->nb[2] == N_EMBD * HC * sizeof(float));
    GGML_ASSERT(nt >= 1 && nt <= COMB_MAX_TOK);
    GGML_ASSERT(res->type == GGML_TYPE_F32 && res->ne[0] == N_EMBD && res->ne[1] == HC && res->ne[2] == nt);
    GGML_ASSERT(blk->type == GGML_TYPE_F32 && blk->ne[0] == N_EMBD && blk->ne[1] == nt && blk->ne[2] == 1);
    GGML_ASSERT(inj->type == GGML_TYPE_F32 && inj->ne[0] == HC && inj->ne[1] == nt);

    float * out = (float *) dst->data;
    int64_t lo = 0, hi = 0;

    // (A) w[t, hc] = 2 * sigmoid(inject[hc, t]), one element per (token, hc), split over threads
    fuse_split(nt * HC, ith, nth, 1, lo, hi);
    for (int64_t k = lo; k < hi; ++k) {
        const int64_t t  = k / HC;
        const int64_t hc = k % HC;
        const float x = *(const float *) ((const char *) inj->data + hc * inj->nb[0] + t * inj->nb[1]);
        const float sg = 1.0f / (1.0f + expf(-x)); // unary-ops.cpp sigmoid
        g_comb_w[k] = 2.0f * sg;                    // ggml_scale(., 2.0f): exact
    }
    bar->wait(nth); // barrier 1

    // (B) out[i0, hc, t] = round(block_out[i0, t] * w[t, hc]) + residual[i0, hc, t], flat index ir = dst linear index
    const int64_t nr = N_EMBD * HC * nt;
    fuse_split(nr, ith, nth, 1, lo, hi);
    for (int64_t ir = lo; ir < hi; ++ir) {
        const int64_t i0   = ir % N_EMBD;
        const int64_t idst = (ir / N_EMBD) % HC;
        const int64_t t    = ir / (N_EMBD * HC);
        const float xv = *(const float *) ((const char *) blk->data + i0 * blk->nb[0] + t * blk->nb[1]);
        const float pv = g_comb_w[t * HC + idst];
        const float rv = *(const float *) ((const char *) res->data + i0 * res->nb[0] + idst * res->nb[1] + t * res->nb[2]);
        const float p = fuse_comb_round(xv * pv); // production: fmul, then fadd (no fused multiply-add)
        out[ir] = p + rv;
    }
}

// True when fuse_build_hc_combine can build the node for these operands (all F32, shapes as ggml_dsv4_hc_post asserts).
bool fuse_hc_combine_supported(const ggml_tensor * residual, const ggml_tensor * block_out, const ggml_tensor * inject) {
    if (!residual || !block_out || !inject) return false;
    const int64_t nt = residual->ne[2];
    return nt >= 1 && nt <= COMB_MAX_TOK &&
           residual->type == GGML_TYPE_F32 && residual->ne[0] == N_EMBD && residual->ne[1] == HC && residual->ne[3] == 1 &&
           block_out->type == GGML_TYPE_F32 && block_out->ne[0] == N_EMBD && block_out->ne[1] == nt && block_out->ne[2] == 1 &&
           inject->type == GGML_TYPE_F32 && inject->ne[0] == HC && inject->ne[1] == nt;
}

// Builds the custom node replacing build_hc_combine(residual, block_out, inject). bar: a FuseBarrier that outlives the
// graph's compute (e.g. fg_ctx::fuse_barrier()). Returns dst F32 [N_EMBD, HC, nt], the same shape and meaning as
// ggml_dsv4_hc_post(block_out, residual, 2*sigmoid(inject), NULL); returns nullptr if the operands are unsupported
// (the caller then keeps build_hc_combine).
ggml_tensor * fuse_build_hc_combine(ggml_context * ctx0, FuseBarrier * bar, ggml_tensor * residual, ggml_tensor * block_out,
                                    ggml_tensor * inject) {
    if (!fuse_hc_combine_supported(residual, block_out, inject)) return nullptr;
    const int64_t nt = residual->ne[2];
    ggml_tensor * args[3] = {residual, block_out, inject};
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, N_EMBD, HC, nt, 1, args, 3, fuse_hc_combine_op, GGML_N_TASKS_MAX, bar);
}

// Caller side, replacing build_hc_combine's body (flash_graph.cpp):
//   ggml_tensor * f = fuse_build_hc_combine(ctx0, c.fuse_barrier(), residual, block_out, inject);
//   if (!f) { /* original: sigmoid, scale 2, ggml_dsv4_hc_post(block_out, residual, w, nullptr) */ }

} // namespace

// Model-free check: fused node vs the ggml subgraph build_hc_combine builds (sigmoid, scale 2, dsv4_hc_post with comb
// NULL), compared bitwise on the full [N_EMBD, HC, nt] output. Inputs include large |inject| (sigmoid saturation to
// 0 and 1, and exp overflow). Returns 0 on pass.
int fg_selftest_fuse_hcpost(int nth) {
    struct Case { int64_t nt; float amp; };
    const Case cases[] = {
        {1, 2.0f},   // moderate inject
        {1, 200.0f}, // saturating inject
        {3, 2.0f},
        {3, 200.0f},
    };
    const int64_t D = N_EMBD, H = HC;
    int fails = 0;
    for (const Case & cs : cases) {
        const int64_t nt = cs.nt;
        const size_t mem = 32u << 20;
        std::vector<uint8_t> buf(mem);
        ggml_init_params ip = {mem, buf.data(), false};
        ggml_context * ctx = ggml_init(ip);
        uint64_t s = 0x2545F4914F6CDD1Dull;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float) ((s >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };

        ggml_tensor * res = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, nt);
        ggml_tensor * blk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, nt);
        ggml_tensor * inj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nt);
        for (int64_t i = 0; i < ggml_nelements(res); ++i) ((float *) res->data)[i] = rnd() * 4.0f;
        for (int64_t i = 0; i < ggml_nelements(blk); ++i) ((float *) blk->data)[i] = rnd() * 4.0f;
        for (int64_t i = 0; i < ggml_nelements(inj); ++i) ((float *) inj->data)[i] = rnd() * cs.amp;
        // exact extremes: sigmoid saturates to 1 (x >> 0), to 0 (x << 0), expf overflows at x < -88.7
        const float ext[] = {1000.0f, -1000.0f, 88.0f, -88.0f, 0.0f, -0.0f, 1e-7f};
        for (int64_t i = 0; i < (int64_t) (sizeof(ext) / sizeof(ext[0])) && i < ggml_nelements(inj); ++i) {
            ((float *) inj->data)[i] = ext[i];
        }

        // reference: the subgraph build_hc_combine builds
        ggml_tensor * w   = ggml_scale(ctx, ggml_sigmoid(ctx, inj), 2.0f);
        ggml_tensor * ref = ggml_dsv4_hc_post(ctx, blk, res, w, nullptr);
        ggml_cgraph * gr = ggml_new_graph(ctx);
        ggml_build_forward_expand(gr, ref);
        ggml_graph_compute_with_ctx(ctx, gr, nth);
        const size_t n_out = (size_t) (D * H * nt);
        std::vector<float> r_out(n_out);
        std::memcpy(r_out.data(), ref->data, n_out * sizeof(float));

        // fused node
        FuseBarrier bar;
        ggml_tensor * f = fuse_build_hc_combine(ctx, &bar, res, blk, inj);
        if (!f) {
            std::fprintf(stderr, "fg_selftest_fuse_hcpost nth %d tokens %d: unsupported operands\n", nth, (int) nt);
            ++fails;
            ggml_free(ctx);
            continue;
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, f);
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        const uint32_t * a = (const uint32_t *) f->data;
        const uint32_t * b = (const uint32_t *) r_out.data();
        size_t diff = 0;
        for (size_t i = 0; i < n_out; ++i) diff += a[i] != b[i];
        const bool ok = diff == 0;
        std::fprintf(stderr, "fg_selftest_fuse_hcpost nth %d tokens %d inject amp %g: bitwise %d (%zu of %zu elements differ)\n",
                     nth, (int) nt, (double) cs.amp, (int) ok, diff, n_out);
        if (!ok) ++fails;
        ggml_free(ctx);
    }
    return fails == 0 ? 0 : 1;
}
