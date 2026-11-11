// flash_mtp.inc (WP-T2): MTP speculative decoding inside the graph port. Included at the end of flash_graph.cpp
// (one translation unit: it uses G, fg_model, fg_ctx). The hooks in flash_graph.cpp are small and marked "T2":
// one more slot in the per-layer arrays (index N_LAYER = the head block blk.48), G::vmode / cols_exact, the
// fg_decode_impl(ext) split, fg_ctx::pend.
//
// 1. Verify (fg_verify / fg_commit). A verify evaluates [t, d_1..d_k] at the next positions with every row
//    computed exactly as that token's one-token decode:
//      - dense matmuls: mm_cols_op, per column production's one-token path (from_float of the column into the
//        weight type's vec_dot_type, then vec_dot nrc = 1 per weight row). ggml's own mul_mat at 2+ columns may
//        take llamafile sgemm (Q8_0 / F32) or the 2-row i8mm kernels (even widths), which round differently;
//        at one column it is exactly this (sgemm returns false for n < 2, the tiled path needs >= 8 columns);
//      - attention: the fixed-chunk routine (row alone == row in a batch, selftest);
//      - experts: the row-wise streaming op (<= 16 rows); GDN: the fused op is token-sequential and its K > 1
//        snapshot path runs the same arithmetic in a scratch state.
//    The recurrent state is not advanced by the graph: the GDN op writes one state snapshot per token into its
//    (output-flagged) result, the conv inputs are kept; fg_commit(m) copies snapshot slot n - m and the conv
//    window [m, m + cols) into the state. KV / indexer cells are written by position; cells at or beyond the
//    committed position are masked by rank and rewritten later (a k-pool is re-pooled when it completes again).
// 2. fg_decode_h: a prefill ubatch that returns h of every row (the head needs every position).
// 3. The head (fg_mtp_attach / fg_mtp_ctx_init / fg_mtp_step): the fork's graph_mtp (qwen4exp.cpp at da5852f):
//    res = eh_proj([enorm(embd(t)) ; hnorm(h)]) per hc stream, one QSA attention block with its own cells, its
//    own MoE (mapped experts, production mul_mat_id), hc_head mixer, output. The head file carries separate
//    hc_*_down / hc_*_inject (the fork's `*_down_inject` is an optional merged form of the same rows).
//    Restricted draft output: fg_mtp_set_vocab copies the chosen rows of output.weight.

struct fg_mtp_model {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> by_name;
    ggml_tensor *eh_proj = nullptr, *enorm = nullptr, *hnorm = nullptr, *hc_head_norm = nullptr, *hc_head_down = nullptr, *hc_head_up = nullptr;
    // restricted draft vocabulary: rows of output.weight
    ggml_context * sub_ctx = nullptr;
    ggml_tensor * out_sub = nullptr;
    void * sub_data = nullptr;
    std::vector<int32_t> sub_ids;
};

struct fg_mtp_ctx {
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_gallocr_t galloc = nullptr; // the head's graphs: never the trunk's allocator (cached decode graph, pending commit)
};

static void fg_mtp_sub_free(fg_mtp_model * mm) {
    if (mm->sub_ctx) ggml_free(mm->sub_ctx);
    std::free(mm->sub_data);
    mm->sub_ctx = nullptr;
    mm->out_sub = nullptr;
    mm->sub_data = nullptr;
    mm->sub_ids.clear();
}

static void fg_mtp_model_free(fg_model * m) {
    if (!m->mtp) return;
    fg_mtp_sub_free(m->mtp);
    if (m->mtp->buf) ggml_backend_buffer_free(m->mtp->buf);
    if (m->mtp->ctx) ggml_free(m->mtp->ctx);
    delete m->mtp;
    m->mtp = nullptr;
}

static void fg_mtp_ctx_free(fg_ctx * c) {
    if (!c->mtp) return;
    if (c->mtp->galloc) ggml_gallocr_free(c->mtp->galloc);
    if (c->mtp->sbuf) ggml_backend_buffer_free(c->mtp->sbuf);
    if (c->mtp->sctx) ggml_free(c->mtp->sctx);
    delete c->mtp;
    c->mtp = nullptr;
    c->k_l[N_LAYER] = c->v_l[N_LAYER] = c->ik_l[N_LAYER] = nullptr;
}

namespace {

void mm_cols_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int64_t K = w->ne[0], M = w->ne[1], n = x->ne[1];
    const auto * tw = ggml_get_type_traits_cpu(w->type);
    const ggml_type vdt = tw->vec_dot_type;
    const ggml_vec_dot_t vec_dot = tw->vec_dot;
    const size_t rs = ggml_row_size(vdt, K);
    // each thread packs the columns for itself (block-wise quantisers: the same bytes as ggml's per-thread
    // block ranges); no shared buffer, no barrier
    thread_local std::vector<char> q;
    const char * cols = (const char *) x->data;
    size_t stride = x->nb[1];
    if (x->type != vdt) {
        if (q.size() < rs * (size_t) n) q.resize(rs * (size_t) n);
        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(vdt)->from_float;
        for (int64_t i = 0; i < n; ++i) from_float((const float *) ((const char *) x->data + i * x->nb[1]), q.data() + i * rs, K);
        cols = q.data();
        stride = rs;
    }
    const int64_t r0 = M * ith / nth, r1 = M * (ith + 1) / nth;
    for (int64_t r = r0; r < r1; ++r) {
        const char * wr = (const char *) w->data + r * w->nb[1];
        for (int64_t i = 0; i < n; ++i) {
            float s;
            vec_dot((int) K, &s, 0, wr, 0, cols + i * stride, 0, 1);
            ((float *) ((char *) dst->data + i * dst->nb[1]))[r] = s;
        }
    }
}

// The attention inputs of one ubatch at cells p0..p0+n-1 (positions, cell indices, the k-pool tables): the same
// values fg_decode_impl writes for the trunk (llama_memory_hybrid_idx_context::set_input_kpool, one sequence,
// cell == position). Kept in step with that block by hand until the two are unified on flash-m3.
void mtp_fill_attn_inputs(G & g, int n_tokens, uint32_t p0, uint32_t n_kv, uint32_t n_pool, uint32_t n_pool_real,
                          const std::vector<uint32_t> & new_pools, uint32_t n_new_g) {
    const uint32_t n_new = (uint32_t) new_pools.size();
    int32_t * pos = (int32_t *) g.inp_pos->data;
    for (int i = 0; i < n_tokens; ++i) {
        pos[i] = pos[n_tokens + i] = pos[2 * n_tokens + i] = (int32_t) (p0 + i);
        pos[3 * n_tokens + i] = 0;
    }
    int64_t * ki = (int64_t *) g.k_idxs->data;
    int64_t * vi = (int64_t *) g.v_idxs->data;
    int64_t * ii = (int64_t *) g.idx_k_idxs->data;
    for (int i = 0; i < n_tokens; ++i) ki[i] = vi[i] = ii[i] = p0 + i;
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
            for (int a = 0; a < 4; ++a) npos[a * n_new_g + i_new] = (int32_t) j;
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
        pcell[ip] = (int32_t) p0;
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

} // namespace

extern "C" {

int fg_verify(fg_ctx * c, const int32_t * tokens, int n, int n_threads, float * logits_all, float * h_all) {
    if (!logits_all) return -6;
    const fg_ext ext = {1, logits_all, h_all};
    const size_t before = c->toks.size();
    const int r = fg_decode_impl(c, tokens, n, n_threads, nullptr, nullptr, &ext);
    if (r != 0 && c->toks.size() > before && !c->pend.active) c->toks.resize(before); // a failed verify leaves no tokens behind
    return r;
}

int fg_commit(fg_ctx * c, int m) {
    auto & p = c->pend;
    if (!p.active) return -1;
    if (m < 1 || m > p.n) return -2;
    const int slot = p.n - m;
    const int ng = (int) p.gdn.size();
    // 36 x 3 MB: only this copy advances the GDN states (the verify graph does not write them)
#pragma omp parallel for schedule(static) num_threads(ng < 8 ? (ng > 0 ? ng : 1) : 8)
    for (int i = 0; i < ng; ++i) std::memcpy(p.gdn[i].state, p.gdn[i].snaps + (size_t) slot * N_EMBD_S, sizeof(float) * N_EMBD_S);
    for (const auto & e : p.conv) {
        const int64_t w = e.cols + p.n; // conv input: [cols + n, ch], column j of channel ch at ch * w + j
        for (int64_t ch = 0; ch < e.ch; ++ch) std::memcpy(e.state + ch * e.cols, e.input + ch * w + m, sizeof(float) * e.cols);
    }
    c->toks.resize(p.p0 + m);
    p.active = false;
    p.gdn.clear();
    p.conv.clear();
    return 0;
}

int fg_decode_h(fg_ctx * c, const int32_t * tokens, int n, int n_threads, float * logits, float * h_all) {
    const fg_ext ext = {2, nullptr, h_all};
    return fg_decode_impl(c, tokens, n, n_threads, logits, nullptr, &ext);
}

void fg_mtp_set_verify_cache(int on) {
    g_mtp_verify_cache = on != 0;
}

void fg_mtp_stats(const fg_ctx * c, double * out4) {
    for (int i = 0; i < 4; ++i) out4[i] = c->mtp_stats[i];
}

int fg_mtp_attach(fg_model * m, const fg_tensor_ref * refs, int n, const fg_region * region) {
    if (m->mtp && m->mtp->ctx) return -3; // one head per model
    if (!m->mtp) m->mtp = new fg_mtp_model();
    fg_mtp_model & H = *m->mtp;
    ggml_init_params ip = {(size_t) (n + 8) * ggml_tensor_overhead(), nullptr, true};
    H.ctx = ggml_init(ip);
    H.buf = ggml_backend_cpu_buffer_from_ptr(region->base, region->size);
    for (int i = 0; i < n; ++i) {
        const fg_tensor_ref & r = refs[i];
        int nd = 4;
        while (nd > 1 && r.ne[nd - 1] == 1) --nd;
        ggml_tensor * t = ggml_new_tensor(H.ctx, (ggml_type) r.type, nd, r.ne);
        ggml_set_name(t, r.name);
        t->data = r.data;
        t->buffer = H.buf;
        H.by_name[r.name] = t;
    }
    bool ok = true;
    const std::string pre = "blk." + std::to_string((int) N_LAYER) + ".";
    auto get = [&](const char * s) -> ggml_tensor * {
        auto it = H.by_name.find(pre + s);
        if (it == H.by_name.end()) {
            std::fprintf(stderr, "flash_mtp: missing head tensor %s%s\n", pre.c_str(), s);
            ok = false;
            return nullptr;
        }
        return it->second;
    };
    auto norm2 = [&](const char * s) -> ggml_tensor * {
        ggml_tensor * t = get(s);
        if (!t) return nullptr;
        if (ggml_nelements(t) != N_EMBD * HC) {
            std::fprintf(stderr, "flash_mtp: %s%s is not [n_embd * hc]\n", pre.c_str(), s);
            ok = false;
            return nullptr;
        }
        t->ne[0] = N_EMBD;
        t->ne[1] = HC;
        t->nb[1] = t->nb[0] * N_EMBD;
        t->nb[2] = t->nb[3] = t->nb[1] * HC;
        return t;
    };
    Layer & L = m->layers[N_LAYER];
    L = Layer{};
    L.hc_attn_norm = norm2("hc_attn_norm.weight");
    L.hc_attn_down = get("hc_attn_down.weight");
    L.hc_attn_up = get("hc_attn_up.weight");
    L.hc_attn_inject = get("hc_attn_inject.weight");
    L.hc_ffn_norm = norm2("hc_ffn_norm.weight");
    L.hc_ffn_down = get("hc_ffn_down.weight");
    L.hc_ffn_up = get("hc_ffn_up.weight");
    L.hc_ffn_inject = get("hc_ffn_inject.weight");
    L.wq = get("attn_q.weight");
    L.wk = get("attn_k.weight");
    L.wv = get("attn_v.weight");
    L.wo = get("attn_output.weight");
    L.attn_q_norm = get("attn_q_norm.weight");
    L.attn_k_norm = get("attn_k_norm.weight");
    L.index_q_proj = get("indexer.q_proj.weight");
    L.index_k_proj = get("indexer.k_proj.weight");
    L.index_q_norm = get("indexer.q_norm.weight");
    L.index_k_norm = get("indexer.k_norm.weight");
    L.ffn_gate_inp = get("ffn_gate_inp.weight");
    L.ffn_up_exps = get("ffn_up_exps.weight");
    L.ffn_gate_exps = get("ffn_gate_exps.weight");
    L.ffn_down_exps = get("ffn_down_exps.weight");
    L.ffn_gate_inp_shexp = get("ffn_gate_inp_shexp.weight");
    L.ffn_gate_shexp = get("ffn_gate_shexp.weight");
    L.ffn_up_shexp = get("ffn_up_shexp.weight");
    L.ffn_down_shexp = get("ffn_down_shexp.weight");
    H.eh_proj = get("nextn.eh_proj.weight");
    H.enorm = get("nextn.enorm.weight");
    H.hnorm = norm2("nextn.hnorm.weight");
    H.hc_head_norm = norm2("nextn.hc_head_norm.weight");
    H.hc_head_down = get("nextn.hc_head_down.weight");
    H.hc_head_up = get("nextn.hc_head_up.weight");
    if (ok && (H.eh_proj->ne[0] != 2 * N_EMBD || H.eh_proj->ne[1] != N_EMBD || L.ffn_up_exps->ne[2] != N_EXPERT)) {
        std::fprintf(stderr, "flash_mtp: head tensor shapes do not match this model\n");
        ok = false;
    }
    if (!ok) {
        m->layers[N_LAYER] = Layer{};
        ggml_backend_buffer_free(H.buf);
        ggml_free(H.ctx);
        H.buf = nullptr;
        H.ctx = nullptr;
        H.by_name.clear();
        H.eh_proj = nullptr;
        return -1;
    }
    return 0;
}

int fg_mtp_set_vocab(fg_model * m, const int32_t * ids, int n) {
    if (!m->mtp) m->mtp = new fg_mtp_model();
    fg_mtp_model & H = *m->mtp;
    fg_mtp_sub_free(&H);
    if (n <= 0) return 0;
    const size_t rs = m->output->nb[1];
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= m->output->ne[1]) return -1;
    H.sub_data = std::aligned_alloc(64, GGML_PAD(rs * (size_t) n + 64, 64));
    if (!H.sub_data) return -2;
    for (int i = 0; i < n; ++i) std::memcpy((char *) H.sub_data + (size_t) i * rs, (const char *) m->output->data + (size_t) ids[i] * rs, rs);
    ggml_init_params ip = {4 * ggml_tensor_overhead(), nullptr, true};
    H.sub_ctx = ggml_init(ip);
    H.out_sub = ggml_new_tensor_2d(H.sub_ctx, m->output->type, m->output->ne[0], n);
    ggml_set_name(H.out_sub, "output_sub.weight");
    H.out_sub->data = H.sub_data;
    H.sub_ids.assign(ids, ids + n);
    return 0;
}

int fg_mtp_ctx_init(fg_ctx * c) {
    if (!c->m->mtp || !c->m->mtp->eh_proj) return -1;
    if (!c->mtp) {
        c->mtp = new fg_mtp_ctx();
        ggml_init_params ip = {8 * ggml_tensor_overhead(), nullptr, true};
        c->mtp->sctx = ggml_init(ip);
        c->k_l[N_LAYER] = ggml_new_tensor_2d(c->mtp->sctx, GGML_TYPE_F16, HEAD_D * N_HEAD_KV, c->n_ctx);
        c->v_l[N_LAYER] = ggml_new_tensor_2d(c->mtp->sctx, GGML_TYPE_F16, HEAD_D * N_HEAD_KV, c->n_ctx);
        c->ik_l[N_LAYER] = ggml_new_tensor_2d(c->mtp->sctx, GGML_TYPE_F16, 2 * IDX_D, c->n_ctx);
        ggml_set_name(c->k_l[N_LAYER], "cache_k_mtp");
        ggml_set_name(c->v_l[N_LAYER], "cache_v_mtp");
        ggml_set_name(c->ik_l[N_LAYER], "cache_idx_k_mtp");
        c->mtp->sbuf = ggml_backend_alloc_ctx_tensors(c->mtp->sctx, c->backend);
        c->mtp->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    }
    ggml_backend_buffer_clear(c->mtp->sbuf, 0);
    return 0;
}

int fg_mtp_step(fg_ctx * c, const int32_t * tokens, const float * h, int n, uint32_t pos0, int n_threads, int want, int32_t * tok,
                float * prob, float * h_out) {
    fg_model & m = *c->m;
    if (!m.mtp || !m.mtp->eh_proj || !c->mtp) return -1;
    if (n <= 0 || pos0 + (uint32_t) n > c->n_ctx) return -2;
    fg_mtp_model & H = *m.mtp;
    const int il = (int) N_LAYER;
    const Layer & L = m.layers[il];
    // sizes: as fg_decode_impl, for the head's own cells
    const uint32_t n_cells = pos0 + (uint32_t) n;
    const uint32_t n_kv = std::min<uint32_t>(c->n_ctx, std::max<uint32_t>(256u, GGML_PAD(n_cells, 256u)));
    const uint32_t n_pool_real = n_cells / KPOOL;
    const uint32_t n_pool = kpool_pad(n_pool_real);
    std::vector<uint32_t> new_pools;
    for (uint32_t p = pos0; p < n_cells; ++p) {
        const uint32_t k = p / KPOOL;
        if (k < n_pool_real && (new_pools.empty() || new_pools.back() != k)) new_pools.push_back(k);
    }
    const uint32_t n_new = (uint32_t) new_pools.size();
    const uint32_t n_pool_max = c->n_ctx / KPOOL;
    const uint32_t bound = (uint32_t) n / KPOOL + 1;
    const uint32_t n_new_g = std::max({n_new, 1u, std::min({bound, n_pool - 1, n_pool_max})});
    const int64_t n_top_pool = std::min<int64_t>(n_pool, IDX_TOP_K / KPOOL);
    const uint32_t n_sel = (uint32_t) (KPOOL * n_top_pool + KPOOL - 1);

    const size_t graph_size = 2048;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * graph_size * 2 + ggml_graph_overhead_custom(graph_size, false));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context * ctx0 = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, graph_size, false);
    G g{m, *c, ctx0, gf, n, 1, n_kv, n_pool, n_new_g, n_sel};
    g.rs_z = -1;
    // the head's experts are ordinary mapped tensors computed by production mul_mat_id: no expert callback here
    const fg_expert_fn saved_fn = c->expert_fn;
    c->expert_fn = nullptr;
    // ... and no S-c fused phases in the head graph: they are sized and tested for the trunk's layers, and their
    // barriers belong to the trunk graph being cached (c->cur_bars). The head only drafts; plain ggml nodes.
    const int saved_fuse = c->opts.fuse;
    c->opts.fuse = 0;

    auto input = [&](ggml_tensor * t) {
        ggml_set_input(t);
        ggml_build_forward_expand(gf, t);
        return t;
    };
    g.inp_tokens = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n));
    ggml_tensor * h_in = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N_EMBD * HC, n));
    g.idx_k_idxs = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n));
    g.pool_cells = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pool));
    g.pool_idxs = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL, n_pool));
    g.pool_mask = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pool, n));
    g.tail_idxs = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL - 1, n));
    g.new_pool_idxs = input(ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, KPOOL, n_new_g));
    g.new_pool_flat = ggml_reshape_1d(ctx0, g.new_pool_idxs, KPOOL * n_new_g);
    g.new_pool_rep = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new_g));
    g.new_pool_pos = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4 * n_new_g));
    g.inp_pos = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n * 4));
    g.inp_out_ids = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1));
    g.k_idxs = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n));
    g.v_idxs = input(ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n));

    // graph_mtp: [enorm(e) ; hnorm(h)_s] -> eh_proj per hc stream (the trunk's token table, as the merged load)
    ggml_tensor * e = ggml_get_rows(ctx0, m.tok_embd, g.inp_tokens);
    g.cb(e, "mtp_tok_embd", il);
    ggml_tensor * h_norm = g.build_norm(ggml_reshape_3d(ctx0, h_in, N_EMBD, HC, n), H.hnorm, il);
    g.cb(h_norm, "mtp_hnorm", il);
    ggml_tensor * e_norm = g.build_norm(e, H.enorm, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, N_EMBD, 1, n), N_EMBD, HC, n, 1);
    g.cb(e_norm, "mtp_enorm", il);
    ggml_tensor * res_hc = g.mm(H.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0)); // [N_EMBD, HC, n]
    g.cb(res_hc, "mtp_eh_proj", il);

    G::hc_mix attn_mix = g.build_hc_mix(res_hc, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, il);
    ggml_tensor * cur = attn_mix.cur;
    ggml_build_forward_expand(gf, cur);
    cur = g.build_layer_attn(cur, il); // writes the head's K / V / indexer cells of these positions
    ggml_build_forward_expand(gf, cur);
    ggml_tensor * out = nullptr;
    ggml_tensor * hrow = nullptr;
    if (want) {
        res_hc = g.build_hc_combine(res_hc, cur, attn_mix.inject, il);
        G::hc_mix ffn_mix = g.build_hc_mix(res_hc, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, il);
        cur = g.build_layer_ffn(ffn_mix.cur, il);
        res_hc = g.build_hc_combine(res_hc, cur, ffn_mix.inject, il);
        // the next draft step reads the last row's residual as its h
        ggml_tensor * flat = ggml_reshape_2d(ctx0, res_hc, N_EMBD * HC, n);
        hrow = ggml_get_rows(ctx0, flat, g.inp_out_ids);
        g.cb(hrow, "h_nextn", il);
        ggml_set_output(hrow);
        ggml_build_forward_expand(gf, hrow);
        G::hc_mix head_mix = g.build_hc_mix(ggml_reshape_3d(ctx0, hrow, N_EMBD, HC, 1), H.hc_head_norm, H.hc_head_down, H.hc_head_up, nullptr, il);
        out = g.mm(H.out_sub ? H.out_sub : m.output, head_mix.cur);
        g.cb(out, "result_output", -1);
        ggml_set_output(out);
        ggml_build_forward_expand(gf, out);
    }
    c->expert_fn = saved_fn;
    c->opts.fuse = saved_fuse;
    // (a catch-up pass, want == 0, stops after the attention block: the cells depend only on the block's input)

    if (!ggml_gallocr_alloc_graph(c->mtp->galloc, gf)) {
        ggml_free(ctx0);
        return -3;
    }
    std::memcpy(g.inp_tokens->data, tokens, sizeof(int32_t) * n);
    std::memcpy(h_in->data, h, sizeof(float) * N_EMBD * HC * n);
    ((int32_t *) g.inp_out_ids->data)[0] = n - 1;
    mtp_fill_attn_inputs(g, n, pos0, n_kv, n_pool, n_pool_real, new_pools, n_new_g);
    ggml_backend_cpu_set_n_threads(c->backend, n_threads);
    const ggml_status st = ggml_backend_graph_compute(c->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return -4;
    }
    c->mtp_stats[3] += 1;
    if (want) {
        const float * l = (const float *) out->data;
        const int64_t nv = out->ne[0];
        int64_t best = 0;
        for (int64_t i = 1; i < nv; ++i)
            if (l[i] > l[best]) best = i;
        double sum = 0.0;
        for (int64_t i = 0; i < nv; ++i) sum += std::exp((double) l[i] - (double) l[best]);
        if (tok) *tok = H.sub_ids.empty() ? (int32_t) best : H.sub_ids[(size_t) best];
        if (prob) *prob = (float) (1.0 / sum);
        if (h_out) std::memcpy(h_out, hrow->data, sizeof(float) * N_EMBD * HC);
    }
    ggml_free(ctx0);
    return 0;
}

// Model-free self-test, 0 = pass:
// (a) mm_cols_op vs a one-column ggml_mul_mat graph per column (bitwise) for F32 / BF16 / Q8_0 / Q4_K / Q5_0 / Q6_K
//     weights at widths 2, 3, 4 (width 2 and 4 are where ggml's own multi-column mul_mat is allowed to differ;
//     its difference from the one-column result is reported, not required);
// (b) GDN: snapshot slot n - m of a K = n pass over n tokens == the final state of a K = 1 pass over the first
//     m tokens, and the first m score rows agree (bitwise), for the model's GDN shape;
// (c) the conv-state window rule of fg_commit == the state write of an m-token graph (bitwise).
int fg_selftest_mtp(int nth) {
    const size_t mem = 512u << 20;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = {mem, buf.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0xD1B54A32D192ED03ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (float) ((x >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    const int64_t K = 2560, M = 640, W = 4;
    bool ok = true;
    ggml_tensor * xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, W);
    for (int64_t i = 0; i < K * W; ++i) ((float *) xin->data)[i] = rnd() * 2.0f;
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K};
    for (ggml_type ty : types) {
        std::vector<float> wf(K * M);
        for (auto & v : wf) v = rnd() * 0.05f;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, ty, K, M);
        ggml_quantize_chunk(ty, wf.data(), w->data, 0, M, K, nullptr);
        // reference: each column alone through production's mul_mat
        std::vector<std::vector<float>> ref(W);
        for (int64_t i = 0; i < W; ++i) {
            ggml_tensor * col = ggml_view_2d(ctx, xin, K, 1, xin->nb[1], i * xin->nb[1]);
            ggml_tensor * y = ggml_mul_mat(ctx, w, col);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            ref[i].assign((float *) y->data, (float *) y->data + M);
        }
        for (int64_t wd = 2; wd <= W; ++wd) {
            ggml_tensor * xs = ggml_view_2d(ctx, xin, K, wd, xin->nb[1], 0);
            ggml_tensor * a[2] = {w, xs};
            ggml_tensor * y = ggml_custom_4d(ctx, GGML_TYPE_F32, M, wd, 1, 1, a, 2, mm_cols_op, GGML_N_TASKS_MAX, nullptr);
            ggml_tensor * yg = ggml_mul_mat(ctx, w, xs);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);
            ggml_build_forward_expand(gf, yg);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            bool same = true, ggml_same = true;
            for (int64_t i = 0; i < wd; ++i) {
                same &= std::memcmp((const char *) y->data + i * y->nb[1], ref[i].data(), M * sizeof(float)) == 0;
                ggml_same &= std::memcmp((const char *) yg->data + i * yg->nb[1], ref[i].data(), M * sizeof(float)) == 0;
            }
            std::fprintf(stderr, "fg_selftest_mtp nth %d: %s width %lld: mm_cols == one-column mul_mat %d (ggml's own %lld-column mul_mat: %d)\n", nth,
                         ggml_type_name(ty), (long long) wd, (int) same, (long long) wd, (int) ggml_same);
            ok &= same;
        }
    }
    // (b) GDN snapshots
    {
        const int64_t S = SSM_D, HV = SSM_VH, HK = SSM_KH, n = 3;
        auto t4 = [&](int64_t a, int64_t b, int64_t c2, float sc, float off) {
            ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b, c2, 1);
            for (int64_t i = 0; i < a * b * c2; ++i) ((float *) t->data)[i] = rnd() * sc + off;
            return t;
        };
        ggml_tensor * q = t4(S, HK, n, 0.09f, 0.0f), * k = t4(S, HK, n, 0.09f, 0.0f), * v = t4(S, HV, n, 1.0f, 0.0f);
        ggml_tensor * gt = t4(1, HV, n, 0.05f, -0.06f), * beta = t4(1, HV, n, 0.4f, 0.5f);
        ggml_tensor * st = t4(S, S, HV, 0.3f, 0.0f);
        ggml_tensor * full = ggml_gated_delta_net(ctx, q, k, v, gt, beta, st, n);
        {
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, full);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
        }
        const float * fs = (const float *) full->data;
        const int64_t snap = S * S * HV;
        for (int64_t m = 1; m <= n; ++m) {
            auto head = [&](ggml_tensor * t) { return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], m, 1, t->nb[1], t->nb[2], t->nb[2] * m, 0); };
            ggml_tensor * part = ggml_gated_delta_net(ctx, head(q), head(k), head(v), head(gt), head(beta), st, 1);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, part);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            const float * ps = (const float *) part->data;
            const bool scores = std::memcmp(ps, fs, sizeof(float) * S * HV * m) == 0;
            const bool state = std::memcmp(ps + S * HV * m, fs + S * HV * n + (n - m) * snap, sizeof(float) * snap) == 0;
            std::fprintf(stderr, "fg_selftest_mtp nth %d: GDN commit %lld of %lld: scores %d, state (slot %lld) %d\n", nth, (long long) m,
                         (long long) n, (int) scores, (long long) (n - m), (int) state);
            ok &= scores && state;
        }
    }
    // (c) conv-state window: fg_commit's index rule (columns [m, m + cols) of the n-token conv input, per channel)
    // == the state an m-token graph writes itself (build_conv_state_at's tail view + cpy), bitwise
    {
        const int64_t cols = 3, ch = 257, n = 4;
        ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols * ch, 1);
        ggml_tensor * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ch, n, 1);
        for (int64_t i = 0; i < cols * ch; ++i) ((float *) state->data)[i] = rnd();
        for (int64_t i = 0; i < ch * n; ++i) ((float *) xs->data)[i] = rnd();
        auto conv_in = [&](int64_t m) {
            ggml_tensor * xm = ggml_view_3d(ctx, xs, ch, m, 1, xs->nb[1], xs->nb[1] * m, 0);
            return ggml_concat(ctx, ggml_reshape_3d(ctx, state, cols, ch, 1), ggml_transpose(ctx, xm), 0);
        };
        ggml_tensor * full = conv_in(n);
        {
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, full);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
        }
        for (int64_t m = 1; m <= n; ++m) {
            ggml_tensor * ci = conv_in(m);
            ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols * ch, 1);
            ggml_tensor * tail = ggml_view_3d(ctx, ci, cols, ch, 1, ci->nb[1], ci->nb[2], ggml_row_size(ci->type, ci->ne[0] - cols));
            ggml_tensor * cp = ggml_cpy(ctx, tail, ggml_view_2d(ctx, dst, cols * ch, 1, dst->nb[1], 0));
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, cp);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            std::vector<float> mine(cols * ch);
            const int64_t w = cols + n;
            for (int64_t c2 = 0; c2 < ch; ++c2) std::memcpy(mine.data() + c2 * cols, (const float *) full->data + c2 * w + m, sizeof(float) * cols);
            const bool same = std::memcmp(mine.data(), dst->data, sizeof(float) * cols * ch) == 0;
            std::fprintf(stderr, "fg_selftest_mtp nth %d: conv window commit %lld of %lld: %d\n", nth, (long long) m, (long long) n, (int) same);
            ok &= same;
        }
    }
    ggml_free(ctx);
    return ok ? 0 : 1;
}

} // extern "C"
