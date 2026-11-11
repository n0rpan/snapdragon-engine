// dump_ref: the llama.cpp oracle for oxe (M1/M2).
//
// Links the pinned llama.cpp (claude/q3k-moe-gemm @ ee6b9a1, KleidiAI OFF, repack ON; see
// tools/oracle/build.sh) and runs a fixed token sequence through it, ONE TOKEN PER llama_decode
// call (the decode path: gemv kernels, fused autoregressive gated delta net), or as one batch
// with --batch. While doing so it can
//   - dump every named intermediate tensor (cb_eval) for selected positions  (--dump-pos)
//   - write the full logits of every evaluated position                      (--logits)
//   - continue greedily for N tokens                                          (--gen N)
// Output directory layout:
//   manifest.json            tokens, generated ids, options
//   pos<P>.tensors           records: u32 name_len, name, u32 type (0 f32, 1 i32), u32 n_dims,
//                            i64 ne[4], payload (contiguous, ne0 fastest)
//   logits.f32               n_positions x n_vocab f32 (if --logits)
//
// Usage: dump_ref --model M.gguf --out DIR (--tokens 1,2,3 | --prompt TEXT | --prompt-file F)
//                 [--threads 4] [--ctx 4096] [--gen 0] [--dump-pos 0-3] [--logits] [--batch]
//                 [--max-elems 600000]
// Prefill oracle (M3.2), with --batch:
//   --ubatch U      n_ubatch (production: 256); n_batch = prompt length (one llama_decode call)
//   --last-only     request logits for the last prompt token only (as llama-server does), so the
//                   last layer's FFN runs on that row only (crop_before_nextn)
//   --state F       after the prefill, write llama_state_seq_get_data(seq 0) to F (KV + recurrent)
//   --n-tokens N    truncate the tokenized prompt to its first N tokens
//   --dump-layers L restrict the tensor dump to layers L (e.g. 0-3,39) plus the non-layer tensors
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Dumper {
    FILE * f = nullptr;
    int64_t max_elems = 600000;
    size_t n_written = 0;
    std::set<int> layers;  // empty: all
};

static bool want_tensor(const ggml_tensor * t, const Dumper * d) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        return false;
    }
    const char * n = t->name;
    if (n[0] == 0) {
        return false;
    }
    const int64_t ne = ggml_nelements(t);
    if (ne > d->max_elems && strcmp(n, "result_output") != 0) {
        return false;
    }
    if (!d->layers.empty()) {
        const char * dash = strrchr(n, '-');
        if (dash && dash[1] >= '0' && dash[1] <= '9' && !d->layers.count(atoi(dash + 1))) {
            return false;
        }
    }
    return true;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user) {
    Dumper * d = (Dumper *) user;
    if (!d || !d->f) {
        return false;
    }
    if (ask) {
        return want_tensor(t, d);
    }
    if (!want_tensor(t, d)) {
        return true;
    }
    const size_t esz = ggml_type_size(t->type);
    std::vector<uint8_t> raw;
    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * base;
    if (host) {
        base = (const uint8_t *) t->data;
    } else {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        base = raw.data();
    }
    std::vector<uint8_t> out((size_t) ggml_nelements(t) * esz);
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    const uint8_t * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    memcpy(&out[o], p, esz);
                    o += esz;
                }
    const uint32_t nl = (uint32_t) strlen(t->name);
    const uint32_t ty = t->type == GGML_TYPE_F32 ? 0 : 1;
    const uint32_t nd = 4;
    fwrite(&nl, 4, 1, d->f);
    fwrite(t->name, 1, nl, d->f);
    fwrite(&ty, 4, 1, d->f);
    fwrite(&nd, 4, 1, d->f);
    fwrite(t->ne, 8, 4, d->f);
    fwrite(out.data(), 1, out.size(), d->f);
    d->n_written++;
    return true;
}

static std::vector<int32_t> parse_ids(const std::string & s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            v.push_back(atoi(item.c_str()));
        }
    }
    return v;
}

static void parse_range(const std::string & s, std::set<int> & out) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto dash = item.find('-');
        if (dash == std::string::npos) {
            out.insert(atoi(item.c_str()));
        } else {
            int a = atoi(item.substr(0, dash).c_str()), b = atoi(item.substr(dash + 1).c_str());
            for (int i = a; i <= b; i++) out.insert(i);
        }
    }
}

int main(int argc, char ** argv) {
    std::string model, out, tokens_s, prompt, prompt_file, dump_pos;
    std::string state_out, dump_layers;
    int threads = 4, n_ctx = 4096, n_gen = 0, n_ubatch = 0, n_tokens_max = 0;
    bool logits = false, batch = false, last_only = false;
    int64_t max_elems = 600000;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--out") out = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--gen") n_gen = atoi(next().c_str());
        else if (a == "--dump-pos") dump_pos = next();
        else if (a == "--logits") logits = true;
        else if (a == "--batch") batch = true;
        else if (a == "--max-elems") max_elems = atoll(next().c_str());
        else if (a == "--ubatch") n_ubatch = atoi(next().c_str());
        else if (a == "--n-tokens") n_tokens_max = atoi(next().c_str());
        else if (a == "--last-only") last_only = true;
        else if (a == "--state") state_out = next();
        else if (a == "--dump-layers") dump_layers = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || out.empty()) {
        fprintf(stderr, "usage: dump_ref --model M --out DIR (--tokens ..|--prompt ..|--prompt-file ..) [opts]\n");
        return 2;
    }
    if (!prompt_file.empty()) {
        std::ifstream f(prompt_file);
        std::stringstream ss;
        ss << f.rdbuf();
        prompt = ss.str();
    }
    std::string mk = "mkdir -p '" + out + "'";
    if (system(mk.c_str()) != 0) return 1;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.no_host = true;    // as production (--no-host)
    mp.load_mtp = false;  // MTP off
    mp.use_extra_bufts = true;  // CPU repack (what --repack does)
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<int32_t> toks = parse_ids(tokens_s);
    if (toks.empty() && !prompt.empty()) {
        std::vector<llama_token> buf(prompt.size() + 16);
        int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), buf.data(), (int) buf.size(), false, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        toks.assign(buf.begin(), buf.begin() + n);
    }
    if (toks.empty()) { fprintf(stderr, "no tokens\n"); return 2; }
    if (n_tokens_max > 0) {
        if ((int) toks.size() < n_tokens_max) { fprintf(stderr, "prompt has only %zu tokens\n", toks.size()); return 2; }
        toks.resize(n_tokens_max);
    }

    Dumper d;
    d.max_elems = max_elems;
    if (!dump_layers.empty()) parse_range(dump_layers, d.layers);
    std::set<int> dpos;
    if (!dump_pos.empty()) parse_range(dump_pos, dpos);

    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = batch ? std::max<int>(toks.size(), 1) : 1;
    cp.n_ubatch = (batch && n_ubatch > 0) ? std::min<int>(n_ubatch, cp.n_batch) : cp.n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.no_perf = true;
    cp.op_offload = false;
    if (!dpos.empty()) {
        cp.cb_eval = cb_eval;
        cp.cb_eval_user_data = &d;
    }
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    FILE * lf = logits ? fopen((out + "/logits.f32").c_str(), "wb") : nullptr;
    std::vector<int32_t> gen;
    auto argmax = [&](const float * lg) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) if (lg[i] > lg[best]) best = i;
        return best;
    };
    int n_eval = 0;
    auto eval_one = [&](int32_t tok, int pos) -> const float * {
        if (dpos.count(pos)) {
            d.f = fopen((out + "/pos" + std::to_string(pos) + ".tensors").c_str(), "wb");
        }
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = tok;
        b.pos[0] = pos;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (d.f) { fclose(d.f); d.f = nullptr; }
        if (rc != 0) { fprintf(stderr, "decode failed at %d\n", pos); exit(1); }
        n_eval++;
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (lf) fwrite(lg, 4, n_vocab, lf);
        return lg;
    };

    const float * last = nullptr;
    if (batch) {
        // whole prompt in one ubatch (prefill path); dumps go to pos<last>.tensors
        int P = (int) toks.size();
        if (!dpos.empty()) d.f = fopen((out + "/pos" + std::to_string(P - 1) + ".batch.tensors").c_str(), "wb");
        llama_batch b = llama_batch_init(P, 0, 1);
        b.n_tokens = P;
        for (int i = 0; i < P; i++) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (!last_only || i == P - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batch decode failed\n"); return 1; }
        llama_batch_free(b);
        if (d.f) { fclose(d.f); d.f = nullptr; }
        for (int i = last_only ? P - 1 : 0; i < P; i++) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (lf) fwrite(lg, 4, n_vocab, lf);
            last = lg;
        }
        if (!state_out.empty()) {
            const size_t sz = llama_state_seq_get_size(ctx, 0);
            std::vector<uint8_t> buf(sz);
            const size_t w = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
            FILE * sf = fopen(state_out.c_str(), "wb");
            if (!sf || fwrite(buf.data(), 1, w, sf) != w) { fprintf(stderr, "state write failed\n"); return 1; }
            fclose(sf);
            fprintf(stderr, "dump_ref: state %zu bytes -> %s\n", w, state_out.c_str());
        }
        n_eval = P;
    } else {
        for (size_t i = 0; i < toks.size(); i++) last = eval_one(toks[i], (int) i);
    }
    int pos = (int) toks.size();
    for (int g = 0; g < n_gen; g++) {
        int t = argmax(last);
        gen.push_back(t);
        if (g + 1 < n_gen) last = eval_one(t, pos++);
    }
    if (lf) fclose(lf);

    FILE * mf = fopen((out + "/manifest.json").c_str(), "w");
    fprintf(mf, "{\n  \"model\": \"%s\",\n  \"n_vocab\": %d,\n  \"threads\": %d,\n  \"n_ctx\": %d,\n  \"batch\": %s,\n  \"n_eval\": %d,\n",
            model.c_str(), n_vocab, threads, n_ctx, batch ? "true" : "false", n_eval);
    fprintf(mf, "  \"n_ubatch\": %d,\n  \"last_only\": %s,\n  \"state\": \"%s\",\n", (int) cp.n_ubatch,
            last_only ? "true" : "false", state_out.c_str());
    fprintf(mf, "  \"tokens\": [");
    for (size_t i = 0; i < toks.size(); i++) fprintf(mf, "%s%d", i ? "," : "", toks[i]);
    fprintf(mf, "],\n  \"generated\": [");
    for (size_t i = 0; i < gen.size(); i++) fprintf(mf, "%s%d", i ? "," : "", gen[i]);
    fprintf(mf, "],\n  \"dump_pos\": [");
    { bool first = true; for (int p : dpos) { fprintf(mf, "%s%d", first ? "" : ",", p); first = false; } }
    fprintf(mf, "],\n  \"logits\": %s\n}\n", logits ? "true" : "false");
    fclose(mf);
    fprintf(stderr, "dump_ref: %d evals, %zu generated, %zu tensors dumped -> %s\n", n_eval, gen.size(), d.n_written, out.c_str());
    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
