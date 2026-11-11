// flash_graph: C API of the qwen4exp ggml graph port (see flash_graph.cpp).
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
typedef struct fg_model fg_model;
typedef struct fg_ctx fg_ctx;

// a memory region weights live in (dense arena, a mapped shard)
typedef struct fg_region {
    void * base;
    size_t size;
} fg_region;

typedef struct fg_tensor_ref {
    const char * name;
    int32_t type;    // ggml_type
    int64_t ne[4];   // file shape
    void * data;
    int32_t region;  // index into the regions, -1 = none
} fg_tensor_ref;

typedef struct fg_ple_params {
    uint64_t mult[3];
    uint32_t head_offsets[16];
    uint32_t head_vocab[16];
    int32_t eos;
} fg_ple_params;

typedef struct fg_dump {
    const char * dir;            // NULL = off; files <dir>/<node name>.bin
    const char * const * names;  // exact, "prefix*" or "*suffix"
    int n_names;
} fg_dump;

// The engine's expert path (streaming cache). With a callback set, the three mul_mat_id of the
// routed experts become custom nodes, in production's order and with production's names:
//   ffn_moe_fetch-<il> (1 thread: look up / queue the selected experts, quantise x)
//   ffn_moe_up-<il>, ffn_moe_gate-<il> [FF, n_used, n_tokens] = W x per selected expert
//   ffn_moe_swiglu-<il> (production ggml_swiglu_split, unchanged)
//   ffn_moe_down-<il> [n_embd, n_used, n_tokens] = W_down act
// Every node, the fetch node included, is called on every graph thread (ith of nth): ggml-cpu ignores a
// custom op's n_tasks at compute time, so the fetch callback must act on ith == 0 only.
// FG_EXP_BATCH: one node for a large ubatch (n_tokens >= fg_options.batch_min_tokens): src = x,
// dst = [n_embd, n_used, n_tokens] down outputs; the callback does fetch/gate/up/swiglu/down itself.
// FG_EXP_CAP: a query at graph build, not a node: the callback writes into dst[0] how many tokens the per-token op
// may take (tokens x n_used experts must fit in the smallest pool with room to spare); wider ubatches use the
// batched op (same bits). F38: a 16-token chunk needs 160 distinct slots under one stamp.
enum { FG_EXP_FETCH = 0, FG_EXP_GATE = 1, FG_EXP_UP = 2, FG_EXP_DOWN = 3, FG_EXP_BATCH = 4, FG_EXP_CAP = 5 };
typedef struct fg_expert_call {
    int kind, il, ith, nth;
    int n_tokens;
    const int32_t * ids;  // selected experts [n_used, n_tokens]
    size_t ids_nb1;       // bytes between tokens
    const float * src;    // x [n_embd, 1, n_tokens] (fetch/gate/up) or act [FF, n_used, n_tokens] (down)
    size_t src_nb1, src_nb2;
    float * dst;          // [rows, n_used, n_tokens]
    size_t dst_nb1, dst_nb2;
} fg_expert_call;
typedef void (*fg_expert_fn)(const fg_expert_call * call, void * userdata);

fg_model * fg_model_new(const fg_tensor_ref * refs, int n, const fg_region * regions, int n_regions, const fg_ple_params * ple);
void fg_model_free(fg_model * m);

fg_ctx * fg_ctx_new(fg_model * m, uint32_t n_ctx);
void fg_ctx_free(fg_ctx * c);
void fg_ctx_clear(fg_ctx * c);
size_t fg_ctx_state_bytes(const fg_ctx * c);
void fg_ctx_set_expert_fn(fg_ctx * c, fg_expert_fn fn, void * userdata);

// Fast-path options (all 0 = the M1 exact graph, production-identical).
#define FG_FUSE_HC 1     // P1 hc-mix (norm, down, silu, up, inject)
#define FG_FUSE_ROUTER 2 // P5 router (logits, softmax, top-k, weights)
#define FG_FUSE_COMB 4   // P6 shared expert + routed-expert combine + final add
#define FG_FUSE_HCPOST 8 // P2 hc-combine (sigmoid, scale 2, hc_post)
#define FG_FUSE_GDN 16   // P3 + P4 GDN pre/post (projections, gates, conv, norms; gated norm, ssm_out; state writes)
#define FG_FUSE_ATTN 32  // P7 attention pre (Q/K/V, q/k norms, gate) and post (sigmoid gate * attention, wo)
typedef struct fg_options {
    int fixed_attn;        // attention in fixed 256-cell blocks merged in order (thread-count / batch independent)
    int shexp_first;       // build the shared expert right after the expert fetch (overlaps the SSD reads)
    int batch_min_tokens;  // >0: ubatches of at least this many tokens use the FG_EXP_BATCH node
    int profile;           // compute node by node and accumulate wall time per node kind (slow; diagnostics)
    int rs_view;           // non-fresh one-sequence ubatches read the recurrent states in place (no 1-row get_rows copy)
    int executor;          // S-b: cached one-token graphs run by ggml's op functions under the engine's barrier schedule
    int fuse;              // S-c fused phases, bitmask FG_FUSE_* (each bitwise = the ggml nodes it replaces)
} fg_options;
// profile table "kind<TAB>calls<TAB>ms" lines, sorted by time; returns the bytes needed (incl. NUL)
size_t fg_ctx_profile_text(const fg_ctx * c, char * buf, size_t cap);
void fg_ctx_profile_reset(fg_ctx * c);
// per-stage wall time of fg_decode since the last reset, ms: graph build, gallocr, inputs, compute; [4] = calls
void fg_ctx_timing(fg_ctx * c, double * out5, int reset);
// one-token decodes reuse the previous call's graph + allocation while the shape is unchanged (default on)
void fg_ctx_set_graph_reuse(fg_ctx * c, int on);
// S-b executor: model-free bitwise check vs ggml_graph_compute (0 = pass); stats = graphs scheduled, nodes, barriers
int fg_selftest_exec(int nth);
// serving cancel: *flag != 0 aborts the running decode between graph nodes -> fg_decode returns -6, positions unchanged,
// the context then returns -7 until fg_ctx_restore_recurrent / fg_ctx_clear. nullptr = off.
void fg_ctx_set_abort(fg_ctx * c, const volatile unsigned char * flag); // a Rust AtomicBool
int fg_selftest_abort(void);
// S-c fused phases, model-free bitwise tests vs the ggml subgraphs (0 = pass)
int fg_selftest_fuse_hc(int nth);
int fg_selftest_fuse_ffn(int nth);
int fg_selftest_fuse_hcpost(int nth);
int fg_selftest_fuse_gdn(int nth);
int fg_selftest_fuse_attn(int nth);
// chunked-prefill split test: which components differ between one chunk and split chunks (flash_split_test.inc)
int fg_selftest_split(int nth);
// model-free dense matvec bandwidth bench at Flash's shapes (prints one line per call)
void fg_bench_dense(int nth, int copies, int reps);
void fg_ctx_exec_stats(const fg_ctx * c, double * out3);
// FD-1 work/wait trace (executor runs only): on/off, and the text table (see flash_graph.cpp)
void fg_ctx_set_trace(fg_ctx * c, int on);
size_t fg_ctx_trace_text(const fg_ctx * c, char * buf, size_t cap);
void fg_ctx_set_options(fg_ctx * c, const fg_options * o);
// self-test of the fixed-chunk attention (no model): 0 = pass
int fg_selftest_attn(int n_kv, int n_rows);
// production swiglu row kernel: y = silu(g) * u
void fg_vec_swiglu(int n, float * y, const float * g, const float * u);
// copy the whole sequence state (KV, indexer, recurrent, PLE history, tokens) of src (same model, n_ctx)
int fg_ctx_copy_state(fg_ctx * dst, const fg_ctx * src);

// position (tokens in the state) and the recurrent snapshot (GDN conv/ssm + PLE history; KV and
// the indexer cache stay in place). restore: pos <= current position; the state becomes that after
// token pos-1 (the snapshot must have been taken at pos).
uint32_t fg_ctx_pos(const fg_ctx * c);
// raw state (fg_ctx_state_bytes bytes) and the token history: save / load a whole sequence state
void * fg_ctx_state_data(fg_ctx * c);
void fg_ctx_get_tokens(const fg_ctx * c, int32_t * out);
void fg_ctx_set_tokens(fg_ctx * c, const int32_t * toks, uint32_t n);
size_t fg_ctx_recurrent_bytes(const fg_ctx * c);
// compact whole state at position pos (KV/indexer rows < pos + recurrent), for persistence
size_t fg_ctx_compact_bytes(fg_ctx * c, uint32_t pos);
void fg_ctx_save_compact(fg_ctx * c, void * buf);                    // at the current position
int fg_ctx_load_compact(fg_ctx * c, const void * buf, uint32_t pos);  // then fg_ctx_set_tokens
void fg_ctx_save_recurrent(const fg_ctx * c, void * buf);
int fg_ctx_restore_recurrent(fg_ctx * c, const void * buf, uint32_t pos);

// production ggml-cpu kernels, for the engine's own ops (exactly what mul_mat_id calls)
int32_t fg_vec_dot_type(int32_t type);
size_t fg_row_size(int32_t type, int64_t n);
void fg_from_float(int32_t vec_dot_type, const float * x, void * y, int64_t n);
void fg_vec_dot(int32_t type, int n, float * s, const void * x, const void * y);

// Evaluate one ubatch (the tokens at the next positions). logits: N_VOCAB floats of the last token.
int fg_decode(fg_ctx * c, const int32_t * tokens, int n_tokens, int n_threads, float * logits, const fg_dump * dump);

#ifdef __cplusplus
}
#endif
