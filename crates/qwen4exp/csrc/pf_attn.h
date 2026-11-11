// pf_attn: fixed-chunk attention part op of prefill ubatches in register tiles (see pf_attn.cpp, PROGRESS-PF.md FP-1).
#pragma once

#ifdef __cplusplus
#include "ggml.h"

// Called first by attn_part_tiled (flash_graph.cpp): true = the unit range of thread ith was computed here (mode 1,
// switch on, supported shapes), bitwise equal to attn_part_tiled; false = not taken, the caller continues.
bool fg_pf_attn_part(ggml_tensor * dst, int ith, int nth);
// the part op of flash_graph.cpp (for the model-free check in pf_attn.cpp)
ggml_custom_op_t fg_attn_part_op_ptr();
// the merge op of flash_graph.cpp (reference of the fused op's check)
ggml_custom_op_t fg_attn_merge_op_ptr();
// W4-I (FP-8): attention output without the partial tensor. fused_ok: switch == 2 and the node is supported;
// fused_op: custom op, dst [DV, heads, rows], src = q, k, v, mask, op_params (scale, mode) as the part node.
bool fg_pf_attn_fused_ok(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask);
void fg_pf_attn_fused_op(ggml_tensor * dst, int ith, int nth, void *);

extern "C" {
#endif

// process-wide switch: 0 = off (flash_graph.cpp's own tiled path), 1 = register tiles in the part op (read when the op
// runs), 2 = 1 + the fused part+merge node for prefill ubatches (read when a graph is built)
void fg_pf_set_attn(int on);
int fg_pf_get_attn(void);
// units (8-row tile x kv head x block) taken by the tile path since the last reset
long fg_pf_attn_units(int reset);
// model-free bitwise check (switch off vs on) + timing of the part op; 0 = equal; out2 = ms off, ms on
int fg_pf_selftest_attn(int n_kv, int n_rows, int nth, int pattern, int reps, double * out2);

#ifdef __cplusplus
}
#endif
