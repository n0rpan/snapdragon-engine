/* oxf_kernels.h: plain C interface to the vendored ggml-flash kernels (see PROVENANCE.md).
 *
 * No ggml headers are needed by includers. Type arguments are ggml type numbers (int); the
 * values below equal the GGML_TYPE_* values of the pinned ggml.h, checked by _Static_assert
 * in oxf_kernels.c. The R4 functions are defined in generated/upstream/src/ggml-cpu/expert-r4.c,
 * whose prototypes take `enum ggml_type`; that enum has int size and ABI on this target, so
 * passing int here is ABI-compatible.
 *
 * Build: GCC, -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme, -std=gnu11 (see PROVENANCE.md).
 */
#ifndef OXF_KERNELS_H
#define OXF_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum oxf_type {
    OXF_TYPE_Q8_0    = 8,
    OXF_TYPE_Q6_K    = 14,
    OXF_TYPE_Q8_K    = 15,
    OXF_TYPE_IQ3_XXS = 18,
    OXF_TYPE_IQ4_NL  = 20,
    OXF_TYPE_IQ2_S   = 22,
    OXF_TYPE_IQ4_XS  = 23,
    OXF_TYPE_Q2_0    = 42,
};

/* ---- dot products: NEON (ggml-cpu/arch/arm/quants.c) and generic (ggml-cpu/quants.c) references ----
 * Same contract as ggml's vec_dot: s = dot(vx row, vy), n = number of weights, nrc = 1 here. */
void oxf_vec_dot_q6_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_q8_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq2_s_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq3_xxs_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq4_xs_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq4_nl_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_q2_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

void oxf_vec_dot_q6_K_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_q8_0_q8_0_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq2_s_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq3_xxs_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq4_xs_q8_K_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_iq4_nl_q8_0_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
void oxf_vec_dot_q2_0_q8_0_generic(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

/* ---- activation quantisers. y must hold k/QK_K block_q8_K (resp. k/32 block_q8_0). ---- */
void oxf_quantize_row_q8_0(const float *x, void *y, int64_t k);
void oxf_quantize_row_q8_K_ref(const float *x, void *y, int64_t k);

/* ---- block sizes in bytes ---- */
size_t oxf_sizeof_block_q8_K(void);
size_t oxf_sizeof_block_q8_0(void);
size_t oxf_sizeof_block_q6_K(void);
size_t oxf_sizeof_block_iq2_s(void);
size_t oxf_sizeof_block_iq3_xxs(void);
size_t oxf_sizeof_block_iq4_xs(void);
size_t oxf_sizeof_block_iq4_nl(void);
size_t oxf_sizeof_block_q2_0(void);
/* byte offset of the fp16 super-scale `d` inside one block of `type`; -1 if the type has none (q8_K). */
int oxf_block_d_offset(int type);

/* ---- R4 expert layout (generated/upstream/src/ggml-cpu/expert-r4.c). type: ggml type number. ---- */
bool ggml_cpu_expert_r4_type_ok(int type, int64_t ne00, int64_t ne01);
bool ggml_cpu_expert_r4_slice_to_r4(int type, void *data, int64_t ne00, int64_t ne01);
bool ggml_cpu_expert_r4_slice_from_r4(int type, void *data, int64_t ne00, int64_t ne01);
void ggml_cpu_expert_r4_gemv(int type, int64_t ne00, float *s, const void *vx, const void *vy, int64_t nrows);

#ifdef __cplusplus
}
#endif

#endif /* OXF_KERNELS_H */
