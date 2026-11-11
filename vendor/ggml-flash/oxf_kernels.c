/* oxf_kernels.c: hand-written glue (not upstream). ONE translation unit.
 *
 * The extracted items are compiled under oxf_-prefixed names so that this object does not clash with
 * any other vendored copy of ggml symbols linked into the same binary. The defines must precede the
 * include of generated/arm_quants_subset.c (which also pulls in the ggml headers).
 * generated/upstream/src/ggml-cpu/expert-r4.c is a separate translation unit, compiled unmodified.
 */
#define ggml_vec_dot_q6_K_q8_K oxf_vec_dot_q6_K_q8_K
#define ggml_vec_dot_q6_K_q8_K_generic oxf_vec_dot_q6_K_q8_K_generic
#define ggml_vec_dot_q8_0_q8_0 oxf_vec_dot_q8_0_q8_0
#define ggml_vec_dot_q8_0_q8_0_generic oxf_vec_dot_q8_0_q8_0_generic
#define ggml_vec_dot_iq2_s_q8_K oxf_vec_dot_iq2_s_q8_K
#define ggml_vec_dot_iq2_s_q8_K_generic oxf_vec_dot_iq2_s_q8_K_generic
#define ggml_vec_dot_iq3_xxs_q8_K oxf_vec_dot_iq3_xxs_q8_K
#define ggml_vec_dot_iq3_xxs_q8_K_generic oxf_vec_dot_iq3_xxs_q8_K_generic
#define ggml_vec_dot_iq4_xs_q8_K oxf_vec_dot_iq4_xs_q8_K
#define ggml_vec_dot_iq4_xs_q8_K_generic oxf_vec_dot_iq4_xs_q8_K_generic
#define ggml_vec_dot_iq4_nl_q8_0 oxf_vec_dot_iq4_nl_q8_0
#define ggml_vec_dot_iq4_nl_q8_0_generic oxf_vec_dot_iq4_nl_q8_0_generic
#define ggml_vec_dot_q2_0_q8_0 oxf_vec_dot_q2_0_q8_0
#define ggml_vec_dot_q2_0_q8_0_generic oxf_vec_dot_q2_0_q8_0_generic
#define quantize_row_q8_0 oxf_quantize_row_q8_0
#define quantize_row_q8_K_ref oxf_quantize_row_q8_K_ref

#include "generated/arm_quants_subset.c"

#include <stddef.h>

/* The enum values used by oxf_kernels.h must equal the pinned ggml.h. */
_Static_assert(GGML_TYPE_Q8_0 == 8, "oxf_type");
_Static_assert(GGML_TYPE_Q6_K == 14, "oxf_type");
_Static_assert(GGML_TYPE_Q8_K == 15, "oxf_type");
_Static_assert(GGML_TYPE_IQ3_XXS == 18, "oxf_type");
_Static_assert(GGML_TYPE_IQ4_NL == 20, "oxf_type");
_Static_assert(GGML_TYPE_IQ2_S == 22, "oxf_type");
_Static_assert(GGML_TYPE_IQ4_XS == 23, "oxf_type");
_Static_assert(GGML_TYPE_Q2_0 == 42, "oxf_type");

size_t oxf_sizeof_block_q8_K(void)   { return sizeof(block_q8_K); }
size_t oxf_sizeof_block_q8_0(void)   { return sizeof(block_q8_0); }
size_t oxf_sizeof_block_q6_K(void)   { return sizeof(block_q6_K); }
size_t oxf_sizeof_block_iq2_s(void)  { return sizeof(block_iq2_s); }
size_t oxf_sizeof_block_iq3_xxs(void){ return sizeof(block_iq3_xxs); }
size_t oxf_sizeof_block_iq4_xs(void) { return sizeof(block_iq4_xs); }
size_t oxf_sizeof_block_iq4_nl(void) { return sizeof(block_iq4_nl); }
size_t oxf_sizeof_block_q2_0(void)   { return sizeof(block_q2_0); }

int oxf_block_d_offset(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0:    return (int) offsetof(block_q8_0, d);
        case GGML_TYPE_Q6_K:    return (int) offsetof(block_q6_K, d);
        case GGML_TYPE_IQ3_XXS: return (int) offsetof(block_iq3_xxs, d);
        case GGML_TYPE_IQ4_NL:  return (int) offsetof(block_iq4_nl, d);
        case GGML_TYPE_IQ2_S:   return (int) offsetof(block_iq2_s, d);
        case GGML_TYPE_IQ4_XS:  return (int) offsetof(block_iq4_xs, d);
        case GGML_TYPE_Q2_0:    return (int) offsetof(block_q2_0, d);
        default:                return -1;
    }
}
