// expert-r4.h -- lossless, size-neutral 4-row interleaved layouts for streamed IQ3_XXS / IQ2_S expert slices,
// and the ARM kernels that consume them (see FLASH-R4.md in the BigMoeOnEdge tree for the design).
//
// An expert slice (ne01 rows of ne00 weights, rows contiguous) is re-encoded IN PLACE, group of 4 rows by group of
// 4 rows: the 4 rows' bytes are replaced by nb = ne00/QK_K "R4" super-blocks of exactly 4x the original block size,
// so every row group starts at the byte offset of its first row and the slice size (nb2) does not change.
//
//   block_iq3_xxs_r4 (392 B = 4 x 98 B)
//     d[4]       fp16 per row
//     qs[256]    grid indices, [ib][row][8]
//     sas[128]   [ib][row][l] = sign7(l) | scale_bit(l) << 7      (8th sign = parity of the 7, as upstream)
//
//   block_iq2_s_r4 (328 B = 4 x 82 B)
//     d[4]       fp16 per row
//     sub[8]     per 32-weight sub-block, 40 bytes:
//                  qi[20]  [row][5]: the four 10-bit grid indices packed little-endian (index l at bits 10l..10l+9)
//                  sg[16]  [row][4]: the four sign bytes (byte l covers weights 8l..8l+7)
//                  sc[4]   [row]   : the scale byte (low nibble: weights 0-15, high nibble: weights 16-31)
//
// Every kernel reproduces the integer sub-block dot products of the upstream NEON vec_dot exactly and then the same
// float operation sequence per (row, token), so its outputs are bitwise identical to ggml_vec_dot_{iq3_xxs,iq2_s}_q8_K
// under the production flags (-ffp-contract=fast, which GCC applies by default under -std=gnu11).
#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t d[4];
    uint8_t  qs[4 * 64];
    uint8_t  sas[4 * 32];
} block_iq3_xxs_r4;

typedef struct {
    uint8_t qi[20];
    uint8_t sg[16];
    uint8_t sc[4];
} block_iq2_s_r4_sub;

typedef struct {
    uint16_t           d[4];
    block_iq2_s_r4_sub sub[8];
} block_iq2_s_r4;

// true when this build has an R4 kernel for `type` and the shape fits it (ne00 % QK_K == 0, ne01 % 4 == 0).
// Only ARM builds with dotprod qualify: elsewhere the upstream vec_dot rounds differently and R4 is not offered.
bool ggml_cpu_expert_r4_type_ok(enum ggml_type type, int64_t ne00, int64_t ne01);

// i8mm 4x4 gemm available in this build (else every token goes through the gemv)
bool ggml_cpu_expert_r4_has_gemm(void);

// in-place transform of one slice of ne01 rows x ne00 weights (rows contiguous), and its exact inverse.
// Returns false (and leaves the bytes untouched) when the type/shape does not qualify or no temp could be had.
bool ggml_cpu_expert_r4_slice_to_r4(enum ggml_type type, void * data, int64_t ne00, int64_t ne01);
bool ggml_cpu_expert_r4_slice_from_r4(enum ggml_type type, void * data, int64_t ne00, int64_t ne01);

// decode (one token): s[r] for the nrows (multiple of 4) rows starting at the R4 group vx; vy = nb q8_K blocks.
void ggml_cpu_expert_r4_gemv(enum ggml_type type, int64_t ne00, float * s, const void * vx, const void * vy, int64_t nrows);

// prefill (four tokens): s[t][r] for the nrows rows starting at vx and tokens vy[0..3]. Requires has_gemm().
void ggml_cpu_expert_r4_gemm4(enum ggml_type type, int64_t ne00, float * const s[4], const void * vx,
                              const void * const vy[4], int64_t nrows);

// the marker the streamer stores in src0->extra; see ggml_cpu_expert_r4_tag() in ggml-cpu.h
bool ggml_cpu_expert_r4_is(const struct ggml_tensor * t);
// true once any tensor has been handed the marker (lets the dispatcher skip its misuse check otherwise)
bool ggml_cpu_expert_r4_in_use(void);

#ifdef __cplusplus
}
#endif
