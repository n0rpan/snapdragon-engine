// Hand-written compatibility header for the vendored ggml subset (not upstream).
// Supplies the handful of macros the extracted items take from ggml.h, ggml-impl.h
// and ggml-cpu/simd-mappings.h at the pinned commit, with the same definitions
// those headers select on AArch64 with NEON + IEEE fp16.
#pragma once

// ggml-base items (quants_base.c) are compiled without -march, like llama.cpp's ggml-base,
// so only NEON is required here; the kernels themselves need dotprod + i8mm to compile.
#if !defined(__aarch64__) || !defined(__ARM_NEON)
#error "kernels-sys targets AArch64 with NEON (Snapdragon X Elite)"
#endif

#include <arm_neon.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#define GGML_RESTRICT __restrict__
#else
#define GGML_RESTRICT restrict
#endif

#define GGML_UNUSED(x) (void)(x)
#define UNUSED GGML_UNUSED

#define GGML_ASSERT(x)                                                         \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "GGML_ASSERT(%s) failed at %s:%d\n", #x, __FILE__, \
                    __LINE__);                                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

typedef uint16_t ggml_fp16_t;

// simd-mappings.h, AArch64 branch: neon_compute_fp16_to_fp32 via __fp16.
static inline float oxe_fp16_to_fp32(ggml_fp16_t h) {
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(ggml_fp16_t));
    return (float) tmp;
}
static inline ggml_fp16_t oxe_fp32_to_fp16(float f) {
    ggml_fp16_t res;
    __fp16 tmp = f;
    memcpy(&res, &tmp, sizeof(ggml_fp16_t));
    return res;
}
#define GGML_CPU_FP16_TO_FP32(x) oxe_fp16_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) oxe_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) oxe_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) oxe_fp32_to_fp16(x)

// ggml-impl.h
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// ggml-cpu-impl.h (__ARM_FEATURE_DOTPROD branch)
#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

// ggml.h: the two activation types the extracted templates compare against (values as upstream)
#ifdef __cplusplus
enum ggml_type { GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q8_K = 15 };
#endif
