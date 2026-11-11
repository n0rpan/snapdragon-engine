/* test_oxf.c: standalone correctness test for the vendored ggml-flash kernels (hand-written).
 *
 * Checks, for each of the 7 weight types (q6_K, iq2_s, iq3_xxs, iq4_xs with K=2560; q8_0, iq4_nl, q2_0 with K=640):
 *   1. NEON dot product (oxf_vec_dot_*) vs generic reference (oxf_vec_dot_*_generic), 64 random rows.
 *      Random weight bytes; every fp16 scale `d` is replaced by a random fp16 in [0.001, 0.05].
 *      Activations: uniform [-1, 1], quantised with oxf_quantize_row_q8_K_ref (q8_K types) or
 *      oxf_quantize_row_q8_0 (q8_0 types). Pass: |neon - ref| / max(|ref|, 1e-3) < 1e-4 for every row.
 *   2. For IQ2_S and IQ3_XXS (640 rows x K=2560): R4 round trip (slice_to_r4 then slice_from_r4 is the identity),
 *      and ggml_cpu_expert_r4_gemv on the R4 layout (all 640 rows) is BITWISE equal (memcmp) to the NEON
 *      oxf_vec_dot on the original layout.
 * Prints one line per check and ALL PASSED at the end; exit status 1 on any failure.
 */
#include "oxf_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t next_u64(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ull;
}

static float frand(float lo, float hi) {
    double u = (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0); /* [0, 1) */
    return lo + (hi - lo) * (float) u;
}

static uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16) f;
    uint16_t b;
    memcpy(&b, &h, sizeof(b));
    return b;
}

static size_t block_size(int type) {
    switch (type) {
        case OXF_TYPE_Q8_0:    return oxf_sizeof_block_q8_0();
        case OXF_TYPE_Q6_K:    return oxf_sizeof_block_q6_K();
        case OXF_TYPE_IQ2_S:   return oxf_sizeof_block_iq2_s();
        case OXF_TYPE_IQ3_XXS: return oxf_sizeof_block_iq3_xxs();
        case OXF_TYPE_IQ4_XS:  return oxf_sizeof_block_iq4_xs();
        case OXF_TYPE_IQ4_NL:  return oxf_sizeof_block_iq4_nl();
        case OXF_TYPE_Q2_0:    return oxf_sizeof_block_q2_0();
        default:               abort();
    }
}

static int block_elems(int type) {
    switch (type) {
        case OXF_TYPE_Q8_0:
        case OXF_TYPE_IQ4_NL:  return 32;
        case OXF_TYPE_Q2_0:    return 64;
        default:               return 256; /* QK_K */
    }
}

/* random valid weights: random bytes, then a sane random fp16 scale in every block */
static void fill_weights(uint8_t *w, int64_t nrows, int64_t K, int type) {
    size_t bsz = block_size(type);
    size_t nblk = (size_t) (K / block_elems(type));
    int doff = oxf_block_d_offset(type);
    if (doff < 0) abort();
    size_t total = (size_t) nrows * nblk;
    for (size_t i = 0; i < total * bsz; i++) w[i] = (uint8_t) next_u64();
    for (size_t b = 0; b < total; b++) {
        uint16_t d = f32_to_f16_bits(frand(0.001f, 0.05f));
        memcpy(w + b * bsz + (size_t) doff, &d, sizeof(d));
    }
}

/* activations: uniform [-1, 1], quantised for the dot product type */
static void make_activation(void *vy, int64_t K, int q8k, float *x) {
    for (int64_t i = 0; i < K; i++) x[i] = frand(-1.0f, 1.0f);
    if (q8k) oxf_quantize_row_q8_K_ref(x, vy, K);
    else     oxf_quantize_row_q8_0(x, vy, K);
}

static int g_failures = 0;

static int check_neon_vs_generic(const char *name, int type, int64_t K, int64_t rows,
                                 vec_dot_fn neon, vec_dot_fn gen, int q8k) {
    size_t bsz = block_size(type);
    size_t rb = (size_t) (K / block_elems(type)) * bsz;
    uint8_t *w = malloc((size_t) rows * rb);
    float *x = malloc((size_t) K * sizeof(float));
    size_t act_sz = (size_t) (K / block_elems(q8k ? OXF_TYPE_Q6_K : OXF_TYPE_Q8_0)) *
                    (q8k ? oxf_sizeof_block_q8_K() : oxf_sizeof_block_q8_0());
    void *vy = malloc(act_sz);
    if (!w || !x || !vy) abort();

    fill_weights(w, rows, K, type);
    make_activation(vy, K, q8k, x);

    double max_rel = 0.0;
    int nonfinite = 0;
    for (int64_t r = 0; r < rows; r++) {
        float a = 0.0f, b = 0.0f;
        neon((int) K, &a, 0, w + (size_t) r * rb, 0, vy, 0, 1);
        gen((int) K, &b, 0, w + (size_t) r * rb, 0, vy, 0, 1);
        if (!isfinite(a) || !isfinite(b)) { nonfinite++; continue; }
        double rel = fabs((double) a - (double) b) / fmax(fabs((double) b), 1e-3);
        if (rel > max_rel) max_rel = rel;
    }
    int pass = nonfinite == 0 && max_rel < 1e-4;
    printf("CHECK neon_vs_generic %-10s K=%-5lld rows=%-4lld max_rel_err=%.3e nonfinite=%d %s\n",
           name, (long long) K, (long long) rows, max_rel, nonfinite, pass ? "PASS" : "FAIL");
    if (!pass) g_failures++;
    free(w);
    free(x);
    free(vy);
    return pass;
}

static int check_r4(const char *name, int type, vec_dot_fn neon) {
    const int64_t K = 2560, rows = 640;
    size_t bsz = block_size(type);
    size_t rb = (size_t) (K / 256) * bsz;
    size_t total = (size_t) rows * rb;
    uint8_t *orig = malloc(total), *r4 = malloc(total), *rt = malloc(total);
    float *x = malloc((size_t) K * sizeof(float));
    void *vy = malloc((size_t) (K / 256) * oxf_sizeof_block_q8_K());
    float *s_r4 = malloc((size_t) rows * sizeof(float));
    float *s_ref = malloc((size_t) rows * sizeof(float));
    if (!orig || !r4 || !rt || !x || !vy || !s_r4 || !s_ref) abort();

    fill_weights(orig, rows, K, type);
    memcpy(r4, orig, total);
    make_activation(vy, K, 1, x);

    int type_ok = ggml_cpu_expert_r4_type_ok(type, K, rows);
    int to_ok = type_ok && ggml_cpu_expert_r4_slice_to_r4(type, r4, K, rows);
    int rt_ok = 0;
    if (to_ok) {
        memcpy(rt, r4, total); /* the R4 layout, converted back below */
        ggml_cpu_expert_r4_slice_from_r4(type, rt, K, rows);
        rt_ok = memcmp(rt, orig, total) == 0;
    }
    int rt_pass = to_ok && rt_ok;
    printf("CHECK r4_roundtrip   %-10s K=%lld rows=%lld type_ok=%d to_r4=%d from_r4_identity=%d %s\n",
           name, (long long) K, (long long) rows, type_ok, to_ok, rt_ok, rt_pass ? "PASS" : "FAIL");
    if (!rt_pass) g_failures++;

    ggml_cpu_expert_r4_gemv(type, K, s_r4, r4, vy, rows);
    for (int64_t r = 0; r < rows; r++)
        neon((int) K, &s_ref[r], 0, orig + (size_t) r * rb, 0, vy, 0, 1);

    int64_t mism = 0, first = -1;
    for (int64_t r = 0; r < rows; r++) {
        if (memcmp(&s_r4[r], &s_ref[r], sizeof(float)) != 0) {
            if (first < 0) first = r;
            mism++;
        }
    }
    int pass = to_ok && mism == 0;
    printf("CHECK r4_bitwise     %-10s K=%lld rows=%lld gemv_vs_neon_memcmp mismatches=%lld first=%lld %s\n",
           name, (long long) K, (long long) rows, (long long) mism, (long long) first, pass ? "PASS" : "FAIL");
    if (!pass) {
        g_failures++;
        if (first >= 0)
            printf("  row %lld: r4=%.9g neon=%.9g\n", (long long) first,
                   (double) s_r4[first], (double) s_ref[first]);
    }

    free(orig); free(r4); free(rt); free(x); free(vy); free(s_r4); free(s_ref);
    return pass;
}

int main(void) {
    const int64_t K_BIG = 2560, K_SMALL = 640, ROWS = 64;

    check_neon_vs_generic("q6_K",    OXF_TYPE_Q6_K,    K_BIG,   ROWS, oxf_vec_dot_q6_K_q8_K,   oxf_vec_dot_q6_K_q8_K_generic,   1);
    check_neon_vs_generic("iq2_s",   OXF_TYPE_IQ2_S,   K_BIG,   ROWS, oxf_vec_dot_iq2_s_q8_K,  oxf_vec_dot_iq2_s_q8_K_generic,  1);
    check_neon_vs_generic("iq3_xxs", OXF_TYPE_IQ3_XXS, K_BIG,   ROWS, oxf_vec_dot_iq3_xxs_q8_K, oxf_vec_dot_iq3_xxs_q8_K_generic, 1);
    check_neon_vs_generic("iq4_xs",  OXF_TYPE_IQ4_XS,  K_BIG,   ROWS, oxf_vec_dot_iq4_xs_q8_K,  oxf_vec_dot_iq4_xs_q8_K_generic,  1);
    check_neon_vs_generic("q8_0",    OXF_TYPE_Q8_0,    K_SMALL, ROWS, oxf_vec_dot_q8_0_q8_0,   oxf_vec_dot_q8_0_q8_0_generic,   0);
    check_neon_vs_generic("iq4_nl",  OXF_TYPE_IQ4_NL,  K_SMALL, ROWS, oxf_vec_dot_iq4_nl_q8_0, oxf_vec_dot_iq4_nl_q8_0_generic, 0);
    check_neon_vs_generic("q2_0",    OXF_TYPE_Q2_0,    K_SMALL, ROWS, oxf_vec_dot_q2_0_q8_0,   oxf_vec_dot_q2_0_q8_0_generic,   0);

    check_r4("iq2_s",   OXF_TYPE_IQ2_S,   oxf_vec_dot_iq2_s_q8_K);
    check_r4("iq3_xxs", OXF_TYPE_IQ3_XXS, oxf_vec_dot_iq3_xxs_q8_K);

    if (g_failures == 0) {
        printf("ALL PASSED\n");
        return 0;
    }
    printf("FAILED: %d check(s)\n", g_failures);
    return 1;
}
