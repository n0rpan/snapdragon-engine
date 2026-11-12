/* SPDX-License-Identifier: MIT
 * HMX fp16 32x32 tile layouts used by oxn (portable scalar C, host + DSP + simulator).
 *
 * The layouts are the ones the ggml-hexagon HMX matmul builds in VTCM (llama.cpp,
 * MIT, ggml/src/ggml-hexagon/htp/hmx-utils.h hmx_interleave_rows_to_tiles and
 * hmx-mm-kernels-tiled.h transfer_activation_row_pair_f16_to_f16 /
 * transfer_output_chunk_fp16_to_fp32). A tile is 1024 fp16 = 16 HVX vectors of 64 halfwords:
 *
 *   activation tile, element (m,k):  (m/2)*64 + 2*k + (m&1)   (rows paired)
 *   weight tile,     element (k,n):  (k/2)*64 + 2*n + (k&1)   (k paired)
 *   output tile,     element (m,n):  (m/2)*64 + 2*n + (m&1)   (rows paired)
 *
 * Tile order in memory:
 *   X tiles [rt][kt]  (T/32 x K/32)
 *   W tiles [ot][kt]  (O/32 x K/32)    W is [O][K] row-major before packing
 *   Y tiles [rt][ot]  (T/32 x O/32)    default; with OXN_GF_YCOL: [ot][rt] (column-major)
 * so a row block of X, a column block of W and a row of Y tiles are contiguous.
 */
#ifndef OXN_TILES_H
#define OXN_TILES_H

#include <stddef.h>
#include <stdint.h>

#define OXN_TILE      32u
#define OXN_TILE_ELMS 1024u
#define OXN_TILE_BYTES 2048u

static inline size_t oxn_act_idx(uint32_t m, uint32_t k) { return (size_t) (m >> 1) * 64u + 2u * k + (m & 1u); }
static inline size_t oxn_wt_idx(uint32_t k, uint32_t n)  { return (size_t) (k >> 1) * 64u + 2u * n + (k & 1u); }
static inline size_t oxn_out_idx(uint32_t m, uint32_t n) { return (size_t) (m >> 1) * 64u + 2u * n + (m & 1u); }

/* fp16 bit helpers (IEEE binary16), so the code does not depend on a compiler half type */
static inline float oxn_h2f(uint16_t h) {
    uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) {
        if (m == 0) { u = s; }
        else {        /* subnormal */
            e = 113; while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3ff; u = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) { u = s | 0x7f800000u | (m << 13); }
    else { u = s | ((e + 112) << 23) | (m << 13); }
    union { uint32_t u; float f; } c = { u };
    return c.f;
}

static inline uint16_t oxn_f2h(float f) {  /* round to nearest even */
    union { float f; uint32_t u; } c = { f };
    uint32_t x = c.u, s = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        uint32_t shift = (uint32_t) (14 - e), hm = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return (uint16_t) (s | hm);
    }
    uint32_t hm = m >> 13, rem = m & 0x1fffu;
    uint32_t h = s | ((uint32_t) e << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) h++;
    return (uint16_t) h;
}

/* row-major X[T][K] (fp16 bits) -> tiles [rt][kt] */
static inline void oxn_pack_act(uint16_t *dst, const uint16_t *x, uint32_t T, uint32_t K) {
    const uint32_t nkt = K / OXN_TILE;
    for (uint32_t m = 0; m < T; m++)
        for (uint32_t k = 0; k < K; k++) {
            uint16_t *tile = dst + ((size_t) (m / OXN_TILE) * nkt + k / OXN_TILE) * OXN_TILE_ELMS;
            tile[oxn_act_idx(m % OXN_TILE, k % OXN_TILE)] = x[(size_t) m * K + k];
        }
}

/* row-major W[O][K] (fp16 bits) -> tiles [ot][kt] */
static inline void oxn_pack_wt(uint16_t *dst, const uint16_t *w, uint32_t O, uint32_t K) {
    const uint32_t nkt = K / OXN_TILE;
    for (uint32_t n = 0; n < O; n++)
        for (uint32_t k = 0; k < K; k++) {
            uint16_t *tile = dst + ((size_t) (n / OXN_TILE) * nkt + k / OXN_TILE) * OXN_TILE_ELMS;
            tile[oxn_wt_idx(k % OXN_TILE, n % OXN_TILE)] = w[(size_t) n * K + k];
        }
}

/* one element of Y from tiles [rt][ot] */
static inline uint16_t oxn_out_get(const uint16_t *ytiles, uint32_t O, uint32_t m, uint32_t n) {
    const uint32_t not_ = O / OXN_TILE;
    const uint16_t *tile = ytiles + ((size_t) (m / OXN_TILE) * not_ + n / OXN_TILE) * OXN_TILE_ELMS;
    return tile[oxn_out_idx(m % OXN_TILE, n % OXN_TILE)];
}

static inline void oxn_out_set(uint16_t *ytiles, uint32_t O, uint32_t m, uint32_t n, uint16_t v) {
    const uint32_t not_ = O / OXN_TILE;
    uint16_t *tile = ytiles + ((size_t) (m / OXN_TILE) * not_ + n / OXN_TILE) * OXN_TILE_ELMS;
    tile[oxn_out_idx(m % OXN_TILE, n % OXN_TILE)] = v;
}

/* column-major Y tiles (OXN_GF_YCOL): tile (rt, ot) at index ot*(T/32) + rt */
static inline uint16_t oxn_out_get_col(const uint16_t *ytiles, uint32_t T, uint32_t m, uint32_t n) {
    const uint32_t nrt = T / OXN_TILE;
    const uint16_t *tile = ytiles + ((size_t) (n / OXN_TILE) * nrt + m / OXN_TILE) * OXN_TILE_ELMS;
    return tile[oxn_out_idx(m % OXN_TILE, n % OXN_TILE)];
}

/* element reads straight from packed tiles (for references computed from the packed buffers) */
static inline uint16_t oxn_act_get(const uint16_t *xt, uint32_t K, uint32_t m, uint32_t k) {
    const uint16_t *tile = xt + ((size_t) (m / OXN_TILE) * (K / OXN_TILE) + k / OXN_TILE) * OXN_TILE_ELMS;
    return tile[oxn_act_idx(m % OXN_TILE, k % OXN_TILE)];
}
static inline uint16_t oxn_wt_get(const uint16_t *wt, uint32_t K, uint32_t n, uint32_t k) {
    const uint16_t *tile = wt + ((size_t) (n / OXN_TILE) * (K / OXN_TILE) + k / OXN_TILE) * OXN_TILE_ELMS;
    return tile[oxn_wt_idx(k % OXN_TILE, n % OXN_TILE)];
}

/* deterministic synthetic data (xorshift32), values in [-scale, scale], fp16 bits */
static inline uint32_t oxn_rng(uint32_t *s) { uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *s = x; }
static inline uint16_t oxn_rand_h(uint32_t *s, float scale) {
    float u = (float) (oxn_rng(s) >> 8) * (1.0f / 16777216.0f);  /* [0,1) */
    return oxn_f2h((2.0f * u - 1.0f) * scale);
}

/* f64 reference of one output element, from packed tiles */
static inline double oxn_ref_elem(const uint16_t *xt, const uint16_t *wt, uint32_t K, uint32_t m, uint32_t n) {
    double acc = 0.0;
    for (uint32_t k = 0; k < K; k++)
        acc += (double) oxn_h2f(oxn_act_get(xt, K, m, k)) * (double) oxn_h2f(oxn_wt_get(wt, K, n, k));
    return acc;
}

#endif
