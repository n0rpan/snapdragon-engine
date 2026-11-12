/* SPDX-License-Identifier: MIT
 * W8 weight format: int8 weights with one fp16 scale per (output, 32-k block).
 * Portable scalar C, host + DSP. The DSP expansion is in dsp/oxn_w8_hvx.h.
 *
 * Tiles are in the same order as the fp16 weight tiles, [ot][kt], tile = ot*(K/32) + kt.
 * One W8 tile record = OXN_W8_TILE_BYTES = 1152 bytes, 128-byte aligned:
 *   bytes    0..1023 : int8 q at byte oxn_wt_idx(k%32, n%32) (same index as the fp16 tile)
 *   bytes 1024..1087 : 32 fp16 scales (little-endian uint16), scale[j] for n = 32*ot + j, k-block kt
 *   bytes 1088..1151 : zero
 * weight(n,k) = h2f(scale) * q.
 *
 * Source format: GGUF Q8_0. Row n is K/32 blocks of 34 bytes { uint16 d; int8 qs[32]; },
 * rows consecutive. weight(n, 32*b + j) = h2f(d) * qs[j].
 */
#ifndef OXN_W8_H
#define OXN_W8_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "oxn_tiles.h"

#define OXN_W8_TILE_BYTES 1152u
#define OXN_W8_SCALE_OFF  1024u
#define OXN_Q8_0_BLOCK    32u
#define OXN_Q8_0_BYTES    34u

static inline size_t oxn_w8_bytes(uint32_t O, uint32_t K) {
    return (size_t) (O / OXN_TILE) * (K / OXN_TILE) * OXN_W8_TILE_BYTES;
}

/* byte offset of the W8 tile that holds element (n, k) */
static inline size_t oxn_w8_tile_off(uint32_t K, uint32_t n, uint32_t k) {
    return ((size_t) (n / OXN_TILE) * (K / OXN_TILE) + k / OXN_TILE) * OXN_W8_TILE_BYTES;
}

static inline uint16_t oxn_w8_scale_bits(const uint8_t * tile, uint32_t j) {
    return (uint16_t) (tile[OXN_W8_SCALE_OFF + 2u * j] | (tile[OXN_W8_SCALE_OFF + 2u * j + 1u] << 8));
}

static inline void oxn_w8_scale_store(uint8_t * tile, uint32_t j, uint16_t s) {
    tile[OXN_W8_SCALE_OFF + 2u * j] = (uint8_t) (s & 0xffu);
    tile[OXN_W8_SCALE_OFF + 2u * j + 1u] = (uint8_t) (s >> 8);
}

/* dst: oxn_w8_bytes(O,K) bytes. Every byte is written, padding included. q8: may be unaligned. */
static inline void oxn_w8_pack_from_q8_0(uint8_t * dst, const uint8_t * q8, uint32_t O, uint32_t K) {
    const uint32_t nkt = K / OXN_TILE, nb = K / OXN_Q8_0_BLOCK;
    memset(dst, 0, oxn_w8_bytes(O, K));
    for (uint32_t n = 0; n < O; n++)
        for (uint32_t kt = 0; kt < nkt; kt++) {
            const uint8_t * blk = q8 + ((size_t) n * nb + kt) * OXN_Q8_0_BYTES;
            uint8_t * tile = dst + ((size_t) (n / OXN_TILE) * nkt + kt) * OXN_W8_TILE_BYTES;
            const uint32_t j = n % OXN_TILE;
            oxn_w8_scale_store(tile, j, (uint16_t) (blk[0] | (blk[1] << 8)));
            for (uint32_t k = 0; k < OXN_TILE; k++)
                tile[oxn_wt_idx(k, j)] = blk[2 + k];
        }
}

static inline float oxn_w8_get(const uint8_t * w8, uint32_t K, uint32_t n, uint32_t k) {
    const uint8_t * tile = w8 + oxn_w8_tile_off(K, n, k);
    const uint32_t j = n % OXN_TILE;
    return oxn_h2f(oxn_w8_scale_bits(tile, j)) * (float) (int8_t) tile[oxn_wt_idx(k % OXN_TILE, j)];
}

/* scalar reference: dst[i] = f2h(h2f(scale[n]) * q[i]), n = (i%64)/2 */
static inline void oxn_w8_tile_to_fp16_ref(uint16_t * dst_tile, const uint8_t * w8_tile) {
    for (uint32_t i = 0; i < OXN_TILE_ELMS; i++) {
        const uint32_t n = (i % 64u) / 2u;
        const float s = oxn_h2f(oxn_w8_scale_bits(w8_tile, n));
        dst_tile[i] = oxn_f2h(s * (float) (int8_t) w8_tile[i]);
    }
}

/* Q8_0 quantisation of w[O][K] (row-major f32) per 32-k block. For tests. */
static inline void oxn_w8_pack_from_f32(uint8_t * dst, const float * w, uint32_t O, uint32_t K) {
    const uint32_t nkt = K / OXN_TILE;
    memset(dst, 0, oxn_w8_bytes(O, K));
    for (uint32_t n = 0; n < O; n++)
        for (uint32_t kt = 0; kt < nkt; kt++) {
            const float * blk = w + (size_t) n * K + kt * OXN_TILE;
            uint8_t * tile = dst + ((size_t) (n / OXN_TILE) * nkt + kt) * OXN_W8_TILE_BYTES;
            const uint32_t j = n % OXN_TILE;
            float amax = 0.0f;
            for (uint32_t i = 0; i < OXN_TILE; i++) {
                const float a = blk[i] < 0.0f ? -blk[i] : blk[i];
                if (a > amax) amax = a;
            }
            const float d = amax / 127.0f;
            oxn_w8_scale_store(tile, j, oxn_f2h(d));
            for (uint32_t k = 0; k < OXN_TILE; k++) {
                int32_t q = 0;
                if (amax > 0.0f) {
                    const float x = blk[k] / d;
                    q = (int32_t) (x >= 0.0f ? x + 0.5f : x - 0.5f);
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                }
                tile[oxn_wt_idx(k, j)] = (uint8_t) (int8_t) q;
            }
        }
}

#endif /* OXN_W8_H */
