/* SPDX-License-Identifier: MIT
 * Native CPU packed Q3_Kx8/Q4_Kx8 -> temporary HMX fp16 tile oracle.
 * Layout reference: llama.cpp ee6b9a1, repack.h/repack.cpp (ggml authors, MIT;
 * vendored with attribution in vendor/ggml-subset/LICENSE and PROVENANCE.md).
 * No resident fp16 weight copy. Portable scalar correctness path, NOT optimized.
 */
#ifndef OXN_NATIVE_QUANT_H
#define OXN_NATIVE_QUANT_H
#include <stddef.h>
#include <stdint.h>
#include "oxn_tiles.h"
#define OXN_NATIVE_Q3K 3u
#define OXN_NATIVE_Q4K 4u

static inline size_t oxn_native_block_bytes(uint32_t kind) {
    return kind == OXN_NATIVE_Q3K ? 880u : kind == OXN_NATIVE_Q4K ? 1152u : 0u;
}
static inline int oxn_native_mul(size_t *x, size_t y) {
    if (y && *x > SIZE_MAX/y) return -1;
    *x *= y; return 0;
}
/* Zero means invalid geometry or overflow. rows are per expert, K is ne0. */
static inline size_t oxn_native_bytes(uint32_t kind, uint32_t K, uint32_t rows, uint32_t experts) {
    size_t n = oxn_native_block_bytes(kind);
    if (!n || !K || K%256u || !rows || rows%8u || !experts ||
        oxn_native_mul(&n,K/256u) || oxn_native_mul(&n,rows/8u) || oxn_native_mul(&n,experts)) return 0;
    return n;
}
static inline uint16_t oxn_native_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1]<<8));
}
/* Inner reader: caller has validated one native block, lane<8, k<256. */
static inline float oxn_native_value(const uint8_t *b, uint32_t kind, uint32_t lane, uint32_t k) {
    const float d = oxn_h2f(oxn_native_u16(b+2u*lane));
    if (kind == OXN_NATIVE_Q3K) {
        const uint8_t *sc=b+16u, *qs=b+112u;
        const uint32_t si=(k/16u)*8u+lane;
        const uint32_t low=si<64u ? sc[si]&15u : sc[si-64u]>>4;
        const int scale=(int)(low | (((sc[64u+si%32u]>>(2u*(si/32u)))&3u)<<4))-32;
        const uint32_t which=(k%64u)/8u, at=(lane%2u)*8u+k%8u;
        const uint8_t *p=qs+((k/64u)*4u+lane/2u)*48u;
        const uint32_t x=p[at],y=p[16u+at],z=p[32u+at];
        uint32_t field;
        switch(which) {
            case 0:field=x>>5;break; case 1:field=(x>>2)&7u;break;
            case 2:field=y>>5;break; case 3:field=(y>>2)&7u;break;
            case 4:field=z>>5;break; case 5:field=(z>>2)&7u;break;
            case 6:field=(x&3u)|((y&1u)<<2);break;
            default:field=((y>>1)&1u)|((z&3u)<<1);break;
        }
        const int q=(int)field-(int)((field&4u)<<1);
        return (d*(float)scale)*(float)q;
    }
    const float dm=oxn_h2f(oxn_native_u16(b+16u+2u*lane));
    const uint8_t *sc=b+32u+(k/32u)*12u;
    const uint32_t c=lane%4u;
    const uint32_t scale=lane<4u ? sc[c]&63u : (sc[8u+c]&15u)|((sc[c]>>6)<<4);
    const uint32_t min=lane<4u ? sc[4u+c]&63u : ((uint32_t)sc[8u+c]>>4)|(((uint32_t)sc[4u+c]>>6)<<4);
    const uint8_t code=b[128u+(k/64u)*256u+((k%32u)/8u)*64u+lane*8u+k%8u];
    const uint32_t q=(code>>(4u*((k%64u)/32u)))&15u;
    return (d*(float)scale)*(float)q-dm*(float)min;
}
/* Invalid metadata is refused before any output write. Source/output must not
 * alias. A non-finite input/result may leave a PARTIAL temporary tile: callers
 * must discard it on error and never publish partial projection outputs.
 * Output has exactly1024 halfs, in oxn_wt_idx(k,n) order. */
static inline int oxn_native_tile_fp16(uint16_t *out, size_t out_halfs,
        const uint8_t *src, size_t src_bytes, uint32_t kind,
        uint32_t K, uint32_t rows, uint32_t experts,
        uint32_t expert, uint32_t row0, uint32_t k0) {
    const size_t total=oxn_native_bytes(kind,K,rows,experts);
    if (!out || out_halfs<1024u || !src || !total || src_bytes<total ||
        expert>=experts || row0%32u || k0%32u || rows<32u || row0>rows-32u || k0>K-32u) return -1;
    const size_t nb=K/256u, bs=oxn_native_block_bytes(kind);
    for (uint32_t n=0;n<32u;n++) {
        const size_t group=(size_t)expert*(rows/8u)+(row0+n)/8u;
        const uint8_t *block=src+(group*nb+k0/256u)*bs;
        for (uint32_t k=0;k<32u;k++) {
            const float v=oxn_native_value(block,kind,(row0+n)%8u,(k0+k)%256u);
            const uint16_t h=oxn_f2h(v);
            if ((h&0x7c00u)==0x7c00u) return -2;
            out[oxn_wt_idx(k,n)]=h;
        }
    }
    return 0;
}
#endif
