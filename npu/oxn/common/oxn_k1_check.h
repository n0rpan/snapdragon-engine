/* SPDX-License-Identifier: MIT
 * Host/DSP shared bounds for our synthetic K1 command. No SDK definitions. */
#ifndef OXN_K1_CHECK_H
#define OXN_K1_CHECK_H
#include <stdint.h>
#include "oxn_proto.h"
#include "oxn_check.h"
#define OXN_K1_CHUNK (128u << 20)
#define OXN_K1_TILE 640u
#define OXN_K1_BLOCK_TILES 8u
/* Fourteen existing wire words; seed and route seed are independent. */
enum {K1_POOL, K1_X, K1_Y, K1_K, K1_O, K1_EXPERTS, K1_TOPK, K1_LAYERS,
      K1_WEIGHT_SEED, K1_FORMAT, K1_THREADS, K1_REPS, K1_ROUTE_SEED, K1_RESERVED};
struct oxn_k1_plan {
    uint32_t matrix_bytes, stride, experts_per_slot, slots_per_layer, pool_slots;
    uint32_t act_bytes, out_bytes, threads, strip_bytes, block_bytes, worker_bytes, vtcm_bytes;
    uint64_t logical_bytes, weight_bytes;
};
static inline uint32_t oxn_k1_align(uint32_t x){return(x+127u)&~127u;}
static inline int oxn_k1_check(const uint32_t*a,const struct oxn_slot_view*sv,uint32_t vtcm,
                               struct oxn_k1_plan*p,uint32_t*detail){
    if(detail)*detail=0;
    uint32_t k=a[K1_K],o=a[K1_O],ne=a[K1_EXPERTS],top=a[K1_TOPK],layers=a[K1_LAYERS];
    if(!k||!o||k%32||o%32||k>4096||o>8192||!(ne==32||ne==128)||!top||top>8||top>ne||
       !layers||layers>16||a[K1_FORMAT]>1||!a[K1_THREADS]||a[K1_THREADS]>4||!a[K1_REPS]||a[K1_REPS]>32||a[K1_RESERVED])return -1;
    p->matrix_bytes=(k/32)*(o/32)*OXN_K1_TILE;
    p->stride=(p->matrix_bytes+4095u)&~4095u;
    if(p->stride<p->matrix_bytes||p->stride>OXN_K1_CHUNK)return -2;
    p->experts_per_slot=OXN_K1_CHUNK/p->stride;
    p->slots_per_layer=(ne+p->experts_per_slot-1)/p->experts_per_slot;
    p->pool_slots=layers*p->slots_per_layer;
    uint32_t base=a[K1_POOL],x=a[K1_X],y=a[K1_Y];
    if(base>=OXN_MAX_SLOTS||p->pool_slots>OXN_MAX_SLOTS-base||x>=OXN_MAX_SLOTS||y>=OXN_MAX_SLOTS||x==y||
       (x>=base&&x<base+p->pool_slots)||(y>=base&&y<base+p->pool_slots))return -3;
    p->act_bytes=(k/32)*1152;p->out_bytes=layers*top*o*4;
    if(sv[x].bytes<p->act_bytes||sv[y].bytes<p->out_bytes)return -4;
    for(uint32_t l=0;l<layers;l++)for(uint32_t b=0;b<p->slots_per_layer;b++){
        uint32_t exp=ne-b*p->experts_per_slot;if(exp>p->experts_per_slot)exp=p->experts_per_slot;
        uint32_t slot=base+l*p->slots_per_layer+b;
        if(sv[slot].bytes<exp*p->stride){if(detail)*detail=slot;return -5;}
    }
    p->threads=a[K1_THREADS]<(o/32)?a[K1_THREADS]:(o/32);
    p->strip_bytes=(k/32)*OXN_K1_TILE;
    uint32_t nrows=((o/32)+p->threads-1)/p->threads;
    if(nrows>OXN_K1_BLOCK_TILES)nrows=OXN_K1_BLOCK_TILES;
    p->block_bytes=nrows*p->strip_bytes;
    p->worker_bytes=2*p->block_bytes+2*OXN_K1_BLOCK_TILES*128;
    p->vtcm_bytes=oxn_k1_align(p->act_bytes)+p->threads*p->worker_bytes;
    if(p->vtcm_bytes>vtcm)return -6;
    uint64_t matrices=(uint64_t)layers*top*a[K1_REPS];
    p->weight_bytes=matrices*p->matrix_bytes;
    p->logical_bytes=matrices*(k/32)*(o/32)*(a[K1_FORMAT]?544u:576u);
    return 0;
}
#endif
