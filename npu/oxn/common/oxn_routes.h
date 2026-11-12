/* SPDX-License-Identifier: MIT
 * Bounded batched MoE routes. Order is the EXISTING CPU prefill order:
 * offsets[E+1], order[T*top], order entry = token*top + slot. Each expert's
 * bucket is ascending. Validate ONCE, hold metadata immutable through the
 * entire phase, then gather/scatter tiles without repeating the full scan.
 * Portable correctness path: no SDK, allocation, worker or device operation.
 */
#ifndef OXN_ROUTES_H
#define OXN_ROUTES_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "oxn_tiles.h"
#define OXN_ROUTE_MAX_TOKENS 8192u
#define OXN_ROUTE_MAX_EXPERTS 1024u
#define OXN_ROUTE_MAX_TOP 32u
#define OXN_ROUTE_MAX_ROWS 65536u
struct oxn_route_view {
    uint32_t tokens,top,experts,total;
    const uint32_t *offsets,*order;
};
/* scratch bits: ceil(total/8), at most8192B. On error v is invalidated.
 * Metadata/scratch/v must not alias. Little-endian host and DSP use u32 words. */
static inline int oxn_routes_validate(struct oxn_route_view*v,
        uint32_t tokens,uint32_t top,uint32_t experts,
        const uint32_t*offsets,size_t n_offsets,const uint32_t*order,size_t n_order,
        uint8_t*seen,size_t seen_bytes) {
    if(!v)return -1;
    memset(v,0,sizeof(*v));
    const uint64_t total=(uint64_t)tokens*top;
    if(!tokens||tokens>OXN_ROUTE_MAX_TOKENS||!top||top>OXN_ROUTE_MAX_TOP||
       !experts||experts>OXN_ROUTE_MAX_EXPERTS||top>experts||total>OXN_ROUTE_MAX_ROWS||
       !offsets||!order||n_offsets!=(size_t)experts+1||n_order!=total||
       !seen||seen_bytes<(total+7)/8||offsets[0]!=0||offsets[experts]!=total)return -1;
    memset(seen,0,(size_t)(total+7)/8);
    for(uint32_t e=0;e<experts;e++){
        const uint32_t a=offsets[e],b=offsets[e+1];
        if(a>b||b>total)return -1;
        uint32_t previous=UINT32_MAX;
        for(uint32_t j=a;j<b;j++){
            const uint32_t id=order[j];
            if(id>=total||(previous!=UINT32_MAX&&(id<=previous||id/top==previous/top))||
               (seen[id/8]&(1u<<(id%8))))return -1;
            seen[id/8]|=(uint8_t)(1u<<(id%8));previous=id;
        }
    }
    *v=(struct oxn_route_view){tokens,top,experts,(uint32_t)total,offsets,order};return 0;
}
/* X is fp16 row-major [T,K] (gate/up) or [T*top,K] (down).
 * Outputs one full activation tile, zero-padding the final row block.
 * v's validated metadata must remain immutable/live, no source/output alias. */
static inline int oxn_routes_gather(uint16_t*out,size_t out_halfs,
        const struct oxn_route_view*v,const uint16_t*x,size_t x_halfs,
        uint32_t K,uint32_t expert,uint32_t m0,uint32_t k0,int per_route) {
    if(!v||!v->offsets||!v->order||!out||out_halfs<1024||!x||!K||K%32||
       expert>=v->experts||m0%32||k0%32||k0>K-32||(per_route!=0&&per_route!=1))return -1;
    const uint64_t need=(uint64_t)(per_route?v->total:v->tokens)*K;
    const uint32_t a=v->offsets[expert],count=v->offsets[expert+1]-a;
    if(x_halfs<need||m0>=count)return -1;
    for(uint32_t r=0;r<32;r++){
        const uint16_t *input=NULL;
        if(r<count-m0){const uint32_t id=v->order[a+m0+r],row=per_route?id:id/v->top;input=x+(size_t)row*K+k0;}
        for(uint32_t k=0;k<32;k++)out[oxn_act_idx(r,k)]=input?input[k]:0;
    }
    return 0;
}
/* One HMX output tile -> canonical CPU route rows [T*top,out_stride] (f32).
 * No expert weighted reduction here: CPU retains its native slot sum order.
 * Padding columns are untouched. Invalid metadata writes nothing; nonfinite
 * values return -2 with partial temporary output, which must be discarded. */
static inline int oxn_routes_scatter(float*out,size_t out_floats,
        const struct oxn_route_view*v,const uint16_t*tile,size_t tile_halfs,
        uint32_t O,uint32_t out_stride,uint32_t expert,uint32_t m0,uint32_t col0) {
    if(!v||!v->offsets||!v->order||!out||!tile||tile_halfs<1024||!O||O%32||
       out_stride<O||expert>=v->experts||m0%32||col0%32||col0>O-32||
       out_floats<(uint64_t)v->total*out_stride)return -1;
    const uint32_t a=v->offsets[expert],count=v->offsets[expert+1]-a;
    if(m0>=count)return -1;
    const uint32_t keep=count-m0<32?count-m0:32;
    for(uint32_t r=0;r<keep;r++)for(uint32_t c=0;c<32;c++){
        const uint16_t h=tile[oxn_out_idx(r,c)];if((h&0x7c00u)==0x7c00u)return -2;
        out[(size_t)v->order[a+m0+r]*out_stride+col0+c]=oxn_h2f(h);
    }
    return 0;
}
#endif
