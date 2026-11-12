/* SPDX-License-Identifier: MIT
 * Bounded native expert-batch pricing operation. Own protocol, no resident f16 weights.
 */
#ifndef OXN_NATIVE_CHECK_H
#define OXN_NATIVE_CHECK_H
#include "oxn_check.h"
#include "oxn_native_quant.h"
#define OXN_OP_NATIVE_BATCH 14u
#define OXN_NATIVE_NARGS 14
/* xslot,xoff,wslot,woff,yslot,yoff,paddedT,K,O,E,kind,vtcmCap,reserved,reserved */
static inline int oxn_native_batch_check(const uint32_t *a,const struct oxn_slot_view*s,
                                        uint32_t vtcm,uint32_t*detail){
    if(!a||!s||!detail)return OXN_E_GEMM_DIMS;
    *detail=0;
    const uint32_t T=a[6],K=a[7],O=a[8],E=a[9],kind=a[10];
    if((T!=32&&T!=128)||!K||K%512||K>8192||!O||O%32||O>8192||!E||E>256||
       (kind!=3&&kind!=4)||a[12]||a[13]){*detail=T;return OXN_E_GEMM_DIMS;}
    uint64_t bytes[3]={(uint64_t)E*T*K*2,oxn_native_bytes(kind,K,O,E),(uint64_t)E*T*O*2};
    for(unsigned i=0;i<3;i++){
        const uint32_t slot=a[2*i],off=a[2*i+1];
        if(slot>=OXN_MAX_SLOTS||!s[slot].bytes){*detail=slot;return OXN_E_GEMM_SLOT_X+i;}
        if(off&127u){*detail=off;return OXN_E_GEMM_ALIGN;}
        if(!bytes[i]||bytes[i]>UINT32_MAX||off>s[slot].bytes||bytes[i]>s[slot].bytes-off){
            *detail=(uint32_t)(bytes[i]>>10);return OXN_E_GEMM_BOUNDS_X+i;}
    }
    for(unsigned i=0;i<2;i++)if(a[4]==a[2*i]){
        uint64_t y0=a[5],y1=y0+bytes[2],x0=a[2*i+1],x1=x0+bytes[i];
        if(y0<x1&&x0<y1){*detail=i;return OXN_E_GEMM_OVERLAP;}
    }
    if(a[11]&&a[11]>vtcm){*detail=a[11];return OXN_E_GEMM_VTCM_CAP;}
    const uint32_t cap=a[11]?a[11]:vtcm;
    /* Kernel uses C<=4, shrinks to1 if necessary; this is its C=1 minimum. */
    uint64_t tk=(uint64_t)(K/32)*2048,sk=(uint64_t)4*(K/256)*oxn_native_block_bytes(kind);
    uint64_t need=4096+(T/32)*tk+2*((sk+2047)&~2047ull)+tk+2*(T/32)*2048;
    if(need>cap){*detail=(uint32_t)need;return OXN_E_GEMM_VTCM_CAP;}
    return 0;
}
#endif
