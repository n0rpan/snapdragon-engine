/* SPDX-License-Identifier: MIT
 * Synthetic, deterministic K1 fixtures. Logical values and packed tiles have
 * independent access paths, so the f64 checker does not decode its own output.
 * Tile layout follows public ggml-hexagon (MIT), pin2cbe0ec, htp-repack.cpp.
 */
#ifndef OXN_K1_FIXTURE_H
#define OXN_K1_FIXTURE_H
#include <stdint.h>
#include <string.h>
#include <math.h>
#define OXN_K1_TILE_BYTES 640u
#define OXN_K1_ACT_BYTES 1152u
static inline uint32_t oxn_k1_hash(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
static inline uint32_t oxn_k1_code(uint32_t seed,uint32_t row,uint32_t col){return oxn_k1_hash(seed^(row*0x9e3779b9u)^(col*0x85ebca6bu))>>28;}
static inline int oxn_k1_a(uint32_t seed,uint32_t col){return (int)(oxn_k1_hash(seed^col)%63)-31;}
static inline uint16_t oxn_k1_half_power(int e){return (uint16_t)((e+15)<<10);}
static inline int oxn_k1_scale_e(uint32_t seed,uint32_t row,uint32_t kt){return (int)(oxn_k1_hash(seed^row^(kt*17u))&3)-5;}
static inline int oxn_k1_mx_e(uint32_t seed,uint32_t row,uint32_t kt){return (int)(oxn_k1_hash(seed^row^(kt*17u))&3)+125;}
static inline double oxn_k1_weight(int mx,uint32_t seed,uint32_t row,uint32_t col){
    static const int lut[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}; /* public MXFP4 E2M1×2 */
    const int q=(int)oxn_k1_code(seed,row,col);
    return mx ? .5*lut[q]*ldexp(1.0,oxn_k1_mx_e(seed,row,col/32)-127)
              : (q-8)*ldexp(1.0,oxn_k1_scale_e(seed,row,col/32));
}
static inline double oxn_k1_reference(int mx,uint32_t ws,uint32_t as,uint32_t row,uint32_t k){
    double y=0;for(uint32_t c=0;c<k;c++)y+=oxn_k1_weight(mx,ws,row,c)*oxn_k1_a(as,c)/64.0;return y;
}
static inline void oxn_k1_pack_weight(uint8_t *dst,int mx,uint32_t seed,uint32_t o,uint32_t k){
    for(uint32_t ct=0;ct<o/32;ct++)for(uint32_t kt=0;kt<k/32;kt++){
        uint8_t*t=dst+((size_t)ct*(k/32)+kt)*OXN_K1_TILE_BYTES;memset(t,0,OXN_K1_TILE_BYTES);
        for(uint32_t cp=0;cp<16;cp++)for(uint32_t row=0;row<32;row++){
            uint32_t r=ct*32+row,c=kt*32+cp*2;
            t[cp*32+row]=(uint8_t)(oxn_k1_code(seed,r,c)|(oxn_k1_code(seed,r,c+1)<<4));
        }
        for(uint32_t row=0;row<32;row++){
            if(mx)t[512+row]=(uint8_t)oxn_k1_mx_e(seed,ct*32+row,kt);
            else{uint16_t h=oxn_k1_half_power(oxn_k1_scale_e(seed,ct*32+row,kt));memcpy(t+512+row*2,&h,2);}
        }
    }
}
static inline void oxn_k1_pack_act(uint8_t*dst,uint32_t seed,uint32_t k){
    for(uint32_t kt=0;kt<k/32;kt++){
        uint8_t*t=dst+(size_t)kt*OXN_K1_ACT_BYTES;memset(t,0,OXN_K1_ACT_BYTES);
        for(uint32_t g=0;g<8;g++)for(uint32_t row=0;row<32;row++)for(uint32_t j=0;j<4;j++)
            t[g*128+row*4+j]=(uint8_t)(int8_t)oxn_k1_a(seed,kt*32+g*4+j);
        for(uint32_t row=0;row<64;row++){uint16_t h=oxn_k1_half_power(-6);memcpy(t+1024+row*2,&h,2);}
    }
}
#endif
