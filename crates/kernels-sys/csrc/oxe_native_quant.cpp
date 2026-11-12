// Own portable reader, tested against the existing independent GGUF dequantizer.
#include "../../../npu/oxn/common/oxn_native_quant.h"
extern "C" int oxe_native_expert_tile(uint16_t *out, size_t out_halfs,
        const uint8_t *src, size_t bytes, uint32_t kind, uint32_t k,
        uint32_t rows, uint32_t experts, uint32_t expert, uint32_t row0, uint32_t k0) {
    return oxn_native_tile_fp16(out,out_halfs,src,bytes,kind,k,rows,experts,expert,row0,k0);
}
