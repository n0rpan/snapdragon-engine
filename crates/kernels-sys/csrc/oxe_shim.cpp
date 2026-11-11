// Hand-written shim (not upstream). Includes the vendored C++ subset in this translation
// unit so the static make_block_* interleavers can be wrapped with row loops identical to
// ggml's repack_<t>_to_<t>_8_bl (minus the ggml_tensor plumbing), and adds small helpers
// used by the M0 timing runner.
#include "repack_subset.cpp"

// NR rows interleaved, blocks of QK values, interleave block IB bytes.
template <typename SRC, typename DST, DST (*MAKE)(SRC *, unsigned int), int NR, int QK, int IB>
static int oxe_repack_rows(const void * src_v, void * dst_v, int64_t nrows, int64_t k) {
    if (nrows % NR != 0 || k % QK != 0) {
        return -1;
    }
    const SRC * src = (const SRC *) src_v;
    DST * dst = (DST *) dst_v;
    const int64_t nblocks = k / QK;
    SRC tmp[NR];
    // same loop as repack_q4_K_to_q4_K_8_bl / repack_q6_K_to_q6_K_8_bl / repack_q3_K_to_q3_K_8_bl /
    // repack_q5_K_to_q5_K_8_bl (NR 8, IB 8) and repack_q8_0_to_q8_0_4_bl (NR 4, IB 8)
    for (int64_t b = 0; b < nrows; b += NR) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < NR; i++) {
                tmp[i] = src[x + i * nblocks];
            }
            *dst++ = MAKE(tmp, IB);
        }
        src += NR * nblocks;
    }
    return 0;
}

extern "C" {

size_t oxe_sizeof_block_q8_K(void) { return sizeof(block_q8_K); }
size_t oxe_sizeof_block_q3_K(void) { return sizeof(block_q3_K); }
size_t oxe_sizeof_block_q4_K(void) { return sizeof(block_q4_K); }
size_t oxe_sizeof_block_q6_K(void) { return sizeof(block_q6_K); }
size_t oxe_sizeof_block_q3_Kx8(void) { return sizeof(block_q3_Kx8); }
size_t oxe_sizeof_block_q4_Kx8(void) { return sizeof(block_q4_Kx8); }
size_t oxe_sizeof_block_q6_Kx8(void) { return sizeof(block_q6_Kx8); }
size_t oxe_sizeof_block_q5_K(void) { return sizeof(block_q5_K); }
size_t oxe_sizeof_block_q5_Kx8(void) { return sizeof(block_q5_Kx8); }
size_t oxe_sizeof_block_q8_0(void) { return sizeof(block_q8_0); }
size_t oxe_sizeof_block_q8_0x4(void) { return sizeof(block_q8_0x4); }

int oxe_repack_q4_K_8x8(const void * src, void * dst, int64_t nrows, int64_t k) {
    return oxe_repack_rows<block_q4_K, block_q4_Kx8, make_block_q4_Kx8, 8, QK_K, 8>(src, dst, nrows, k);
}
int oxe_repack_q6_K_8x8(const void * src, void * dst, int64_t nrows, int64_t k) {
    return oxe_repack_rows<block_q6_K, block_q6_Kx8, make_block_q6_Kx8, 8, QK_K, 8>(src, dst, nrows, k);
}
int oxe_repack_q5_K_8x8(const void * src, void * dst, int64_t nrows, int64_t k) {
    return oxe_repack_rows<block_q5_K, block_q5_Kx8, make_block_q5_Kx8, 8, QK_K, 8>(src, dst, nrows, k);
}
int oxe_repack_q8_0_4x8(const void * src, void * dst, int64_t nrows, int64_t k) {
    return oxe_repack_rows<block_q8_0, block_q8_0x4, make_block_q8_0x4, 4, QK8_0, 8>(src, dst, nrows, k);
}
int oxe_repack_q3_K_8x8(const void * src, void * dst, int64_t nrows, int64_t k) {
    return oxe_repack_rows<block_q3_K, block_q3_Kx8, make_block_q3_Kx8, 8, QK_K, 8>(src, dst, nrows, k);
}

void oxe_quantize_row_q8_K(const float * x, void * y, int64_t k) {
    quantize_row_q8_K_ref(x, (block_q8_K *) y, k);
}

// f32 -> f16 row conversion (KV cache writes).
void oxe_f32_to_f16(const float * x, uint16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(x + i)), vcvt_f16_f32(vld1q_f32(x + i + 4)));
        vst1q_u16(y + i, vreinterpretq_u16_f16(h));
    }
    for (; i < n; i++) {
        y[i] = oxe_fp32_to_fp16(x[i]);
    }
}

} // extern "C"

// ---- M3.2 batched prefill helpers (hand-written glue around the vendored templates)
extern "C" {

// ggml_repack_interleave_q8_rows_x4<8, Q8_K>: 4 from_float q8_K rows -> one block_q8_Kx4 tile
// (repack.cpp forward_mul_mat_id, gemm path).
void oxe_interleave_q8_K_rows_x4(const void * const rows[4], void * out, int64_t n_per_row) {
    ggml_repack_interleave_q8_rows_x4<8, GGML_TYPE_Q8_K>(rows, out, n_per_row);
}
// ggml_repack_interleave_q8_rows_x4<8, Q8_0> (the q8_0_4x8 trait has INTER_SIZE 8).
void oxe_interleave_q8_0_rows_x4(const void * const rows[4], void * out, int64_t n_per_row) {
    ggml_repack_interleave_q8_rows_x4<8, GGML_TYPE_Q8_0>(rows, out, n_per_row);
}
size_t oxe_sizeof_block_q8_Kx4(void) { return sizeof(block_q8_Kx4); }

} // extern "C"

// ---- moe-prefill: multi-token SMMLA tiles (bitwise equal to the vendored gemm kernels)
#include "oxe_pf_tiles.inc"

// ---- verify-lane GEMVs (OD-6b): weights read and unpacked once for 2-4 lanes, bitwise the GEMV per lane
#include "oxe_vl.inc"
