# Vendored ggml subset: provenance

- Upstream: llama.cpp (ggml CPU backend), MIT licence (see `LICENSE`).
- Local clone: `<llama.cpp checkout>`, branch `claude/q3k-moe-gemm`.
- Pinned commit: `ee6b9a1caf38138cafcfa3d6104f93d8a32db19a`
  (HEAD of `claude/q3k-moe-gemm`: "ggml-cpu: make the Q4_K/Q5_K MUL_MAT_ID gemm bitwise identical to gemv";
  it contains the Q3_K 8x8 repack patch `1c148ab` "ggml-cpu: add Q3_K 8x8 repack with dotprod gemv and i8mm gemm").
- Regenerate: `python3 vendor/ggml-subset/extract.py <llama.cpp checkout> ee6b9a1caf38138cafcfa3d6104f93d8a32db19a`.
  The script reads `git show <commit>:<path>` only and copies each item verbatim (brace-matched).
- Hand-written glue (not upstream): `compat.h` (the few ggml-impl/simd-mappings macros the items need),
  `ggml_subset.h` (C declarations of the extracted functions), `crates/kernels-sys/csrc/oxe_shim.cpp`
  (includes `generated/repack_subset.cpp` and adds row-loop wrappers around the static `make_block_*`)
  and `crates/kernels-sys/csrc/oxe_ops.cpp` (the ported decode ops; see M1.md).
- `generated/upstream/` holds whole upstream files verbatim (ggml-cpu `vec.{h,cpp}`, `simd-mappings.h` and the
  headers they include); `vec.cpp` is compiled unmodified.
- Build flags (crates/kernels-sys/build.rs) are llama.cpp's, per library, from the pinned build.ninja:
  ggml-cpu items `-O3 -DNDEBUG -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme`; ggml-base items
  (`quants_base.c`, from ggml-quants.c) `-O3 -DNDEBUG` without -march. Same compiler (GCC 16) required for
  bit-exactness: scalar float code is FMA-contracted/vectorised by the compiler.

## Extracted items

| Output | Upstream file | Line | Kind | Name |
|---|---|---:|---|---|
| `generated/ggml-common.h` | `ggml/src/ggml-common.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml.h` | `ggml/include/ggml.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-cpu.h` | `ggml/include/ggml-cpu.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-backend.h` | `ggml/include/ggml-backend.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-alloc.h` | `ggml/include/ggml-alloc.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/gguf.h` | `ggml/include/gguf.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-impl.h` | `ggml/src/ggml-impl.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-common.h` | `ggml/src/ggml-common.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/ggml-cpu-impl.h` | `ggml/src/ggml-cpu/ggml-cpu-impl.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/simd-mappings.h` | `ggml/src/ggml-cpu/simd-mappings.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/vec.h` | `ggml/src/ggml-cpu/vec.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/vec.cpp` | `ggml/src/ggml-cpu/vec.cpp` | 1 | file | `(verbatim)` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 48 | struct | `block_q4_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 91 | struct | `block_q6_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 103 | struct | `block_q3_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 80 | struct | `block_q5_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 112 | struct | `block_q8_Kx4` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 13 | fn | `QK_0` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 26 | tstruct | `block` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.h` | 44 | line | `using block_q8_0x4 =` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 358 | fn | `ggml_gemv_q6_K_NxM_q8_K_generic_impl` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 960 | fn | `ggml_gemv_q4_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1120 | fn | `ggml_gemv_q6_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1126 | fn | `ggml_q3x8_raw_scale` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1134 | fn | `ggml_q3x8_value` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1153 | fn | `ggml_gemv_q3_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 552 | fn | `ggml_gemv_q5_K_NxM_q8_K_generic_impl` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1111 | fn | `ggml_gemv_q5_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 1387 | fn | `ggml_gemv_q8_0_4x8_q8_0_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3284 | fn | `make_block_q4_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3540 | fn | `make_block_q6_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3606 | fn | `q3k_raw_scales` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3620 | fn | `q3k_fields` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3636 | fn | `make_block_q3_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3455 | fn | `make_block_q5_Kx8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 3129 | fn | `make_block_q8_0x4` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 29 | fn | `decode_q_Kx8_6bit_scales` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 781 | fn | `ggml_gemv_q4_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1570 | fn | `ggml_gemv_q6_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1774 | fn | `ggml_q3x8_decode_scales` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1792 | define | `GGML_Q3X8_DECODE_TRIPLE` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1816 | fn | `ggml_q3x8_dot_row_dotprod` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1882 | fn | `ggml_gemv_q3_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1094 | fn | `ggml_gemv_q5_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 1969 | fn | `ggml_gemv_q8_0_4x8_q8_0` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 29 | fn | `nearest_int` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 264 | fn | `ggml_quantize_mat_q8_K_4x8_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 137 | fn | `ggml_quantize_mat_q8_0_4x8` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 175 | fn | `ggml_quantize_mat_q8_0_4x8_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 4540 | fn | `ggml_repack_interleave_q8_rows_x4` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 448 | fn | `ggml_gemm_q6_K_NxM_q8_K_generic_impl` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 648 | fn | `ggml_gemm_q5_K_NxM_q8_K_generic_impl` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 2098 | fn | `ggml_gemm_q4_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 2273 | fn | `ggml_gemm_q5_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 2281 | fn | `ggml_gemm_q6_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 2285 | fn | `ggml_gemm_q3_K_8x8_q8_K_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/repack.cpp` | 2568 | fn | `ggml_gemm_q8_0_4x8_q8_0_generic` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 4155 | fn | `ggml_gemm_q4_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 4675 | fn | `ggml_gemm_q5_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 4932 | fn | `ggml_gemm_qK_8x8_q8_K_gemv_order` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5093 | fn | `ggml_gemm_q4_K_8x8_q8_K_gemv_order` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5110 | fn | `ggml_gemm_q5_K_8x8_q8_K_gemv_order` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5329 | fn | `ggml_gemm_q6_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5733 | fn | `ggml_gemm_q3_K_8x8_q8_K` |
| `generated/repack_subset.cpp` | `ggml/src/ggml-cpu/arch/arm/repack.cpp` | 5919 | fn | `ggml_gemm_q8_0_4x8_q8_0` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 630 | fn | `nearest_int` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 2784 | fn | `quantize_row_q8_K_ref` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 889 | fn | `get_scale_min_k4` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 556 | fn | `dequantize_row_q8_0` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 1315 | fn | `dequantize_row_q3_K` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 1539 | fn | `dequantize_row_q4_K` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 1741 | fn | `dequantize_row_q5_K` |
| `generated/quants_base.c` | `ggml/src/ggml-quants.c` | 1949 | fn | `dequantize_row_q6_K` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 41 | fn | `quantize_row_q8_0` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 665 | fn | `ggml_vec_dot_q3_K_q8_K_generic` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 744 | fn | `ggml_vec_dot_q4_K_q8_K_generic` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 899 | fn | `ggml_vec_dot_q6_K_q8_K_generic` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 819 | fn | `ggml_vec_dot_q5_K_q8_K_generic` |
| `generated/quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 499 | fn | `ggml_vec_dot_q8_0_q8_0_generic` |

## Checksums of generated files

- `ggml-common.h` sha256 `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`
- `quants_base.c` sha256 `2d4f5301dac900b19f511928cb2a2a2a3a45665c0184fc9feeec6501fe4f5690`
- `quants_subset.c` sha256 `b8d9f2993d86b3cf0be0a552c97fd9dc474b0f1d54052992d65bde093c1706b5`
- `repack_subset.cpp` sha256 `0f5b081886863267c6cd455d1ee9f8d98fa6e6b03f60a8a90800350fbbd6b3a1`
- `upstream/include/ggml-alloc.h` sha256 `d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d`
- `upstream/include/ggml-backend.h` sha256 `34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a`
- `upstream/include/ggml-cpu.h` sha256 `316279e004cdeb8e6ef78599acb602bf79a8abdf897fed9fd1914808c1518c6e`
- `upstream/include/ggml.h` sha256 `590e00a522b8791235353a8c7388d67ff2fe929554f1834c58528b82afd67392`
- `upstream/include/gguf.h` sha256 `e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4`
- `upstream/src/ggml-common.h` sha256 `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`
- `upstream/src/ggml-cpu/ggml-cpu-impl.h` sha256 `1dc9a672a756a1184c965d8b58679236292285ce5a2479317d51a7e1a0f7dc19`
- `upstream/src/ggml-cpu/simd-mappings.h` sha256 `5b1106263759a83e779185b72567d629cb690eb0ccdb0040d635b156a605d465`
- `upstream/src/ggml-cpu/vec.cpp` sha256 `41113028bc5df753a06935d51f702e757d9e30b54450c6b3463980e87ebde12a`
- `upstream/src/ggml-cpu/vec.h` sha256 `6afaaa70b626987e15dc37594e434a0cec28437b02391e8f72367239654ec768`
- `upstream/src/ggml-impl.h` sha256 `75465b01eba2b1ecae8634898508e71294fbff0f7f96ee9ad2caf5bae5b6c5fd`
