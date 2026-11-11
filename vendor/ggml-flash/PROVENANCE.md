# Vendored ggml-flash kernels: provenance

- Upstream: llama.cpp fork (ggml CPU backend, R4 expert kernels), MIT licence (see `LICENSE`, verbatim below).
- Local clone: `<BigMoeOnEdge checkout>/third_party/llama.cpp`, branch `claude/flash-r4`.
- Pinned commit: `800986470f752d49f20e8eaa5d00c8d3aee373be` (short `8009864`).
- Regenerate: `python3 vendor/ggml-flash/extract.py <BigMoeOnEdge checkout>/third_party/llama.cpp 800986470f752d49f20e8eaa5d00c8d3aee373be`.
  The script reads `git show <commit>:<path>` only, copies whole files verbatim (with the transitive closure of
  their quoted `#include`s) and copies each item verbatim (brace-matched).
- Compiler: GCC is required (not clang). The numerics depend on GCC's FMA contraction of the scalar float code
  (`-ffp-contract=fast` is GCC's default under `-std=gnu11`), so bit-exact results are only claimed for GCC.
- Compile flags for the engine: `-O3 -DNDEBUG -std=gnu11 -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme`
  (same as the ggml-cpu items of `vendor/ggml-subset`). Include paths: `-I vendor/ggml-flash/generated/upstream/include
  -I vendor/ggml-flash/generated/upstream/src -I vendor/ggml-flash/generated/upstream/src/ggml-cpu`.
  The test build additionally uses `-D_GNU_SOURCE -lm`.

## Hand-written glue (not upstream)

- `oxf_kernels.c`: one translation unit. `#define`s every extracted external function name to an `oxf_` name,
  then includes `generated/arm_quants_subset.c`; adds `oxf_sizeof_block_*()` and `oxf_block_d_offset()`.
  It does not include `oxf_kernels.h`, because `quantize_row_q8_K_ref` takes `block_q8_K *` and the header takes `void *`.
- `oxf_kernels.h`: plain C declarations (no ggml headers) for the `oxf_*` functions, the three R4 entry points, and
  the `OXF_TYPE_*` numbers (equal to the `GGML_TYPE_*` values of the pinned ggml.h; checked by `_Static_assert`s
  in `oxf_kernels.c`).
- `test/test_oxf.c`: standalone correctness test (see the header comment of that file for the checks).
- `extract.py` (this script): generator, not part of the build.

`expert-r4.c` is compiled unmodified as its own translation unit. Its `ggml_cpu_expert_r4_*` names are unique.

## Extracted items

| Output | Upstream file | Line | Kind | Name |
|---|---|---:|---|---|
| `generated/upstream/src/ggml-cpu/expert-r4.c` | `ggml/src/ggml-cpu/expert-r4.c` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/expert-r4.h` | `ggml/src/ggml-cpu/expert-r4.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/quants.h` | `ggml/src/ggml-cpu/quants.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/ggml-cpu-impl.h` | `ggml/src/ggml-cpu/ggml-cpu-impl.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-cpu/simd-mappings.h` | `ggml/src/ggml-cpu/simd-mappings.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-quants.h` | `ggml/src/ggml-quants.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-common.h` | `ggml/src/ggml-common.h` | 1 | file | `(verbatim)` |
| `generated/upstream/src/ggml-impl.h` | `ggml/src/ggml-impl.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml.h` | `ggml/include/ggml.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-cpu.h` | `ggml/include/ggml-cpu.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-backend.h` | `ggml/include/ggml-backend.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/ggml-alloc.h` | `ggml/include/ggml-alloc.h` | 1 | file | `(verbatim)` |
| `generated/upstream/include/gguf.h` | `ggml/include/gguf.h` | 1 | file | `(verbatim)` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-quants.c` | 630 | fn | `nearest_int` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-quants.c` | 2784 | fn | `quantize_row_q8_K_ref` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 24 | define | `UNUSED` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 3598 | var | `keven_signs_q2xs` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 41 | fn | `quantize_row_q8_0` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 2969 | fn | `ggml_vec_dot_q6_K_q8_K` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 1153 | fn | `ggml_vec_dot_q8_0_q8_0` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 3770 | fn | `ggml_vec_dot_iq2_s_q8_K` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 3867 | fn | `ggml_vec_dot_iq3_xxs_q8_K` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 4259 | fn | `ggml_vec_dot_iq4_xs_q8_K` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 4199 | fn | `ggml_vec_dot_iq4_nl_q8_0` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/arch/arm/quants.c` | 222 | fn | `ggml_vec_dot_q2_0_q8_0` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 899 | fn | `ggml_vec_dot_q6_K_q8_K_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 499 | fn | `ggml_vec_dot_q8_0_q8_0_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 1046 | fn | `ggml_vec_dot_iq2_s_q8_K_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 1098 | fn | `ggml_vec_dot_iq3_xxs_q8_K_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 1331 | fn | `ggml_vec_dot_iq4_xs_q8_K_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 1302 | fn | `ggml_vec_dot_iq4_nl_q8_0_generic` |
| `generated/arm_quants_subset.c` | `ggml/src/ggml-cpu/quants.c` | 182 | fn | `ggml_vec_dot_q2_0_q8_0_generic` |

## Verbatim whole files (sha256)

- `arm_quants_subset.c` sha256 `aa7ae70b5a4b01b92bc373db98292aebec31a291ad4579b5425bed40d9103c35`
- `upstream/include/ggml-alloc.h` sha256 `d114067f9099c4a0e2bbc0625c676d4894d9cf0ea27168ee072c7de0e71bb51d`
- `upstream/include/ggml-backend.h` sha256 `34464da4b46314c710cd634626578c6f2a6811431d41eddaf5e8aa77082fea2a`
- `upstream/include/ggml-cpu.h` sha256 `662295a5e330a4cfb4df6b2c6cf198db2c78725a8859b35629b194e1022bcce3`
- `upstream/include/ggml.h` sha256 `1aef7a5f465b1c19de305dd6f9a0c303f88e485133a94b9a0ee1cce6345c9101`
- `upstream/include/gguf.h` sha256 `e56714aab702e5ce62ee587a409643c08f7e93e8fbb77f48ef7cc85075f96fa4`
- `upstream/src/ggml-common.h` sha256 `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`
- `upstream/src/ggml-cpu/expert-r4.c` sha256 `5c5a610fa56361c01c109c1c4e22f11248e5755e89bcb3a47e890fad61d4ed4f`
- `upstream/src/ggml-cpu/expert-r4.h` sha256 `e9498d19d2a042216ae866fa4ab3227907d0772b6ac813b174dbdb1f131fcb06`
- `upstream/src/ggml-cpu/ggml-cpu-impl.h` sha256 `1dc9a672a756a1184c965d8b58679236292285ce5a2479317d51a7e1a0f7dc19`
- `upstream/src/ggml-cpu/quants.h` sha256 `ec1212bb61b1dafca4f9f4375577fd5f2a9db3dab509cc60dde51cb50c29b680`
- `upstream/src/ggml-cpu/simd-mappings.h` sha256 `5b1106263759a83e779185b72567d629cb690eb0ccdb0040d635b156a605d465`
- `upstream/src/ggml-impl.h` sha256 `c84c8f9e3c246504602f5cd388ba54e36691e46bf150e4d70ae857bb823c985f`
- `upstream/src/ggml-quants.h` sha256 `d8ed2c13734cbbc5016011611b6acf14905a54a7d1a3b2c6c3389164dc068418`

## LICENSE (verbatim from the fork)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
