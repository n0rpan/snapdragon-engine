#!/usr/bin/env python3
"""Regenerate the vendored ggml subset from a pinned llama.cpp commit.

Usage:
  python3 vendor/ggml-subset/extract.py [LLAMA_REPO] [COMMIT]

Reads files with `git show COMMIT:path` (never the working tree), extracts the
named top-level definitions verbatim by brace matching, and writes them into
vendor/ggml-subset/generated/. Nothing is edited by hand inside generated/;
the only hand-written glue is compat.h and ../../crates/kernels-sys/csrc/.
PROVENANCE.md lists the commit and every extracted item.
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1])  # path of the llama.cpp checkout
COMMIT = sys.argv[2] if len(sys.argv) > 2 else 'ee6b9a1caf38138cafcfa3d6104f93d8a32db19a'

CPU = 'ggml/src/ggml-cpu'
# (source file, kind, name). kind: fn = function (with a preceding template line if any),
# struct = struct definition, define = multi-line #define macro.
CPP_ITEMS = [
    (f'{CPU}/repack.h', 'struct', 'block_q4_Kx8'),
    (f'{CPU}/repack.h', 'struct', 'block_q6_Kx8'),
    (f'{CPU}/repack.h', 'struct', 'block_q3_Kx8'),
    (f'{CPU}/repack.h', 'struct', 'block_q5_Kx8'),
    (f'{CPU}/repack.h', 'struct', 'block_q8_Kx4'),
    (f'{CPU}/repack.h', 'fn', 'QK_0'),
    (f'{CPU}/repack.h', 'tstruct', 'block'),
    (f'{CPU}/repack.h', 'line', 'using block_q8_0x4 ='),
    # generic (scalar) reference kernels on the interleaved layouts
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q6_K_NxM_q8_K_generic_impl'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q4_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q6_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_q3x8_raw_scale'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_q3x8_value'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q3_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q5_K_NxM_q8_K_generic_impl'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q5_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemv_q8_0_4x8_q8_0_generic'),
    # load-time repack (block interleavers)
    (f'{CPU}/repack.cpp', 'fn', 'make_block_q4_Kx8'),
    (f'{CPU}/repack.cpp', 'fn', 'make_block_q6_Kx8'),
    (f'{CPU}/repack.cpp', 'fn', 'q3k_raw_scales'),
    (f'{CPU}/repack.cpp', 'fn', 'q3k_fields'),
    (f'{CPU}/repack.cpp', 'fn', 'make_block_q3_Kx8'),
    (f'{CPU}/repack.cpp', 'fn', 'make_block_q5_Kx8'),
    (f'{CPU}/repack.cpp', 'fn', 'make_block_q8_0x4'),
    # NEON dotprod/i8mm kernels used at decode
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'decode_q_Kx8_6bit_scales'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemv_q4_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemv_q6_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_q3x8_decode_scales'),
    (f'{CPU}/arch/arm/repack.cpp', 'define', 'GGML_Q3X8_DECODE_TRIPLE'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_q3x8_dot_row_dotprod'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemv_q3_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemv_q5_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemv_q8_0_4x8_q8_0'),
    # ---- M3.2 batched prefill (repack forward_mul_mat / forward_mul_mat_id) ----
    # 4-row activation quantisers (ggml_quantize_mat_t<8, Q8_K> is the generic one on AArch64,
    # arch-fallback.h maps ggml_quantize_mat_q8_K_4x8_generic -> ggml_quantize_mat_q8_K_4x8)
    (f'{CPU}/repack.cpp', 'fn', 'nearest_int'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_quantize_mat_q8_K_4x8_generic'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_quantize_mat_q8_0_4x8'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_quantize_mat_q8_0_4x8_generic'),
    # 4-row tile interleaver of from_float rows (MUL_MAT_ID gemm path)
    (f'{CPU}/repack.cpp', 'fn', 'ggml_repack_interleave_q8_rows_x4'),
    # generic gemm references (and link-time fallbacks of the NEON kernels)
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q6_K_NxM_q8_K_generic_impl'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q5_K_NxM_q8_K_generic_impl'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q4_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q5_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q6_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q3_K_8x8_q8_K_generic'),
    (f'{CPU}/repack.cpp', 'fn', 'ggml_gemm_q8_0_4x8_q8_0_generic'),
    # NEON i8mm gemm kernels (MUL_MAT: gemm<>; MUL_MAT_ID: gemm_mmid<>)
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q4_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q5_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_qK_8x8_q8_K_gemv_order'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q4_K_8x8_q8_K_gemv_order'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q5_K_8x8_q8_K_gemv_order'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q6_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q3_K_8x8_q8_K'),
    (f'{CPU}/arch/arm/repack.cpp', 'fn', 'ggml_gemm_q8_0_4x8_q8_0'),
]
# ggml-base items (ggml/src/ggml-quants.c): llama.cpp compiles these WITHOUT -march flags.
BASE_ITEMS = [
    # activation quantisation (on ARM, quantize_row_q8_K is this ref function)
    ('ggml/src/ggml-quants.c', 'fn', 'nearest_int'),
    ('ggml/src/ggml-quants.c', 'fn', 'quantize_row_q8_K_ref'),
    # token embedding rows (get_rows -> to_float)
    ('ggml/src/ggml-quants.c', 'fn', 'get_scale_min_k4'),
    ('ggml/src/ggml-quants.c', 'fn', 'dequantize_row_q8_0'),
    ('ggml/src/ggml-quants.c', 'fn', 'dequantize_row_q3_K'),
    ('ggml/src/ggml-quants.c', 'fn', 'dequantize_row_q4_K'),
    ('ggml/src/ggml-quants.c', 'fn', 'dequantize_row_q5_K'),
    ('ggml/src/ggml-quants.c', 'fn', 'dequantize_row_q6_K'),
]
# ggml-cpu C items: compiled with the ggml-cpu -march flags.
C_ITEMS = [
    # Q8_0 activation quantisation (NEON version, from_float of the q8_0 repack path)
    (f'{CPU}/arch/arm/quants.c', 'fn', 'quantize_row_q8_0'),
    # row-major (non-repacked) references: the FFI tests check repack + kernel against these
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q3_K_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q4_K_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q6_K_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q5_K_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q8_0_q8_0_generic'),
]
VERBATIM = ['ggml/src/ggml-common.h']
# Whole files copied verbatim into generated/upstream/ (same relative layout) so that
# ggml-cpu/vec.cpp (dot products, silu/swiglu, softmax, scale/mad) compiles unmodified and
# the ported ops in csrc/oxe_ops.cpp use exactly ggml's inline vector helpers.
UPSTREAM_TREE = [
    'ggml/include/ggml.h',
    'ggml/include/ggml-cpu.h',
    'ggml/include/ggml-backend.h',
    'ggml/include/ggml-alloc.h',
    'ggml/include/gguf.h',
    'ggml/src/ggml-impl.h',
    'ggml/src/ggml-common.h',
    'ggml/src/ggml-cpu/ggml-cpu-impl.h',
    'ggml/src/ggml-cpu/simd-mappings.h',
    'ggml/src/ggml-cpu/vec.h',
    'ggml/src/ggml-cpu/vec.cpp',
]


def show(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{COMMIT}:{path}'],
                          check=True, capture_output=True, text=True).stdout


def find_start(lines, kind, name):
    if kind == 'struct':
        pat = re.compile(rf'^struct {re.escape(name)}\s*{{')
    elif kind == 'tstruct':
        pat = re.compile(rf'^template <[^>]*> struct {re.escape(name)}\s*{{')
    elif kind == 'define':
        pat = re.compile(rf'^#define {re.escape(name)}\b')
    elif kind == 'line':
        pat = re.compile(rf'^{re.escape(name)}')
    else:
        pat = re.compile(rf'^(?!\s)(?:static\s+)?(?:inline\s+)?[\w:<>,\s\*]*?\b{re.escape(name)}\s*\(')
    hits = [i for i, l in enumerate(lines) if pat.match(l)]
    # a definition, not a declaration: the signature must reach '{' before ';'
    defs = []
    for i in hits:
        j = i
        while j < len(lines) and '{' not in lines[j] and ';' not in lines[j]:
            j += 1
        if j < len(lines) and '{' in lines[j] or kind in ('define', 'line'):
            defs.append(i)
    if len(defs) != 1:
        raise SystemExit(f'{name}: expected exactly one definition, found {len(defs)}')
    start = defs[0]
    if kind == 'fn' and start > 0 and lines[start - 1].startswith('template'):
        start -= 1
    return start


def extract(text, kind, name):
    lines = text.split('\n')
    start = find_start(lines, kind, name)
    if kind == 'line':
        return lines[start], start + 1
    if kind == 'define':
        end = start
        while lines[end].rstrip().endswith('\\'):
            end += 1
        return '\n'.join(lines[start:end + 1]), start + 1
    depth, seen, end = 0, False, start
    for end in range(start, len(lines)):
        code = re.sub(r'//.*', '', lines[end])
        depth += code.count('{') - code.count('}')
        seen = seen or '{' in code
        if seen and depth == 0:
            break
    body = '\n'.join(lines[start:end + 1])
    return body, start + 1


def emit(items, header, out_name, provenance):
    cache = {}
    parts = [header]
    for path, kind, name in items:
        if path not in cache:
            cache[path] = show(path)
        body, line = extract(cache[path], kind, name)
        parts.append(f'// ---- {kind} {name}  ({path}:{line} @ {COMMIT[:10]})\n{body}\n')
        provenance.append((out_name, path, line, kind, name))
    (HERE / 'generated' / out_name).write_text('\n'.join(parts))


def main():
    (HERE / 'generated').mkdir(exist_ok=True)
    prov = []
    for path in VERBATIM:
        (HERE / 'generated' / Path(path).name).write_text(show(path))
        prov.append((Path(path).name, path, 1, 'file', '(verbatim)'))
    for path in UPSTREAM_TREE:
        rel = Path(path).relative_to('ggml')
        dst = HERE / 'generated' / 'upstream' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(show(path))
        prov.append((f'upstream/{rel}', path, 1, 'file', '(verbatim)'))
    banner = ('// GENERATED by vendor/ggml-subset/extract.py from llama.cpp {c}. Do not edit.\n'
              '// MIT licence, (c) the ggml authors; see vendor/ggml-subset/LICENSE.\n').format(c=COMMIT)
    emit(CPP_ITEMS, banner + '#include "ggml_subset.h"\n', 'repack_subset.cpp', prov)
    emit(BASE_ITEMS, banner + '#include "ggml_subset.h"\n', 'quants_base.c', prov)
    emit(C_ITEMS, banner + '#include "ggml_subset.h"\n', 'quants_subset.c', prov)
    rows = '\n'.join(f'| `generated/{o}` | `{p}` | {l} | {k} | `{n}` |' for o, p, l, k, n in prov)
    sums = '\n'.join(f'- `{f.relative_to(HERE / "generated")}` sha256 `{hashlib.sha256(f.read_bytes()).hexdigest()}`'
                     for f in sorted((HERE / 'generated').rglob('*')) if f.is_file())
    (HERE / 'PROVENANCE.md').write_text(f"""# Vendored ggml subset: provenance

- Upstream: llama.cpp (ggml CPU backend), MIT licence (see `LICENSE`).
- Local clone: `<llama.cpp checkout>`, branch `claude/q3k-moe-gemm`.
- Pinned commit: `{COMMIT}`
  (HEAD of `claude/q3k-moe-gemm`: "ggml-cpu: make the Q4_K/Q5_K MUL_MAT_ID gemm bitwise identical to gemv";
  it contains the Q3_K 8x8 repack patch `1c148ab` "ggml-cpu: add Q3_K 8x8 repack with dotprod gemv and i8mm gemm").
- Regenerate: `python3 vendor/ggml-subset/extract.py <llama.cpp checkout> {COMMIT}`.
  The script reads `git show <commit>:<path>` only and copies each item verbatim (brace-matched).
- Hand-written glue (not upstream): `compat.h` (the few ggml-impl/simd-mappings macros the items need),
  `ggml_subset.h` (C declarations of the extracted functions), `crates/kernels-sys/csrc/oxe_shim.cpp`
  (includes `generated/repack_subset.cpp` and adds row-loop wrappers around the static `make_block_*`)
  and `crates/kernels-sys/csrc/oxe_ops.cpp` (the ported decode ops; see M1.md).
- `generated/upstream/` holds whole upstream files verbatim (ggml-cpu `vec.{{h,cpp}}`, `simd-mappings.h` and the
  headers they include); `vec.cpp` is compiled unmodified.
- Build flags (crates/kernels-sys/build.rs) are llama.cpp's, per library, from the pinned build.ninja:
  ggml-cpu items `-O3 -DNDEBUG -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme`; ggml-base items
  (`quants_base.c`, from ggml-quants.c) `-O3 -DNDEBUG` without -march. Same compiler (GCC 16) required for
  bit-exactness: scalar float code is FMA-contracted/vectorised by the compiler.

## Extracted items

| Output | Upstream file | Line | Kind | Name |
|---|---|---:|---|---|
{rows}

## Checksums of generated files

{sums}
""")
    print(f'wrote {len(prov)} items')


if __name__ == '__main__':
    main()
