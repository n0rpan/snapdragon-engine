#!/usr/bin/env python3
"""Regenerate the vendored ggml-flash kernels (q6_K/q8_0/iq2_s/iq3_xxs/iq4_xs/iq4_nl/q2_0 NEON + generic
references, the q8 activation quantisers, and the R4 expert kernels) from a pinned llama.cpp fork commit.

Usage:
  python3 vendor/ggml-flash/extract.py [LLAMA_REPO] [COMMIT]

Reads files with `git show COMMIT:path` only (never the working tree), copies whole upstream files verbatim
(plus the transitive closure of their quoted #includes), and extracts the named top-level items verbatim by
brace matching into vendor/ggml-flash/generated/. Nothing under generated/ is edited by hand.
Hand-written glue (oxf_kernels.c, oxf_kernels.h, test/test_oxf.c) is not produced by this script.
PROVENANCE.md and LICENSE are written by this script too.
"""
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('BIGMOE_LLAMA_DIR', 'third_party/llama.cpp'))
COMMIT = sys.argv[2] if len(sys.argv) > 2 else '800986470f752d49f20e8eaa5d00c8d3aee373be'
OUT = HERE / 'generated'
ARM_QUANTS = 'arm_quants_subset.c'

CPU = 'ggml/src/ggml-cpu'
# (source file, kind, name). kind: fn = function (with a preceding template line if any),
# define = single or multi-line #define, var = file-scope static const table.
ARM_ITEMS = [
    (f'{CPU}/arch/arm/quants.c', 'define', 'UNUSED'),
    (f'{CPU}/arch/arm/quants.c', 'var', 'keven_signs_q2xs'),   # used by ggml_vec_dot_iq3_xxs_q8_K
    (f'{CPU}/arch/arm/quants.c', 'fn', 'quantize_row_q8_0'),   # NEON activation quantiser (q8_0 types)
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_q6_K_q8_K'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_q8_0_q8_0'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_iq2_s_q8_K'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_iq3_xxs_q8_K'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_iq4_xs_q8_K'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_iq4_nl_q8_0'),
    (f'{CPU}/arch/arm/quants.c', 'fn', 'ggml_vec_dot_q2_0_q8_0'),
]
GENERIC_ITEMS = [
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q6_K_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q8_0_q8_0_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_iq2_s_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_iq3_xxs_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_iq4_xs_q8_K_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_iq4_nl_q8_0_generic'),
    (f'{CPU}/quants.c', 'fn', 'ggml_vec_dot_q2_0_q8_0_generic'),
]
# ggml-quants.c: q8_K activation quantiser used by the q8_K dot products (+ its static helper)
BASE_ITEMS = [
    ('ggml/src/ggml-quants.c', 'fn', 'nearest_int'),
    ('ggml/src/ggml-quants.c', 'fn', 'quantize_row_q8_K_ref'),
]
# Whole upstream files that the extracted code and the R4 kernels need, copied verbatim.
# The closure of their quoted #includes is added automatically (see closure()).
VERBATIM_ROOTS = [
    'ggml/src/ggml-cpu/expert-r4.c',
    'ggml/src/ggml-cpu/expert-r4.h',
    'ggml/src/ggml-cpu/quants.h',
    'ggml/src/ggml-cpu/ggml-cpu-impl.h',
    'ggml/src/ggml-cpu/simd-mappings.h',
    'ggml/src/ggml-quants.h',
    'ggml/src/ggml-common.h',
    'ggml/src/ggml-impl.h',
    'ggml/include/ggml.h',
    'ggml/include/ggml-cpu.h',
    'ggml/include/ggml-backend.h',
    'ggml/include/ggml-alloc.h',
    'ggml/include/gguf.h',
]
# include search order for quoted includes that are not relative to the including file
INCLUDE_DIRS = ['ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']
# Preamble of generated/arm_quants_subset.c: the include block of ggml/src/ggml-cpu/arch/arm/quants.c
# (lines 1-16 of that file, minus the upstream-only arch-fallback include) so the extracted items see the same
# declarations and macros as upstream.
PREAMBLE = """#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
"""


def show(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{COMMIT}:{path}'],
                          check=True, capture_output=True, text=True).stdout


def exists(path):
    return subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e', f'{COMMIT}:{path}'],
                          capture_output=True).returncode == 0


def resolve_include(owner, inc):
    cands = [os.path.normpath(os.path.join(os.path.dirname(owner), inc))]
    cands += [os.path.normpath(os.path.join(d, inc)) for d in INCLUDE_DIRS]
    for c in cands:
        if exists(c):
            return c
    raise SystemExit(f'unresolved #include "{inc}" in {owner}')


def closure(roots):
    seen, order, stack = set(), [], list(roots)
    while stack:
        p = stack.pop(0)
        if p in seen:
            continue
        seen.add(p)
        order.append(p)
        for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', show(p), flags=re.M):
            stack.append(resolve_include(p, inc))
    return order


def find_start(lines, kind, name):
    if kind == 'define':
        pat = re.compile(rf'^#define {re.escape(name)}\b')
    elif kind == 'var':
        pat = re.compile(rf'^static const [\w\s]+\b{re.escape(name)}\s*\[')
    else:
        pat = re.compile(rf'^(?!\s)(?:static\s+)?(?:inline\s+)?[\w:<>,\s\*]*?\b{re.escape(name)}\s*\(')
    hits = [i for i, l in enumerate(lines) if pat.match(l)]
    # a definition, not a declaration: the signature must reach '{' before ';'
    defs = []
    for i in hits:
        j = i
        while j < len(lines) and '{' not in lines[j] and ';' not in lines[j]:
            j += 1
        if (j < len(lines) and '{' in lines[j]) or kind == 'define':
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
    return '\n'.join(lines[start:end + 1]), start + 1


def emit(items, parts, prov, out_name):
    cache = {}
    for path, kind, name in items:
        if path not in cache:
            cache[path] = show(path)
        body, line = extract(cache[path], kind, name)
        parts.append(f'// ---- {kind} {name}  ({path}:{line} @ {COMMIT[:10]})\n{body}\n')
        prov.append((out_name, path, line, kind, name))


def main():
    OUT.mkdir(exist_ok=True)
    prov = []  # (output, upstream file, line, kind, name)
    upstream = []  # (output relative to generated/, upstream path)
    for path in closure(VERBATIM_ROOTS):
        assert path.startswith('ggml/')
        rel = Path(path).relative_to('ggml')
        dst = OUT / 'upstream' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(show(path))
        upstream.append((f'upstream/{rel}', path))
        prov.append((f'upstream/{rel}', path, 1, 'file', '(verbatim)'))

    banner = (f'// GENERATED by vendor/ggml-flash/extract.py from llama.cpp fork commit {COMMIT}. Do not edit.\n'
              '// MIT licence, (c) the ggml authors; see vendor/ggml-flash/LICENSE.\n')
    parts = [banner, PREAMBLE]
    emit(BASE_ITEMS, parts, prov, ARM_QUANTS)
    emit(ARM_ITEMS, parts, prov, ARM_QUANTS)
    emit(GENERIC_ITEMS, parts, prov, ARM_QUANTS)
    (OUT / ARM_QUANTS).write_text('\n'.join(parts))

    lic = show('LICENSE')
    (HERE / 'LICENSE').write_text(lic)

    rows = '\n'.join(f'| `generated/{o}` | `{p}` | {l} | {k} | `{n}` |' for o, p, l, k, n in prov)
    sums = '\n'.join(f'- `{f.relative_to(OUT)}` sha256 `{hashlib.sha256(f.read_bytes()).hexdigest()}`'
                     for f in sorted(OUT.rglob('*')) if f.is_file())
    (HERE / 'PROVENANCE.md').write_text(f"""# Vendored ggml-flash kernels: provenance

- Upstream: llama.cpp fork (ggml CPU backend, R4 expert kernels), MIT licence (see `LICENSE`, verbatim below).
- Local clone: `<BigMoeOnEdge checkout>/third_party/llama.cpp`, branch `claude/flash-r4`.
- Pinned commit: `{COMMIT}` (short `{COMMIT[:7]}`).
- Regenerate: `python3 vendor/ggml-flash/extract.py <BigMoeOnEdge checkout>/third_party/llama.cpp {COMMIT}`.
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
  then includes `generated/{ARM_QUANTS}`; adds `oxf_sizeof_block_*()` and `oxf_block_d_offset()`.
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
{rows}

## Verbatim whole files (sha256)

{sums}

## LICENSE (verbatim from the fork)

```
{lic.rstrip()}
```
""".replace('{ARM_QUANTS}', ARM_QUANTS))
    print(f'wrote {len(prov)} items, {len(upstream)} upstream files')


if __name__ == '__main__':
    main()
