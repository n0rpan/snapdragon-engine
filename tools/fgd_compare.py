#!/usr/bin/env python3
"""Compare two directories of FGD1 tensor dumps (flash-m1 --dump-dir / production --tensor-dump).

FGD1 file: b"FGD1", i32 ggml type, 4 x i64 ne, then the elements (contiguous, ggml type size).
Per tensor: identical / shape or type mismatch / max-abs-diff and first differing index (f32 only).
Files are walked in layer order (suffix -L), so the first divergent node is the first line.

usage: python3 -I tools/fgd_compare.py A_DIR B_DIR [--all]
exit 0 if every tensor present in A is byte-identical in B.
"""
import os
import struct
import sys


def read(p):
    with open(p, 'rb') as f:
        b = f.read()
    if b[:4] != b'FGD1':
        raise ValueError(f'{p}: not FGD1')
    ty, = struct.unpack_from('<i', b, 4)
    ne = struct.unpack_from('<4q', b, 8)
    return ty, ne, b[40:]


def key(n):
    stem = n[:-4] if n.endswith('.bin') else n
    tail = stem.rsplit('-', 1)
    layer = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 99
    return (layer, stem)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show_all = '--all' in sys.argv
    if len(args) != 2:
        print(__doc__)
        return 2
    a, b = args
    names = sorted((n for n in os.listdir(a) if n.endswith('.bin') and n != 'tokens.bin'), key=key)
    same = 0
    bad = 0
    for n in names:
        pb = os.path.join(b, n)
        if not os.path.exists(pb):
            print(f'MISSING {n}')
            bad += 1
            continue
        ta, na, da = read(os.path.join(a, n))
        tb, nb, db = read(pb)
        if (ta, na) != (tb, nb):
            print(f'SHAPE   {n}: {ta} {na} vs {tb} {nb}')
            bad += 1
            continue
        if da == db:
            same += 1
            if show_all:
                print(f'same    {n}')
            continue
        bad += 1
        msg = f'DIFF    {n}'
        if ta == 0:  # f32
            k = len(da) // 4
            xa = struct.unpack(f'<{k}f', da)
            xb = struct.unpack(f'<{k}f', db)
            first = next(i for i in range(k) if struct.pack('<f', xa[i]) != struct.pack('<f', xb[i]))
            mad = max(abs(x - y) for x, y in zip(xa, xb))
            msg += f': max|d| {mad:.3e}, first index {first} ({xa[first]!r} vs {xb[first]!r})'
        print(msg)
    print(f'{same}/{len(names)} identical, {bad} differ/missing')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
