#!/usr/bin/env python3
"""GGUF v2/v3 header parser (stdlib only). Reads metadata + tensor info table; never reads tensor data.

usage: python3 -I tools/gguf_header.py FILE [--kv] [--tensors] [--grep PATTERN]
"""
import argparse, json, re, struct, sys

SCALAR = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2), 4: ('I', 4), 5: ('i', 4),
          6: ('f', 4), 7: ('?', 1), 10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
# ggml_type -> (name, elements per block, bytes per block); only types the task needs are sized.
GGML = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 8: ('Q8_0', 32, 34), 12: ('Q4_K', 256, 144),
        14: ('Q6_K', 256, 210), 30: ('BF16', 1, 2)}
OTHER = {2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 13: 'Q5_K',
         15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
         22: 'IQ2_S', 23: 'IQ4_XS', 29: 'IQ1_M', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'}


def tname(t):
    return GGML[t][0] if t in GGML else OTHER.get(t, f'type{t}')


def tbytes(t, ne):
    n = 1
    for d in ne:
        n *= d
    if t not in GGML:
        return None
    _, blk, bsz = GGML[t]
    return n // blk * bsz if n % blk == 0 else None


def parse(path):
    with open(path, 'rb', buffering=0) as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n:
                raise EOFError('truncated header')
            return b

        def u(fmt):
            return struct.unpack('<' + fmt, rd(struct.calcsize(fmt)))[0]

        def s():
            return rd(u('Q')).decode('utf-8', 'replace')

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et = u('I')
                cnt = u('Q')
                if et == 8:
                    return [s() for _ in range(cnt)]
                fmt, sz = SCALAR[et]
                return list(struct.unpack('<%d%s' % (cnt, fmt), rd(cnt * sz))) if cnt else []
            fmt, _ = SCALAR[t]
            return u(fmt)

        if rd(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        ver = u('I')
        n_t, n_kv = u('Q'), u('Q')
        kv = {}
        for _ in range(n_kv):
            k = s()
            kv[k] = val(u('I'))
        tensors = []
        for _ in range(n_t):
            name = s()
            ne = [u('Q') for _ in range(u('I'))]
            t = u('I')
            off = u('Q')
            tensors.append({'name': name, 'ne': ne, 'type': t, 'type_name': tname(t),
                            'offset': off, 'bytes': tbytes(t, ne)})
        align = kv.get('general.alignment', 32)
        data_start = (f.tell() + align - 1) // align * align
    return {'version': ver, 'kv': kv, 'tensors': tensors, 'data_start': data_start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--kv', action='store_true', help='print metadata key/values')
    ap.add_argument('--tensors', action='store_true', help='print tensor table')
    ap.add_argument('--grep', default=None, help='regex filter on tensor and key names')
    a = ap.parse_args()
    h = parse(a.file)
    pat = re.compile(a.grep) if a.grep else None
    keep = lambda n: pat is None or pat.search(n)
    tot = sum(t['bytes'] or 0 for t in h['tensors'])
    unk = sum(1 for t in h['tensors'] if t['bytes'] is None)
    print(f"GGUF v{h['version']}  tensors={len(h['tensors'])}  kv={len(h['kv'])}  "
          f"data_start={h['data_start']}  sized_bytes={tot}  unsized_tensors={unk}")
    if a.kv or not (a.tensors or a.grep):
        for k, v in h['kv'].items():
            if not keep(k):
                continue
            if isinstance(v, list):
                v = f"<array len={len(v)}> " + json.dumps(v[:6]) + (' ...' if len(v) > 6 else '')
            print(f"KV {k} = {v}")
    if a.tensors or a.grep or not a.kv:
        for t in h['tensors']:
            if not keep(t['name']):
                continue
            print(f"T {t['name']} shape={t['ne']} type={t['type_name']}({t['type']}) "
                  f"bytes={t['bytes'] if t['bytes'] is not None else 'unknown'} offset={t['offset']}")


if __name__ == '__main__':
    main()
