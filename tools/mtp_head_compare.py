#!/usr/bin/env python3
"""Compare MTP-head and trunk tensors between two GGUF checkpoints (WP-D5a, offline).

Question: is model A's MTP head (blk.40.*) unchanged from model B's (within quantisation
noise) while the trunk moved?

Method
  For every tensor name present in both files with the same shape, both tensors are dequantised
  to float32 in chunks along the outermost dimension (--chunk-mb per side, default 256 MB) and
  streamed into float64 sums. Per tensor:
    qtype A/B, shape, ||A||, ||B||, cos(A,B), rel = ||A-B|| / ||B||.
  3-D tensors (MoE experts; expert index = outermost dim) also get per-expert cos and rel;
  the summary is min/median/max of cos and median/max of rel over the selected experts.
  --expert-sample N restricts 3-D tensors to N evenly spaced experts. Global stats of those
  tensors then cover only the sampled experts.

Quantisation-noise reference (noise_ref)
  rel(B, requant_A(B)): B's dequantised values are requantised to A's quant type, then
  dequantised again. Available only when gguf.quants.quantize supports that type and A's type is
  lossy. In this gguf install that means Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. The K-quants (Q2_K..Q6_K)
  and IQ* types are NOT supported, and neither are the lossless F32/F16/BF16 types as a reference.
  When there is no exact noise_ref, two proxies are reported instead:
  proxies = {Q4_0: rel(B, requant_Q4_0(B)), Q8_0: rel(B, requant_Q8_0(B))}, which bracket the
  typical 4-bit and 8-bit requantisation noise. Their absolute values are not the noise of the
  real K-quant types. Use them as a scale only.

Interpretation rule
  ratio = rel / max(noise_ref, 1e-4)
    ratio <= 1.5  -> "unchanged": the A-vs-B difference is about what requantisation alone gives
    ratio >= 3    -> "trained":   the difference is well above the quantisation noise
    otherwise     -> "borderline"
    no exact noise_ref -> "n/a": judge by rel against the proxies and the same-type rows.
  Caveat: when A and B share the quant type, B already lies on A's grid, so noise_ref is ~0 and
  any real difference reads as "trained". For such pairs the question is whether rel is ~0,
  which means the two checkpoints store identical bytes for that tensor.

Selection (--set)
  mtp           every blk.40.* tensor (MTP block, incl. nextn.*)
  trunk-sample  token_embd.weight, output.weight, output_norm.weight, and every tensor of
                layers 0, 3, 10, 20, 30, 38, 39
  all           every tensor present in both files with the same shape

Output
  Table on stdout (per tensor), group/layer summary, optional --json OUT.
  Groups: embd_head, trunk (blk.N.*, N < 40), mtp (blk.40.*), other.

Usage
  tools/mtp_head_compare.py --a A.gguf --b B.gguf --set mtp [--expert-sample 8] [--json OUT]
  Heavy: dequantises full tensors of the real models. Run only inside an allowed window.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGML_QUANT_SIZES, GGUFReader
from gguf.quants import dequantize, quantize

LOSSLESS = (QT.F32, QT.F16, QT.BF16)
PROXY_TYPES = (QT.Q4_0, QT.Q8_0)
UNCHANGED_RATIO = 1.5
TRAINED_RATIO = 3.0
NOISE_FLOOR = 1e-4
DEFAULT_CHUNK_MB = 256.0
TRUNK_SAMPLE_LAYERS = (0, 3, 10, 20, 30, 38, 39)
MTP_PREFIX = "blk.40."
EMBD_HEAD = ("token_embd.weight", "output.weight", "output_norm.weight")
GROUP_ORDER = ("embd_head", "trunk", "mtp", "other")


# ----------------------------------------------------------------------------- helpers

def supports_quantize(qt):
    """True when gguf.quants.quantize can produce type qt (K-quants and IQ* cannot here)."""
    try:
        quantize(np.zeros((1, 256), np.float32), qt)
        return True
    except NotImplementedError:
        return False


def can_quantize(qt, cols):
    """True when the last (contiguous) dim of length cols is a whole number of qt blocks."""
    return supports_quantize(qt) and cols % GGML_QUANT_SIZES[qt][0] == 0


def nshape(t):
    """Numpy-order shape of a GGUF tensor (GGUFReader keeps ggml ne order)."""
    return tuple(int(d) for d in t.shape[::-1])


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def group_of(name):
    if name.startswith(MTP_PREFIX):
        return "mtp"
    if name in EMBD_HEAD:
        return "embd_head"
    if layer_of(name) is not None:
        return "trunk"
    return "other"


def select_names(names, which):
    """Filter names (already present in both files with equal shape) by the --set group."""
    out = []
    for n in names:
        if which == "all":
            ok = True
        elif which == "mtp":
            ok = n.startswith(MTP_PREFIX)
        elif which == "trunk-sample":
            L = layer_of(n)
            ok = n in EMBD_HEAD or (L is not None and L in TRUNK_SAMPLE_LAYERS)
        else:
            raise ValueError(f"unknown --set {which}")
        if ok:
            out.append(n)
    key = lambda n: (GROUP_ORDER.index(group_of(n)), layer_of(n) or -1, n)  # noqa: E731
    return sorted(out, key=key)


def outer_selection(shape, expert_sample):
    """Indices along the outermost dim to process. 1-D tensors use a single unit."""
    ndim = len(shape)
    if ndim < 2:
        return np.arange(1, dtype=np.int64)
    n0 = shape[0]
    if ndim == 3 and expert_sample:
        k = min(int(expert_sample), n0)
        return np.unique(np.round(np.linspace(0, n0 - 1, k)).astype(np.int64))
    return np.arange(n0, dtype=np.int64)


def dequant_block(t, idx):
    """Dequantise rows idx (outer dim) of tensor t to float32, shape (len(idx), -1)."""
    if len(nshape(t)) < 2:
        raw = t.data
        rows = 1
    else:
        raw = t.data[idx]
        rows = len(idx)
    raw = np.ascontiguousarray(raw)
    out = np.asarray(dequantize(raw, t.tensor_type), dtype=np.float32)
    return out.reshape(rows, -1)


def _med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


# ----------------------------------------------------------------------------- core

def compare_tensor(ta, tb, sel, chunk_mb=DEFAULT_CHUNK_MB):
    """Stream-compare two tensors of equal shape. Returns a dict of statistics (see module doc)."""
    shape = nshape(ta)
    ndim = len(shape)
    cols = int(np.prod(shape[1:])) if ndim >= 2 else int(shape[0])
    m = max(1, int(chunk_mb * 2**20) // (cols * 4))
    qa = ta.tensor_type
    exact_q = qa if (qa not in LOSSLESS and can_quantize(qa, cols)) else None
    targets = [exact_q] if exact_q is not None else [q for q in PROXY_TYPES if can_quantize(q, cols)]

    K = len(sel)
    dot = np.zeros(K)
    na = np.zeros(K)
    nb = np.zeros(K)
    dd = np.zeros(K)
    noise_num = {q: 0.0 for q in targets}
    idx_arg_1d = ndim < 2

    for pos in range(0, K, m):
        idx = sel[pos:pos + m]
        k = len(idx)
        idx_arg = None if idx_arg_1d else idx
        A = dequant_block(ta, idx_arg)
        B = dequant_block(tb, idx_arg)
        if A.shape != B.shape:
            raise ValueError(f"shape mismatch after dequantisation: {A.shape} vs {B.shape}")
        D = A - B
        sl = slice(pos, pos + k)
        dot[sl] = np.sum(A * B, axis=1, dtype=np.float64)
        na[sl] = np.sum(A * A, axis=1, dtype=np.float64)
        nb[sl] = np.sum(B * B, axis=1, dtype=np.float64)
        dd[sl] = np.sum(D * D, axis=1, dtype=np.float64)
        for q in targets:
            R = np.asarray(dequantize(quantize(B, q), q), dtype=np.float32).reshape(B.shape)
            noise_num[q] += float(np.sum((B - R) ** 2, dtype=np.float64))
        del A, B, D

    T_dot, T_na, T_nb, T_dd = dot.sum(), na.sum(), nb.sum(), dd.sum()
    cos = float(T_dot / math.sqrt(T_na * T_nb)) if T_na > 0 and T_nb > 0 else None
    rel = math.sqrt(T_dd) / math.sqrt(T_nb) if T_nb > 0 else None

    noise_ref = None
    proxies = None
    if exact_q is not None:
        if T_nb > 0:
            noise_ref = math.sqrt(noise_num[exact_q]) / math.sqrt(T_nb)
    elif T_nb > 0 and targets:
        proxies = {q.name: math.sqrt(noise_num[q]) / math.sqrt(T_nb) for q in targets}

    if noise_ref is None or rel is None:
        ratio, verdict = None, "n/a"
    else:
        ratio = rel / max(noise_ref, NOISE_FLOOR)
        if ratio <= UNCHANGED_RATIO:
            verdict = "unchanged"
        elif ratio >= TRAINED_RATIO:
            verdict = "trained"
        else:
            verdict = "borderline"

    experts = None
    if ndim == 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            cosk = dot / np.sqrt(na * nb)
            relk = np.sqrt(dd) / np.sqrt(nb)
        cosk = cosk[np.isfinite(cosk)]
        relk = relk[np.isfinite(relk)]
        experts = {
            "n_total": int(shape[0]),
            "n_selected": int(K),
            "selected": [int(i) for i in sel] if K < shape[0] else None,
            "cos_min": float(cosk.min()) if cosk.size else None,
            "cos_median": float(np.median(cosk)) if cosk.size else None,
            "cos_max": float(cosk.max()) if cosk.size else None,
            "rel_median": float(np.median(relk)) if relk.size else None,
            "rel_max": float(relk.max()) if relk.size else None,
        }

    return {
        "shape": list(shape),
        "norm_a": math.sqrt(T_na),
        "norm_b": math.sqrt(T_nb),
        "cos": cos,
        "rel": rel,
        "noise_ref": noise_ref,
        "noise_basis": f"requant_to_A_type:{exact_q.name}" if exact_q is not None else None,
        "noise_proxies": proxies,  # keys: Q4_0 / Q8_0 (absent when the block size does not divide cols)
        "ratio": ratio,
        "verdict": verdict,
        "experts": experts,
    }


def compare_record(name, ta, tb, sel, chunk_mb=DEFAULT_CHUNK_MB):
    r = compare_tensor(ta, tb, sel, chunk_mb)
    r.update({
        "name": name,
        "group": group_of(name),
        "layer": layer_of(name),
        "qtype_a": ta.tensor_type.name,
        "qtype_b": tb.tensor_type.name,
        "same_type": ta.tensor_type == tb.tensor_type,
    })
    return r


def summarise(records):
    by_group = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    out = {}
    for g in GROUP_ORDER:
        rs = by_group.get(g)
        if not rs:
            continue
        same = [r["rel"] for r in rs if r["same_type"]]
        out[g] = {
            "n": len(rs),
            "median_cos": _med([r["cos"] for r in rs]),
            "median_rel": _med([r["rel"] for r in rs]),
            "n_exact_noise": sum(r["noise_ref"] is not None for r in rs),
            "median_noise_ref": _med([r["noise_ref"] for r in rs]),
            "n_same_type": len(same),
            "median_rel_same_type": _med(same),
            "verdicts": dict(Counter(r["verdict"] for r in rs)),
        }
        if g == "trunk":
            layers = {}
            for r in rs:
                layers.setdefault(r["layer"], []).append(r)
            out[g]["by_layer"] = {
                str(L): {
                    "n": len(v),
                    "median_cos": _med([r["cos"] for r in v]),
                    "median_rel": _med([r["rel"] for r in v]),
                }
                for L, v in sorted(layers.items())
            }
    return out


# ----------------------------------------------------------------------------- output

def _f(x, spec):
    return "-" if x is None else format(x, spec)


def print_table(records):
    hdr = (f"{'group':9} {'tensor':34} {'qA':7} {'qB':7} {'shape':24} {'|A|':>10} {'|B|':>10} "
           f"{'cos':>8} {'rel':>8} {'noise':>8} {'ratio':>8} {'verdict':10} "
           f"{'proxy Q4_0/Q8_0':>17} {'expert cos min/med/max':>26}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        shape = "x".join(str(d) for d in r["shape"])
        px = r["noise_proxies"] or {}
        pxs = "/".join(_f(px.get(q.name), ".4f") for q in PROXY_TYPES) if px else "-"
        e = r["experts"]
        ex = "-" if not e else f"{_f(e['cos_min'], '.5f')}/{_f(e['cos_median'], '.5f')}/{_f(e['cos_max'], '.5f')}"
        print(f"{r['group']:9} {r['name'][:34]:34} {r['qtype_a']:7} {r['qtype_b']:7} {shape[:24]:24} "
              f"{_f(r['norm_a'], '10.4g'):>10} {_f(r['norm_b'], '10.4g'):>10} "
              f"{_f(r['cos'], '8.5f'):>8} {_f(r['rel'], '8.4f'):>8} {_f(r['noise_ref'], '8.4f'):>8} "
              f"{_f(r['ratio'], '8.2f'):>8} {r['verdict']:10} {pxs:>17} {ex:>26}")


def print_summary(summary):
    print()
    print(f"{'group':10} {'n':>4} {'med cos':>9} {'med rel':>9} {'exact noise n':>13} "
          f"{'med noise':>10} {'same-type n':>11} {'med rel same':>12}  verdicts")
    for g, s in summary.items():
        print(f"{g:10} {s['n']:>4} {_f(s['median_cos'], '9.5f'):>9} {_f(s['median_rel'], '9.4f'):>9} "
              f"{s['n_exact_noise']:>13} {_f(s['median_noise_ref'], '10.4f'):>10} "
              f"{s['n_same_type']:>11} {_f(s['median_rel_same_type'], '12.4f'):>12}  {s['verdicts']}")
        if "by_layer" in s:
            for L, v in s["by_layer"].items():
                print(f"    layer {L:>3}: n={v['n']:<3} med cos {_f(v['median_cos'], '.5f'):>9} "
                      f"med rel {_f(v['median_rel'], '.4f'):>8}")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="model A gguf (Ornith)")
    ap.add_argument("--b", required=True, help="model B gguf (Qwen3.6 base)")
    ap.add_argument("--set", dest="which", choices=("mtp", "trunk-sample", "all"), default="mtp")
    ap.add_argument("--expert-sample", type=int, default=0,
                    help="N evenly spaced experts for 3-D tensors (0 = all experts)")
    ap.add_argument("--chunk-mb", type=float, default=DEFAULT_CHUNK_MB,
                    help="float32 MB per side per chunk (default %(default)s)")
    ap.add_argument("--json", dest="json_out", help="write full results here")
    args = ap.parse_args(argv)
    if args.expert_sample < 0:
        ap.error("--expert-sample must be >= 0")
    if args.chunk_mb <= 0:
        ap.error("--chunk-mb must be > 0")

    ta = {t.name: t for t in GGUFReader(args.a, "r").tensors}
    tb = {t.name: t for t in GGUFReader(args.b, "r").tensors}
    common = [n for n in ta if n in tb]
    mismatch = [n for n in common if nshape(ta[n]) != nshape(tb[n])]
    same_shape = [n for n in common if n not in mismatch]
    names = select_names(same_shape, args.which)
    if not names:
        sys.exit(f"no tensors selected by --set {args.which}")

    records = []
    for n in names:
        sel = outer_selection(nshape(ta[n]), args.expert_sample)
        records.append(compare_record(n, ta[n], tb[n], sel, args.chunk_mb))

    skipped = {
        "only_in_a": sorted(n for n in ta if n not in tb),
        "only_in_b": sorted(n for n in tb if n not in ta),
        "shape_mismatch": sorted(mismatch),
    }
    summary = summarise(records)

    print(f"A: {args.a}\nB: {args.b}\nset={args.which} expert_sample={args.expert_sample or 'all'} "
          f"chunk_mb={args.chunk_mb} compared={len(records)} "
          f"only_in_a={len(skipped['only_in_a'])} only_in_b={len(skipped['only_in_b'])} "
          f"shape_mismatch={len(skipped['shape_mismatch'])}\n")
    print_table(records)
    print_summary(summary)

    if args.json_out:
        payload = {
            "a": args.a, "b": args.b, "set": args.which,
            "expert_sample": args.expert_sample, "chunk_mb": args.chunk_mb,
            "thresholds": {"unchanged_ratio": UNCHANGED_RATIO, "trained_ratio": TRAINED_RATIO,
                           "noise_floor": NOISE_FLOOR},
            "summary": summary, "tensors": records, "skipped": skipped,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
