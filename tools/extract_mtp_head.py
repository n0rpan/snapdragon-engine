#!/usr/bin/env python3
"""Extract a model's MTP block (blk.{N}.*, incl. the nextn.* tensors) into a small sidecar GGUF,
byte-for-byte (no requantisation), for use as another model's draft head (WP-D5: Qwen3.6's head on
Ornith's trunk: `oxe ... --mtp-from SIDECAR`, server `--mtp-head SIDECAR`).

  ~/.cache/local-ai/venvs/d3a/bin/python -I tools/extract_mtp_head.py \
      --from Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf --out Ornith-1.5-35B-A3B.qwen36-mtp-head.gguf [--target ORNITH.gguf]

With --target, the shapes of every extracted tensor are checked against the target's own blk.{N}.*
(same names must exist with the same shapes), and the target's block_count must equal N.
Metadata written: general.architecture, <arch>.block_count, <arch>.nextn_predict_layers, plus
oxe.mtp_head.source (file name) and oxe.mtp_head.source_sha256 (provenance). The output is
verified by re-reading it and comparing every tensor's bytes with the source.
"""
import argparse, hashlib, os, sys

import numpy as np
import gguf


def field_value(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(v).decode()
    return v.tolist()[0] if hasattr(v, "tolist") else v


def sha256(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--draft-vocab", default=None, help="token-frequency ranking (one id per line): store the first --draft-k ids as oxe.draft_vocab.ids (restricted draft head, WP-D3b)")
    ap.add_argument("--draft-k", type=int, default=32768)
    a = ap.parse_args()

    r = gguf.GGUFReader(a.src, "r")
    arch = field_value(r, "general.architecture")
    n_layer_total = int(field_value(r, f"{arch}.block_count"))
    n_nextn = int(field_value(r, f"{arch}.nextn_predict_layers") or 0)
    if n_nextn != 1:
        sys.exit(f"{a.src}: nextn_predict_layers = {n_nextn}, expected 1")
    il = n_layer_total - n_nextn  # the MTP block index (= the trunk's n_layer)
    pre = f"blk.{il}."
    ts = [t for t in r.tensors if t.name.startswith(pre)]
    if not ts:
        sys.exit(f"{a.src}: no {pre}* tensors")

    if a.target:
        rt = gguf.GGUFReader(a.target, "r")
        if field_value(rt, "general.architecture") != arch or int(field_value(rt, f"{arch}.block_count")) != n_layer_total:
            sys.exit("target architecture / block_count differ")
        tt = {t.name: t for t in rt.tensors}
        for t in ts:
            if t.name not in tt:
                sys.exit(f"target lacks {t.name}")
            if list(tt[t.name].shape) != list(t.shape):
                sys.exit(f"{t.name}: shape {list(t.shape)} vs target {list(tt[t.name].shape)}")

    src_sha = sha256(a.src)
    w = gguf.GGUFWriter(a.out, arch)
    w.add_uint32(f"{arch}.block_count", n_layer_total)
    w.add_uint32(f"{arch}.nextn_predict_layers", n_nextn)
    w.add_string("oxe.mtp_head.source", os.path.basename(a.src))
    w.add_string("oxe.mtp_head.source_sha256", src_sha)
    for t in ts:
        # byte-shaped (outer dims..., bytes per row) uint8 view; the writer converts the last dim back
        # to elements for raw_dtype (works for F32/BF16 and block-quantised types alike)
        d = np.ascontiguousarray(t.data)
        raw = d.view(np.uint8).reshape(d.shape[:-1] + (-1,))
        w.add_tensor(t.name, raw, raw_dtype=t.tensor_type)
    if a.draft_vocab:
        with open(a.draft_vocab) as f:
            rk = [int(l.split()[0]) for l in f if l.strip()][: a.draft_k]
        w.add_string("oxe.draft_vocab.source", os.path.basename(a.draft_vocab))
        w.add_tensor("oxe.draft_vocab.ids", np.asarray(sorted(set(rk)), dtype=np.int32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # verify: same names, types, shapes, bytes
    o = gguf.GGUFReader(a.out, "r")
    ot = {t.name: t for t in o.tensors}
    nbytes = 0
    for t in ts:
        u = ot.get(t.name)
        if u is None or u.tensor_type != t.tensor_type or list(u.shape) != list(t.shape):
            sys.exit(f"verify: {t.name} differs in name/type/shape")
        if not np.array_equal(np.asarray(u.data).view(np.uint8).reshape(-1), np.asarray(t.data).view(np.uint8).reshape(-1)):
            sys.exit(f"verify: {t.name} bytes differ")
        nbytes += int(t.n_bytes)
    dv = ot.get("oxe.draft_vocab.ids")
    extra = f"; draft vocabulary {int(dv.n_elements)} ids" if dv is not None else ""
    print(f"wrote {a.out}: {len(ts)} tensors of {pre}*, {nbytes / 1e6:.1f} MB, source sha256 {src_sha[:16]}..; verified byte-identical{extra}")


if __name__ == "__main__":
    main()
