#!/usr/bin/env python3
"""MTP auto-tune (MTPLX-style): per context bucket, time every MTP depth against plain decode on THIS
machine and enable a depth only where it wins. "Off" (depth 0) is a valid result: on a MoE,
speculation can lose. Writes the server's tune table (`oxe-server --mtp-tune FILE`, or
`<model>.mtp-tune.json` next to the model).

Runs `oxe bench` (one process per arm; Codex lane only: it loads the model and needs the inference
lease and a TIMING decode window). Per bucket the arms run forward then reversed (ABBA):
  d0, d1, d2, d3, d3, d2, d1, d0
A depth qualifies only if it beats plain by at least --min-gain on BOTH passes (plain: median ms per
token; MTP: effective ms per produced token); the fastest qualifying depth wins, else 0. All arms of a
bucket must produce the same `generated_hash` (exactness), or the bucket is marked invalid and set 0.

  python3 tools/mtp_tune.py --oxe ./target/release/oxe --model ORN.gguf --mtp-from SIDECAR.gguf \
      --bucket 4096:ornith-p1-code --bucket 12288:ornith-lc8k --bucket 32768:ornith-lc8k*2 \
      --depths 0,1,2,3 --out ORN.mtp-tune.json

Bucket = MAX_CTX:PROMPT. PROMPT is an oracle dir under --oracle (its manifest's prompt tokens), with
`*N` to repeat it N times (longer contexts; repetition is stated in the output). Contexts above the
last bucket use the last bucket's depth.
"""
import argparse, hashlib, json, os, statistics, subprocess, sys, tempfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def prompt_ids(oracle, spec):
    name, rep = (spec.split("*") + ["1"])[:2]
    toks = json.load(open(os.path.join(oracle, name, "manifest.json")))["tokens"]
    return toks * int(rep)


def run_arm(a, ids, depth, tag):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        out = t.name
    cmd = [a.oxe, "bench", "--model", a.model, "--cpu-list", a.cpus, "--ctx", str(a.ctx), "--batched-prompt",
           "--warmup", "16", "--tokens", str(a.tokens), "--prompt-ids", ",".join(map(str, ids)), "--out", out]
    if depth > 0:
        cmd += ["--spec-draft-n-max", str(depth)]
        if a.mtp_from:
            cmd += ["--mtp-from", a.mtp_from]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    if r.returncode != 0:
        raise SystemExit(f"{tag} depth {depth}: oxe bench failed rc {r.returncode}: {r.stderr[-800:]}")
    j = json.load(open(out))
    os.unlink(out)
    mpt = j["ms_per_token"]
    ms = mpt["median"] if depth == 0 else mpt.get("effective", mpt.get("median"))
    print(f"{tag} depth {depth}: {ms:.3f} ms/token  hash {j.get('generated_hash')}", flush=True)
    return {"depth": depth, "ms": ms, "hash": j.get("generated_hash"), "tokens_per_verify": (j.get("mtp") or {}).get("tokens_per_verify")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp-from", default=None, help="sidecar MTP head (else the model's own)")
    ap.add_argument("--oracle", default=os.path.expanduser("~/.cache/local-ai/snapdragon-engine/oracle"))
    ap.add_argument("--bucket", action="append", required=True, help="MAX_CTX:PROMPT (oracle dir, optional *N)")
    ap.add_argument("--depths", default="0,1,2,3")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--cpus", default="0-7")
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    if 0 not in depths:
        depths = [0] + depths
    result = {"tool": "mtp_tune", "model": os.path.basename(a.model), "mtp_from": a.mtp_from and os.path.basename(a.mtp_from),
              "oxe_sha256": sha256(a.oxe), "min_gain": a.min_gain, "tokens": a.tokens, "buckets": [], "raw": []}
    for b in sorted(a.bucket, key=lambda s: int(s.split(":")[0])):
        max_ctx, spec = b.split(":", 1)
        ids = prompt_ids(a.oracle, spec)
        order = depths + depths[::-1]
        runs = [run_arm(a, ids, d, f"[{max_ctx}] pass {1 if i < len(depths) else 2}") for i, d in enumerate(order)]
        passes = [runs[: len(depths)], runs[len(depths):][::-1]]
        hashes = {r["hash"] for r in runs}
        best, valid = 0, len(hashes) == 1
        if valid:
            plain = [p[0]["ms"] for p in passes]  # depth 0 is first in both (pass 2 reversed back)
            cands = []
            for k, d in enumerate(depths):
                if d == 0:
                    continue
                ms = [p[k]["ms"] for p in passes]
                if all(m <= pl * (1 - a.min_gain) for m, pl in zip(ms, plain)):
                    cands.append((statistics.mean(ms), d))
            if cands:
                best = min(cands)[1]
        result["buckets"].append({"max_ctx": int(max_ctx), "depth": best, "prompt": spec, "prompt_tokens": len(ids), "exact": valid})
        result["raw"].append({"max_ctx": int(max_ctx), "runs": runs})
        print(f"[{max_ctx}] -> depth {best}{'' if valid else ' (hash mismatch: forced 0)'}", flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {a.out}: {[(b['max_ctx'], b['depth']) for b in result['buckets']]}")


if __name__ == "__main__":
    main()
