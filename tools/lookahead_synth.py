#!/usr/bin/env python3
"""Synthetic X.lae + X.routes.csv for testing lookahead_train.py / lookahead_replay.py.

Not real model data. The hidden process: a 32-dim AR(1) latent Z_t drives the hidden
state h_t = Z_t @ M; tokens (300-word vocab, 16-dim latent each) drive embd_t = T[tok_t] @ Qe.
Routes at step s are the top-10 per layer of  [Z_{s-1}, T[tok_s]] @ G + noise, so the
experts are a noisy linear function of exactly the predictor's inputs (h_{s-1}, embd_s).
Previous-token and frequency baselines should therefore lose to a trained MLP.

Usage: ~/.cache/local-ai/venv-lookahead/bin/python -I tools/lookahead_synth.py \
           --out-dir DIR --steps 2000 --prefix 64 --seed 1
"""
import argparse
import os
import struct

import numpy as np

D_E, D_H, N_L, N_E, TOP_K = 2560, 10240, 48, 512, 10
REC = np.dtype([("step", "<i4"), ("embd", "<f4", (D_E,)), ("h", "<f4", (D_H,))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000, help="records (decode steps)")
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.35)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    P, N = a.prefix, a.steps
    T = P + N  # steps 0..T-1 exist in the routes; records cover P..T-1

    LAT, VOC, TL = 32, 300, 16
    M = (rng.standard_normal((LAT, D_H)) / np.sqrt(LAT)).astype(np.float32)
    Qe = (rng.standard_normal((TL, D_E)) / np.sqrt(TL)).astype(np.float32)
    TT = rng.standard_normal((VOC, TL)).astype(np.float32)
    G = (rng.standard_normal((LAT + TL, N_L * N_E)) / np.sqrt(LAT + TL)).astype(np.float32)

    Z = np.zeros((T, LAT), np.float32)
    z = rng.standard_normal(LAT).astype(np.float32)
    for t in range(T):
        z = 0.9 * z + np.sqrt(1 - 0.81) * rng.standard_normal(LAT).astype(np.float32)
        Z[t] = z
    tok = rng.integers(0, VOC, size=T)

    os.makedirs(a.out_dir, exist_ok=True)
    # routes: step 0 has no previous latent (zeros)
    lines = ["turn,phase,step,layer,slot,expert"]
    for s in range(T):
        zprev = Z[s - 1] if s > 0 else np.zeros(LAT, np.float32)
        x = np.concatenate([zprev, TT[tok[s]]])
        sc = (x @ G).reshape(N_L, N_E) + a.noise * rng.standard_normal((N_L, N_E)).astype(np.float32)
        top = np.argsort(-sc, axis=1)[:, :TOP_K]
        ph = 0 if s < P else 1
        for l in range(N_L):
            for k in range(TOP_K):
                lines.append(f"0,{ph},{s},{l},{k},{int(top[l, k])}")
    with open(os.path.join(a.out_dir, "X.routes.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    recs = np.empty(N, REC)
    for i in range(N):
        t = P + i
        recs[i]["step"] = t
        recs[i]["h"] = Z[t] @ M
        recs[i]["embd"] = TT[tok[t]] @ Qe
    with open(os.path.join(a.out_dir, "X.lae"), "wb") as f:
        f.write(b"LAE1" + struct.pack("<II", D_E, D_H))
        recs.tofile(f)
    print(f"wrote {a.out_dir}/X.lae ({N} records, prefix {P}) and X.routes.csv ({len(lines) - 1} rows)")


if __name__ == "__main__":
    main()
