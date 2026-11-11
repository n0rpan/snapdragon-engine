#!/usr/bin/env python3
"""Learned expert lookahead: offline trainer (numpy only; no model, GPU, NPU or network).

Interpreter: ~/.cache/local-ai/venv-lookahead/bin/python (numpy 2.x). Stdlib + numpy only.

Data
----
X.lae           b"LAE1", u32 embd_dim (2560), u32 h_dim (10240), then records of
                i32 step, f32 embd[2560], f32 h[10240]. Consecutive steps from `prefix`.
X.routes.csv    turn,phase,step,layer,slot,expert. phase 0 = prefill (steps 0..prefix-1,
                may be sparse: a (step, layer) group can be missing), phase 1 = decode
                (every (step, layer) has 10 rows, distinct experts, 0..511).

Sample for step s (decode, s >= prefix+1):
    x_s = concat(rmsnorm(h_{s-1}), rmsnorm(embd_s))     (12800 dims, eps 1e-6, no weight)
    y_s = routes of step s: 48 layers x 10 experts       (multi-hot, 10 per layer)

Split: samples are ordered by step. The LAST 20% (at least 1) are held out and never
used for gradient steps. Warm-up and replay use the same split (see lookahead_replay.py).

Model: x -> Linear(12800, W) -> SiLU -> Linear(W, 24576) -> 48 x 512 logits.
Loss: per layer, softmax over 512 experts against the target normalised to sum 1
(0.1 on each of the 10 experts); mean over layers and samples (48 independent 512-way CEs).
Early stopping on the held-out split (as specified); the reported held-out numbers are
therefore slightly optimistic (one selected epoch count). Use --patience to limit that.

Usage
-----
  ~/.cache/local-ai/venv-lookahead/bin/python -I tools/lookahead_train.py \
      --lae X.lae --routes X.routes.csv --width 128 --out lookahead-w128.npz
"""
import argparse
import array
import csv
import math
import os
import struct
import time

import numpy as np

D_E, D_H = 2560, 10240
D_IN = D_E + D_H
N_L, N_E, TOP_K = 48, 512, 10
D_OUT = N_L * N_E
LAE_MAGIC = b"LAE1"
LAE_HDR = 12
REC = np.dtype([("step", "<i4"), ("embd", "<f4", (D_E,)), ("h", "<f4", (D_H,))])
REC_SIZE = REC.itemsize  # 51204, packed
RMS_EPS = 1e-6
ALPHA = 0.5  # additive smoothing for the frequency / previous-token baselines
HELD_FRAC = 0.2
ROUTE_HDR = ["turn", "phase", "step", "layer", "slot", "expert"]


def open_lae(path):
    with open(path, "rb") as f:
        hdr = f.read(LAE_HDR)
    if len(hdr) != LAE_HDR or hdr[:4] != LAE_MAGIC:
        raise ValueError(f"{path}: not an LAE1 file")
    embd_dim, h_dim = struct.unpack("<II", hdr[4:12])
    if (embd_dim, h_dim) != (D_E, D_H):
        raise ValueError(f"{path}: dims {embd_dim}/{h_dim}, expected {D_E}/{D_H}")
    body = os.path.getsize(path) - LAE_HDR
    if body % REC_SIZE:
        raise ValueError(f"{path}: truncated (body {body} not a multiple of {REC_SIZE})")
    return np.memmap(path, dtype=REC, mode="r", offset=LAE_HDR, shape=(body // REC_SIZE,))


def read_routes(path):
    """Return (routes int16 [S, 48, 10] with -1 for absent slots, prefix, n_steps)."""
    cols = {k: array.array("i") for k in ("phase", "step", "layer", "slot", "expert")}
    with open(path, newline="") as f:
        rd = csv.reader(f)
        if next(rd) != ROUTE_HDR:
            raise ValueError(f"{path}: unexpected header")
        for r in rd:
            cols["phase"].append(int(r[1]))
            cols["step"].append(int(r[2]))
            cols["layer"].append(int(r[3]))
            cols["slot"].append(int(r[4]))
            cols["expert"].append(int(r[5]))
    ph = np.frombuffer(cols["phase"], dtype=np.int32)
    st = np.frombuffer(cols["step"], dtype=np.int32)
    la = np.frombuffer(cols["layer"], dtype=np.int32)
    sl = np.frombuffer(cols["slot"], dtype=np.int32)
    ex = np.frombuffer(cols["expert"], dtype=np.int32)
    if st.size == 0:
        raise ValueError(f"{path}: no rows")
    if la.min() < 0 or la.max() >= N_L or sl.min() < 0 or sl.max() >= TOP_K:
        raise ValueError(f"{path}: layer/slot out of range")
    if ex.min() < 0 or ex.max() >= N_E:
        raise ValueError(f"{path}: expert out of range")
    S = int(st.max()) + 1
    routes = np.full((S, N_L, TOP_K), -1, dtype=np.int16)
    flat = (st.astype(np.int64) * N_L + la) * TOP_K + sl
    if np.unique(flat).size != flat.size:
        raise ValueError(f"{path}: duplicate (step, layer, slot) rows")
    routes.reshape(-1)[flat] = ex.astype(np.int16)
    phase_of = np.full(S, -1, dtype=np.int32)
    phase_of[st] = ph
    if (phase_of < 0).any():
        raise ValueError(f"{path}: steps without rows")
    n_pre = int(np.sum(phase_of == 0))
    if not np.array_equal(phase_of[:n_pre], np.zeros(n_pre, np.int32)):
        raise ValueError(f"{path}: prefill steps must be 0..prefix-1")
    if not np.array_equal(phase_of[n_pre:], np.ones(S - n_pre, np.int32)):
        raise ValueError(f"{path}: decode steps must be contiguous after prefill")
    dec = routes[n_pre:]
    if (dec < 0).any():
        raise ValueError(f"{path}: decode step with a missing slot")
    srt = np.sort(dec, axis=2)
    if (srt[:, :, 1:] == srt[:, :, :-1]).any():
        raise ValueError(f"{path}: decode step with a repeated expert in one layer")
    return routes, n_pre, S


def rmsnorm_rows(a):
    ms = np.mean(a * a, axis=1, keepdims=True, dtype=np.float32)
    return (a / np.sqrt(ms + np.float32(RMS_EPS))).astype(np.float32)


class Dataset:
    pass


def load_dataset(lae_path, routes_path, log=print):
    t0 = time.time()
    recs = open_lae(lae_path)
    T = recs.shape[0]
    routes, prefix, S = read_routes(routes_path)
    rec_steps = np.asarray(recs["step"]).astype(np.int64)
    if T < 3:
        raise ValueError("need at least 3 records")
    if rec_steps[0] != prefix or not np.array_equal(rec_steps, np.arange(prefix, prefix + T)):
        raise ValueError(f"records must be consecutive steps from prefix={prefix}")
    if S < prefix + T:
        raise ValueError("routes do not cover all record steps")

    hn = np.empty((T, D_H), np.float32)
    en = np.empty((T, D_E), np.float32)
    for a in range(0, T, 512):
        b = min(T, a + 512)
        hn[a:b] = rmsnorm_rows(np.array(recs["h"][a:b], dtype=np.float32))
        en[a:b] = rmsnorm_rows(np.array(recs["embd"][a:b], dtype=np.float32))

    n = T - 1                      # samples: records 1..T-1 (step prefix+1 .. prefix+T-1)
    n_hold = max(1, int(round(HELD_FRAC * n)))
    n_train = n - n_hold
    ds = Dataset()
    ds.prefix, ds.T, ds.S = prefix, T, S
    ds.routes = routes
    ds.hn, ds.en = hn, en
    ds.sample_rec = np.arange(1, T, dtype=np.int64)
    ds.n, ds.n_train, ds.n_hold = n, n_train, n_hold
    ds.held0 = prefix + 1 + n_train  # first held-out step
    ds.prefill_warm = []
    pre = routes[:prefix]
    for l in range(N_L):
        seq = pre[:, l, :].reshape(-1)
        seq = seq[seq >= 0]
        ds.prefill_warm.append(np.array(list(dict.fromkeys(seq.tolist())), dtype=np.int64))
    log(f"[data] records={T} prefix={prefix} decode_steps={S - prefix} samples={n} "
        f"train={n_train} held_out={n_hold} held_steps={ds.held0}..{prefix + T - 1} "
        f"load={time.time() - t0:.1f}s")
    return ds


def sample_inputs(ds, sidx):
    rec = ds.sample_rec[sidx]
    return np.concatenate([ds.hn[rec - 1], ds.en[rec]], axis=1)


def sample_targets(ds, sidx):
    steps = ds.prefix + ds.sample_rec[sidx]
    return ds.routes[steps].astype(np.int64)


def sample_prev_targets(ds, sidx):
    steps = ds.prefix + ds.sample_rec[sidx] - 1
    return ds.routes[steps].astype(np.int64)


def dense_q(Y):
    B = Y.shape[0]
    q = np.zeros((B, N_L, N_E), np.float32)
    q[np.arange(B)[:, None, None], np.arange(N_L)[None, :, None], Y] = 1.0 / TOP_K
    return q


def log_softmax(lg):
    m = lg.max(axis=2, keepdims=True)
    sh = lg - m
    return sh - np.log(np.exp(sh).sum(axis=2, keepdims=True))


def ce_loss(logp, q):
    """Mean over (sample, layer) of the 512-way cross-entropy against q (rows sum to 1)."""
    return float(-(q * logp).sum() / (logp.shape[0] * N_L))


def topk_hit(scores, Y):
    """Fraction of target experts found in the per-layer top-10 of `scores` (B,48,512)."""
    top = np.argpartition(-scores, TOP_K, axis=2)[:, :, :TOP_K]
    mask = np.zeros(scores.shape, bool)
    np.put_along_axis(mask, top, True, axis=2)
    return float(np.take_along_axis(mask, Y, axis=2).mean())


def freq_logprobs_from_train(ds):
    """Per-layer expert frequency in the training split, additively smoothed."""
    idx = np.arange(0, ds.n_train)
    Y = sample_targets(ds, idx)
    flat = (np.arange(N_L)[None, :, None] * N_E + Y).reshape(-1)
    cnt = np.bincount(flat, minlength=D_OUT).reshape(N_L, N_E).astype(np.float64)
    p = (cnt + ALPHA) / (cnt.sum(axis=1, keepdims=True) + ALPHA * N_E)
    return np.log(p).astype(np.float32)


def prev_logprobs(Yprev):
    """Previous-token baseline: step t's experts are the prediction for step t+1."""
    B = Yprev.shape[0]
    ind = np.zeros((B, N_L, N_E), np.float64)
    np.put_along_axis(ind, Yprev, 1.0, axis=2)
    p = (ind + ALPHA) / (TOP_K + ALPHA * N_E)
    return np.log(p).astype(np.float32)


# ---------------------------------------------------------------- MLP


def silu_sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class MLP:
    def __init__(self, width, seed):
        rng = np.random.default_rng(seed)
        self.width = width
        self.p = {
            "W1": (rng.standard_normal((D_IN, width)) / math.sqrt(D_IN)).astype(np.float32),
            "b1": np.zeros(width, np.float32),
            "W2": (rng.standard_normal((width, D_OUT)) / math.sqrt(width) * 0.1).astype(np.float32),
            "b2": np.zeros(D_OUT, np.float32),
        }

    def forward(self, X):
        z1 = X @ self.p["W1"] + self.p["b1"]
        s = silu_sig(z1)
        a = z1 * s
        lg = a @ self.p["W2"] + self.p["b2"]
        return lg.reshape(-1, N_L, N_E), (X, z1, s, a)

    def loss_grad(self, X, q):
        lg, (X_, z1, s, a) = self.forward(X)
        B = X.shape[0]
        logp = log_softmax(lg)
        p = np.exp(logp)
        loss = ce_loss(logp, q)
        dl = ((p - q) / np.float32(B * N_L)).reshape(B, D_OUT)
        g = {"W2": a.T @ dl, "b2": dl.sum(axis=0)}
        da = dl @ self.p["W2"].T
        dz1 = da * (s + z1 * s * (1.0 - s))
        g["W1"] = X.T @ dz1
        g["b1"] = dz1.sum(axis=0)
        return loss, g

    def logprobs(self, X):
        lg, _ = self.forward(X)
        return log_softmax(lg)


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            params[k] -= (self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)).astype(np.float32)


def load_mlp(path):
    z = np.load(path)
    m = MLP.__new__(MLP)
    m.p = {k: z[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2")}
    m.width = int(m.p["W1"].shape[1])
    return m


def mlp_probs_for_samples(model, ds, sidx, batch=256):
    out = np.empty((len(sidx), N_L, N_E), np.float32)
    for a in range(0, len(sidx), batch):
        b = sidx[a:a + batch]
        out[a:a + len(b)] = np.exp(model.logprobs(sample_inputs(ds, b)))
    return out


def train(ds, width, epochs, patience, lr, batch, seed, log=print):
    rng = np.random.default_rng(seed + 1)
    model = MLP(width, seed)
    opt = Adam(model.p, lr)
    held = np.arange(ds.n_train, ds.n)
    best, best_params, best_ep, bad = float("inf"), None, -1, 0
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        perm = rng.permutation(ds.n_train)  # shuffles TRAINING samples only
        run, nb = 0.0, 0
        for a in range(0, len(perm), batch):
            b = np.sort(perm[a:a + batch])
            X = sample_inputs(ds, b)
            q = dense_q(sample_targets(ds, b))
            loss, g = model.loss_grad(X, q)
            opt.step(model.p, g)
            run += loss
            nb += 1
        hl = held_ce(model, ds, held)
        hist.append({"epoch": ep, "train_ce": run / nb, "held_ce": hl})
        log(f"[train] W={width} epoch {ep:3d} train_ce={run / nb:.5f} held_ce={hl:.5f} "
            f"({time.time() - t0:.1f}s)")
        if hl < best - 1e-5:
            best, best_ep, bad = hl, ep, 0
            best_params = {k: v.copy() for k, v in model.p.items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"[train] early stop at epoch {ep} (best epoch {best_ep})")
                break
    model.p = best_params
    return model, best, best_ep, hist


def held_ce(model, ds, held, batch=256):
    tot, n = 0.0, 0
    for a in range(0, len(held), batch):
        b = held[a:a + batch]
        logp = model.logprobs(sample_inputs(ds, b))
        tot += -float((dense_q(sample_targets(ds, b)) * logp).sum())
        n += len(b)
    return tot / (n * N_L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lae", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--width", type=int, choices=[64, 128, 256], required=True)
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(args.lae, args.routes)
    held = np.arange(ds.n_train, ds.n)
    Yh = sample_targets(ds, held)
    qh = dense_q(Yh)

    # baselines on the held-out split (training-split statistics only)
    fl = freq_logprobs_from_train(ds)
    fl_h = np.broadcast_to(fl, (len(held), N_L, N_E))
    pl_h = prev_logprobs(sample_prev_targets(ds, held))
    base = {
        "frequency": (ce_loss(fl_h, qh), topk_hit(np.exp(fl_h), Yh)),
        "previous_token": (ce_loss(pl_h, qh), topk_hit(np.exp(pl_h), Yh)),
    }
    for k, (ce, hit) in base.items():
        print(f"[baseline] {k:15s} held_ce={ce:.5f} top10_hit={hit:.4f}")
    print(f"[baseline] uniform      held_ce={math.log(N_E):.5f} top10_hit={TOP_K / N_E:.4f}")

    t0 = time.time()
    model, best, best_ep, hist = train(ds, args.width, args.epochs, args.patience,
                                       args.lr, args.batch, args.seed)
    lp = model.logprobs(sample_inputs(ds, held))
    hit = topk_hit(np.exp(lp), Yh)
    print(f"[result] MLP W={args.width} best_epoch={best_ep} held_ce={best:.5f} "
          f"top10_hit={hit:.4f} train_time={time.time() - t0:.1f}s")
    beats = best < min(c for c, _ in base.values())
    print(f"[result] MLP beats both baselines on held-out CE: {beats}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, W1=model.p["W1"], b1=model.p["b1"], W2=model.p["W2"], b2=model.p["b2"],
             width=np.int64(args.width), prefix=np.int64(ds.prefix), n_train=np.int64(ds.n_train),
             n_hold=np.int64(ds.n_hold), held0=np.int64(ds.held0), seed=np.int64(args.seed),
             best_epoch=np.int64(best_ep), held_ce=np.float64(best))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
