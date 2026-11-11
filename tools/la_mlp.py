#!/usr/bin/env python3
"""Residual-based expert predictors for lookahead, cross-fitted (numpy only, no model).

Input: X.lae + X.routes.csv (read by lookahead_train.load_dataset). Decode token j (0..D-1,
D = number of LAE records, step = prefix + j) has features x_j = concat(rmsnorm(h_{j-1}),
rmsnorm(embd_j)) (12800 dims) and target = the 48 x 10 routes of token j. Token 0 has no
features: its score is 0 everywhere and it is excluded from every metric.

Folds: K = 5 contiguous blocks, fold[j] = j * 5 // D. For fold k the training part is the set of
tokens j >= 1 not in fold k. Its LAST 10% (by step order) is an inner validation set, used only for
early stopping (MLP) and for choosing the ridge penalty. Fold k is never used for either.
Predictions for the tokens of fold k come from the model fitted on the training part of fold k.

Predictors (--sets, each writes one SCORE file <out-prefix><name>.npz):
  mlp-w64, mlp-w128   lookahead_train.MLP (12800 -> W -> SiLU -> 48 x 512 logits), per-layer softmax
                      CE against the 10 routed experts, Adam lr 1e-3, batch 128, <= 30 epochs,
                      patience 3 on the inner-validation CE. score = min(1, 10 * softmax prob).
  ridge-r128, ridge-r256
                      x centred on the training part, projected onto the top-r right singular
                      vectors of the training matrix (r coords standardised, n_train < r -> all
                      components), closed-form ridge (with unpenalised intercept) to the 24576 0/1
                      targets. lambda = c * n_train / 1000, c in {1,10,100,1000}, chosen by the
                      inner-validation MSE (one lambda for all outputs), then refitted on the whole
                      training part. score = clip(prediction, 0, 1).

SCORE file: score float16 [D,48,512], fold int16 [D], lead = 0, name, prefix, D.

Printed (out-of-fold over tokens j >= 1): mean per-layer top-10 hit rate, per-fold values, train time,
against two baselines computed the same way (training-part frequency, previous-token). With --sim
SIM.npz (its res/dem; D and prefix must match): miss-aware numbers among NON-RESIDENT experts ranked
by score: fraction demanded in the top 1/3/10 and recall of all misses at top 3/10.

Usage (wrap in the memory scope):
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 \\
      $PY -I tools/la_mlp.py --lae X.lae --routes X.routes.csv --out-prefix results/la/score-NAME- \\
      [--sim SIM.npz] [--sets mlp-w64,mlp-w128,ridge-r128,ridge-r256] [--seed 0]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import os
import sys
import time

import numpy as np

# `python -I` drops the script directory from sys.path, so import the sibling module explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookahead_train import (ALPHA, MLP, N_E, N_L, TOP_K, Adam, dense_q, held_ce,  # noqa: E402
                             load_dataset, sample_inputs, sample_targets)

N_FOLD = 5
INNER_FRAC = 0.1
MLP_EPOCHS, MLP_PATIENCE, MLP_LR, MLP_BATCH = 30, 3, 1e-3, 128
RIDGE_RANKS = {"ridge-r128": 128, "ridge-r256": 256}
LAMBDA_MULT = (1, 10, 100, 1000)
D_OUT = N_L * N_E
ALL_SETS = ("mlp-w64", "mlp-w128", "ridge-r128", "ridge-r256")
CHUNK = 256
MISS_KS = (1, 3, 10)


def fold_of(D):
    return (np.arange(D) * N_FOLD) // D


def fold_split(D, fold, k):
    """Sample indices of the training part, and the test tokens of fold k (token j = sample j - 1 + 1)."""
    j = np.arange(1, D)
    tr_tok = j[fold[1:] != k]
    te_tok = j[fold[1:] == k]
    return tr_tok - 1, te_tok


# ---------------------------------------------------------------- MLP


def mlp_fit(ds, tr_s, width, seed, log):
    if len(tr_s) < 10:
        raise ValueError(f"training part too small ({len(tr_s)} samples)")
    n_val = max(1, int(round(INNER_FRAC * len(tr_s))))
    fit_s, val_s = tr_s[:-n_val], tr_s[-n_val:]   # LAST 10% by step order = inner validation
    rng = np.random.default_rng(seed + 1)
    model = MLP(width, seed)
    opt = Adam(model.p, MLP_LR)
    best, best_p, best_ep, bad, ep = np.inf, None, -1, 0, -1
    for ep in range(MLP_EPOCHS):
        perm = rng.permutation(fit_s)
        for a in range(0, len(perm), MLP_BATCH):
            b = np.sort(perm[a:a + MLP_BATCH])
            _, g = model.loss_grad(sample_inputs(ds, b), dense_q(sample_targets(ds, b)))
            opt.step(model.p, g)
        vl = held_ce(model, ds, val_s)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            best_p = {k: v.copy() for k, v in model.p.items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    model.p = best_p
    log(f"      epochs_run={ep + 1} best_epoch={best_ep} inner_val_ce={best:.5f} "
        f"n_fit={len(fit_s)} n_inner_val={len(val_s)}")
    return model


def mlp_predict(ds, model, te_tok, te_s, score):
    for a in range(0, len(te_s), CHUNK):
        s = te_s[a:a + CHUNK]
        p = np.exp(model.logprobs(sample_inputs(ds, s)))
        score[te_tok[a:a + CHUNK]] = np.minimum(1.0, 10.0 * p)


# ---------------------------------------------------------------- ridge


def y_dense(ds, sidx):
    """Multi-hot float64 targets (B, 48*512) for the given sample indices."""
    Y = sample_targets(ds, sidx)
    out = np.zeros((len(sidx), D_OUT), np.float64)
    cols = (np.arange(N_L)[None, :, None] * N_E + Y).reshape(len(sidx), -1)
    np.put_along_axis(out, cols, 1.0, axis=1)
    return out


def ridge_basis(ds, tr_s, rmax):
    """Centred PCA basis of the training inputs (top rmax components only are kept)."""
    X = sample_inputs(ds, tr_s)
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    X -= mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    del X
    r = min(rmax, S.size)
    Zraw = U[:, :r].astype(np.float64) * S[:r]          # training coords (= X Vt^T)
    std = np.maximum(Zraw.std(axis=0), 1e-12)
    return {"mu": mu, "Vt": Vt[:r].copy(), "Z": Zraw, "std": std}


def ridge_fit(ds, tr_s, basis, rr, log):
    n = len(tr_s)
    n_in = n - max(1, int(round(INNER_FRAC * n)))   # first 90% fit, last 10% inner validation
    Z = basis["Z"][:, :rr] / basis["std"][:rr]
    A = np.hstack([Z, np.ones((n, 1))])
    P = np.eye(rr + 1)
    P[-1, -1] = 0.0                                  # intercept unpenalised
    G_in = np.zeros((rr + 1, rr + 1)); B_in = np.zeros((rr + 1, D_OUT))
    G_all = np.zeros((rr + 1, rr + 1)); B_all = np.zeros((rr + 1, D_OUT))
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        Ac = A[a:b]
        Yc = y_dense(ds, tr_s[a:b])
        G_all += Ac.T @ Ac
        B_all += Ac.T @ Yc
        ia = min(b, n_in) - a
        if ia > 0:
            G_in += Ac[:ia].T @ Ac[:ia]
            B_in += Ac[:ia].T @ Yc[:ia]
    n_val = n - n_in
    table = []
    for c in LAMBDA_MULT:
        lam = c * n / 1000.0
        W = np.linalg.solve(G_in + lam * P, B_in)
        sse = 0.0
        for a in range(n_in, n, CHUNK):
            b = min(n, a + CHUNK)
            d = A[a:b] @ W - y_dense(ds, tr_s[a:b])
            sse += float(np.sum(d * d))
        table.append((c, lam, sse / (n_val * D_OUT)))
    c_best, lam_best, _ = min(table, key=lambda t: t[2])
    log("      lambda inner-val MSE: " + "  ".join(f"c={c}:{m:.6f}" for c, _, m in table)
        + f"  -> c={c_best}")
    return np.linalg.solve(G_all + lam_best * P, B_all)


def ridge_predict(ds, basis, rr, W, te_tok, te_s, score):
    for a in range(0, len(te_s), CHUNK):
        s = te_s[a:a + CHUNK]
        x = sample_inputs(ds, s) - basis["mu"]
        z = (x @ basis["Vt"][:rr].T).astype(np.float64) / basis["std"][:rr]
        A = np.hstack([z, np.ones((len(s), 1))])
        pred = np.clip(A @ W, 0.0, 1.0)
        score[te_tok[a:a + CHUNK]] = pred.reshape(len(s), N_L, N_E)


# ---------------------------------------------------------------- baselines


def freq_probs(ds, tr_s):
    Y = sample_targets(ds, tr_s)
    flat = (np.arange(N_L)[None, :, None] * N_E + Y).reshape(-1)
    cnt = np.bincount(flat, minlength=D_OUT).reshape(N_L, N_E).astype(np.float64)
    return ((cnt + ALPHA) / (cnt.sum(axis=1, keepdims=True) + ALPHA * N_E)).astype(np.float32)


# ---------------------------------------------------------------- evaluation


def token_hits(score, Ytrue, D):
    """Per-token mean top-10 hit rate over layers and slots (NaN for token 0)."""
    h = np.full(D, np.nan)
    for a in range(1, D, CHUNK):
        b = min(D, a + CHUNK)
        s = score[a:b].astype(np.float16).astype(np.float32)
        top = np.argpartition(-s, TOP_K - 1, axis=2)[:, :, :TOP_K]
        mask = np.zeros(s.shape, bool)
        np.put_along_axis(mask, top, True, axis=2)
        h[a:b] = np.take_along_axis(mask, Ytrue[a:b], axis=2).mean(axis=(1, 2))
    return h


def sim_counts(score, res, miss, D):
    """Non-resident candidates ranked by score: demanded-count in top k and miss recall (tokens j >= 1)."""
    valid_n = {k: 0 for k in MISS_KS}
    hit_n = {k: 0 for k in MISS_KS}
    miss_tot = 0
    rank = np.arange(TOP_K)
    for a in range(1, D, CHUNK):
        b = min(D, a + CHUNK)
        s = score[a:b].astype(np.float16).astype(np.float32)
        r, m = res[a:b], miss[a:b]
        s = np.where(r, np.float32(-np.inf), s)
        nnr = (~r).sum(axis=2)
        part = np.argpartition(-s, TOP_K - 1, axis=2)[:, :, :TOP_K]
        vals = np.take_along_axis(s, part, axis=2)
        order = np.argsort(-vals, axis=2, kind="stable")
        top = np.take_along_axis(part, order, axis=2)
        valid = rank[None, None, :] < np.minimum(nnr, TOP_K)[:, :, None]
        hitm = np.take_along_axis(m, top, axis=2) & valid
        for k in MISS_KS:
            valid_n[k] += int(valid[:, :, :k].sum())
            hit_n[k] += int(hitm[:, :, :k].sum())
        miss_tot += int(m.sum())
    return valid_n, hit_n, miss_tot


def fmt_miss(valid_n, hit_n, miss_tot):
    prec = " ".join(f"top{k}={hit_n[k] / max(1, valid_n[k]):.4f}" for k in MISS_KS)
    rec = " ".join(f"@{k}={hit_n[k] / max(1, miss_tot):.4f}" for k in (3, 10))
    return f"demanded-fraction among non-resident {prec} | miss recall {rec} (misses={miss_tot})"


def report(label, h, fold, D, extra=""):
    overall = float(np.nanmean(h))
    per = [float(np.mean(h[1:][fold[1:] == k])) for k in range(N_FOLD)]
    print(f"[{label}] top10_hit={overall:.4f} folds=[" + " ".join(f"{v:.4f}" for v in per) + f"]{extra}")
    return overall, per


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lae", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--out-prefix", required=True, help="writes <out-prefix><name>.npz per predictor")
    ap.add_argument("--sim", help="SIM .npz (res, dem, prefix, D) for miss-aware numbers")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    bad = [s for s in sets if s not in ALL_SETS]
    if bad:
        ap.error(f"unknown sets {bad}; choose from {ALL_SETS}")

    ds = load_dataset(args.lae, args.routes)
    prefix, D = ds.prefix, ds.T              # D = number of LAE records = decode tokens used
    if ds.S - prefix != D:
        print(f"[note] routes have {ds.S - prefix} decode steps, using the first D={D} (= LAE records)")
    Ytrue = ds.routes[prefix:prefix + D].astype(np.int64)
    fold = fold_of(D)

    sim = None
    if args.sim:
        z = np.load(args.sim)
        if int(z["D"]) != D or int(z["prefix"]) != prefix:
            raise SystemExit(f"error: --sim has D={int(z['D'])} prefix={int(z['prefix'])}, "
                             f"need D={D} prefix={prefix}")
        res = z["res"].astype(bool)
        dem = z["dem"].astype(bool)
        if res.shape != (D, N_L, N_E) or dem.shape != res.shape:
            raise SystemExit(f"error: --sim res/dem shape {res.shape}, expected {(D, N_L, N_E)}")
        sim = (res, dem & ~res)              # miss = demanded and not resident
        print(f"[sim] {args.sim}: misses (j>=1) = {int(sim[1][1:].sum())}")
    print(f"[setup] D={D} prefix={prefix} folds={[int((fold == k).sum()) for k in range(N_FOLD)]} "
          f"seed={args.seed} sets={sets}")

    def run_eval(label, score):
        h = token_hits(score, Ytrue, D)
        overall, per = report(label, h, fold, D)
        if sim is not None:
            print(f"[{label}] miss-aware: " + fmt_miss(*sim_counts(score, sim[0], sim[1], D)))
        return overall, per

    # baselines, out-of-fold, same protocol
    base_scores = {}
    fr = np.zeros((D, N_L, N_E), np.float32)
    pv = np.zeros((D, N_L, N_E), np.float32)
    prev_rows = np.arange(1, D)
    np.put_along_axis(pv[1:], ds.routes[prefix + prev_rows - 1].astype(np.int64), 1.0, axis=2)
    for k in range(N_FOLD):
        tr_s, te_tok = fold_split(D, fold, k)
        fr[te_tok] = freq_probs(ds, tr_s)
    print("[baseline]")
    base_scores["frequency"] = fr
    base_scores["previous-token"] = pv
    base_res = {}
    for name, sc in base_scores.items():
        base_res[name] = run_eval(f"baseline {name}", sc)
    del fr, pv

    ridge_cache = {}
    summary = []
    for name in sets:
        print(f"[{name}] fitting")
        score = np.zeros((D, N_L, N_E), np.float32)
        t_train = 0.0
        for k in range(N_FOLD):
            tr_s, te_tok = fold_split(D, fold, k)
            te_s = te_tok - 1
            t0 = time.time()
            print(f"  fold {k}: n_train={len(tr_s)} n_test={len(te_tok)}")
            if name.startswith("mlp"):
                width = int(name.split("-w")[1])
                model = mlp_fit(ds, tr_s, width, args.seed + k, print)
                mlp_predict(ds, model, te_tok, te_s, score)
            else:
                rr = RIDGE_RANKS[name]
                if k not in ridge_cache:
                    ridge_cache[k] = ridge_basis(ds, tr_s, max(RIDGE_RANKS.values()))
                basis = ridge_cache[k]
                rr_eff = min(rr, basis["Z"].shape[1])
                W = ridge_fit(ds, tr_s, basis, rr_eff, print)
                ridge_predict(ds, basis, rr_eff, W, te_tok, te_s, score)
                print(f"      r={rr_eff}")
            t_train += time.time() - t0
        overall, per = run_eval(name, score)
        out = f"{args.out_prefix}{name}.npz"
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.savez_compressed(out, score=score.astype(np.float16), fold=fold.astype(np.int16),
                            lead=np.int64(0), name=np.str_(name), prefix=np.int64(prefix),
                            D=np.int64(D))
        print(f"[{name}] train_time={t_train:.1f}s wrote {out}")
        summary.append((name, overall, per, t_train))
        del score

    print("== summary (out-of-fold top-10 hit, j>=1) ==")
    for name, (ov, _) in base_res.items():
        print(f"  baseline {name:15s} {ov:.4f}")
    for name, overall, per, t in summary:
        print(f"  {name:15s} {overall:.4f}  train_time={t:.1f}s")


if __name__ == "__main__":
    main()
