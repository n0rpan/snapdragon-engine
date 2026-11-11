#!/usr/bin/env python3
"""Route-only expert-demand predictors, cross-fitted, writing SCORE files (F-LOOKAHEAD).

Question answered: P(expert e is demanded by decode token j at layer L | e is NOT resident).
Residency is taken from the SIM file (res), demand from the SIM file (dem). A cache miss is an
expert that is not resident, so "same experts as the previous token" carries no information
about misses; the features below are chosen for that.

Features per candidate (decode token j, layer L, expert e); t = prefix + j; prefill steps
count as history where present (rows of -1 are skipped):
  causal history (true routes of earlier steps only):
    h1 = log(1 + gap), gap = steps since e was last used at layer L (cap 500, never = 500)
    h2 = sum_{s<t} 0.5 ** ((t-s)/50) * [e used at (s, L)]
    h3 = same with half-life 400
  count tables (cross-fitted: a token of fold k sees tables built from prefill steps plus the
  decode tokens NOT in fold k):
    f0 = log p0[L,e], p0 = (count + 0.5) / (n_L + 256)   (per-access frequency)
    freq = 10 * p0                                       (probability that e is among the 10)
    m1 = log(1e-6 + mean_{e' in route(t-1,L)} P(e at L in step s | e' at L in step s-1))
    w_d = log(1e-6 + mean_{e' in route(t,L-d)} P(e at L | e' at L-d, same step)), d = 1..6
         (for L < d: w_d := f0)
    x_d = log(1e-6 + MAX_{e' in route(t,L-d)} P(e at L | e' at L-d, same step)), d = 1..3
         (for L < d: x_d := f0; used by `intra1-max-pooled` only)
    P = (C[e',e] + a * freq[e]) / (n[e'] + a), a = 5, counts over eligible pairs/steps.
Fold k = j * 5 // D.  Tables for fold k: prefill steps + decode steps with fold != k, plus any
--extra-routes trace (always eligible; it never gets scored).
--train-frac F (0 < F <= 1, default 1): for fold k, the tables AND the regression use only a
deterministic evenly spaced subset (by index) of the eligible decode tokens (fold != k), of size
round(F * #eligible). Prefill steps and --extra-routes stay in the tables for every F. A count
pair (s-1, s) is used only when both steps are in the subset, so for small F few transition
pairs survive (printed as "transition pairs"). Features of a token are still computed from the
full history. With F < 1 the set name gets the suffix -f<round(100 F)> (e.g. -f25, -f50, -f75).

Models: per layer L, L2-penalised logistic regression (intercept unpenalised, LAM = 1.0 on the
standardised non-intercept weights, penalty applied to the SUMMED log-loss, 25 Newton/IRLS
iterations max). Training rows: decode tokens not in fold k x experts with res False; label dem.
Predictions: all 512 experts of the tokens of fold k.
--pooled variants (`intra1-pooled`, `markov-pooled`): one weight vector shared by all layers plus
a per-layer intercept (fitted jointly).

Memory (streamed): no array holds all candidates x features. Features are built one target layer
at a time, as float32 blocks [D, 512, F]; the causal features are built for groups of 8 layers.
Every pass over the layers rebuilds the blocks from the route trace (tables are recomputed per
layer and fold, so nothing of size D x 48 x 512 x F is kept). Pooled fits accumulate the Newton
sums X^T W X, X^T (y - p), and the per-layer intercept sums layer by layer; the five folds are
iterated in lock-step, so one feature pass serves one IRLS iteration of all folds. The
standardisation mean and sd come from one streaming pass (Chan merge of per-block mean and M2).
Non-pooled fits take the per-layer block directly (same arithmetic as before).

Feature sets (lead = minimum within-token layer distance used):
  freq (lead 0): f0            hist (0): f0,h1,h2,h3        markov (0): hist + m1
  intra1 (1): markov + w1,w2,w3     intra2 (2): markov + w2,w3,w4     intra3 (3): markov + w3,w4
  intra1-max-pooled (1, pooled): intra1 + x1,x2,x3     intra1-wide-pooled (1, pooled): markov + w1..w6

Outputs (np.savez_compressed, one file per set): PREFIX<name>.npz with
  score float16 [D,48,512], fold int16 [D], lead int, name str.

--shuffle-check (no files written): with stand-in residency (res = e used in the previous 25
steps), compares hist and intra1 on the true routes and on routes whose decode tokens are
permuted independently per layer (dem and res recomputed from the permuted routes). The w_d
features should carry no signal on the permuted trace.

--extra-routes CSV (repeatable): extra trace used only to add counts to the tables (always
eligible, never scored). Its history features are not used.

--extra-score FILE.npz (repeatable): SCORE file from another predictor (same D, same fold array;
its scores are already out-of-fold; checked, exit on mismatch). Adds ONE feature per file and
candidate, z = ln(1e-4 + max(score, 0)), named z_<file stem>. Used by the stacking sets only
(these require --extra-score):
  stack1-pooled (lead 1): intra1-max-pooled features + z
  stack0-pooled (lead 0): markov-pooled features + z
  extra-pooled  (lead 0): f0 + z            (calibrated version of the extra score alone)
Stacking sets are pooled (one weight vector, per-layer intercepts); their standardised weights,
averaged over the folds, are printed by name.

Usage (numpy only; run under the lookahead venv, wrapped in the memory limit):
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 \
    ~/.cache/local-ai/venv-lookahead/bin/python -I tools/la_predict.py \
    --routes EXPERT_IDS.csv \
    --sim results/la/sim-fixed353-lfu-builtin-20.npz --out-prefix results/la/score-fixed353- \
    [--sets freq,hist,markov,intra1,intra2,intra3,intra1-pooled,markov-pooled,
                intra1-max-pooled,intra1-wide-pooled] [--train-frac F] [--no-write]
    [--extra-score OTHER-score.npz ... --sets stack1-pooled,stack0-pooled,extra-pooled]
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
from lookahead_train import N_E, N_L, TOP_K, read_routes  # noqa: E402

K_FOLDS = 5
GAP_CAP = 500
SMOOTH_A = 5.0
LAM = 1.0
MAX_ITER = 25
EPS = 1e-6
WIN = 25  # stand-in residency window (steps)
MAX_D = 6      # mean-based w_d, d = 1..MAX_D
MAX_X_D = 3    # max-based x_d, d = 1..MAX_X_D
GROUP = 8      # layers per causal-feature group
CHUNK = 32     # tokens per chunk in metrics and output conversion

MARKOV = ["f0", "h1", "h2", "h3", "m1"]
# name -> (lead, feature columns, pooled)
SETS = {
    "freq": (0, ["f0"], False),
    "hist": (0, ["f0", "h1", "h2", "h3"], False),
    "markov": (0, MARKOV, False),
    "intra1": (1, MARKOV + ["w1", "w2", "w3"], False),
    "intra2": (2, MARKOV + ["w2", "w3", "w4"], False),
    "intra3": (3, MARKOV + ["w3", "w4"], False),
    "intra1-pooled": (1, MARKOV + ["w1", "w2", "w3"], True),
    "markov-pooled": (0, MARKOV, True),
    "intra1-max-pooled": (1, MARKOV + ["w1", "w2", "w3", "x1", "x2", "x3"], True),
    "intra1-wide-pooled": (1, MARKOV + ["w%d" % d for d in range(1, MAX_D + 1)], True),
}
# stacking sets (need --extra-score): name -> SETS entry supplying lead and base features; always pooled
STACK_SETS = {
    "stack1-pooled": "intra1-max-pooled",
    "stack0-pooled": "markov-pooled",
    "extra-pooled": "freq",
}


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def usage(routes):
    """uint8 [S,48,512]: 1 if expert e is routed at (step, layer)."""
    S = routes.shape[0]
    U = np.zeros((S, N_L, N_E), np.uint8)
    mask = routes >= 0
    s, l, _ = np.nonzero(mask)
    U[s, l, routes[mask]] = 1
    return U


def causal_group(Ug, prefix, D):
    """h1, h2, h3 for decode tokens of a group of layers: float32 [D,G,512,3]. Ug uint8 [S,G,512]."""
    S, G = Ug.shape[0], Ug.shape[1]
    H = np.zeros((D, G, N_E, 3), np.float32)
    never = -10 ** 9
    last = np.full((G, N_E), never, np.int64)
    p2 = np.zeros((G, N_E))
    p3 = np.zeros((G, N_E))
    r2 = 0.5 ** (1 / 50)
    r3 = 0.5 ** (1 / 400)
    for t in range(S):
        if t >= prefix:
            j = t - prefix
            gap = np.where(last > never, np.minimum(t - last, GAP_CAP), GAP_CAP)
            H[j, :, :, 0] = np.log1p(gap)
            H[j, :, :, 1] = p2
            H[j, :, :, 2] = p3
        p2 = r2 * (p2 + Ug[t])
        p3 = r3 * (p3 + Ug[t])
        last[Ug[t] > 0] = t
    return H


def standin_res(U, prefix, D, win=WIN):
    """Stand-in residency: e used at layer L within the previous `win` steps (prefill included)."""
    S = U.shape[0]
    cs = np.zeros((S + 1, N_L, N_E), np.int32)
    np.cumsum(U, axis=0, dtype=np.int32, out=cs[1:])
    t = prefix + np.arange(D)
    lo = np.maximum(t - win, 0)
    return (cs[t] - cs[lo]) > 0


def pair_counts(A, B, w):
    """One layer pair. A [n,10] source experts (-1 absent), B [n,10] target experts, w [n] row weights.

    Returns C float64 [512,512], C[e',e] = sum_n w_n [e' in A_n] [e in B_n], and nA float64 [512],
    nA[e'] = sum_n w_n [e' in A_n].
    """
    va = A >= 0
    vb = B >= 0
    a = np.where(va, A, 0).astype(np.int64)
    b = np.where(vb, B, 0).astype(np.int64)
    idx = a[:, :, None] * N_E + b[:, None, :]
    wt = w[:, None, None] * (va[:, :, None] & vb[:, None, :])
    C = np.bincount(idx.ravel(), weights=wt.ravel(), minlength=N_E * N_E).reshape(N_E, N_E)
    nA = np.bincount(a.ravel(), weights=(w[:, None] * va).ravel(), minlength=N_E)
    return C, nA


def _mean_gather(P, ids):
    """P [512,512], ids [n,10] (-1 invalid) -> (mean over valid ids of P[id, :], has_any)."""
    valid = ids >= 0
    idc = np.where(valid, ids, 0).astype(np.intp)
    G = P[idc]  # [n,10,512]
    G = np.where(valid[..., None], G, 0.0).sum(axis=1)
    cnt = valid.sum(axis=1)
    return G / np.maximum(cnt, 1)[:, None], cnt > 0


def _max_gather(P, ids):
    """As _mean_gather, but MAX over the valid ids. Returns (max, has_any)."""
    valid = ids >= 0
    idc = np.where(valid, ids, 0).astype(np.intp)
    G = P[idc]
    G = np.where(valid[..., None], G, -np.inf).max(axis=1)
    return G, valid.any(axis=1)


def train_masks(fold, F):
    """sel[k] bool [D]: decode tokens of fold != k used by fold k (all of them when F == 1).

    Evenly spaced by index among the eligible tokens, round(F * n) of the n eligible ones.
    """
    D = fold.shape[0]
    sel = np.zeros((K_FOLDS, D), bool)
    for k in range(K_FOLDS):
        elig = np.nonzero(fold != k)[0]
        n = elig.size
        if F >= 1.0:
            sel[k, elig] = True
            continue
        m = max(1, int(round(F * n)))
        pick = elig[np.round(np.linspace(0, n - 1, m)).astype(np.int64)]
        if np.unique(pick).size != m:
            raise ValueError("train_masks: non-unique subset")
        sel[k, pick] = True
    return sel


def step_elig(prefix, S, sel_k):
    """Per-step eligibility of the main trace for one fold: prefill steps always, decode per sel_k."""
    el = np.ones(S, np.float64)
    el[prefix:] = sel_k
    return el


class LayerSource:
    """Feature blocks, one target layer at a time: X [D, 512, len(cols)] float32.

    Count tables are cross-fitted per fold exactly as before: tokens of fold m use the tables built
    from prefill steps and decode steps with fold != m (and the extra traces).
    """

    def __init__(self, routes, U, prefix, D, fold, extra_routes, sel, extra_scores):
        self.routes = routes
        self.U = U
        self.prefix = prefix
        self.D = D
        self.S = U.shape[0]
        self.fold = fold
        self.sel = sel
        self.extra_R = extra_routes
        self.extra_sc = extra_scores  # name -> SCORE array float16 [D,48,512]
        self.js = [np.nonzero(fold == m)[0] for m in range(K_FOLDS)]
        self.f0 = np.zeros((K_FOLDS, N_L, N_E), np.float32)
        self.freq = np.zeros((K_FOLDS, N_L, N_E), np.float64)
        for m in range(K_FOLDS):
            cnt = np.zeros((N_L, N_E))
            for R, el in zip(self._segments(), self._els(m)):
                cnt += np.bincount(
                    (np.arange(N_L)[None, :, None] * N_E + np.where(R >= 0, R, 0)).ravel(),
                    weights=(el[:, None, None] * (R >= 0)).ravel(), minlength=N_L * N_E).reshape(N_L, N_E)
            p0 = (cnt + 0.5) / (cnt.sum(axis=1, keepdims=True) + 0.5 * N_E)
            self.f0[m] = np.log(p0).astype(np.float32)
            self.freq[m] = TOP_K * p0

    def _segments(self):
        return [self.routes] + self.extra_R

    def _els(self, m):
        return [step_elig(self.prefix, self.S, self.sel[m])] + [np.ones(R.shape[0], np.float64) for R in self.extra_R]

    def p_table(self, m, L, d):
        """Smoothed transition table P [512,512] float32 for target layer L of fold m.

        d = 0: same-layer transitions between consecutive steps (m1). d >= 1: same-step, source L-d.
        """
        C = np.zeros((N_E, N_E))
        n = np.zeros(N_E)
        for R, el in zip(self._segments(), self._els(m)):
            if d == 0:
                if R.shape[0] <= 1:
                    continue
                c, na = pair_counts(R[:-1, L], R[1:, L], el[:-1] * el[1:])
            else:
                c, na = pair_counts(R[:, L - d], R[:, L], el)
            C += c
            n += na
        freq = self.freq[m, L]
        return ((C + SMOOTH_A * freq[None, :]) / (n[:, None] + SMOOTH_A)).astype(np.float32)

    def blocks(self, cols):
        """Yield (L, X) for L = 0..47, X float32 [D, 512, len(cols)] with the columns named in cols."""
        need_h = any(c in ("h1", "h2", "h3") for c in cols)
        ds = sorted({int(c[1:]) for c in cols if c[0] in "wx" and c[1:].isdigit()})
        routes = self.routes
        for g0 in range(0, N_L, GROUP):
            g1 = min(N_L, g0 + GROUP)
            Hg = causal_group(self.U[:, g0:g1], self.prefix, self.D) if need_h else None
            for L in range(g0, g1):
                X = np.empty((self.D, N_E, len(cols)), np.float32)
                for i, c in enumerate(cols):
                    if c.startswith("z_"):
                        sc = self.extra_sc[c]
                        X[:, :, i] = np.log(1e-4 + np.maximum(sc[:, L, :].astype(np.float64), 0.0)).astype(np.float32)
                for m in range(K_FOLDS):
                    js = self.js[m]
                    ts = self.prefix + js
                    f0m = self.f0[m, L]
                    freqm = self.freq[m, L]
                    P1 = self.p_table(m, L, 0) if "m1" in cols else None
                    Pd = {d: self.p_table(m, L, d) for d in ds if L >= d}
                    for i, c in enumerate(cols):
                        if c == "f0":
                            X[js, :, i] = f0m
                        elif c in ("h1", "h2", "h3"):
                            X[js, :, i] = Hg[js, L - g0, :, int(c[1]) - 1]
                        elif c == "m1":
                            mean, has = _mean_gather(P1, routes[ts - 1, L])
                            X[js, :, i] = np.log(EPS + np.where(has[:, None], mean, freqm[None, :])).astype(np.float32)
                        elif c[0] in "wx" and c[1:].isdigit():
                            d = int(c[1:])
                            if L < d:
                                X[js, :, i] = f0m
                            else:
                                ids = routes[ts, L - d]
                                if c[0] == "w":
                                    mean, has = _mean_gather(Pd[d], ids)
                                else:
                                    mean, has = _max_gather(Pd[d], ids)
                                X[js, :, i] = np.log(EPS + np.where(has[:, None], mean, freqm[None, :])).astype(np.float32)
                        elif c.startswith("z_"):
                            pass  # filled above for all tokens
                        else:
                            raise ValueError("unknown feature column %r" % c)
                yield L, X


def fit_logreg(X, y, counts):
    """Newton/IRLS for sum-NLL + 0.5*LAM*||beta||^2 with one unpenalised intercept per group.

    Rows are contiguous per group: group g owns counts[g] consecutive rows (all counts > 0).
    """
    N, F = X.shape
    counts = np.asarray(counts, np.int64)
    G = counts.size
    if (counts <= 0).any() or counts.sum() != N:
        raise ValueError("fit_logreg: empty group or bad counts")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    beta = np.zeros(F)
    alpha = np.zeros(G)
    I = np.eye(F)
    conv = False
    it = 0

    def gsum(V):
        return np.add.reduceat(V, starts, axis=0)

    for it in range(1, MAX_ITER + 1):
        eta = X @ beta + np.repeat(alpha, counts)
        p = expit(eta)
        r = y - p
        v = p * (1.0 - p)
        XV = X * v[:, None]
        g = np.concatenate([X.T @ r - LAM * beta, gsum(r)])
        Hbb = XV.T @ X + LAM * I
        Hba = gsum(XV).T  # [F, G]
        Haa = gsum(v) + 1e-10
        H = np.block([[Hbb, Hba], [Hba.T, np.diag(Haa)]])
        step = np.linalg.solve(H, g)
        beta += step[:F]
        alpha += step[F:]
        if np.max(np.abs(step)) < 1e-7:
            conv = True
            break
    return beta, alpha, it, conv


def fit_pooled(src, fold, nonres, dem, cols, sel):
    """Pooled logistic fits, streamed over layers; the K folds are iterated in lock-step.

    Standardisation: mean and population sd over the fit's training rows (Chan merge per layer block).
    Newton: the same Hessian/gradient/step/convergence rule as fit_logreg, accumulated per layer.
    Returns score float32 [D,48,512], weights, stats.
    """
    F = len(cols)
    D = fold.shape[0]
    K = K_FOLDS
    ntr = np.zeros((K, N_L), np.int64)
    npos = np.zeros((K, N_L), np.int64)

    # 1. standardisation statistics (one streaming pass)
    n = np.zeros(K)
    mean = np.zeros((K, F))
    M2 = np.zeros((K, F))
    for L, X in src.blocks(cols):
        for k in range(K):
            m = sel[k][:, None] & nonres[:, L, :]
            Xr = X[m].astype(np.float64)
            nb = Xr.shape[0]
            if nb == 0:
                continue
            mb = Xr.mean(axis=0)
            M2b = ((Xr - mb) ** 2).sum(axis=0)
            na = n[k]
            delta = mb - mean[k]
            nn = na + nb
            mean[k] += delta * nb / nn
            M2[k] += M2b + delta ** 2 * na * nb / nn
            n[k] = nn
            ntr[k, L] = nb
            npos[k, L] = int(dem[:, L, :][m].sum())
    mu = mean
    sd = np.sqrt(M2 / n[:, None])
    sd = np.where(sd < 1e-9, 1.0, sd)

    # 2. IRLS, lock-step over folds: one feature pass per iteration serves all active folds
    beta = np.zeros((K, F))
    alpha = np.zeros((K, N_L))
    conv = np.zeros(K, bool)
    I = np.eye(F)
    for _ in range(1, MAX_ITER + 1):
        act = [k for k in range(K) if not conv[k]]
        if not act:
            break
        acc = {k: {"Hbb": np.zeros((F, F)), "gb": np.zeros(F), "Hba": np.zeros((F, N_L)),
                   "Haa": np.zeros(N_L), "ga": np.zeros(N_L)} for k in act}
        for L, X in src.blocks(cols):
            for k in act:
                m = sel[k][:, None] & nonres[:, L, :]
                Z = (X[m].astype(np.float64) - mu[k]) / sd[k]
                y = dem[:, L, :][m].astype(np.float64)
                eta = Z @ beta[k] + alpha[k, L]
                p = expit(eta)
                r = y - p
                v = p * (1.0 - p)
                Zv = Z * v[:, None]
                a = acc[k]
                a["Hbb"] += Zv.T @ Z
                a["gb"] += Z.T @ r
                a["Hba"][:, L] += Zv.sum(axis=0)
                a["Haa"][L] += v.sum()
                a["ga"][L] += r.sum()
        for k in act:
            a = acc[k]
            g = np.concatenate([a["gb"] - LAM * beta[k], a["ga"]])
            H = np.block([[a["Hbb"] + LAM * I, a["Hba"]],
                          [a["Hba"].T, np.diag(a["Haa"] + 1e-10)]])
            step = np.linalg.solve(H, g)
            beta[k] += step[:F]
            alpha[k] += step[F:]
            if np.max(np.abs(step)) < 1e-7:
                conv[k] = True
    nonconv = int((~conv).sum())

    # 3. test predictions (one feature pass): fold k is scored by fit k
    score = np.zeros((D, N_L, N_E), np.float32)
    for L, X in src.blocks(cols):
        for k in range(K):
            te = fold == k
            Zte = (X[te].astype(np.float64) - mu[k]) / sd[k]
            score[te, L] = expit(Zte @ beta[k] + alpha[k, L])
    weights = [(k, None, beta[k].copy()) for k in range(K)]
    stats = {"ntr": ntr, "npos": npos, "nonconv": nonconv}
    return score, weights, stats


def fit_set(src, fold, nonres, dem, cols, pooled, sel):
    """Cross-fitted predictions. Returns score float32 [D,48,512], weights, stats.

    Fit k trains on sel[k] (decode tokens of fold != k, subsampled by --train-frac) and predicts
    all tokens of fold k.
    """
    if pooled:
        return fit_pooled(src, fold, nonres, dem, cols, sel)
    D = fold.shape[0]
    score = np.zeros((D, N_L, N_E), np.float32)
    weights = []   # per fit: (k, L, beta[F])
    ntr = np.zeros((K_FOLDS, N_L), np.int64)
    npos = np.zeros((K_FOLDS, N_L), np.int64)
    nonconv = 0
    for L, X in src.blocks(cols):
        for k in range(K_FOLDS):
            tr = sel[k]
            te = fold == k
            m = tr[:, None] & nonres[:, L, :]
            Xr = X[m].astype(np.float64)
            y = dem[:, L, :][m].astype(np.float64)
            mu = Xr.mean(axis=0)
            sd = Xr.std(axis=0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            Z = (Xr - mu) / sd
            beta, alpha, _, conv = fit_logreg(Z, y, [len(y)])
            nonconv += 0 if conv else 1
            Zte = (X[te].astype(np.float64) - mu) / sd
            score[te, L] = expit(Zte @ beta + alpha[0])
            weights.append((k, L, beta))
            ntr[k, L] = len(y)
            npos[k, L] = int(y.sum())
    stats = {"ntr": ntr, "npos": npos, "nonconv": nonconv}
    return score, weights, stats


def metrics(score, dem, res, chunk=CHUNK):
    """Same metrics as before, accumulated over token chunks (bounded temporaries)."""
    D = score.shape[0]
    ll = 0.0
    n_nr = 0
    n_pos = 0
    psum = {1: 0.0, 3: 0.0, 10: 0.0}
    msum = {3: 0.0, 10: 0.0}
    mcnt = {3: 0, 10: 0}
    for i in range(0, D, chunk):
        s32 = score[i:i + chunk].astype(np.float32)
        dm = dem[i:i + chunk]
        nr = ~res[i:i + chunk]
        p = np.clip(s32[nr].astype(np.float64), EPS, 1.0 - EPS)
        y = dm[nr].astype(np.float64)
        ll += float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        n_nr += int(nr.sum())
        n_pos += int(y.sum())
        sm = np.where(nr, s32, -np.inf)
        # top-10 among non-resident per (token, layer), ordered by score (ties arbitrary)
        part = np.argpartition(-sm, TOP_K - 1, axis=2)[:, :, :TOP_K]
        pv = np.take_along_axis(sm, part, axis=2)
        o = np.argsort(-pv, axis=2, kind="stable")
        order = np.take_along_axis(part, o, axis=2)
        hit = np.take_along_axis(dm, order, axis=2)
        for k in (1, 3, 10):
            psum[k] += float(hit[:, :, :k].mean(axis=2).sum())
        miss = dm & nr
        mhit = np.take_along_axis(miss, order, axis=2)
        nm = miss.sum(axis=2)
        ok = nm > 0
        for k in (3, 10):
            v = mhit[:, :, :k].sum(axis=2)[ok] / nm[ok]
            msum[k] += float(v.sum())
            mcnt[k] += int(v.size)
    out = {"logloss": float(-ll / n_nr), "base": float(n_pos / n_nr)}
    for k in (1, 3, 10):
        out["p@%d" % k] = psum[k] / (D * N_L)
    for k in (3, 10):
        out["mrec@%d" % k] = msum[k] / mcnt[k]
    return out


def print_header():
    print("%-15s %4s %9s %8s %7s %7s %7s %8s %9s %8s" % (
        "set", "lead", "logloss", "base", "p@1", "p@3", "p@10", "mrec@3", "mrec@10", "nonconv"))


def print_row(name, lead, m, nonconv):
    print("%-15s %4d %9.5f %8.5f %7.4f %7.4f %7.4f %8.4f %9.4f %8d" % (
        name, lead, m["logloss"], m["base"], m["p@1"], m["p@3"], m["p@10"],
        m["mrec@3"], m["mrec@10"], nonconv))


def print_weights(weights, cols, head=None):
    W = np.stack([b for (_, _, b) in weights])  # [K*48, F]
    if head is None:
        head = "standardised weights, intra model (mean over layers and folds; sd over the %d fits):" % W.shape[0]
    print(head)
    for c, mu, sd in zip(cols, W.mean(axis=0), W.std(axis=0)):
        print("  %-4s %+9.4f  (sd %.4f)" % (c, mu, sd))


def load_extra_score(path, D, fold):
    """SCORE file from another predictor: must have this trace's D and fold array. Returns float16 [D,48,512]."""
    z = np.load(path)
    for key in ("score", "fold"):
        if key not in z.files:
            raise SystemExit("--extra-score %s: no '%s' array (not a SCORE file)" % (path, key))
    sc = z["score"]
    if sc.shape != (D, N_L, N_E):
        raise SystemExit("--extra-score %s: score shape %s, expected %s (D=%d)" % (
            path, sc.shape, (D, N_L, N_E), D))
    fz = z["fold"]
    if fz.shape != fold.shape or not np.array_equal(fz.astype(np.int64), fold.astype(np.int64)):
        raise SystemExit("--extra-score %s: fold array differs from the fold array of this trace" % path)
    if not np.isfinite(sc).all():
        raise SystemExit("--extra-score %s: non-finite scores" % path)
    return sc


def extra_name(path):
    stem = os.path.basename(path)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    return "z_" + stem


def shuffle_routes(routes, prefix, rng):
    """Permute decode tokens independently per layer (each token's 10-set moves as a unit)."""
    S = routes.shape[0]
    D = S - prefix
    R = routes.copy()
    perm = np.stack([rng.permutation(D) for _ in range(N_L)], axis=1)  # [D,48]
    R[prefix:] = routes[prefix:][perm, np.arange(N_L)[None, :], :]
    return R


def main():
    ap = argparse.ArgumentParser(description="Route-only cross-fitted expert-demand predictors (SCORE files).")
    ap.add_argument("--routes", required=True, help="route trace CSV (turn,phase,step,layer,slot,expert)")
    ap.add_argument("--sim", default=None, help="SIM npz (res, dem); required unless --standin")
    ap.add_argument("--standin", action="store_true", help="use stand-in residency (previous 25 steps) instead of --sim")
    ap.add_argument("--out-prefix", default="results/la/score-fixed353-")
    ap.add_argument("--sets", default=",".join(SETS), help="comma-separated subset of: " + ",".join(SETS)
                    + " (stacking, needs --extra-score: " + ",".join(STACK_SETS) + ")")
    ap.add_argument("--extra-routes", action="append", default=[],
                    help="extra route CSV used only to add counts to the tables (repeatable)")
    ap.add_argument("--extra-score", action="append", default=[],
                    help="SCORE npz from another predictor (same D and folds) used as a stacking feature "
                         "z = ln(1e-4 + max(score, 0)) (repeatable)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="learning curve: fraction (0 < F <= 1) of the eligible decode tokens used by the "
                         "tables and the regression of each fold (evenly spaced by index); F < 1 adds the "
                         "suffix -f<100F> to set names and output files")
    ap.add_argument("--no-write", action="store_true", help="print metrics only")
    ap.add_argument("--shuffle-check", action="store_true",
                    help="sanity check (a): hist and intra1 on true vs per-layer-permuted routes, stand-in residency")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    names = [s.strip() for s in args.sets.split(",") if s.strip()]
    for n in names:
        if n not in SETS and n not in STACK_SETS:
            ap.error("unknown set %r" % n)
    if any(n in STACK_SETS for n in names) and not args.extra_score:
        ap.error("stacking sets (%s) need --extra-score" % ",".join(n for n in names if n in STACK_SETS))
    if not args.standin and args.sim is None:
        ap.error("--sim is required (or pass --standin)")
    F = args.train_frac
    if not (0.0 < F <= 1.0):
        ap.error("--train-frac must satisfy 0 < F <= 1")
    suffix = ""
    if F < 1.0:
        pct = int(round(100 * F))
        if not (1 <= pct <= 99):
            ap.error("--train-frac < 1 must round to 1..99 percent")
        suffix = "-f%02d" % pct

    t0 = time.time()
    routes, prefix, S = read_routes(args.routes)
    D = S - prefix
    fold = (np.arange(D) * K_FOLDS) // D
    sel = train_masks(fold, F)
    extra_R = []
    for p in args.extra_routes:
        er, _, _ = read_routes(p)
        extra_R.append(er)
    extra_S = [load_extra_score(p, D, fold) for p in args.extra_score]
    extra_names = [extra_name(p) for p in args.extra_score]
    if len(set(extra_names)) != len(extra_names):
        raise SystemExit("--extra-score: duplicate file stems %s" % extra_names)
    print("trace: %s  prefix=%d D=%d S=%d extra_traces=%d" % (args.routes, prefix, D, S, len(extra_R)))
    for nm, p in zip(extra_names, args.extra_score):
        print("extra score feature %s from %s (checked: D, fold array)" % (nm, p))
    if F < 1.0:
        n_pairs = [int(np.sum(step_elig(prefix, S, sel[k])[:-1] * step_elig(prefix, S, sel[k])[1:]))
                   for k in range(K_FOLDS)]
        print("train-frac F=%.3f: decode tokens per fit (train) mean %.1f of eligible mean %.1f; "
              "transition pairs in count tables per fit: mean %.1f (min %d max %d)" % (
                  F, sel.sum(axis=1).mean(), np.mean([(fold != k).sum() for k in range(K_FOLDS)]),
                  np.mean(n_pairs), min(n_pairs), max(n_pairs)))

    res = None
    sim_dem = None
    if args.sim is not None and not args.standin:
        z = np.load(args.sim)
        if z["res"].shape != (D, N_L, N_E) or int(z["prefix"]) != prefix or int(z["D"]) != D:
            raise SystemExit("SIM file shape/prefix/D does not match the routes")
        res = z["res"].astype(bool)
        sim_dem = z["dem"].astype(bool)
        print("residency: SIM file %s (policy=%s, quota_scheme=%s, cache_gib=%s)" % (
            args.sim, z["policy"], z["quota_scheme"], z["cache_gib"]))

    U = usage(routes)
    dem = U[prefix:].astype(bool)
    if sim_dem is not None:
        mism = int(np.sum(sim_dem != dem))
        print("check: sim dem vs routes dem mismatches = %d" % mism)
        if mism:
            raise SystemExit("sim dem does not match the route trace")
    if res is None:
        res = standin_res(U, prefix, D)
        print("residency: STAND-IN (e used in the previous %d steps)" % WIN)
    src = LayerSource(routes, U, prefix, D, fold, extra_R, sel, dict(zip(extra_names, extra_S)))
    print("feature source ready (streamed per layer) in %.1fs" % (time.time() - t0))

    nonres = ~res
    print("non-resident candidates: %d of %d (%.3f); demanded among them: %d" % (
        nonres.sum(), nonres.size, nonres.mean(), int((dem & nonres).sum())))
    print()
    print_header()
    stats_printed = False
    for name in names:
        if name in STACK_SETS:
            lead, feats, _ = SETS[STACK_SETS[name]]
            cols, pooled = feats + extra_names, True
        else:
            lead, cols, pooled = SETS[name]
        t1 = time.time()
        score, weights, stats = fit_set(src, fold, nonres, dem, cols, pooled, sel)
        m = metrics(score, dem, res)
        print_row(name + suffix, lead, m, stats["nonconv"])
        if not pooled and not stats_printed:
            ntr = stats["ntr"].ravel()
            npos = stats["npos"].ravel()
            print("  training candidates per (layer, fold) fit: min %d mean %.1f max %d" % (
                ntr.min(), ntr.mean(), ntr.max()))
            print("  positives (demanded) per (layer, fold) fit: min %d mean %.1f max %d" % (
                npos.min(), npos.mean(), npos.max()))
            stats_printed = True
        if name == "intra1":
            print_weights(weights, cols)
        if name in STACK_SETS:
            print_weights(weights, cols, head="standardised weights, stacked pooled model (mean over the %d folds; "
                                              "sd over folds):" % len(weights))
        if not args.no_write:
            outdir = os.path.dirname(args.out_prefix)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
            path = args.out_prefix + name + suffix + ".npz"
            out16 = np.empty((D, N_L, N_E), np.float16)
            for i in range(0, D, 4 * CHUNK):
                out16[i:i + 4 * CHUNK] = np.clip(score[i:i + 4 * CHUNK], 0.0, 1.0).astype(np.float16)
            np.savez_compressed(path, score=out16,
                                fold=fold.astype(np.int16), lead=np.int64(lead),
                                name=np.str_(name + suffix))
            print("  wrote %s (%.1fs)" % (path, time.time() - t1))
        del score
    print()

    if args.shuffle_check:
        rng = np.random.default_rng(args.seed)
        R2 = shuffle_routes(routes, prefix, rng)
        print("shuffle check (stand-in residency; dem and residency recomputed from each trace):")
        print_header()
        for label, R in (("true", routes), ("shuffled", R2)):
            Ux = usage(R)
            demx = Ux[prefix:].astype(bool)
            resx = standin_res(Ux, prefix, D)
            srcx = LayerSource(R, Ux, prefix, D, fold, extra_R, sel, {})
            for name in ("hist", "intra1"):
                lead, cols, pooled = SETS[name]
                score, weights, stats = fit_set(srcx, fold, ~resx, demx, cols, pooled, sel)
                m = metrics(score, demx, resx)
                print_row("%s/%s" % (name, label), lead, m, stats["nonconv"])
                if name == "intra1" and label == "shuffled":
                    print_weights(weights, cols)
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
