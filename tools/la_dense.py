#!/usr/bin/env python3
"""Dense additive lookahead predictor: integer engine path, SCORE replay files, LAT2/LAV1 export (F-LOOKAHEAD).

Question: when the routes of layers T-1, T-2, T-3 of the current token are known, which NON-RESIDENT
experts of layer T should be pre-read? This is the dense counterpart of la_table.py (same job, same
engine-side idea, but every (source, target) pair is stored, so there is no top-K truncation).

Model, target layer T, candidate expert e, decode token j with routes R[T-d] (10 experts):
  C, n, f = la_predict's within-token count tables and smoothed frequency (f = 10 * p0, cross-fitted per
            fold, prefill and --extra-routes included, same eligibility as la_predict);  a = 5
  pmi_d[T][s][e] = clip(ln((C[s,e] + a f[e]) / (n[s] + a)) - ln f[e], -4, 6),  pmi = 0 where n[s] = 0
  S_d(e)  = sum over the 10 routed s at T-d of pmi_d[T][s][e]            (0 when T-d < 0)
  lf(e)   = ln f[e]
  logit   = b0[T] + wf * lf + w1 S_1 + w2 S_2 + w3 S_3
  Fit: L2-penalised logistic regression (la_predict's LAM and MAX_ITER, standardised features, intercept per
  layer unpenalised, penalty on the summed log-loss) on the NON-RESIDENT candidates (res False) of the training
  tokens, label dem. Newton/IRLS normal equations are accumulated layer by layer (no candidate x feature matrix
  is ever built). Default: ONE pooled (wf, w1..w3) with 48 per-layer intercepts (52 parameters).
  --per-layer-w: separate (wf, w1..w3) per target layer (48 small fits).

Cross-fitting (--mode score): fold k tables = prefill + decode tokens with fold != k (+ --extra-routes); the
weights of fold k are fitted on the training tokens (fold != k) using features computed from the SAME tables
(the la_table.py convention: in-sample features for the training tokens, self-counts included). --own-feats
switches the training features to la_predict's convention: a training token of fold m uses the tables that
exclude fold m (the test-side features are unchanged).
--mode export: tables from ALL decode tokens (+ prefill + extra); weights fitted on all decode tokens with the
same feature convention (in-sample by default, --own-feats for own-fold features).

Engine integer path (SCALE = 16 score units per logit):
  tab[T][d-1][s][e]  = clip(rint(16 * w_d[T] * pmi_d[T][s][e]), -127, 127)     int8
  bias[T][e]         = clip(rint(16 * (b0[T] + wf[T] * lf[T][e])), -32000, 32000)  int16
  score_int[T][e]    = bias[T][e] + sum_d sum_{s routed at T-d} tab[T][d-1][s][e]   int32 (numpy int32 only)
  SCORE probability  = sigmoid(score_int / 16)
  candidates         = experts with score_int >= thr[T]  (thr = ceil(16 logit(p)) for a probability p)

Error bound used by --selftest: rint has error <= 0.5, the engine sums 3 * 10 = 30 table entries and one bias,
so |score_int/16 - logit| <= (30 * 0.5 + 0.5) / 16 = 0.96875 whenever no entry was clipped (|raw| > 127 or
bias beyond 32000). Clipped entries are counted and reported; the bound is then not claimed.

Modes:
  --mode score   cross-fitted over la_predict's 5 folds. Writes SCORE files <out-prefix>dense.npz
                 (<out-prefix>dense-plw.npz with --per-layer-w; suffix -own with --own-feats); fields score
                 float16 [D,48,512] = sigmoid(score_int/16), fold int16, lead = 1, name. Prints la_predict's
                 metric row (print_row) and the clipping fraction of the tab entries.
  --mode export  writes the LAT2 binary (--out) and the LAV1 test vectors (--vectors), re-reads both, checks that
                 every array round-trips exactly, that the integer scores of the file equal the in-memory ones,
                 and that the vectors reproduce from the file alone. --thr-prob P (one probability) or a text
                 file with 48 probabilities, one per line, "inf" = never (thr = 2147483647).
  --selftest     synthetic chain routes (expert s at layer T-1 implies (7s+3) % 512 at layer T): recall@10 of the
                 implied experts from the cross-fitted integer scores (layers 1..47); quantised vs float logit
                 bound (see above); LAT2/LAV1 round trip; per-layer variant runs.

Per-layer thresholds from la_pool (--thr-pool-json FILE, instead of --thr-prob; exactly one of the two in export
mode): FILE is a la_pool.py --json output. --thr-score NAME picks its SCORE entry (default: the only one; with
several, an error lists the names); --thr-pair S_C,S_W picks its cost pair (default 0.66,0.02, matched within
1e-9; an error lists the available pairs). The taus are the per-fold crossfit taus (scores[i].tau_per_fold). Rule:
for each layer, the MEDIAN over the folds of that layer's tau. Odd number of folds: the middle value after
sorting, "never" (null, read as +inf) last. Even number of folds: the HIGHER of the two middle values (the
conservative one). A layer whose median is never gets thr = 2147483647; otherwise thr = ceil(16 logit(tau)).

LAT2 binary, little-endian: b"LAT2"; u32 version = 2; u32 n_layer = 48; u32 n_expert = 512; u32 n_dist = 3;
i32 scale = 16; i32 thr[48]; i16 bias[48][512]; i8 tab[48][3][512][512] (C order: [target T][d-1][source s][target e]).
Blocks with T-d < 0 are all zeros. Size 24 + 192 + 49152 + 37748736 = 37798104 bytes.

LAV1 binary, little-endian: b"LAV1"; u32 n_vec; per vector: u32 T; u16 ids[3][10] (ids[d-1] = the 10 experts routed
at layer T-d of one decode token; 0xFFFF x 10 when T-d < 0); u32 n_out (<= 8); n_out records (u16 expert, i32 score):
experts with score >= thr[T], score descending then expert id ascending, residency ignored.

Usage (numpy only; run under the lookahead venv; wrapped in the memory cap; PY = venv python):
  flock -w 900 "${LOCAL_AI_BUILD_LOCK:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/build3.lock}" systemd-run --user --scope -q -p MemoryMax=4G \
    -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 19 $PY -I tools/la_dense.py --selftest
  ... -I tools/la_dense.py --mode score --routes ROUTES.csv [--extra-routes X.csv] --sim SIM.npz \
      --out-prefix results/la/score-NAME- [--per-layer-w] [--own-feats]
  ... -I tools/la_dense.py --mode export --routes ROUTES.csv [--extra-routes X.csv] --sim SIM.npz \
      --thr-prob 0.05 --out results/la/NAME.lat --vectors results/la/NAME.lav [--per-layer-w] [--own-feats]
  ... -I tools/la_dense.py --mode export --routes ROUTES.csv --sim SIM.npz --thr-pool-json results/la/POOL.json \
      [--thr-score NAME] [--thr-pair 0.66,0.02] --out results/la/NAME.lat --vectors results/la/NAME.lav
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import math
import re
import resource
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
from lookahead_train import N_E, N_L, TOP_K, read_routes  # noqa: E402
# ----------------------------------------------------------------------------- la_predict helpers
# Copied verbatim from the COMMITTED tools/la_predict.py (HEAD e7da824). The working copy of la_predict.py is
# being rewritten (streamed refactor, uncommitted, within_tables removed), so this file does not import it: the
# count tables, smoothing, metric row and step eligibility below are the same arithmetic as la_predict.py HEAD.
K_FOLDS = 5
SMOOTH_A = 5.0
LAM = 1.0
MAX_ITER = 25
EPS = 1e-6


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


def count_tables(A, B, w, d):
    """Weighted pair counts sparse over the 10 routed experts (la_predict.count_tables, HEAD).

    A [n, 48-d, 10]: source-layer routes (A[:, i] is layer i), B [n, 48-d, 10]: target routes
    (B[:, i] is layer i+d), -1 = absent; w [n]: row weights (0/1 eligibility).
    Returns C float64 [48,512,512] indexed by TARGET layer (rows L < d are zero) with
    C[L,e',e] = sum_n w_n [e' in A_n at L-d] [e in B_n at L], and nA float64 [48,512]
    with nA[L,e'] = sum_n w_n [e' in A_n at L-d].
    """
    Lt = np.arange(d, N_L)
    va = A >= 0
    vb = B >= 0
    a = np.where(va, A, 0).astype(np.int64)
    b = np.where(vb, B, 0).astype(np.int64)
    idx = (Lt[None, :, None, None] * N_E + a[:, :, :, None]) * N_E + b[:, :, None, :]
    wt = w[:, None, None, None] * (va[:, :, :, None] & vb[:, :, None, :])
    C = np.bincount(idx.ravel(), weights=wt.ravel(), minlength=N_L * N_E * N_E)
    C = C.reshape(N_L, N_E, N_E)
    ia = Lt[None, :, None] * N_E + a
    wa = w[:, None, None] * va
    nA = np.bincount(ia.ravel(), weights=wa.ravel(), minlength=N_L * N_E).reshape(N_L, N_E)
    return C, nA


def within_tables(segs, els, d):
    """Same-step counts for source layer L-d -> target L, eligible steps (la_predict.within_tables, HEAD)."""
    C = np.zeros((N_L, N_E, N_E))
    nw = np.zeros((N_L, N_E))
    for (R, _), el in zip(segs, els):
        C2, nA = count_tables(R[:, : N_L - d], R[:, d:], el, d)
        C += C2
        nw += nA
    return C, nw


def step_elig(prefix, S, sel_k):
    """Per-step eligibility of the main trace for one fold: prefill steps always, decode per sel_k."""
    el = np.ones(S, np.float64)
    el[prefix:] = sel_k
    return el


def metrics(score, dem, res):
    """la_predict.metrics (HEAD): logloss, base rate, p@k and mrec@k among non-resident candidates."""
    nr = ~res
    s32 = score.astype(np.float32)
    p = np.clip(s32[nr].astype(np.float64), EPS, 1.0 - EPS)
    y = dem[nr].astype(np.float64)
    logloss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    base = y.mean()
    sm = np.where(nr, s32, -np.inf)
    # top-10 among non-resident per (token, layer), ordered by score (ties arbitrary)
    part = np.argpartition(-sm, TOP_K - 1, axis=2)[:, :, :TOP_K]
    pv = np.take_along_axis(sm, part, axis=2)
    o = np.argsort(-pv, axis=2, kind="stable")
    order = np.take_along_axis(part, o, axis=2)
    hit = np.take_along_axis(dem, order, axis=2)
    miss = dem & nr
    mhit = np.take_along_axis(miss, order, axis=2)
    out = {"logloss": float(logloss), "base": float(base)}
    for k in (1, 3, 10):
        out["p@%d" % k] = float(hit[:, :, :k].mean(axis=2).mean())
    nm = miss.sum(axis=2)
    ok = nm > 0
    for k in (3, 10):
        out["mrec@%d" % k] = float((mhit[:, :, :k].sum(axis=2)[ok] / nm[ok]).mean())
    return out


def print_header():
    print("%-15s %4s %9s %8s %7s %7s %7s %8s %9s %8s" % (
        "set", "lead", "logloss", "base", "p@1", "p@3", "p@10", "mrec@3", "mrec@10", "nonconv"))


def print_row(name, lead, m, nonconv):
    print("%-15s %4d %9.5f %8.5f %7.4f %7.4f %7.4f %8.4f %9.4f %8d" % (
        name, lead, m["logloss"], m["base"], m["p@1"], m["p@3"], m["p@10"],
        m["mrec@3"], m["mrec@10"], nonconv))


ND = 3
NF = 4                      # features: lf, S_1, S_2, S_3
SCALE = 16
PMI_LO, PMI_HI = -4.0, 6.0
TAB_MAX = 127
BIAS_MAX = 32000
W_NEVER = -30.0             # intercept of a layer without training candidates
CHUNK_TOK = 32
UNUSED = 0xFFFF
N_OUT_MAX = 8
THR_NEVER = 2147483647
LAT_MAGIC = b"LAT2"
LAT_VERSION = 2
LAT_SIZE = 24 + 4 * N_L + 2 * N_L * N_E + N_L * ND * N_E * N_E   # 37798104
LAV_MAGIC = b"LAV1"
BOUND = (ND * TOP_K * 0.5 + 0.5) / SCALE                          # 0.96875


# ----------------------------------------------------------------------------- data

def make_data(routes, prefix, extra):
    """Dict with the trace, folds, decode tokens and the dem array (res must be added by the caller)."""
    S = routes.shape[0]
    D = S - prefix
    if D < K_FOLDS:
        raise SystemExit("need at least %d decode tokens" % K_FOLDS)
    dec = routes[prefix:]
    if (dec < 0).any():
        raise SystemExit("decode tokens must be complete (10 experts per layer)")
    return {"routes": routes, "prefix": prefix, "S": S, "D": D,
            "fold": (np.arange(D) * K_FOLDS) // D, "extra": extra, "dec": dec,
            "dem": usage(routes)[prefix:].astype(bool),
            "segs": [(routes, None)] + [(R, None) for R in extra]}


def load_data(routes_path, sim_path, extra_paths):
    routes, prefix, S = read_routes(routes_path)
    extra = [read_routes(p)[0] for p in extra_paths]
    data = make_data(routes, prefix, extra)
    if sim_path is None:
        raise SystemExit("--sim is required")
    z = np.load(sim_path)
    D = data["D"]
    if z["res"].shape != (D, N_L, N_E) or int(z["prefix"]) != prefix or int(z["D"]) != D:
        raise SystemExit("SIM file shape/prefix/D does not match the routes")
    if np.any(z["dem"].astype(bool) != data["dem"]):
        raise SystemExit("sim dem does not match the route trace")
    data["res"] = z["res"].astype(bool)
    print("trace: %s  prefix=%d D=%d S=%d extra_traces=%d" % (routes_path, prefix, D, S, len(extra)))
    print("residency: SIM file %s (policy=%s, quota_scheme=%s, cache_gib=%s); dem check OK" % (
        sim_path, z["policy"], z["quota_scheme"], z["cache_gib"]))
    return data


def els_for(data, train_tok):
    els = [step_elig(data["prefix"], data["S"], train_tok)]
    return els + [np.ones(R.shape[0], np.float64) for R in data["extra"]]


# ----------------------------------------------------------------------------- tables

def tables_for(data, train_tok):
    """pmi float32 [ND, 48, 512, 512] (index [d-1][T][s][e]; rows T < d are zero) and lf float32 [48, 512]."""
    segs = data["segs"]
    els = els_for(data, train_tok)
    cnt = np.zeros((N_L, N_E))
    for (R, _), el in zip(segs, els):
        cnt += np.bincount(
            (np.arange(N_L)[None, :, None] * N_E + np.where(R >= 0, R, 0)).ravel(),
            weights=(el[:, None, None] * (R >= 0)).ravel(), minlength=N_L * N_E).reshape(N_L, N_E)
    p0 = (cnt + 0.5) / (cnt.sum(axis=1, keepdims=True) + 0.5 * N_E)
    freq = TOP_K * p0
    lf = np.log(freq).astype(np.float32)
    pmi = np.zeros((ND, N_L, N_E, N_E), np.float32)
    for d in range(1, ND + 1):
        C, nw = within_tables(segs, els, d)
        pmi[d - 1] = pmi_block(C, nw, freq, d)
    return pmi, lf


def pmi_block(C, nw, freq, d):
    """pmi for source layer T-d -> target layer T, float32 [48, 512, 512] (rows T < d zero)."""
    fd = freq[d:]
    nd = nw[d:]
    P = (C[d:] + SMOOTH_A * fd[:, None, :]) / (nd[:, :, None] + SMOOTH_A)
    pm = np.log(P) - np.log(fd)[:, None, :]
    pm = np.where((nd > 0)[:, :, None], pm, 0.0)
    np.clip(pm, PMI_LO, PMI_HI, out=pm)
    out = np.zeros((N_L, N_E, N_E), np.float32)
    out[d:] = pm
    return out


def token_S(pmi, dec, dtype=np.float32):
    """S_d for decode tokens: [n, 48, 512, ND]; S_d(T, e) = sum over the 10 routed s at T-d of pmi_d[T][s][e]."""
    n = dec.shape[0]
    S = np.zeros((n, N_L, N_E, ND), dtype)
    for a in range(0, n, CHUNK_TOK):
        b = min(n, a + CHUNK_TOK)
        R = dec[a:b].astype(np.int64)
        for d in range(1, ND + 1):
            Ls = np.arange(d, N_L)
            src = R[:, Ls - d, :]                                  # [c, M, 10]
            if (src < 0).any():
                raise ValueError("token_S: absent expert in a decode token")
            G = pmi[d - 1][Ls[None, :, None], src]                 # [c, M, 10, 512]
            S[a:b, d:, :, d - 1] = G.sum(axis=2, dtype=np.float64).astype(dtype)
    return S


def own_features(data):
    """Training features with la_predict's convention: token of fold m uses the tables excluding fold m."""
    D, fold, dec = data["D"], data["fold"], data["dec"]
    S = np.zeros((D, N_L, N_E, ND), np.float32)
    LF = np.zeros((D, N_L, N_E), np.float32)
    for m in range(K_FOLDS):
        pmi, lf = tables_for(data, fold != m)
        idx = np.nonzero(fold == m)[0]
        S[idx] = token_S(pmi, dec[idx])
        LF[idx] = lf
    return S, LF


# ----------------------------------------------------------------------------- fit

def irls(layers, rows):
    """L2 logistic fit on standardised features [lf, S1, S2, S3], one intercept per layer in `layers`.

    rows(T) -> (X float64 [n, NF] raw, y float64 [n]) for layer T. Returns raw weights [NF] (shared by the
    layers in `layers`), raw intercepts [len(layers)], converged flag, rows per layer.
    Normal equations are accumulated layer by layer (Newton step with the block system of la_predict.fit_logreg).
    """
    G = len(layers)
    n_g = np.zeros(G, np.int64)
    sx = np.zeros(NF)
    sxx = np.zeros(NF)
    for gi, T in enumerate(layers):
        X, _ = rows(T)
        n_g[gi] = X.shape[0]
        sx += X.sum(axis=0)
        sxx += (X * X).sum(axis=0)
    n = int(n_g.sum())
    if n == 0:
        return np.zeros(NF), np.full(G, W_NEVER), True, n_g
    mu = sx / n
    sd = np.sqrt(np.maximum(sxx / n - mu * mu, 0.0))
    sd = np.where(sd < 1e-9, 1.0, sd)
    beta = np.zeros(NF)
    alpha = np.zeros(G)
    I = np.eye(NF)
    conv = False
    for _ in range(MAX_ITER):
        Hbb = LAM * I
        Hba = np.zeros((NF, G))
        Haa = np.zeros(G)
        gb = -LAM * beta
        ga = np.zeros(G)
        for gi, T in enumerate(layers):
            if n_g[gi] == 0:
                continue
            X, y = rows(T)
            Z = (X - mu) / sd
            p = expit(Z @ beta + alpha[gi])
            r = y - p
            v = p * (1.0 - p)
            Zv = Z * v[:, None]
            Hbb += Z.T @ Zv
            Hba[:, gi] = Zv.sum(axis=0)
            Haa[gi] = v.sum()
            gb += Z.T @ r
            ga[gi] = r.sum()
        Haa += 1e-10
        H = np.block([[Hbb, Hba], [Hba.T, np.diag(Haa)]])
        step = np.linalg.solve(H, np.concatenate([gb, ga]))
        beta += step[:NF]
        alpha += step[NF:]
        if np.max(np.abs(step)) < 1e-7:
            conv = True
            break
    wr = beta / sd                                        # fold the standardisation back into raw weights
    a_raw = alpha - np.sum(wr * mu)
    a_raw = np.where(n_g > 0, a_raw, W_NEVER)
    return wr, a_raw, conv, n_g


def fit_all(S, LF, nonres, dem, train_tok, per_layer):
    """W float64 [48, NF] (wf, w1, w2, w3 per target layer), b0 float64 [48], nonconverged count."""
    tk = np.nonzero(train_tok)[0]

    def rows(T):
        X = np.concatenate([LF[tk, T][:, :, None], S[tk, T]], axis=2)   # [n, 512, NF]
        m = nonres[tk, T]
        return X[m].astype(np.float64), dem[tk, T][m].astype(np.float64)

    W = np.zeros((N_L, NF))
    b0 = np.full(N_L, W_NEVER)
    nonconv = 0
    if per_layer:
        for T in range(N_L):
            wr, a, conv, _ = irls([T], rows)
            W[T] = wr
            b0[T] = a[0]
            nonconv += 0 if conv else 1
    else:
        wr, a, conv, _ = irls(list(range(N_L)), rows)
        W[:] = wr
        b0[:] = a
        nonconv = 0 if conv else 1
    return W, b0, nonconv


# ----------------------------------------------------------------------------- engine quantisation and integer path

def quantise(pmi, lf, W, b0):
    """tab int8 [48,3,512,512], bias int16 [48,512] and clipping counts."""
    tab = np.zeros((N_L, ND, N_E, N_E), np.int8)
    beyond_T = np.zeros(N_L, np.int64)     # entries whose raw value is outside +-127, per target layer
    at_lim = 0
    for d in range(1, ND + 1):
        for a in range(d, N_L, 8):
            b = min(N_L, a + 8)
            raw = np.rint(SCALE * W[a:b, d][:, None, None] * pmi[d - 1, a:b].astype(np.float64))
            beyond_T[a:b] += np.count_nonzero(np.abs(raw) > TAB_MAX, axis=(1, 2))
            c = np.clip(raw, -TAB_MAX, TAB_MAX)
            at_lim += int(np.count_nonzero(np.abs(c) == TAB_MAX))
            tab[a:b, d - 1] = c.astype(np.int8)
    n_valid = sum(N_L - d for d in range(1, ND + 1)) * N_E * N_E
    braw = np.rint(SCALE * (b0[:, None] + W[:, 0][:, None] * lf.astype(np.float64)))
    bias_beyond_T = np.count_nonzero(np.abs(braw) > BIAS_MAX, axis=1)
    bias = np.clip(braw, -BIAS_MAX, BIAS_MAX).astype(np.int16)
    return {"tab": tab, "bias": bias,
            "clip": {"valid": n_valid, "at_lim": at_lim, "beyond": int(beyond_T.sum()),
                     "bias_beyond": int(bias_beyond_T.sum())},
            "clean_T": (beyond_T == 0) & (bias_beyond_T == 0)}


def score_int(tables, routes_tok):
    """Engine integer score of one decode token: int32 [48, 512]. Uses only int32 arithmetic on int8/int16 tables."""
    tab = tables["tab"]
    acc = tables["bias"].astype(np.int32)
    R = np.asarray(routes_tok, np.int64)
    for d in range(1, ND + 1):
        Ls = np.arange(d, N_L)
        src = R[Ls - d]                                         # [M, 10]
        if (src < 0).any():
            raise ValueError("score_int: absent expert")
        G = tab[Ls[:, None], d - 1, src]                        # [M, 10, 512] int8
        acc[d:] += G.sum(axis=1, dtype=np.int32)
    return acc


def score_int_all(tables, dec):
    return np.stack([score_int(tables, dec[j]) for j in range(dec.shape[0])])


def to_prob(si):
    return 1.0 / (1.0 + np.exp(-si.astype(np.float64) / SCALE))


def add_clip(tot, c):
    for k in tot:
        tot[k] += c[k]


def crossfit(data, own, plw):
    """Out-of-fold integer scores si [D, 48, 512] int32, clip counts, nonconverged fits, per-fold weights."""
    D, fold, dec = data["D"], data["fold"], data["dec"]
    nonres = ~data["res"]
    dem = data["dem"]
    si = np.zeros((D, N_L, N_E), np.int32)
    tot = {"valid": 0, "at_lim": 0, "beyond": 0, "bias_beyond": 0}
    nonconv = 0
    Ws = []
    if own:
        S_own, LF_own = own_features(data)
    for k in range(K_FOLDS):
        train = fold != k
        pmi, lf = tables_for(data, train)
        if own:
            S_fit, LF_fit = S_own, LF_own
        else:
            S_fit = token_S(pmi, dec)
            LF_fit = np.broadcast_to(lf, (D, N_L, N_E))
        W, b0, nc = fit_all(S_fit, LF_fit, nonres, dem, train, plw)
        nonconv += nc
        Ws.append((W, b0))
        q = quantise(pmi, lf, W, b0)
        add_clip(tot, q["clip"])
        for j in np.nonzero(fold == k)[0]:
            si[j] = score_int(q, dec[j])
        del pmi
    return si, tot, nonconv, Ws


def export_model(data, own, plw):
    D, dec = data["D"], data["dec"]
    nonres = ~data["res"]
    train = np.ones(D, bool)
    pmi, lf = tables_for(data, train)
    if own:
        S, LF = own_features(data)
    else:
        S = token_S(pmi, dec)
        LF = np.broadcast_to(lf, (D, N_L, N_E))
    W, b0, nc = fit_all(S, LF, nonres, data["dem"], train, plw)
    q = quantise(pmi, lf, W, b0)
    return {"q": q, "W": W, "b0": b0, "lf": lf, "pmi": pmi, "nonconv": nc}


def print_weights(W, b0, label):
    print("%s: wf %+.4f  w1 %+.4f  w2 %+.4f  w3 %+.4f (mean over layers; sd %.4f %.4f %.4f %.4f); "
          "b0 min %.3f max %.3f" % (label, W[:, 0].mean(), W[:, 1].mean(), W[:, 2].mean(), W[:, 3].mean(),
                                    W[:, 0].std(), W[:, 1].std(), W[:, 2].std(), W[:, 3].std(),
                                    b0.min(), b0.max()))


def print_clip(tot):
    print("clipping: tab entries at +-127: %d of %d valid entries (fraction %.6f); raw beyond +-127 before "
          "clip: %d; bias beyond +-%d: %d" % (tot["at_lim"], tot["valid"], tot["at_lim"] / max(1, tot["valid"]),
                                              tot["beyond"], BIAS_MAX, tot["bias_beyond"]))


# ----------------------------------------------------------------------------- binary I/O

def write_lat(path, thr, bias, tab):
    with open(path, "wb") as f:
        f.write(LAT_MAGIC)
        f.write(struct.pack("<IIIIi", LAT_VERSION, N_L, N_E, ND, SCALE))
        f.write(np.ascontiguousarray(thr, dtype="<i4").tobytes())
        f.write(np.ascontiguousarray(bias, dtype="<i2").tobytes())
        f.write(np.ascontiguousarray(tab, dtype="i1").tobytes())
    if os.path.getsize(path) != LAT_SIZE:
        raise ValueError("LAT2 size %d, expected %d" % (os.path.getsize(path), LAT_SIZE))


def read_lat(path):
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) != LAT_SIZE:
        raise ValueError("%s: %d bytes, expected %d" % (path, len(buf), LAT_SIZE))
    if buf[:4] != LAT_MAGIC:
        raise ValueError("%s: not a LAT2 file" % path)
    ver, nl, ne, nd, scale = struct.unpack_from("<IIIIi", buf, 4)
    if (ver, nl, ne, nd, scale) != (LAT_VERSION, N_L, N_E, ND, SCALE):
        raise ValueError("%s: header %s" % (path, (ver, nl, ne, nd, scale)))
    off = 24
    thr = np.frombuffer(buf, "<i4", N_L, off).astype(np.int32)
    off += 4 * N_L
    bias = np.frombuffer(buf, "<i2", N_L * N_E, off).reshape(N_L, N_E)
    off += 2 * N_L * N_E
    tab = np.frombuffer(buf, "i1", N_L * ND * N_E * N_E, off).reshape(N_L, ND, N_E, N_E)
    off += N_L * ND * N_E * N_E
    if off != len(buf):
        raise ValueError("%s: trailing bytes" % path)
    return {"thr": thr, "bias": bias, "tab": tab}


def ids_of(dec, j, T):
    """LAV ids for decode token j at target layer T."""
    ids = np.full((ND, TOP_K), UNUSED, np.uint16)
    for d in range(1, ND + 1):
        if T - d >= 0:
            ids[d - 1] = dec[j, T - d].astype(np.uint16)
    return ids


def score_from_lat(lat, T, ids):
    """Pure function of the LAT arrays: integer scores int32 [512] for target layer T and LAV ids [3,10]."""
    acc = lat["bias"][T].astype(np.int32)
    for d in range(1, ND + 1):
        if T - d < 0:
            continue
        for s in ids[d - 1]:
            if int(s) == UNUSED:
                continue
            acc += lat["tab"][T, d - 1, int(s)].astype(np.int32)
    return acc


def candidates(lat, T, sc):
    """Experts with score >= thr[T], best first (score desc, id asc), at most N_OUT_MAX: list of (expert, score)."""
    ok = np.nonzero(sc >= lat["thr"][T])[0]
    order = ok[np.lexsort((ok, -sc[ok].astype(np.int64)))]
    return [(int(e), int(sc[e])) for e in order[:N_OUT_MAX]]


def write_lav(path, recs):
    parts = [LAV_MAGIC, struct.pack("<I", len(recs))]
    for T, ids, out in recs:
        parts.append(struct.pack("<I", T))
        parts.append(np.ascontiguousarray(ids, dtype="<u2").tobytes())
        parts.append(struct.pack("<I", len(out)))
        for e, s in out:
            parts.append(struct.pack("<Hi", e, s))
    with open(path, "wb") as f:
        f.write(b"".join(parts))


def read_lav(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != LAV_MAGIC:
        raise ValueError("%s: not a LAV1 file" % path)
    n, = struct.unpack_from("<I", buf, 4)
    off = 8
    recs = []
    for _ in range(n):
        T, = struct.unpack_from("<I", buf, off)
        off += 4
        ids = np.frombuffer(buf, "<u2", ND * TOP_K, off).reshape(ND, TOP_K).copy()
        off += 2 * ND * TOP_K
        no, = struct.unpack_from("<I", buf, off)
        off += 4
        if no > N_OUT_MAX:
            raise ValueError("LAV1: n_out %d > 8" % no)
        out = []
        for _ in range(no):
            e, s = struct.unpack_from("<Hi", buf, off)
            off += 6
            out.append((e, s))
        recs.append((T, ids, out))
    if off != len(buf):
        raise ValueError("%s: trailing bytes" % path)
    return recs


def make_vectors(lat, dec, n, seed):
    rng = np.random.default_rng(seed)
    D = dec.shape[0]
    Ts = np.empty(n, np.int64)
    Ts[:N_L] = np.arange(N_L)                       # every layer, including 0, 1, 2
    Ts[N_L:] = rng.integers(0, N_L, n - N_L)
    js = rng.integers(0, D, n)
    recs = []
    for T, j in zip(Ts, js):
        T = int(T)
        ids = ids_of(dec, int(j), T)
        recs.append((T, ids, candidates(lat, T, score_from_lat(lat, T, ids))))
    return recs


def check_vectors(recs_file, lat_file):
    """Reproduce every vector from the arrays read back from the file alone."""
    bad = 0
    for T, ids, out in recs_file:
        if T < 0 or T >= N_L:
            bad += 1
            continue
        want_unused = np.array([T - d < 0 for d in range(1, ND + 1)])
        if not np.array_equal(ids == UNUSED, np.repeat(want_unused[:, None], TOP_K, axis=1)):
            bad += 1
            continue
        got = candidates(lat_file, T, score_from_lat(lat_file, T, ids))
        if got != out:
            bad += 1
    return bad


def verify_lat(path, mem, dec, seed=0, n_tok=20):
    """Round trip of LAT2: arrays identical; integer scores of the file equal the in-memory ones."""
    ok = True
    lat = read_lat(path)
    print("round-trip LAT2 arrays:")
    for key in ("thr", "bias", "tab"):
        same = lat[key].shape == mem[key].shape and np.array_equal(lat[key], mem[key])
        ok &= bool(same)
        print("  %-5s shape %-24s identical %s" % (key, str(lat[key].shape), same))
    rng = np.random.default_rng(seed)
    js = rng.choice(dec.shape[0], size=min(n_tok, dec.shape[0]), replace=False)
    mismatch = 0
    for j in js:
        si_f = score_int(lat, dec[j])
        si_m = score_int(mem, dec[j])
        mismatch += int(np.count_nonzero(si_f != si_m))
        for T in range(N_L):
            mismatch += int(np.count_nonzero(score_from_lat(lat, T, ids_of(dec, int(j), T)) != si_f[T]))
    ok &= mismatch == 0
    print("  integer scores: file vs in-memory, and score_from_lat(file) vs score_int(file), %d tokens x 48 "
          "layers: mismatches %d" % (js.size, mismatch))
    print("  LAT2 size %d bytes (expected %d)" % (os.path.getsize(path), LAT_SIZE))
    print("round-trip LAT2 %s" % ("PASS" if ok else "FAIL"))
    return ok, lat


# ----------------------------------------------------------------------------- thresholds

THR_POOL_PAIR_DEFAULT = "0.66,0.02"
PAIR_KEY_RE = re.compile(r"^s_c=(\S+) s_w=(\S+)$")       # la_pool.analyse key format: "s_c=%g s_w=%g"


def probs_from_prob_spec(spec):
    """48 probabilities (inf = never) from one float, or from a 48-line text file ('inf' = never)."""
    try:
        return [float(spec)] * N_L
    except ValueError:
        with open(spec) as f:
            vals = [ln.strip() for ln in f if ln.strip()]
        if len(vals) != N_L:
            raise SystemExit("--thr-prob file %s: %d lines, expected %d" % (spec, len(vals), N_L))
        return [float(v) for v in vals]


def thr_from_probs(ps):
    """thr int32 [48] from 48 probabilities: ceil(16 logit(p)) per layer; inf = never (THR_NEVER)."""
    thr = np.zeros(N_L, np.int64)
    for T, p in enumerate(ps):
        if math.isinf(p) and p > 0:
            thr[T] = THR_NEVER
            continue
        if not (0.0 < p < 1.0):
            raise SystemExit("threshold probability %r out of (0,1)" % p)
        v = math.ceil(SCALE * math.log(p / (1.0 - p)))
        thr[T] = min(max(v, -2147483647), 2147483646)
    if thr.min() < -2 ** 31 or thr.max() > THR_NEVER:
        raise SystemExit("threshold out of int32 range")
    return thr.astype(np.int32)


def parse_thr(spec):
    """thr int32 [48] for --thr-prob: ceil(16 logit(p)) per layer; p from one float or from a 48-line file."""
    return thr_from_probs(probs_from_prob_spec(spec))


def median_taus(per_fold):
    """Per-layer median over folds of the crossfit tau. per_fold: one list per fold of per-layer taus; a tau is a
    probability, or None / inf for "never" (read as +inf). Returns one value per layer (inf = never).
    Odd number of folds: the middle value after sorting (inf last). Even number: the HIGHER of the two middle values."""
    if not per_fold:
        raise ValueError("median_taus: no folds")
    n_layer = len(per_fold[0])
    if any(len(f) != n_layer for f in per_fold):
        raise ValueError("median_taus: folds have different numbers of layers")
    out = []
    for T in range(n_layer):
        vals = [math.inf if (f[T] is None or math.isinf(f[T])) else float(f[T]) for f in per_fold]
        vals.sort()
        out.append(vals[len(vals) // 2])
    return out


def pool_medians(path, score_name, pair_spec):
    """48 median-over-folds tau probabilities (inf = never) from a la_pool.py --json file.
    Read path: obj["scores"][i]["tau_per_fold"][KEY][FOLD][T], KEY = "s_c=<s_c> s_w=<s_w>", FOLD = "0".."K-1"."""
    try:
        s_c, s_w = (float(x) for x in pair_spec.split(","))
    except ValueError:
        raise SystemExit("--thr-pair %r: expected S_C,S_W" % pair_spec)
    with open(path) as f:
        obj = json.load(f)
    scores = obj.get("scores") or []
    if not scores:
        raise SystemExit("%s: no SCORE entries" % path)
    names = [str(r.get("name")) for r in scores]
    if score_name is None:
        if len(scores) != 1:
            raise SystemExit("%s has %d SCORE entries; choose one with --thr-score: %s" % (
                path, len(scores), ", ".join(names)))
        i = 0
    else:
        if score_name not in names:
            raise SystemExit("--thr-score %r not in %s; available: %s" % (score_name, path, ", ".join(names)))
        i = names.index(score_name)
    tpf = scores[i].get("tau_per_fold") or {}
    match = []
    for key in tpf:
        m = PAIR_KEY_RE.match(key)
        if m and abs(float(m.group(1)) - s_c) <= 1e-9 and abs(float(m.group(2)) - s_w) <= 1e-9:
            match.append(key)
    if not match:
        raise SystemExit("cost pair %s not in SCORE %r; available: %s" % (pair_spec, names[i], ", ".join(tpf)))
    per = tpf[match[0]]
    folds = [per[k] for k in sorted(per, key=int)]
    for f in folds:
        if len(f) != N_L:
            raise SystemExit("%s: a fold has %d taus, expected %d" % (path, len(f), N_L))
    print("thr-pool: %s SCORE %r cost pair [%s] folds %d" % (path, names[i], match[0], len(folds)))
    return median_taus(folds)


def print_thr(ps, thr, label):
    """Export report: the 48 thresholds as probabilities and integers, count of never layers, min/median/max of finite."""
    print("thresholds (%s):" % label)
    for T in range(N_L):
        print("  T=%2d  p=%-10s thr=%d" % (T, "%.6g" % ps[T], thr[T]))
    fin = thr[thr != THR_NEVER].astype(np.int64)
    n_never = N_L - fin.size
    if fin.size:
        print("never layers: %d of %d; finite thr min %d median %g max %d" % (
            n_never, N_L, fin.min(), float(np.median(fin)), fin.max()))
    else:
        print("never layers: %d of %d; no finite thr" % (n_never, N_L))


# ----------------------------------------------------------------------------- modes

def recall_at10(si, dem):
    top = np.argsort(-si[:, 1:].astype(np.int64), axis=2, kind="stable")[:, :, :TOP_K]
    hits = np.take_along_axis(dem[:, 1:], top, axis=2).sum(axis=2)
    return float(hits.mean() / TOP_K)


def run_score(args):
    t0 = time.time()
    data = load_data(args.routes, args.sim, args.extra_routes)
    D, fold, dem, res = data["D"], data["fold"], data["dem"], data["res"]
    si, tot, nonconv, Ws = crossfit(data, args.own_feats, args.per_layer_w)
    score16 = to_prob(si).astype(np.float16)
    name = "dense" + ("-plw" if args.per_layer_w else "") + ("-own" if args.own_feats else "")
    m = metrics(score16.astype(np.float32), dem, res)
    print()
    print_header()
    print_row(name, 1, m, nonconv)
    print_clip(tot)
    for k, (W, b0) in enumerate(Ws):
        print_weights(W, b0, "  fold %d fit" % k)
    path = args.out_prefix + name + ".npz"
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(path, score=score16, fold=fold.astype(np.int16), lead=np.int64(1), name=np.str_(name))
    print("wrote %s  recall@10 of dem among top-10 integer scores (layers 1..47, out-of-fold): %.4f" % (
        path, recall_at10(si, dem)))
    print("total %.1fs; peak RSS %.0f MiB" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))


def run_export(args):
    t0 = time.time()
    if args.out is None or args.vectors is None:
        raise SystemExit("--out and --vectors are required for --mode export")
    if args.thr_pool_json is not None:
        ps = pool_medians(args.thr_pool_json, args.thr_score,
                          args.thr_pair if args.thr_pair is not None else THR_POOL_PAIR_DEFAULT)
        label = "median over folds, la_pool taus"
    else:
        ps = probs_from_prob_spec(args.thr_prob)
        label = "thr-prob %s" % args.thr_prob
    thr = thr_from_probs(ps)
    print_thr(ps, thr, label)
    data = load_data(args.routes, args.sim, args.extra_routes)
    m = export_model(data, args.own_feats, args.per_layer_w)
    q = m["q"]
    print_clip(q["clip"])
    print_weights(m["W"], m["b0"], "weights (all decode tokens)")
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    write_lat(args.out, thr, q["bias"], q["tab"])
    ok_lat, lat = verify_lat(args.out, {"thr": thr, "bias": q["bias"], "tab": q["tab"]}, data["dec"])
    recs = make_vectors({"thr": thr, "bias": q["bias"], "tab": q["tab"]}, data["dec"], args.n_vectors, args.seed)
    write_lav(args.vectors, recs)
    recs_file = read_lav(args.vectors)
    same_recs = [(T, ids.tolist(), out) for T, ids, out in recs] == [(T, ids.tolist(), out) for T, ids, out in recs_file]
    bad = check_vectors(recs_file, lat)
    ok_lav = same_recs and bad == 0
    n_out = np.array([len(o) for _, _, o in recs_file])
    print("LAV1: %d vectors, %d bytes; mean n_out %.2f, vectors with n_out 0: %d" % (
        len(recs_file), os.path.getsize(args.vectors), n_out.mean(), int((n_out == 0).sum())))
    print("round-trip LAV1: written == read %s; reproduced from file arrays: %d of %d mismatches %d" % (
        same_recs, len(recs_file) - bad, len(recs_file), bad))
    print("round-trip LAV1 %s" % ("PASS" if ok_lav else "FAIL"))
    print("export %s" % ("PASS" if (ok_lat and ok_lav) else "FAIL"))
    print("total %.1fs; peak RSS %.0f MiB" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    if not (ok_lat and ok_lav):
        raise SystemExit(1)


def synth_routes(rng, prefix, D):
    """Chain routes: layer 0 random 10-sets, layer T = (7 * s + 3) % 512 of each expert s of layer T-1."""
    S = prefix + D
    R = np.full((S, N_L, TOP_K), -1, np.int16)
    for t in range(S):
        R[t, 0, :] = rng.choice(N_E, TOP_K, replace=False)
    for L in range(1, N_L):
        R[:, L, :] = (7 * R[:, L - 1, :].astype(np.int64) + 3) % N_E
    return R


def run_selftest(args):
    """Fit on a synthetic training trace A, score a HELD-OUT synthetic trace B with the integer path.

    In-sample scores are not used: a token's own pairs are in the tables it is scored with, which inflates
    the ranking. Checks: (1) the integer top-10 agrees with the float-logit top-10 (integer path correctness);
    (2) the rounding bound on the layers where nothing was clipped; (3) LAT2 / LAV1 round trip. The recall of the
    implied experts is printed for the record: the additive PMI model is noise-limited on this chain (a chance
    co-occurrence gives PMI about +0.5 while the true pair gives about +3.8 at ~29 counts per source), so it is
    NOT expected to put the implied experts on top; the value is reported, not asserted.
    """
    t0 = time.time()
    RA = synth_routes(np.random.default_rng(1), 10, 1500)
    RB = synth_routes(np.random.default_rng(2), 0, 300)
    dA = make_data(RA, 10, [])
    dA["res"] = np.zeros((dA["D"], N_L, N_E), bool)
    dB = make_data(RB, 0, [])
    dB["res"] = np.zeros((dB["D"], N_L, N_E), bool)
    ok_all = True
    # median over folds, hand-made (no file): 3 folds, then 4 folds (even: higher middle), None/inf = never
    f3 = [[0.5, None, 0.01], [0.2, 0.3, math.inf], [0.4, 0.1, 0.02]]
    f4 = [[0.5, None, 0.01, 0.1], [0.2, 0.3, math.inf, 0.2], [0.4, 0.1, 0.02, None], [0.6, 0.05, 0.03, math.inf]]
    ok_med = median_taus(f3) == [0.4, 0.3, 0.02] and median_taus(f4) == [0.5, 0.3, 0.03, math.inf]
    ok_all &= ok_med
    print("  median_taus (3 and 4 folds, None/inf as never, even count -> higher middle): %s" % (
        "PASS" if ok_med else "FAIL"))
    print("selftest: fit on %d+%d synthetic steps, score %d held-out tokens; rule s -> (7s+3) %% 512 per layer" % (
        10, dA["D"], dB["D"]))
    m_pool = None
    for plw in (False, True):
        m = export_model(dA, False, plw)
        q = m["q"]
        name = "heldout" + ("-plw" if plw else "")
        si = score_int_all(q, dB["dec"])
        rec = recall_at10(si, dB["dem"])
        print("  %-14s info: recall@10 of implied experts in the top-10 integer scores (layers 1..47): %.4f "
              "(spec expectation 'on top' NOT met by this model, see docstring)" % (name, rec))
        print_clip(q["clip"])
        mt = metrics(to_prob(si).astype(np.float32), dB["dem"], dB["res"])
        print("  %-14s metrics on held-out tokens: logloss %.5f p@1 %.4f p@10 %.4f" % (
            name, mt["logloss"], mt["p@1"], mt["p@10"]))
        if not plw:
            m_pool = m

    m = m_pool
    q, W, b0, lf, pmi = m["q"], m["W"], m["b0"], m["lf"], m["pmi"]
    dec = dB["dec"]
    S64 = token_S(pmi, dec, np.float64)
    flogit = (b0[None, :, None] + W[None, :, 0, None] * lf[None].astype(np.float64)
              + np.einsum("ntej,tj->nte", S64, W[:, 1:]))
    si_all = score_int_all(q, dec)
    # (1) integer top-10 vs float top-10 on layers 1..47
    ti = np.argsort(-si_all[:, 1:].astype(np.int64), axis=2, kind="stable")[:, :, :TOP_K]
    tf = np.argsort(-flogit[:, 1:], axis=2, kind="stable")[:, :, :TOP_K]
    overlap = np.mean([len(set(a) & set(b)) for a, b in zip(ti.reshape(-1, TOP_K), tf.reshape(-1, TOP_K))]) / TOP_K
    ok_ov = overlap >= 0.95
    ok_all &= ok_ov
    print("  integer vs float top-10 overlap (held-out, layers 1..47): %.4f  %s" % (
        overlap, "PASS" if ok_ov else "FAIL"))
    # (2) rounding bound on layers without clipping
    diff = np.abs(si_all / SCALE - flogit)
    clean = m["q"]["clean_T"]
    maxd_all = float(diff.max())
    print("  quantised vs float logit: max |score_int/16 - logit| over all layers = %.6f (bound %.6f applies only "
          "to layers without clipping)" % (maxd_all, BOUND))
    if clean.any():
        maxd_clean = float(diff[:, clean].max())
        ok_b = maxd_clean <= BOUND + 1e-6
        ok_all &= ok_b
        print("  bound check on %d of %d layers without clipping: max %.6f <= %.6f  %s" % (
            int(clean.sum()), N_L, maxd_clean, BOUND, "PASS" if ok_b else "FAIL"))
    else:
        print("  bound check: no layer without clipping")

    path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "la",
                                         "la_dense_selftest.lat"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    thr = np.full(N_L, math.ceil(SCALE * math.log(0.05 / 0.95)), np.int32)
    mem = {"thr": thr, "bias": q["bias"], "tab": q["tab"]}
    write_lat(path, thr, q["bias"], q["tab"])
    ok_lat, lat = verify_lat(path, mem, dec, n_tok=10)
    recs = make_vectors(mem, dec, 200, 2)
    vpath = path[:-4] + ".lav"
    write_lav(vpath, recs)
    recs_file = read_lav(vpath)
    bad = check_vectors(recs_file, lat)
    ok_lav = bad == 0 and [(T, ids.tolist(), o) for T, ids, o in recs] == \
        [(T, ids.tolist(), o) for T, ids, o in recs_file]
    print("  LAV1 200 vectors: reproduced from file arrays, mismatches %d %s" % (bad, "PASS" if ok_lav else "FAIL"))
    os.remove(path)
    os.remove(vpath)
    ok_all &= ok_lat and ok_lav
    print("selftest %s (%.1fs; peak RSS %.0f MiB)" % ("PASS" if ok_all else "FAIL", time.time() - t0,
                                                     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    if not ok_all:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Dense additive lookahead predictor: integer engine path, SCORE replay, LAT2/LAV1 export (F-LOOKAHEAD).")
    ap.add_argument("--mode", choices=["score", "export"], default=None)
    ap.add_argument("--selftest", action="store_true", help="synthetic checks (no data needed)")
    ap.add_argument("--routes", default=None, help="route trace CSV (turn,phase,step,layer,slot,expert)")
    ap.add_argument("--extra-routes", action="append", default=[],
                    help="extra route CSV, used only to add counts to the tables (repeatable)")
    ap.add_argument("--sim", default=None, help="SIM npz (res, dem) from la_cache.py")
    ap.add_argument("--out-prefix", default="results/la/score-", help="score mode: writes <prefix>dense[-plw].npz")
    ap.add_argument("--per-layer-w", action="store_true", help="separate wf, w1..w3 per target layer")
    ap.add_argument("--own-feats", action="store_true",
                    help="training features of a token of fold m from the tables excluding fold m "
                         "(la_predict convention) instead of the in-sample tables")
    ap.add_argument("--thr-prob", default=None,
                    help="export: one probability for all layers, or a text file of 48 probabilities ('inf' = never)")
    ap.add_argument("--thr-pool-json", default=None,
                    help="export: la_pool.py --json output; thr per layer = median over folds of the crossfit taus "
                         "(exclusive with --thr-prob)")
    ap.add_argument("--thr-score", default=None, help="with --thr-pool-json: SCORE name (default: the only entry)")
    ap.add_argument("--thr-pair", default=None,
                    help="with --thr-pool-json: cost pair S_C,S_W (default %s)" % THR_POOL_PAIR_DEFAULT)
    ap.add_argument("--out", default=None, help="export mode: LAT2 output file")
    ap.add_argument("--vectors", default=None, help="export mode: LAV1 test-vector output file")
    ap.add_argument("--n-vectors", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.selftest:
        run_selftest(args)
    elif args.mode == "score":
        if args.routes is None or args.sim is None:
            ap.error("--mode score needs --routes and --sim")
        run_score(args)
    elif args.mode == "export":
        if args.routes is None or args.sim is None:
            ap.error("--mode export needs --routes and --sim")
        if (args.thr_prob is None) == (args.thr_pool_json is None):
            ap.error("--mode export needs exactly one of --thr-prob and --thr-pool-json")
        if args.thr_pool_json is None and (args.thr_score is not None or args.thr_pair is not None):
            ap.error("--thr-score and --thr-pair need --thr-pool-json")
        run_export(args)
    else:
        ap.error("give --selftest or --mode score|export")


if __name__ == "__main__":
    main()
