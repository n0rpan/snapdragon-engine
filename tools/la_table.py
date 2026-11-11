#!/usr/bin/env python3
"""Engine-side "table path" lookahead predictor: python reference, replay, and binary export (F-LOOKAHEAD).

Question: when the routes of layers L-1, L-2, L-3 of the current token are known, which NON-RESIDENT
experts of layer L should be pre-read? The engine can afford only a small sparse table and a few adds.

Table path, target layer L, candidate expert e (decode token with routes R[L-d] = its 10 experts):
  S_d(e) = sum over the 10 experts e' routed at layer L-d of THIS token of P_d[L][e'][e],  d = 1, 2, 3
  P_d[L][e'] is SPARSE: only the top-K successors e of each source e' are stored (ties: lower id first).
  A pair that is not stored contributes 0. A source e' never seen in the tables gets an empty list.
  x_d(e) = ln(1e-6 + S_d(e) / 10)     if L - d >= 0,     else x_d(e) = lf[L][e]
  lf[L][e] = ln of the smoothed frequency (the f0 of la_predict, cross-fitted per fold in --mode score)
  logit(e) = w0[L] + wf[L] lf[L][e] + w1[L] x_1(e) + w2[L] x_2(e) + w3[L] x_3(e)   (raw features)
  listed(e) = S_1 + S_2 + S_3 > 0 (e appears in at least one stored successor list); score = sigmoid(logit)
  for listed experts, 0 otherwise. Layer 0 never lists anything (no L-d >= 0).
  P_d[L][e'][e] = (C[e', e] + a freq[L][e]) / (n[L-d][e'] + a), a = 5, counts from la_predict's
  within-token tables (same eligibility, same cross-fitting). Sums are accumulated in float64 and
  rounded to float32; table values and weights are float32.

Weights: one logistic regression per layer (la_predict.fit_logreg: L2 on the standardised weights,
LAM = 1), fitted on the LISTED non-resident candidates of the training tokens, label dem, with features
computed through the same truncated table. --variant pooled shares wf, w1..w3 across layers with a
per-layer intercept. Standardised weights are folded back into raw weights.

Modes:
  --mode score   cross-fitted over the 5 folds of la_predict (fold k: tables and weights from decode
                 tokens not in fold k, plus prefill and --extra-routes; scores fold k). Writes SCORE files
                 (score float16 [D,48,512], fold, lead = 1, name table-k<K>[-pooled]) to
                 <out-prefix>k<K>[-pooled].npz and prints la_predict's metric row, plus the mean number of
                 listed candidates per (token, layer) and of listed non-resident candidates.
                 --variant layer|pooled|both (both fits both weight variants on the same tables).
  --mode export  tables and weights from ALL decode tokens (+ prefill + extra); writes the binary
                 LAT1 file (--out), re-reads it, and checks: every array round-trips exactly, and
                 score_from_lat(file arrays) equals the in-memory table path on 20 random decode tokens
                 to 1e-6. Prints file size and PASS/FAIL. --tau P sets tau_logit = logit(P) for all layers.
  --selftest     synthetic chain routes (layer L expert = (7e+3) % 512 of layer L-1): recall of the implied
                 experts, naive per-token reference vs vectorised scoring, K = 512 sums vs dense sums,
                 export/read round trip.

LAT1 binary, little-endian: magic b"LAT1"; u32 version = 1; u32 n_layer = 48; u32 n_expert = 512;
u32 n_dist = 3; u32 K; f32 w0[48]; f32 wf[48]; f32 w[48][3] (w1, w2, w3); f32 tau_logit[48];
f32 lf[48][512]; u16 succ_id[48][3][512][K] (unused = 65535); f32 succ_p[48][3][512][K] (unused = 0).
Layer index = TARGET layer L; succ_*[L][d-1][e'] lists successors at layer L of source e' at layer L-d.

Not implemented: --taus-json. The la_net JSON stores tuned taus only per (score, s_c, s_w, stage row),
not as an unambiguous per-layer tau vector, so export takes one probability threshold via --tau.

Usage (numpy only; wrapped in the memory limit; the build slot is held for the whole run):
  ( exec 9>"${LOCAL_AI_BUILD_LOCK:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/build2.lock}"; flock -n 9 || flock -w 600 9 || exit 99
    systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=4 \
      OMP_NUM_THREADS=4 nice -n 19 ~/.cache/local-ai/venv-lookahead/bin/python -I tools/la_table.py \
      --selftest )
  ... -I tools/la_table.py --mode score --K 16 --variant both --routes ROUTES.csv --sim SIM.npz
  ... -I tools/la_table.py --mode export --K 16 --tau 0.1 --routes ROUTES.csv --sim SIM.npz \
      --out results/la/lookahead-fixed353-k16.lat
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
from lookahead_train import N_E, N_L, TOP_K, read_routes  # noqa: E402
import la_predict as lp  # noqa: E402

K_FOLDS = lp.K_FOLDS
ND = 3
EPS = 1e-6
UNUSED_ID = 65535
W_NEVER = -30.0   # intercept of a layer with no training candidates (scores ~ 0)
CHUNK = 64
MAGIC = b"LAT1"
VERSION = 1
NF = 4            # features: lf, x1, x2, x3
LAT_ARRAYS = ("w0", "wf", "w", "tau_logit", "lf", "sid", "sp")


def expit32(x):
    x = np.clip(np.asarray(x, np.float32), -80.0, 80.0)  # saturated in float32 beyond this
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def chunks(n, size):
    for a in range(0, n, size):
        yield a, min(n, a + size)


# ----------------------------------------------------------------------------- tables

def build_tables(segs, els, K, keep_dense=False):
    """Sparse table (and optionally the dense P_d) from route segments.

    segs: list of (R, _) with R int16 [S,48,10] (-1 absent); els: per-step eligibility weights.
    Returns T = {K, lf f32 [48,512], sid u16 [48,3,512,K], sp f32 [48,3,512,K]} and, if keep_dense,
    the list of dense P_d float32 [48,512,512] (rows of unseen sources and L < d are zero).
    """
    cnt = np.zeros((N_L, N_E))
    for (R, _), el in zip(segs, els):
        cnt += np.bincount(
            (np.arange(N_L)[None, :, None] * N_E + np.where(R >= 0, R, 0)).ravel(),
            weights=(el[:, None, None] * (R >= 0)).ravel(), minlength=N_L * N_E).reshape(N_L, N_E)
    p0 = (cnt + 0.5) / (cnt.sum(axis=1, keepdims=True) + 0.5 * N_E)
    freq = TOP_K * p0
    T = {"K": K,
         "lf": np.log(p0).astype(np.float32),
         "sid": np.full((N_L, ND, N_E, K), UNUSED_ID, np.uint16),
         "sp": np.zeros((N_L, ND, N_E, K), np.float32)}
    dense = []
    for d in range(1, ND + 1):
        C, nw = lp.within_tables(segs, els, d)
        P = lp.smoothed(C[d:], nw[d:], freq[d:])                     # [48-d, 512, 512] f32
        order = np.argsort(-P, axis=2, kind="stable")[:, :, :K]     # ties: lower id first
        vals = np.take_along_axis(P, order, axis=2)
        empty = nw[d:] <= 0
        ids = order.astype(np.uint16)
        ids[empty] = UNUSED_ID
        vals[empty] = 0.0
        T["sid"][d:, d - 1] = ids
        T["sp"][d:, d - 1] = vals
        if keep_dense:
            Pf = np.zeros((N_L, N_E, N_E), np.float32)
            Pf[d:] = np.where(empty[:, :, None], np.float32(0), P)
            dense.append(Pf)
    return (T, dense) if keep_dense else T


# ----------------------------------------------------------------------------- engine scoring

def table_feats(T, R):
    """Table path features for token routes R int [n,48,10] (-1 absent).

    Returns xs float32 [n,48,512,4] (lf, x1, x2, x3), Sd float32 [n,3,48,512] (S_1..S_3),
    listed bool [n,48,512] (S_1 + S_2 + S_3 > 0).
    """
    R = np.asarray(R)
    n = R.shape[0]
    Sd = np.zeros((n, ND, N_L, N_E), np.float32)
    for d in range(1, ND + 1):
        M = N_L - d
        Ls = np.arange(d, N_L)
        src = R[:, Ls - d, :].astype(np.int64)                          # [n, M, 10]
        ok = src >= 0
        srcc = np.where(ok, src, 0)
        ids = T["sid"][Ls[None, :, None], d - 1, srcc].astype(np.int64)  # [n, M, 10, K]
        pr = T["sp"][Ls[None, :, None], d - 1, srcc]                     # [n, M, 10, K]
        valid = ok[..., None] & (ids != UNUSED_ID)
        base = (np.arange(n)[:, None, None, None] * M + np.arange(M)[None, :, None, None]) * N_E
        flat = (base + ids)[valid]
        acc = np.bincount(flat, weights=pr[valid].astype(np.float64), minlength=n * M * N_E)
        Sd[:, d - 1, d:, :] = acc.reshape(n, M, N_E)
    listed = Sd.sum(axis=1) > 0
    lf = T["lf"]
    xs = np.empty((n, N_L, N_E, NF), np.float32)
    xs[..., 0] = lf[None]
    for d in range(1, ND + 1):
        x = np.log(np.float32(EPS) + Sd[:, d - 1] / np.float32(10.0)).astype(np.float32)
        x[:, :d] = lf[None, :d]
        xs[..., d] = x
    return xs, listed, Sd


def apply_model(W, xs, listed):
    """Score = sigmoid(logit) on listed candidates, 0 otherwise. W: w0 [48], wf [48], w [48,3]."""
    w = W["w"]
    lg = (W["w0"][:, None] + W["wf"][:, None] * xs[..., 0]
          + w[:, 0, None] * xs[..., 1] + w[:, 1, None] * xs[..., 2] + w[:, 2, None] * xs[..., 3])
    sc = expit32(lg.astype(np.float32))
    return np.where(listed, sc, np.float32(0)).astype(np.float32)


def score_from_lat(lat, routes):
    """Pure function of the LAT arrays (dict with w0, wf, w, lf, sid, sp). routes int [n,48,10] or [48,10].

    Returns (score float32 [n,48,512], listed bool [n,48,512]).
    """
    R = np.asarray(routes)
    if R.ndim == 2:
        R = R[None]
    xs, listed, _ = table_feats(lat, R)
    return apply_model(lat, xs, listed), listed


def naive_score_token(T, route):
    """Slow per-token loop over the spec formula (selftest reference). route int [48,10]."""
    K = T["K"]
    S = np.zeros((ND, N_L, N_E), np.float64)
    for L in range(N_L):
        for d in range(1, ND + 1):
            if L - d < 0:
                continue
            for e_src in route[L - d]:
                if e_src < 0:
                    continue
                for k in range(K):
                    i = int(T["sid"][L, d - 1, e_src, k])
                    if i == UNUSED_ID:
                        continue
                    S[d - 1, L, i] += float(T["sp"][L, d - 1, e_src, k])
    out = np.zeros((N_L, N_E), np.float64)
    for L in range(N_L):
        for e in range(N_E):
            tot = S[:, L, e].sum()
            if tot <= 0:
                continue
            x = [float(T["lf"][L, e])]
            for d in range(1, ND + 1):
                if L - d < 0:
                    x.append(float(T["lf"][L, e]))
                else:
                    x.append(np.log(EPS + S[d - 1, L, e] / 10.0))
            lg = float(T["w0"][L]) + sum(float(c) * xi for c, xi in
                                         zip([T["wf"][L], T["w"][L, 0], T["w"][L, 1], T["w"][L, 2]], x))
            out[L, e] = 1.0 / (1.0 + np.exp(-lg))
    return out


def dense_sums(Pfull, R):
    """Dense S_d [n,3,48,512] from dense P_d (list of [48,512,512]) for routes R [n,48,10]."""
    n = R.shape[0]
    out = np.zeros((n, ND, N_L, N_E), np.float64)
    for d in range(1, ND + 1):
        Ls = np.arange(d, N_L)
        src = R[:, Ls - d, :].astype(np.int64)
        ok = src >= 0
        G = Pfull[d - 1][Ls[None, :, None], np.where(ok, src, 0)]     # [n, M, 10, 512]
        out[:, d - 1, d:] = np.where(ok[..., None], G, 0.0).sum(axis=2)
    return out


# ----------------------------------------------------------------------------- weights

def _fit_group(X, y, counts):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    beta, alpha, _, conv = lp.fit_logreg(Z, y, counts)
    return mu, sd, beta, alpha, conv


def fit_weights(Lr, X, y, pooled):
    """Logistic fits on listed non-resident candidates. Returns raw weights dict (float32) and #nonconv."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    w0 = np.full(N_L, W_NEVER, np.float64)
    wr_all = np.zeros((N_L, NF))
    nonconv = 0
    if not pooled:
        for L in range(N_L):
            m = Lr == L
            if not m.any():
                continue
            mu, sd, beta, alpha, conv = _fit_group(X[m], y[m], [int(m.sum())])
            nonconv += 0 if conv else 1
            wr = beta / sd
            wr_all[L] = wr
            w0[L] = alpha[0] - np.sum(wr * mu)
    else:
        o = np.argsort(Lr, kind="stable")
        Ls, Xs, ys = Lr[o], X[o], y[o]
        cnt_all = np.bincount(Ls, minlength=N_L)
        groups = np.nonzero(cnt_all)[0]
        mu, sd, beta, alpha, conv = _fit_group(Xs, ys, cnt_all[groups])
        nonconv += 0 if conv else 1
        wr = beta / sd
        for gi, L in enumerate(groups):
            wr_all[L] = wr
            w0[L] = alpha[gi] - np.sum(wr * mu)
    W = {"w0": w0.astype(np.float32), "wf": wr_all[:, 0].astype(np.float32),
         "w": wr_all[:, 1:].astype(np.float32)}
    return W, nonconv


def collect_rows(T, dec, nonres_dec, dem_dec, train_tok):
    """Listed non-resident candidates of training tokens: (layer, X [n,4], y)."""
    Lr, Xr, yr = [], [], []
    for a, b in chunks(dec.shape[0], CHUNK):
        xs, listed, _ = table_feats(T, dec[a:b])
        m = listed & nonres_dec[a:b] & train_tok[a:b, None, None]
        i, L, e = np.nonzero(m)
        Lr.append(L)
        Xr.append(xs[i, L, e])
        yr.append(dem_dec[a + i, L, e])
    return np.concatenate(Lr), np.concatenate(Xr), np.concatenate(yr)


# ----------------------------------------------------------------------------- binary I/O

def write_lat(path, T):
    K = T["K"]
    shapes = {"w0": (N_L,), "wf": (N_L,), "w": (N_L, ND), "tau_logit": (N_L,), "lf": (N_L, N_E),
              "sid": (N_L, ND, N_E, K), "sp": (N_L, ND, N_E, K)}
    dtypes = {"w0": "<f4", "wf": "<f4", "w": "<f4", "tau_logit": "<f4", "lf": "<f4", "sid": "<u2", "sp": "<f4"}
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5I", VERSION, N_L, N_E, ND, K))
        for name in LAT_ARRAYS:
            a = np.ascontiguousarray(T[name], dtype=dtypes[name])
            if a.shape != shapes[name]:
                raise ValueError("%s: shape %s, expected %s" % (name, a.shape, shapes[name]))
            f.write(a.tobytes())


def read_lat(path):
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != MAGIC:
        raise ValueError("%s: not a LAT1 file" % path)
    if len(buf) < 24:
        raise ValueError("%s: truncated header" % path)
    ver, nl, ne, nd, K = struct.unpack_from("<5I", buf, 4)
    if (ver, nl, ne, nd) != (VERSION, N_L, N_E, ND):
        raise ValueError("%s: version/dims %s, expected %s" % (path, (ver, nl, ne, nd), (VERSION, N_L, N_E, ND)))
    off = 24
    lat = {"K": int(K)}

    def take(name, dtype, shape):
        nonlocal off
        cnt = int(np.prod(shape))
        arr = np.frombuffer(buf, dtype=dtype, count=cnt, offset=off).reshape(shape)
        off += cnt * np.dtype(dtype).itemsize
        lat[name] = arr

    take("w0", "<f4", (N_L,))
    take("wf", "<f4", (N_L,))
    take("w", "<f4", (N_L, ND))
    take("tau_logit", "<f4", (N_L,))
    take("lf", "<f4", (N_L, N_E))
    take("sid", "<u2", (N_L, ND, N_E, K))
    take("sp", "<f4", (N_L, ND, N_E, K))
    if off != len(buf):
        raise ValueError("%s: %d trailing bytes" % (path, len(buf) - off))
    return lat


def verify_roundtrip(path, Tm, dec, n_tokens=20, seed=0):
    lat = read_lat(path)
    ok = True
    print("round-trip: array check")
    for key in LAT_ARRAYS:
        same = lat[key].shape == np.asarray(Tm[key]).shape and np.array_equal(
            lat[key].astype(np.float64), np.asarray(Tm[key]).astype(np.float64))
        ok &= bool(same)
        print("  %-10s shape %-24s identical %s" % (key, str(lat[key].shape), same))
    ok &= lat["K"] == Tm["K"]
    rng = np.random.default_rng(seed)
    js = rng.choice(dec.shape[0], size=min(n_tokens, dec.shape[0]), replace=False)
    s_file, l_file = score_from_lat(lat, dec[js])
    s_mem, l_mem = score_from_lat(Tm, dec[js])
    maxdiff = float(np.max(np.abs(s_file.astype(np.float64) - s_mem.astype(np.float64))))
    same_listed = bool(np.array_equal(l_file, l_mem))
    ok &= maxdiff <= 1e-6 and same_listed
    print("  score_from_lat(file) vs in-memory table path on %d random decode tokens: max |diff| = %.3e, "
          "listed sets equal %s, mean listed per (token,layer) = %.2f" % (
              js.size, maxdiff, same_listed, l_file.sum(axis=2).mean()))
    print("round-trip %s" % ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------------- data

def load_data(routes_path, sim_path, extra_paths):
    routes, prefix, S = read_routes(routes_path)
    D = S - prefix
    if D < K_FOLDS:
        raise SystemExit("need at least %d decode tokens" % K_FOLDS)
    fold = (np.arange(D) * K_FOLDS) // D
    extra = [read_routes(p)[0] for p in extra_paths]
    dem = lp.usage(routes)[prefix:].astype(bool)
    if sim_path is None:
        raise SystemExit("--sim is required for this mode")
    z = np.load(sim_path)
    if z["res"].shape != (D, N_L, N_E) or int(z["prefix"]) != prefix or int(z["D"]) != D:
        raise SystemExit("SIM file shape/prefix/D does not match the routes")
    res = z["res"].astype(bool)
    if np.any(z["dem"].astype(bool) != dem):
        raise SystemExit("sim dem does not match the route trace")
    print("trace: %s  prefix=%d D=%d S=%d extra_traces=%d" % (routes_path, prefix, D, S, len(extra)))
    print("residency: SIM file %s (policy=%s, quota_scheme=%s, cache_gib=%s); dem check OK" % (
        sim_path, z["policy"], z["quota_scheme"], z["cache_gib"]))
    return {"routes": routes, "prefix": prefix, "S": S, "D": D, "fold": fold, "extra": extra,
            "dem": dem, "res": res, "dec": routes[prefix:],
            "segs": [(routes, None)] + [(R, None) for R in extra]}


def els_for(data, train_dec):
    els = [lp.step_elig(data["prefix"], data["S"], train_dec)]
    return els + [np.ones(R.shape[0], np.float64) for R in data["extra"]]


# ----------------------------------------------------------------------------- modes

def run_score(args):
    t0 = time.time()
    data = load_data(args.routes, args.sim, args.extra_routes)
    K, D, fold, dem, res = args.K, data["D"], data["fold"], data["dem"], data["res"]
    prefix = data["prefix"]
    nonres_dec = ~res  # SIM res is already decode-only [D,48,512]
    variants = ["layer", "pooled"] if args.variant == "both" else [args.variant]
    out = {v: np.zeros((D, N_L, N_E), np.float32) for v in variants}
    listed_all = np.zeros((D, N_L, N_E), bool)
    nonconv = dict.fromkeys(variants, 0)
    wsum = {v: [] for v in variants}
    for k in range(K_FOLDS):
        tk = time.time()
        train_tok = fold != k
        T = build_tables(data["segs"], els_for(data, train_tok), K)
        Lr, X, y = collect_rows(T, data["dec"], nonres_dec, dem, train_tok)
        js = np.nonzero(~train_tok)[0]
        tests = []
        for a, b in chunks(js.size, CHUNK):
            jj = js[a:b]
            xs, listed, _ = table_feats(T, data["dec"][jj])
            tests.append((jj, xs, listed))
            listed_all[jj] = listed
        for v in variants:
            W, nc = fit_weights(Lr, X, y, pooled=(v == "pooled"))
            nonconv[v] += nc
            wsum[v].append(np.concatenate([W["wf"][:, None], W["w"]], axis=1))
            for jj, xs, listed in tests:
                out[v][jj] = apply_model(W, xs, listed)
        print("fold %d: training candidates %d (positives %d); %.1fs" % (
            k, Lr.size, int(y.sum()), time.time() - tk))
    print()
    lp.print_header()
    for v in variants:
        name = "table-k%d" % K + ("-pooled" if v == "pooled" else "")
        score16 = np.clip(out[v], 0.0, 1.0).astype(np.float16)
        m = lp.metrics(score16.astype(np.float32), dem, res)
        lp.print_row(name, 1, m, nonconv[v])
        ws = np.mean(np.stack(wsum[v]), axis=0)  # [48, 4] mean over folds
        print("  %s: mean over folds and layers of raw weights wf %+.4f w1 %+.4f w2 %+.4f w3 %+.4f" % (
            name, ws[:, 0].mean(), ws[:, 1].mean(), ws[:, 2].mean(), ws[:, 3].mean()))
        nr = ~res
        lst = listed_all.sum(axis=2)
        lstn = (listed_all & nr).sum(axis=2)
        nr_dem = int((dem & nr).sum())
        cov = int((listed_all & nr & dem).sum())
        print("  %s: listed per (token,layer) mean %.2f (layers 1..47: %.2f); listed non-resident per "
              "(token,layer) mean %.2f; demanded non-resident experts listed: %d of %d (%.3f)" % (
                  name, lst.mean(), lst[:, 1:].mean(), lstn.mean(), cov, nr_dem,
                  cov / max(1, nr_dem)))
        path = args.out_prefix + "k%d%s.npz" % (K, "-pooled" if v == "pooled" else "")
        outdir = os.path.dirname(path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        np.savez_compressed(path, score=score16, fold=fold.astype(np.int16), lead=np.int64(1),
                            name=np.str_(name))
        print("  wrote %s" % path)
    print("total %.1fs" % (time.time() - t0))


def run_export(args):
    t0 = time.time()
    if args.out is None:
        raise SystemExit("--out is required for --mode export")
    if args.tau is None or not (0.0 < args.tau < 1.0):
        raise SystemExit("--tau P (0 < P < 1) is required for --mode export")
    if args.variant == "both":
        raise SystemExit("--variant must be layer or pooled for export")
    data = load_data(args.routes, args.sim, args.extra_routes)
    D, prefix = data["D"], data["prefix"]
    nonres_dec = ~data["res"]
    T = build_tables(data["segs"], els_for(data, np.ones(D, bool)), args.K)
    Lr, X, y = collect_rows(T, data["dec"], nonres_dec, data["dem"], np.ones(D, bool))
    W, nc = fit_weights(Lr, X, y, pooled=(args.variant == "pooled"))
    tau_logit = np.float64(np.log(args.tau / (1.0 - args.tau)))
    Tm = {"K": args.K, "w0": W["w0"], "wf": W["wf"], "w": W["w"],
          "tau_logit": np.full(N_L, tau_logit, np.float32), "lf": T["lf"], "sid": T["sid"], "sp": T["sp"]}
    print("fit: variant %s, training candidates %d (positives %d), nonconverged %d, built in %.1fs" % (
        args.variant, Lr.size, int(y.sum()), nc, time.time() - t0))
    print("tau P = %.4f  tau_logit = %.6f (all layers)" % (args.tau, tau_logit))
    print("weights: w0 min %.3f max %.3f | wf mean %.4f | w1 %.4f w2 %.4f w3 %.4f (means over layers)" % (
        W["w0"].min(), W["w0"].max(), W["wf"].mean(), W["w"][:, 0].mean(), W["w"][:, 1].mean(), W["w"][:, 2].mean()))
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    write_lat(args.out, Tm)
    size = os.path.getsize(args.out)
    expect = 24 + 4 * (N_L + N_L + N_L * ND + N_L + N_L * N_E) + N_L * ND * N_E * args.K * (2 + 4)
    print("wrote %s: %d bytes (expected %d)" % (args.out, size, expect))
    ok = verify_roundtrip(args.out, Tm, data["dec"])
    print("export %s" % ("PASS" if ok and size == expect else "FAIL"))
    print("total %.1fs" % (time.time() - t0))
    if not ok or size != expect:
        raise SystemExit(1)


def naive_check(T, routes_dec, js):
    worst = 0.0
    for j in js:
        ref = naive_score_token(T, routes_dec[j])
        sc, _ = score_from_lat(T, routes_dec[j])
        worst = max(worst, float(np.max(np.abs(sc[0].astype(np.float64) - ref))))
    return worst


def run_selftest(args):
    rng = np.random.default_rng(1)
    S = D = 60
    prefix = 0
    R = np.full((S, N_L, TOP_K), -1, np.int16)
    R[:, 0, :] = np.stack([rng.choice(N_E, TOP_K, replace=False) for _ in range(S)])
    for L in range(1, N_L):
        R[:, L, :] = (7 * R[:, L - 1, :].astype(np.int64) + 3) % N_E
    dem = lp.usage(R).astype(bool)
    res = np.zeros((D, N_L, N_E), bool)
    fold = (np.arange(D) * K_FOLDS) // D
    data = {"routes": R, "prefix": prefix, "S": S, "D": D, "fold": fold, "extra": [], "dem": dem,
            "res": res, "dec": R, "segs": [(R, None)]}
    ok_all = True
    K = args.K
    T = build_tables(data["segs"], els_for(data, np.ones(D, bool)), K)
    Lr, X, y = collect_rows(T, R, ~res, dem, np.ones(D, bool))
    W, nc = fit_weights(Lr, X, y, pooled=(args.variant == "pooled"))
    Tm = {"K": K, "w0": W["w0"], "wf": W["wf"], "w": W["w"], "tau_logit": np.full(N_L, -2.0, np.float32),
          "lf": T["lf"], "sid": T["sid"], "sp": T["sp"]}
    sc, listed = score_from_lat(Tm, R)
    # recall: implied experts (= routes at L, by construction) among the top-10 listed by score, L >= 1
    top = np.argsort(-sc, axis=2, kind="stable")[:, :, :TOP_K]
    hit = np.take_along_axis(dem, top, axis=2)[:, 1:].sum(axis=2) / TOP_K
    recall = float(hit.mean())
    print("selftest: %d steps, K=%d, weights %s; mean listed per (token,layer) %.2f (layers 1..47 %.2f)" % (
        S, K, args.variant, listed.sum(axis=2).mean(), listed[:, 1:].sum(axis=2).mean()))
    print("  recall of implied experts in top-10 listed (layers 1..47): %.4f" % recall)
    r_ok = recall >= 0.95
    ok_all &= r_ok
    print("  recall check %s" % ("PASS" if r_ok else "FAIL"))
    nv = naive_check(Tm, R, [0, 7, 30, 59])
    n_ok = nv <= 1e-5
    ok_all &= n_ok
    print("  naive per-token reference vs vectorised table path: max |diff| = %.3e  %s" % (
        nv, "PASS" if n_ok else "FAIL"))
    T512, dense = build_tables(data["segs"], els_for(data, np.ones(D, bool)), N_E, keep_dense=True)
    xs512, listed512, Sd = table_feats(T512, R[:8])
    Sdense = dense_sums(dense, R[:8])
    sd_diff = float(np.max(np.abs(Sd.astype(np.float64) - Sdense)))
    sd_max = float(np.max(np.abs(Sdense)))
    sd_ok = sd_diff <= 1e-6 * max(1.0, sd_max)
    ok_all &= sd_ok
    print("  K=512 sparse S_1..S_3 vs dense table sums (8 tokens): max |diff| = %.3e (max |S| %.4f), listed "
          "sets equal dense %s  %s" % (sd_diff, sd_max,
                                       bool(np.array_equal(listed512, Sdense.sum(axis=1) > 0)),
                                       "PASS" if sd_ok else "FAIL"))
    path = os.path.join(os.getcwd(), "results", "la", "la_table_selftest.lat")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_lat(path, Tm)
    rt = verify_roundtrip(path, Tm, R, n_tokens=20)
    os.remove(path)
    ok_all &= rt
    print("selftest %s" % ("PASS" if ok_all else "FAIL"))
    if not ok_all:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Engine-side table-path lookahead predictor: replay SCORE files and LAT1 export (F-LOOKAHEAD).")
    ap.add_argument("--mode", choices=["score", "export"], default=None)
    ap.add_argument("--selftest", action="store_true", help="synthetic checks (no data needed)")
    ap.add_argument("--routes", default=None, help="route trace CSV (turn,phase,step,layer,slot,expert)")
    ap.add_argument("--sim", default=None, help="SIM npz (res, dem) from la_cache.py")
    ap.add_argument("--extra-routes", action="append", default=[],
                    help="extra route CSV, used only to add counts to the tables (repeatable)")
    ap.add_argument("--K", type=int, default=16, help="stored successors per (L, d, source expert)")
    ap.add_argument("--variant", choices=["layer", "pooled", "both"], default="layer",
                    help="weights: one fit per layer, shared-weight pooled fit, or both (score mode only)")
    ap.add_argument("--out-prefix", default="results/la/score-fixed353-table-",
                    help="score mode: writes <prefix>k<K>[-pooled].npz")
    ap.add_argument("--out", default=None, help="export mode: LAT1 output file")
    ap.add_argument("--tau", type=float, default=None, help="export mode: probability threshold for all layers")
    args = ap.parse_args()
    if not 1 <= args.K <= N_E:
        ap.error("--K must be in 1..512")
    if args.selftest:
        run_selftest(args)
    elif args.mode == "score":
        if args.routes is None:
            ap.error("--routes is required")
        run_score(args)
    elif args.mode == "export":
        if args.routes is None:
            ap.error("--routes is required")
        run_export(args)
    else:
        ap.error("give --selftest or --mode score|export")


if __name__ == "__main__":
    main()
