#!/usr/bin/env python3
"""NET-TIME evaluator for expert prefetch scores (numpy only; offline).

Reads one SIM file (res/dem/miss from la_cache.py) and one or more SCORE files (score/fold/lead
from la_predict.py). For each SCORE file it prints the net ms/token a prefetcher driven by that
score would save, using per-layer thresholds tau_l (or one global tau).

No staging (default), per token j, layer l:
  staged[j,l,e]  = ~res & (score >= tau_l)
  covered        = staged & miss            (a demand miss that was staged)
  wasted         = staged & ~dem            (staged, not demanded)
  net ms/token   = covered/tok * s_c * g - wasted/tok * s_w
  g              = 1 if lead == 0 else min(1, lead * (decode_ms/48) / read_ms)

Staging pool (--stage-slots M, one block of rows per M): each layer has a staging list of at most M
experts, FIFO by insertion token. Per token j and layer l, in order:
  1. candidates = non-resident experts with score >= tau, not already staged, in descending score;
     insert them one by one (one read each) until M inserts at this token; when the list is full,
     evict the oldest entry, but never one inserted at this same token (then drop the candidate).
     Evicted, never-demanded entries are wasted reads at the token of eviction.
  2. demand: each miss that is staged is covered and leaves the list: "fresh" if inserted at j,
     "kept" if inserted at an earlier token. Covered-fresh weighs g*s_c, covered-kept weighs s_c.
  3. entries still staged at the end of the trace are wasted at the last token. (A staged expert
     that becomes resident is removed without cost; this cannot occur, since only demands make
     experts resident and demanded staged experts were covered in step 2.)
  net ms/token = (cov_fresh*g + cov_kept) * s_c - wasted * s_w.
  Staging memory = M * 48 * 2049280 bytes (slot = TOTAL_BYTES / (48*512)).
The staging state is sequential, so each (layer, tau) is simulated over the WHOLE trace once. The
cross-fit then selects taus on those per-token counts. The state carry-over across fold boundaries
is accepted as an approximation: a fold's tokens inherit staged entries from the preceding tokens,
whatever fold those belong to.

Threshold modes:
  crossfit         headline: per layer, tau maximising net over the tokens NOT in fold k, applied to fold k.
  global           one tau for all layers, cross-fitted the same way.
  oracle-insample  per-layer tau maximising net over all tokens (optimistic upper bound).

Cost grid: --stall-ms (s_c) x --wasted-ms (s_w); thresholds are tuned at each pair. Classic GO pair:
thresholds tuned at the smallest --wasted-ms (0.20), evaluated at that value (net_opt) and at the
largest one (0.82, net_pes) without retuning. GO per score, s_c and staging block (crossfit, classic
pair): net >= 0.03*decode_ms AND net_pes >= 0 AND reads/tok (misses + prefetch reads) <= --max-reads.
With --sim-base SIMBASE.npz, the GO decision and the bootstrap/fold nets use
net_vs_base = net - (misses/tok of --sim - misses/tok of --sim-base) * s_c, because the main SIM
has a smaller cache that leaves room for the staging pool. Otherwise they use the plain net.
Bootstrap: 1000 token resamples (seed 0) of the crossfit per-token net_opt; per-fold nets printed.

Usage:
  ~/.cache/local-ai/venv-lookahead/bin/python -I tools/la_net.py --sim SIM.npz --score A.npz [--score B.npz ...] \
      [--stage-slots 4,8,16] [--sim-base SIMBASE.npz] [--json OUT] [--decode-ms 110] [--read-ms 2.5] \
      [--stall-ms 1.7,0.82] [--wasted-ms 0.20,0.82] [--max-reads 200]
  ~/.cache/local-ai/venv-lookahead/bin/python -I tools/la_net.py --selftest
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

N_L, N_E, TOP_K = 48, 512, 10
TAUS = [None] + [float(x) for x in np.geomspace(0.9, 1e-4, 40)]  # index 0 = never prefetch
T_N = len(TAUS)
MODES = ("crossfit", "global", "oracle-insample")
BOOT_N = 1000
GO_FRAC = 0.03
NAN = float("nan")
TOTAL_BYTES = 50363105280
SLOT_BYTES = TOTAL_BYTES // (N_L * N_E)  # 2049280


def lead_factor(lead, decode_ms, read_ms):
    if lead == 0:
        return 1.0
    return min(1.0, lead * (decode_ms / N_L) / read_ms)


def load_sim(src):
    if isinstance(src, dict):
        return dict(src)
    z = np.load(src)
    sim = {k: z[k] for k in ("res", "dem", "miss")}
    sim["D"] = int(z["D"]) if "D" in z.files else sim["res"].shape[0]
    sim["prefix"] = int(z["prefix"]) if "prefix" in z.files else -1
    for k in ("policy", "quota_scheme"):
        sim[k] = str(z[k]) if k in z.files else "?"
    sim["cache_gib"] = float(z["cache_gib"]) if "cache_gib" in z.files else NAN
    sim["quota"] = np.asarray(z["quota"]) if "quota" in z.files else None
    return sim


def check_sim(sim):
    res, dem, miss = sim["res"], sim["dem"], sim["miss"]
    D = res.shape[0]
    if res.shape != (D, N_L, N_E) or dem.shape != res.shape or miss.shape != res.shape:
        raise SystemExit(f"SIM shapes wrong: {res.shape} {dem.shape} {miss.shape}")
    # A non-resident demanded expert is always a miss, and a miss is never resident. Hence
    # staged & dem == staged & miss for every staged set (staged is a subset of ~res).
    if not np.array_equal(dem & ~res, miss):
        raise SystemExit("SIM inconsistent: dem & ~res != miss")
    sim["D"] = D


def load_score(src):
    if isinstance(src, dict):
        return dict(src)
    z = np.load(src)
    if "fold" not in z.files:
        raise SystemExit(f"{src}: SCORE file has no 'fold' array")
    return {"score": z["score"], "fold": z["fold"].astype(np.int64),
            "lead": int(z["lead"]) if "lead" in z.files else 0,
            "name": str(z["name"]) if "name" in z.files else os.path.basename(src)}


def precount(sim, score):
    """No staging. Per threshold index t, token j, layer l: prefetch, covered, wasted (int32 [T, D, 48])."""
    res, dem, miss = sim["res"], sim["dem"], sim["miss"]
    S = score["score"]
    D = res.shape[0]
    pre = np.zeros((T_N, D, N_L), np.int32)
    cov = np.zeros((T_N, D, N_L), np.int32)
    wst = np.zeros((T_N, D, N_L), np.int32)
    for l in range(N_L):  # layer by layer: never materialise [D, 48, 512] temporaries
        nonres = ~res[:, l, :]
        m = miss[:, l, :]
        nd = ~dem[:, l, :]
        s = S[:, l, :].astype(np.float32)
        for t in range(1, T_N):
            st = nonres & (s >= np.float32(TAUS[t]))
            pre[t, :, l] = np.count_nonzero(st, axis=1)
            cov[t, :, l] = np.count_nonzero(st & m, axis=1)
            wst[t, :, l] = np.count_nonzero(st & nd, axis=1)
    return pre, cov, cov * 0, wst  # (reads, covered_fresh, covered_kept = 0, wasted)


def sim_layer(order, scs, miss_l, D, tau, M):
    """Staging simulation for one layer and one tau over the whole trace (pure python, small M).

    order[j], scs[j]: this token's non-resident experts by descending score (window of at most 2M entries
    suffices, see layer_windows). miss_l[j]: missed experts at token j. Returns per-token lists
    reads, cov_fresh, cov_kept, wasted (see module docstring for the rules)."""
    reads = [0] * D
    cf = [0] * D
    ck = [0] * D
    wst = [0] * D
    stage = {}  # expert -> token of insertion; dict order = FIFO (oldest first)
    for j in range(D):
        ins = 0
        for e, v in zip(order[j], scs[j]):
            if v < tau or ins >= M:
                break
            if e in stage:
                continue
            if len(stage) < M:
                stage[e] = j
            else:
                old = next(iter(stage))
                if stage[old] == j:  # full with entries from this token only: nothing evictable
                    break
                del stage[old]
                wst[j] += 1
                stage[e] = j
            ins += 1
        reads[j] = ins
        for e in miss_l[j]:
            t = stage.pop(e, None)
            if t is not None:
                if t == j:
                    cf[j] += 1
                else:
                    ck[j] += 1
    wst[D - 1] += len(stage)
    return reads, cf, ck, wst


def layer_windows(S_l, res_l, W):
    """Per token: top-W non-resident experts by score (descending), as lists, plus their scores."""
    masked = np.where(~res_l, S_l.astype(np.float32), np.float32(-np.inf))
    order = np.argsort(-masked, axis=1, kind="stable")[:, :W]
    scs = np.take_along_axis(masked, order, axis=1)
    return order, scs


def precount_stage(sim, score, M):
    """Staging counts per threshold index t, token j, layer l: reads, covered fresh, covered kept, wasted."""
    res, dem, miss = sim["res"], sim["dem"], sim["miss"]
    S = score["score"]
    D = res.shape[0]
    W = min(N_E, 2 * M)  # at most M staged entries, so 2M window entries always hold M non-staged ones
    rd = np.zeros((T_N, D, N_L), np.int32)
    cf = np.zeros((T_N, D, N_L), np.int32)
    ck = np.zeros((T_N, D, N_L), np.int32)
    wst = np.zeros((T_N, D, N_L), np.int32)
    for l in range(N_L):
        order, scs = layer_windows(S[:, l, :], res[:, l, :], W)
        top = float(scs[:, 0].max()) if D else -np.inf
        order_l, scs_l = order.tolist(), scs.tolist()
        miss_l = [np.flatnonzero(row).tolist() for row in miss[:, l, :]]
        for t in range(1, T_N):
            tau = float(np.float32(TAUS[t]))
            if top < tau:  # nothing is staged at this tau in this layer
                continue
            r, a, b, w = sim_layer(order_l, scs_l, miss_l, D, tau, M)
            rd[t, :, l] = r
            cf[t, :, l] = a
            ck[t, :, l] = b
            wst[t, :, l] = w
    return rd, cf, ck, wst


def select(arrs, coeffs, fold, mode):
    """Threshold index per (token, layer). arrs: [T, D, L] int arrays, coeffs: their weights in the net.
    Integer sums are used, so a true zero stays exactly zero (ties resolved toward lower index)."""
    D, L = arrs[0].shape[1], arrs[0].shape[2]
    tots = [a.sum(axis=1, dtype=np.int64) for a in arrs]  # [T, L]
    if mode == "oracle-insample":
        V = sum(c * t for c, t in zip(coeffs, tots))
        best = V.argmax(axis=0)
        return np.broadcast_to(best, (D, L)).astype(np.int64)
    idx = np.zeros((D, L), np.int64)
    for k in np.unique(fold):
        test = fold == k
        tr = [t - a[:, test, :].sum(axis=1, dtype=np.int64) for t, a in zip(tots, arrs)]
        V = sum(c * t for c, t in zip(coeffs, tr))
        if mode == "crossfit":
            best = V.argmax(axis=0)
        else:  # global: one tau for all layers
            best = np.full(L, int(V.sum(axis=1).argmax()), np.int64)
        idx[test, :] = best
    return idx


def gather(arrs, idx):
    D, L = idx.shape
    jj = np.arange(D)[:, None]
    ll = np.arange(L)[None, :]
    return [a[idx, jj, ll].astype(np.int64) for a in arrs]


def tau_report(idx, fold, mode):
    if mode == "oracle-insample":
        return {"per_layer": [TAUS[int(t)] for t in idx[0]]}
    return {"per_fold": {str(int(k)): [TAUS[int(t)] for t in idx[fold == k][0]] for k in np.unique(fold)}}


def make_row(mode, s_c, tune, s_w, g, G, miss_pl, D, shift):
    cf_t = float(G["cf"].sum()) / D
    ck_t = float(G["ck"].sum()) / D
    wst_t = float(G["wst"].sum()) / D
    pre_t = float(G["pre"].sum()) / D
    mis_t = float(miss_pl.sum()) / D
    cov_t = cf_t + ck_t
    net = (cf_t * g + ck_t) * s_c - wst_t * s_w
    return {
        "mode": mode, "s_c": float(s_c), "tuned_s_w": float(tune), "s_w": float(s_w), "g": float(g),
        "prefetch_per_tok": pre_t, "covered_per_tok": cov_t, "wasted_per_tok": wst_t,
        "cov_fresh_per_tok": cf_t, "cov_kept_per_tok": ck_t,
        "misses_per_tok": mis_t,
        "recall": cov_t / mis_t if mis_t else NAN,
        "precision": cov_t / pre_t if pre_t else NAN,
        "net_ms": net,
        "net_vs_base": net - shift * s_c,
        "reads_per_tok": mis_t + pre_t,
        "covered_per_layer": ((G["cf"] + G["ck"]).sum(axis=0) / D).tolist(),
        "wasted_per_layer": (G["wst"].sum(axis=0) / D).tolist(),
        "prefetch_per_layer": (G["pre"].sum(axis=0) / D).tolist(),
        "misses_per_layer": (miss_pl.sum(axis=0) / D).tolist(),
    }


def find(rows, mode, s_c, tuned, s_w):
    for r in rows:
        if (r["mode"] == mode and math.isclose(r["s_c"], s_c) and math.isclose(r["tuned_s_w"], tuned)
                and math.isclose(r["s_w"], s_w)):
            return r
    raise KeyError((mode, s_c, tuned, s_w))


def run_block(sim, fold, g, cfg, counts, miss_pl, shift, stage_M):
    D = sim["D"]
    pre, cf, ck, wst = counts
    s_opt, s_pes = min(cfg["wasted"]), max(cfg["wasted"])
    cache, rows = {}, []

    def get(mode, s_c, tune):
        key = (mode, float(s_c), float(tune))
        if key not in cache:
            names = [cf, ck, wst, pre]
            idx = select([cf, ck, wst], [s_c * g, s_c, -tune], fold, mode)
            cache[key] = (idx, dict(zip(("cf", "ck", "wst", "pre"), gather(names, idx))))
        return cache[key]

    def add(mode, s_c, tune, s_w):
        idx, G = get(mode, s_c, tune)
        row = make_row(mode, s_c, tune, s_w, g, G, miss_pl, D, shift)
        row["tau"] = tau_report(idx, fold, mode)
        rows.append(row)

    for mode in MODES:
        for s_c in cfg["stall"]:
            for s_w in cfg["wasted"]:
                add(mode, s_c, s_w, s_w)              # thresholds tuned at this pair
            add(mode, s_c, s_opt, s_pes)              # classic pes: tuned at s_opt, evaluated at s_pes

    go, boot, folds = {}, {}, {}
    for s_c in cfg["stall"]:
        key = f"{s_c:g}"
        idx, G = get("crossfit", s_c, s_opt)
        x = (G["cf"].sum(axis=1) * g + G["ck"].sum(axis=1)) * s_c - G["wst"].sum(axis=1) * s_opt
        x = x - shift * s_c  # per-token net_vs_base (a constant shift when a base SIM is given)
        row_opt = find(rows, "crossfit", s_c, s_opt, s_opt)
        row_pes = find(rows, "crossfit", s_c, s_opt, s_pes)
        net_opt = row_opt["net_vs_base"] if shift else row_opt["net_ms"]
        net_pes = row_pes["net_vs_base"] if shift else row_pes["net_ms"]
        reads = row_opt["reads_per_tok"]
        go[key] = {
            "basis": "net_vs_base" if shift else "net",
            "net_opt": net_opt, "net_pes": net_pes, "reads_per_tok": reads,
            "need_opt": GO_FRAC * cfg["decode_ms"], "max_reads": cfg["max_reads"],
            "go": bool(net_opt >= GO_FRAC * cfg["decode_ms"] and net_pes >= 0 and reads <= cfg["max_reads"]),
        }
        rng = np.random.default_rng(0)
        samp = rng.integers(0, D, size=(BOOT_N, D))
        means = x[samp].mean(axis=1)
        boot[key] = {"point": float(x.mean()), "boot_mean": float(means.mean()),
                     "p5": float(np.percentile(means, 5)), "p95": float(np.percentile(means, 95))}
        folds[key] = [{"fold": int(k), "n": int(np.count_nonzero(fold == k)),
                       "net_opt": float(x[fold == k].mean())} for k in np.unique(fold)]
    return {"stage_slots": stage_M, "staging_gib": (stage_M * N_L * SLOT_BYTES / 2**30) if stage_M else None,
            "rows": rows, "go": go, "boot": boot, "folds": folds}


def evaluate(sim, score, cfg, stage_list=()):
    """Returns {"name", "lead", "g", "n_folds", "nostage": block, "stage": {M: block}}."""
    D = sim["D"]
    fold = np.asarray(score["fold"], dtype=np.int64)
    if fold.shape != (D,):
        raise SystemExit(f"{score['name']}: fold has shape {fold.shape}, SIM has D={D}")
    if tuple(score["score"].shape) != (D, N_L, N_E):
        raise SystemExit(f"{score['name']}: score has shape {score['score'].shape}, expected {(D, N_L, N_E)}")
    g = lead_factor(score["lead"], cfg["decode_ms"], cfg["read_ms"])
    miss_pl = sim["miss"].sum(axis=2, dtype=np.int64)  # [D, 48]
    shift = cfg.get("delta_misses") or 0.0
    out = {"name": score["name"], "lead": score["lead"], "g": g,
           "n_folds": int(len(np.unique(fold)))}
    pre, cov, _, wst = precount(sim, score)
    out["nostage"] = run_block(sim, fold, g, cfg, (pre, cov, cov * 0, wst), miss_pl, shift, None)
    out["stage"] = {}
    for M in stage_list:
        out["stage"][M] = run_block(sim, fold, g, cfg, precount_stage(sim, score, M), miss_pl, shift, M)
    return out


def print_sim(sim):
    D = sim["D"]
    miss, dem = sim["miss"], sim["dem"]
    print(f"[SIM] D={D} prefix={sim['prefix']} policy={sim['policy']} quota_scheme={sim['quota_scheme']} "
          f"cache_gib={sim['cache_gib']:g}")
    print(f"[SIM] misses/token={miss.sum() / D:.2f}  demand accesses/token={dem.sum() / D:.1f}")
    if sim.get("quota") is not None:
        q = sim["quota"]
        print(f"[SIM] quota per layer: mean {q.mean():.1f} min {q.min()} max {q.max()}")


def print_block(b, cfg, wide, has_base):
    dec = cfg["decode_ms"]
    if b["stage_slots"] is None:
        title = "no staging (classic: a prefetched expert not demanded at its token is wasted)"
    else:
        title = (f"staging pool M={b['stage_slots']} slots/layer: {b['staging_gib']:.3f} GiB "
                 f"(M*48*{SLOT_BYTES} bytes)")
    print(f"-- {title}")
    head = (f"{'mode':<16s} {'s_c':>5s} {'tuned@':>6s} {'s_w':>5s} {'pref/tok':>9s} ")
    if wide:
        head += f"{'reads/tok':>9s} {'fresh/tok':>9s} {'kept/tok':>8s} "
    head += f"{'cov/tok':>8s} {'recall':>7s} {'wst/tok':>8s} {'prec':>6s} {'g':>6s} {'net ms':>9s}"
    if has_base:
        head += f" {'net_vs_base':>11s}"
    print(head)
    for row in b["rows"]:
        line = (f"{row['mode']:<16s} {row['s_c']:5.2f} {row['tuned_s_w']:6.2f} {row['s_w']:5.2f} "
                f"{row['prefetch_per_tok']:9.2f} ")
        if wide:
            line += f"{row['reads_per_tok']:9.2f} {row['cov_fresh_per_tok']:9.2f} {row['cov_kept_per_tok']:8.2f} "
        line += (f"{row['covered_per_tok']:8.2f} {row['recall']:7.3f} {row['wasted_per_tok']:8.2f} "
                 f"{row['precision']:6.3f} {row['g']:6.4f} {row['net_ms']:9.3f}")
        if has_base:
            line += f" {row['net_vs_base']:11.3f}"
        print(line)
    for key, go in b["go"].items():
        verdict = "GO" if go["go"] else "NO-GO"
        print(f"GO s_c={key} (crossfit, tuned@{min(cfg['wasted']):g}, basis {go['basis']}): {verdict}  "
              f"net_opt={go['net_opt']:.3f} (need >= {go['need_opt']:.2f})  net_pes={go['net_pes']:.3f} (need >= 0)  "
              f"reads/tok={go['reads_per_tok']:.1f} (need <= {go['max_reads']:g})")
        bb = b["boot"][key]
        print(f"  bootstrap crossfit {go['basis']} s_c={key}: point {bb['point']:.3f}  boot mean {bb['boot_mean']:.3f}  "
              f"5th {bb['p5']:.3f}  95th {bb['p95']:.3f}  ({BOOT_N} resamples, seed 0)")
        print("  per-fold crossfit net_opt: " + ", ".join(
            f"fold {f['fold']} (n={f['n']}) {f['net_opt']:.3f}" for f in b["folds"][key]))


def print_score(r, cfg):
    print(f"\n== SCORE {r['name']}: lead={r['lead']} g={r['g']:.4f} folds={r['n_folds']} ==")
    wide = bool(r["stage"])
    has_base = cfg.get("delta_misses") is not None
    print_block(r["nostage"], cfg, wide, has_base)
    for M in r["stage"]:
        print()
        print_block(r["stage"][M], cfg, wide, has_base)


def synth_sim(D=200, seed=0, window=20):
    rng = np.random.default_rng(seed)
    base = 1.0 / np.arange(1, N_E + 1) ** 1.1
    base /= base.sum()
    logits = np.empty((N_L, N_E))
    for l in range(N_L):
        logits[l, rng.permutation(N_E)] = np.log(base)  # skewed, layer-specific expert popularity
    keys = logits[None] + rng.gumbel(size=(D, N_L, N_E))
    top = np.argpartition(-keys, TOP_K, axis=2)[:, :, :TOP_K]
    dem = np.zeros((D, N_L, N_E), bool)
    dem[np.arange(D)[:, None, None], np.arange(N_L)[None, :, None], top] = True
    res = np.zeros_like(dem)
    for j in range(D):  # trivial residency: used in the previous `window` tokens of this layer
        res[j] = dem[max(0, j - window):j].any(axis=0)
    return {"res": res, "dem": dem, "miss": dem & ~res, "D": D, "prefix": 0,
            "policy": "synthetic-window20", "quota_scheme": "none", "cache_gib": NAN, "quota": None}


def find_row(r, mode, s_c, tuned, s_w, block="nostage"):
    return find(r[block]["rows"], mode, s_c, tuned, s_w)


def selftest():
    cfg = {"decode_ms": 110.0, "read_ms": 2.5, "stall": [1.7, 0.82], "wasted": [0.20, 0.82], "max_reads": 200.0,
           "delta_misses": None}
    ok_all = True

    def report(tag, ok, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {tag}: {msg}")

    sim = synth_sim(200)
    check_sim(sim)
    D = sim["D"]
    misses = float(sim["miss"].sum()) / D
    fold = (np.arange(D) * 5 // D).astype(np.int64)
    rng = np.random.default_rng(1)
    dem_f = sim["dem"].astype(np.float16)
    noisy = (0.6 * sim["dem"].astype(np.float32) + 0.3 * rng.random((D, N_L, N_E), dtype=np.float32)).astype(np.float16)
    scores = {
        "dem-lead0": {"score": dem_f, "fold": fold, "lead": 0, "name": "dem-lead0"},
        "random": {"score": rng.random((D, N_L, N_E)).astype(np.float16), "fold": fold, "lead": 0, "name": "random"},
        "dem-lead1": {"score": dem_f, "fold": fold, "lead": 1, "name": "dem-lead1"},
        "noisy-dem": {"score": noisy, "fold": fold, "lead": 0, "name": "noisy-dem"},
    }
    results = {k: evaluate(sim, v, cfg) for k, v in scores.items()}

    report("(0) SIM consistency", True, "dem & ~res == miss on the synthetic SIM "
           f"(D={D}, misses/token={misses:.3f})")

    r = find_row(results["dem-lead0"], "crossfit", 1.7, 0.2, 0.2)
    ok = (abs(r["covered_per_tok"] - misses) < 1e-9 and abs(r["wasted_per_tok"]) < 1e-12
          and abs(r["net_ms"] - misses * 1.7) < 1e-9)
    report("(i) score=dem, lead 0", ok, f"covered/tok={r['covered_per_tok']:.6f} misses/tok={misses:.6f} "
           f"wasted/tok={r['wasted_per_tok']:.6f} net={r['net_ms']:.6f} expected={misses * 1.7:.6f}")

    rr = results["random"]["nostage"]["rows"]
    net_opt = find_row(results["random"], "crossfit", 1.7, 0.2, 0.2)["net_ms"]
    max_cf = max(row["net_ms"] for row in rr if row["mode"] == "crossfit")
    ok = abs(net_opt) <= 0.5 and max_cf <= 0.5
    report("(ii) score=random uniform", ok, f"crossfit net_opt(1.7,0.20)={net_opt:.4f} ms/tok, "
           f"max crossfit net over all grid rows={max_cf:.4f} (need |net|<=0.5 and max<=0.5)")

    r1 = results["dem-lead1"]
    g = lead_factor(1, cfg["decode_ms"], cfg["read_ms"])
    net1 = find_row(r1, "crossfit", 1.7, 0.2, 0.2)["net_ms"]
    net0 = find_row(results["dem-lead0"], "crossfit", 1.7, 0.2, 0.2)["net_ms"]
    ok = abs(r1["g"] - g) < 1e-12 and abs(net1 - g * misses * 1.7) < 1e-9 and abs(net1 - g * net0) < 1e-9
    report("(iii) score=dem, lead 1", ok, f"g={r1['g']:.6f} net={net1:.6f} expected g*misses*1.7={g * misses * 1.7:.6f} "
           f"g*net(lead0)={g * net0:.6f}")

    ok = True
    worst = -1e9
    for name, res_ in results.items():
        for s_c in cfg["stall"]:
            for s_w in cfg["wasted"]:
                cf_ = find_row(res_, "crossfit", s_c, s_w, s_w)["net_ms"]
                orc = find_row(res_, "oracle-insample", s_c, s_w, s_w)["net_ms"]
                worst = max(worst, cf_ - orc)
                ok = ok and cf_ <= orc + 1e-9
    report("(iv) crossfit <= oracle-insample", ok, f"all scores x grid pairs; max(crossfit - oracle) = {worst:.3e}")

    # (v) M >= 512 with score = dem, lead 0: staging must reproduce the non-staging result exactly
    rv = evaluate(sim, scores["dem-lead0"], cfg, stage_list=(512,))
    a_rows, b_rows = rv["nostage"]["rows"], rv["stage"][512]["rows"]
    diff = 0.0
    for ra, rb in zip(a_rows, b_rows):
        for k in ("net_ms", "covered_per_tok", "prefetch_per_tok", "wasted_per_tok"):
            diff = max(diff, abs(ra[k] - rb[k]))
        if ra["mode"] != rb["mode"] or ra["tau"] != rb["tau"]:
            diff = max(diff, 1.0)
    report("(v) staging M=512, score=dem == non-staging", diff < 1e-9 and len(a_rows) == len(b_rows),
           f"rows={len(a_rows)}/{len(b_rows)} max abs diff over net/cov/pref/wst = {diff:.3e}")

    # (vi) hand-built tiny trace: D=4 tokens, layer 0 active, M=4, tau=0.5, residency empty
    D6 = 4
    dem6 = np.zeros((D6, N_L, N_E), bool)
    score6 = np.zeros((D6, N_L, N_E), np.float16)
    for j, es in enumerate([[0, 1], [2, 3], [5, 6], [4, 8]]):
        for e in es:
            dem6[j, 0, e] = True
    for j, sc in enumerate([{0: .9, 1: .8}, {5: .9, 4: .7}, {4: .9, 5: .9, 6: .8, 7: .6}, {8: .9, 9: .7, 2: .6}]):
        for e, v in sc.items():
            score6[j, 0, e] = v
    res6 = np.zeros_like(dem6)
    miss6 = dem6 & ~res6
    order6, scs6 = layer_windows(score6[:, 0, :], res6[:, 0, :], N_E)
    miss_l6 = [np.flatnonzero(row).tolist() for row in miss6[:, 0, :]]
    got = sim_layer(order6.tolist(), scs6.tolist(), miss_l6, D6, 0.5, 4)
    exp = ([2, 2, 2, 3], [2, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 4])
    ok = all(list(g_) == list(e_) for g_, e_ in zip(got, exp))
    report("(vi) hand-built staging trace (M=4, tau=0.5)", ok,
           f"reads={list(got[0])} fresh={list(got[1])} kept={list(got[2])} wasted={list(got[3])} "
           f"expected reads={list(exp[0])} fresh={list(exp[1])} kept={list(exp[2])} wasted={list(exp[3])}")

    # (vii) staging M=512 >= non-staging for the same tau in every layer (kept covers add, no evictions at M=512).
    # Scores are active in layers 0..3 only, to keep the python loop short; other layers are zero for both.
    S7 = noisy.copy()
    S7[:, 4:, :] = 0
    sc7 = {"score": S7, "fold": fold, "lead": 0, "name": "noisy-4layers"}
    pre_n, cov_n, _, wst_n = precount(sim, sc7)
    _, cf_s, ck_s, wst_s = precount_stage(sim, sc7, 512)
    net_n = cov_n.sum(axis=1) * 1.7 - wst_n.sum(axis=1) * 0.2     # [T, L]
    net_s = (cf_s.sum(axis=1) + ck_s.sum(axis=1)) * 1.7 - wst_s.sum(axis=1) * 0.2
    deficit = float(np.max(net_n - net_s))
    report("(vii) staging M=512 >= non-staging per (layer, tau)", deficit <= 1e-9,
           f"max(non-staging - staging) over all (layer, tau) at (1.7, 0.20) = {deficit:.3e}; "
           f"kept covers total over grid = {int(ck_s.sum())}")

    print("[selftest] " + ("ALL PASS" if ok_all else "SOME FAILED"))
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", help="SIM .npz (from la_cache.py)")
    ap.add_argument("--score", action="append", default=[], help="SCORE .npz (repeatable)")
    ap.add_argument("--stage-slots", default="", help="comma list of staging pool sizes M per layer, e.g. 4,8,16")
    ap.add_argument("--sim-base", help="SIM .npz without lookahead at the full cache size (enables net_vs_base GO)")
    ap.add_argument("--json", help="write everything to this JSON file")
    ap.add_argument("--decode-ms", type=float, default=110.0)
    ap.add_argument("--read-ms", type=float, default=2.5, help="flash read latency used for the lead factor")
    ap.add_argument("--stall-ms", default="1.7,0.82", help="comma list of s_c (ms stall saved per covered miss)")
    ap.add_argument("--wasted-ms", default="0.20,0.82", help="comma list of s_w (ms per wasted read); min = optimistic, "
                    "max = pessimistic classic pair")
    ap.add_argument("--max-reads", type=float, default=200.0, help="bandwidth cap: misses + prefetch reads per token")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        sys.exit(selftest())
    if not args.sim or not args.score:
        ap.error("--sim and at least one --score are required (or --selftest)")
    stage_list = [int(x) for x in args.stage_slots.split(",") if x.strip()]
    cfg = {"decode_ms": args.decode_ms, "read_ms": args.read_ms,
           "stall": [float(x) for x in args.stall_ms.split(",") if x],
           "wasted": [float(x) for x in args.wasted_ms.split(",") if x], "max_reads": args.max_reads,
           "delta_misses": None}
    t0 = time.time()
    sim = load_sim(args.sim)
    check_sim(sim)
    print_sim(sim)
    misses_main = float(sim["miss"].sum()) / sim["D"]
    base_info = None
    if args.sim_base:
        base = load_sim(args.sim_base)
        if base["D"] != sim["D"]:
            raise SystemExit(f"--sim-base has D={base['D']}, --sim has D={sim['D']}")
        base_misses = float(base["miss"].sum()) / base["D"]
        cfg["delta_misses"] = misses_main - base_misses
        base_info = {"file": args.sim_base, "misses_per_tok": base_misses, "delta_misses_per_tok": cfg["delta_misses"],
                     "cache_gib": base["cache_gib"]}
        print(f"[SIM-BASE] {args.sim_base}: misses/token={base_misses:.2f} cache_gib={base['cache_gib']:g}; "
              f"delta misses/token (main - base) = {cfg['delta_misses']:.2f}; GO uses net_vs_base")
        del base
    out = {"sim": {"file": args.sim, "D": sim["D"], "prefix": sim["prefix"], "policy": sim["policy"],
                   "quota_scheme": sim["quota_scheme"], "cache_gib": sim["cache_gib"],
                   "misses_per_tok": misses_main},
           "sim_base": base_info, "cfg": cfg, "stage_slots": stage_list, "scores": []}
    for path in args.score:
        r = evaluate(sim, load_score(path), cfg, stage_list)
        r["file"] = path
        print_score(r, cfg)
        out["scores"].append(r)
    print(f"\n[la_net] done in {time.time() - t0:.1f}s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[la_net] wrote {args.json}")


if __name__ == "__main__":
    main()
