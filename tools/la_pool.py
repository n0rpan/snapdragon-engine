#!/usr/bin/env python3
"""In-pool speculation simulator for expert prefetch (F-LOOKAHEAD; numpy + stdlib only; offline).

Question: does speculating an expert INTO the layer's ordinary cache pool pay? The engine's
prefetch_guarded puts a speculated expert into the same LFU pool that demand uses (evicting a victim),
so it can displace a useful resident. la_net.py models separate staging slots instead; this file
replays the in-pool version exactly.

Per layer l (layers are independent), one threshold tau, decode tokens j = 0..D-1 in order. The pool is
la_cache's engine LFU (class Lfu) with the layer's quota, warmed up from the prefill routes exactly as in
la_cache.simulate, pin "all". Per token j:
  1. SPECULATE (before the layer's demand accesses of token j): candidates = experts NOT in the pool with
     score[j,l,e] >= tau, highest score first, at most --max-spec of them. Each is inserted WITHOUT
     touching LFU counts, tick or aging. Victim: an empty slot if any, else the resident expert with the
     fewest counts then least recently used (the LFU key, ties to the lowest slot); never an expert
     speculated at this same token (this excludes the other candidates too). The inserted slot is stamped
     with the current tick (no increment), as the engine stamps an inserted slot. The inserted expert is
     "speculated, unused". If the victim was itself "speculated, unused", that earlier read was WASTED
     (counted at token j).
  2. DEMAND: the routed experts in slot order through Lfu.access (pinned = the token's experts). A demand
     hit on a "speculated, unused" expert = COVERED (covered_fresh if speculated at j, covered_kept
     otherwise; the mark is cleared). A demand miss = MISS (normal LFU insert); if its victim was
     "speculated, unused", that read was WASTED (counted at j).
  Speculated-unused experts left at the end of the trace are wasted (counted at the last token).
Per token and layer the simulator records misses, spec_reads, covered_fresh, covered_kept, wasted.

With tau = +inf there is no speculation and the misses equal la_cache.simulate exactly. This is asserted
for every layer on every run (it is the baseline base_miss[j, l]).

Net per token and layer: net = (base_miss - miss) * s_c - wasted * s_w (no lead factor, g = 1: a covered
expert costs no stall; evicting a useful resident shows up as extra misses later and is charged there).

Threshold selection follows la_net.py (its select/gather/load_score are imported): grid = [inf] + 40 values
geomspace(0.9, 1e-4); modes crossfit (per layer, tau chosen on the tokens outside the token's fold, using
the SCORE file's fold array), global (one tau for all layers, cross-fitted) and oracle-insample. Each
(s_c, s_w) pair is tuned and evaluated at the same pair. Carry-over: each tau is simulated over the WHOLE
trace once, so pool state flows across fold boundaries. This is accepted as an approximation: the tokens
of a fold inherit the pool state left by the preceding tokens, whatever fold those belong to.

GO line per cost pair: GO iff the crossfit net (ms/token) >= 0.03 * decode_ms.
Bootstrap: 1000 token resamples (seed 0) of the crossfit per-token net; per-fold crossfit nets printed;
saturation = fraction of (token, layer) cells with demand misses + spec reads > 3 (crossfit selection).

Speed: the candidate order of each (token, layer) is a full stable argsort of the score row (exact; it
subsumes a top-32 window with fallback). Candidates are filtered by residency and tau in the loop. A tau
that selects no candidate anywhere in a layer reuses the tau=inf run (identical by construction).

Usage (wrap every run; PY = ~/.cache/local-ai/venv-lookahead/bin/python):
  flock -w 600 "${LOCAL_AI_BUILD_LOCK:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/build.lock}" systemd-run --user --scope -q -p MemoryMax=4G \
      -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 nice -n 19 $PY -I tools/la_pool.py \
      --routes CSV --cache-gib 20 --quota builtin --score A.npz [--score B.npz ...] [--max-spec 2] \
      [--decode-ms 90] [--stall-ms 0.66,0.5] [--wasted-ms 0.02,0.10,0.20] [--json OUT]
  ... tools/la_pool.py --selftest
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir
from la_cache import Lfu, N_E, N_L, quotas, simulate  # noqa: E402
from la_net import MODES, gather, load_score, select  # noqa: E402
from lookahead_train import TOP_K, read_routes  # noqa: E402

TAUS = [math.inf] + [float(x) for x in np.geomspace(0.9, 1e-4, 40)]  # index 0 = never speculate
BOOT_N = 1000
GO_FRAC = 0.03
SAT_LIMIT = 3  # the device saturates beyond about 3 records per layer
NAMES = ("dm", "miss", "spec", "cf", "ck", "wst")
COUNT_NAMES = ("miss", "spec", "cf", "ck", "wst")


def prep_layer(sc):
    """sc float32 [D, N_E] -> per token the expert ids by descending score (stable on ties)."""
    return np.argsort(-sc, axis=1, kind="stable").tolist()


def warm_lists(routes, prefix):
    """Per layer: distinct prefill experts in first-use order (step, then slot), as la_cache.simulate."""
    out = []
    for l in range(N_L):
        seen, w = set(), []
        for s in range(prefix):
            for x in routes[s, l].tolist():
                if x < 0 or x in seen:
                    continue
                seen.add(x)
                w.append(x)
        out.append(w)
    return out


def run_layer(dem_l, order_l, ngt, warm, quota, max_spec):
    """One layer, one tau. dem_l[j]: the token's routed experts (slot order). order_l[j]: expert ids by
    descending score; ngt[j]: how many leading entries of order_l[j] have score >= tau (candidates).
    Returns five lists of length D: miss, spec, covered_fresh, covered_kept, wasted."""
    D = len(dem_l)
    pol = Lfu(int(quota))
    for x in warm:
        pol.access(x, frozenset())
    spec_at = {}  # expert -> token of its speculative insert, while still "speculated, unused"
    miss, spec, cf, ck, wst = [0] * D, [0] * D, [0] * D, [0] * D, [0] * D
    for j in range(D):
        n = ngt[j]
        if n and max_spec:
            row = order_l[j]
            k = 0
            for i in range(n):
                e = row[i]
                if e in pol.map:  # only non-resident experts are candidates
                    continue
                best, bk = -1, None
                for h in range(pol.n):  # victim: empty slot first, else LFU key, never speculated at j
                    o = pol.owner[h]
                    if o < 0:
                        best = h
                        break
                    if spec_at.get(o) == j:
                        continue
                    key = (pol.cnt[o], pol.last[h])
                    if bk is None or key < bk:
                        best, bk = h, key
                if best < 0:
                    raise RuntimeError(f"speculation at token {j}: no evictable slot")
                o = pol.owner[best]
                if o >= 0:
                    del pol.map[o]
                    if spec_at.pop(o, None) is not None:  # evicted while speculated, unused
                        wst[j] += 1
                pol.owner[best] = e
                pol.map[e] = best
                pol.last[best] = pol.tick  # no tick increment, no count change
                spec_at[e] = j
                spec[j] += 1
                k += 1
                if k == max_spec:
                    break
        pinned = frozenset(dem_l[j])
        for e in dem_l[j]:
            t = spec_at.get(e)
            if pol.access(e, pinned):
                if t is not None:  # speculated, unused, now demanded
                    if spec_at.pop(e) == j:
                        cf[j] += 1
                    else:
                        ck[j] += 1
            else:
                miss[j] += 1
                if spec_at:  # the victim of this miss, if it was speculated and unused
                    for o in [o for o in spec_at if o not in pol.map]:
                        del spec_at[o]
                        wst[j] += 1
    wst[D - 1] += len(spec_at)  # still speculated and unused at the end of the trace
    return miss, spec, cf, ck, wst


def pool_sim(routes, prefix, quota, score, max_spec, base_pl, taus=TAUS):
    """All layers, all taus. routes int16 [S, N_L, TOP_K]; score [D, N_L, N_E]; base_pl int [D, N_L] =
    la_cache baseline misses. Returns dict of int32 [len(taus), D, N_L] arrays (miss, spec, cf, ck, wst).
    taus[0] must be +inf; its misses are asserted equal to base_pl."""
    assert math.isinf(taus[0]) and taus[0] > 0
    D = routes.shape[0] - prefix
    T = len(taus)
    arr = {k: np.zeros((T, D, N_L), np.int32) for k in COUNT_NAMES}
    warm = warm_lists(routes, prefix)
    dem_all = routes[prefix:]
    for l in range(N_L):
        dem_l = dem_all[:, l, :].tolist()
        sc = score[:, l, :].astype(np.float32)
        order_l = prep_layer(sc)
        ngt0 = [0] * D
        base = run_layer(dem_l, order_l, ngt0, warm[l], quota[l], max_spec)
        for k, v in zip(COUNT_NAMES, base):
            arr[k][0, :, l] = v
        if not np.array_equal(np.asarray(base[0]), base_pl[:, l]):
            raise AssertionError(f"layer {l}: tau=inf misses differ from la_cache.simulate")
        for t in range(1, T):
            cnt = np.count_nonzero(sc >= np.float32(taus[t]), axis=1)
            if not cnt.any():  # no candidate in this layer at this tau: identical to the baseline
                for k in COUNT_NAMES:
                    arr[k][t, :, l] = arr[k][0, :, l]
                continue
            res = run_layer(dem_l, order_l, cnt.tolist(), warm[l], quota[l], max_spec)
            for k, v in zip(COUNT_NAMES, res):
                arr[k][t, :, l] = v
    return arr


def analyse(arr, base_pl, fold, cfg):
    """Threshold selection (la_net.select) and the net table for every (s_c, s_w) pair."""
    miss, spec, cf, ck, wst = (arr[k] for k in COUNT_NAMES)
    taus = cfg["taus"]
    D = miss.shape[1]
    dm = base_pl.astype(np.int32)[None] - miss  # [T, D, L] misses avoided (negative = extra misses)
    out = {"n_folds": int(len(np.unique(fold))), "rows": [], "go": {}, "boot": {}, "folds": {},
           "sat": {}, "tau_per_fold": {}}
    for s_c in cfg["stall"]:
        for s_w in cfg["wasted"]:
            key = f"s_c={s_c:g} s_w={s_w:g}"
            for mode in MODES:
                idx = select([dm, wst], [s_c, -s_w], fold, mode)  # tuned at this pair
                g = dict(zip(NAMES, gather([dm, miss, spec, cf, ck, wst], idx)))
                net_tok = (s_c * g["dm"] - s_w * g["wst"]).sum(axis=1).astype(np.float64)  # per token
                out["rows"].append({
                    "mode": mode, "s_c": float(s_c), "s_w": float(s_w),
                    "spec_per_tok": float(g["spec"].sum()) / D,
                    "reads_per_tok": float((g["miss"] + g["spec"]).sum()) / D,
                    "misses_per_tok": float(g["miss"].sum()) / D,
                    "cov_fresh_per_tok": float(g["cf"].sum()) / D,
                    "cov_kept_per_tok": float(g["ck"].sum()) / D,
                    "wasted_per_tok": float(g["wst"].sum()) / D,
                    "net_ms": float(net_tok.mean()),
                })
                if mode != "crossfit":
                    continue
                net = float(net_tok.mean())
                need = GO_FRAC * cfg["decode_ms"]
                out["go"][key] = {"s_c": float(s_c), "s_w": float(s_w), "net_ms": net, "need": need,
                                  "go": bool(net >= need)}
                rng = np.random.default_rng(0)
                samp = rng.integers(0, D, size=(BOOT_N, D))
                means = net_tok[samp].mean(axis=1)
                out["boot"][key] = {"point": net, "boot_mean": float(means.mean()),
                                    "p5": float(np.percentile(means, 5)),
                                    "p95": float(np.percentile(means, 95))}
                out["folds"][key] = [{"fold": int(k), "n": int(np.count_nonzero(fold == k)),
                                      "net_ms": float(net_tok[fold == k].mean())}
                                     for k in np.unique(fold)]
                cells = (g["miss"] + g["spec"]) > SAT_LIMIT
                out["sat"][key] = float(cells.mean())
                out["tau_per_fold"][key] = {str(int(k)): [None if math.isinf(taus[int(t)]) else taus[int(t)]
                                                          for t in idx[fold == k][0]]
                                            for k in np.unique(fold)}
    return out


def print_score(name, res, cfg, base_tok, D):
    print(f"\n== SCORE {name}: folds={res['n_folds']} D={D} base misses/tok={base_tok:.3f} ==")
    print(f"{'mode':<16s} {'s_c':>5s} {'s_w':>5s} {'spec/tok':>9s} {'reads/tok':>9s} {'miss/tok':>9s} "
          f"{'fresh/tok':>9s} {'kept/tok':>9s} {'wst/tok':>8s} {'net ms/tok':>10s}")
    for r in res["rows"]:
        print(f"{r['mode']:<16s} {r['s_c']:5.2f} {r['s_w']:5.2f} {r['spec_per_tok']:9.3f} "
              f"{r['reads_per_tok']:9.3f} {r['misses_per_tok']:9.3f} {r['cov_fresh_per_tok']:9.3f} "
              f"{r['cov_kept_per_tok']:9.3f} {r['wasted_per_tok']:8.3f} {r['net_ms']:10.4f}")
    for key, go in res["go"].items():
        verdict = "GO" if go["go"] else "NO-GO"
        print(f"GO {key} (crossfit): {verdict}  net={go['net_ms']:.3f} ms/tok "
              f"(need >= {go['need']:.2f} = 0.03 x decode_ms {cfg['decode_ms']:g})")
        b = res["boot"][key]
        print(f"  bootstrap crossfit net {key}: point {b['point']:.3f}  boot mean {b['boot_mean']:.3f}  "
              f"5th {b['p5']:.3f}  95th {b['p95']:.3f}  ({BOOT_N} token resamples, seed 0)")
        print("  per-fold crossfit net: " + ", ".join(
            f"fold {f['fold']} (n={f['n']}) {f['net_ms']:.3f}" for f in res["folds"][key]))
        print(f"  saturation {key}: fraction of (token,layer) cells with demand misses + spec reads > "
              f"{SAT_LIMIT} (crossfit) = {res['sat'][key]:.4f}")


def synth_routes(prefix, D, seed):
    """Synthetic complete trace: skewed, layer-specific popularity; 10 distinct experts per (step, layer)."""
    rng = np.random.default_rng(seed)
    S = prefix + D
    base = 1.0 / np.arange(1, N_E + 1) ** 1.1
    base /= base.sum()
    logits = np.empty((N_L, N_E))
    for l in range(N_L):
        logits[l, rng.permutation(N_E)] = np.log(base)
    keys = logits[None] + rng.gumbel(size=(S, N_L, N_E))
    top = np.argpartition(-keys, TOP_K, axis=2)[:, :, :TOP_K]
    return top.astype(np.int16)


def selftest():
    ok_all = True

    def report(tag, ok, msg):
        nonlocal ok_all
        ok_all = ok_all and bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} {tag}: {msg}", flush=True)

    cfg = {"taus": TAUS, "stall": [0.66, 0.5], "wasted": [0.02, 0.10, 0.20], "decode_ms": 90.0}
    prefix, D = 20, 60
    routes = synth_routes(prefix, D, seed=0)
    base_pl_of = lambda q: simulate(routes, prefix, "lfu", q, "all")  # noqa: E731

    # (i) tau = inf reproduces la_cache.simulate (small quota: plenty of evictions)
    quota12 = np.full(N_L, 12, np.int64)
    _, dem12, base12 = base_pl_of(quota12)
    base12_pl = base12.sum(axis=2).astype(np.int64)
    rng = np.random.default_rng(3)
    sc_rand = rng.random((D, N_L, N_E), dtype=np.float32).astype(np.float16)
    arr = pool_sim(routes, prefix, quota12, sc_rand, 2, base12_pl, [math.inf, 0.5])
    same = np.array_equal(arr["miss"][0], base12_pl.astype(np.int32)[None][0])
    report("(i) tau=inf == la_cache.simulate", same,
           f"per (token, layer) misses equal on synthetic D={D}, quota 12/layer, "
           f"misses/tok={base12_pl.sum() / D:.3f}; speculation at tau=0.5 gives misses/tok="
           f"{arr['miss'][1].sum() / D:.3f} (info)")

    # (ii) score = dem (perfect oracle), max-spec 10, quota = N_E (no evictions): misses -> 0
    quota_big = np.full(N_L, N_E, np.int64)
    _, dem_b, base_b = base_pl_of(quota_big)
    base_b_pl = base_b.sum(axis=2).astype(np.int64)
    arr2 = pool_sim(routes, prefix, quota_big, dem_b.astype(np.float16), 10, base_b_pl, [math.inf, 0.5])
    m1 = int(arr2["miss"][1].sum())
    w1 = int(arr2["wst"][1].sum())
    s1 = int(arr2["spec"][1].sum())
    cf1 = int(arr2["cf"][1].sum())
    ck1 = int(arr2["ck"][1].sum())
    tot = int(base_b_pl.sum())
    net = 0.66 * (tot - m1) - 0.10 * w1
    ok = m1 == 0 and w1 == 0 and s1 == tot and cf1 == tot and ck1 == 0 and abs(net - 0.66 * tot) < 1e-9
    report("(ii) score=dem, max-spec 10, quota N_E", ok,
           f"base misses={tot} after speculation misses={m1} spec reads={s1} covered_fresh={cf1} "
           f"covered_kept={ck1} wasted={w1}; net(0.66,0.10)={net:.4f} expected base*0.66={0.66 * tot:.4f}")
    arr2s = pool_sim(routes, prefix, quota12, dem_b.astype(np.float16), 10, base12_pl, [math.inf, 0.5])
    print(f"  info (not asserted): score=dem with quota 12/layer: base misses/tok={base12_pl.sum() / D:.3f}, "
          f"misses/tok with in-pool speculation={arr2s['miss'][1].sum() / D:.3f}, "
          f"wasted/tok={arr2s['wst'][1].sum() / D:.3f} (a speculated expert can evict a routed resident that "
          f"is not yet demanded, which is then a miss that no candidate covers)", flush=True)

    # (iii) score = uniform random: crossfit net must not be strongly positive
    fold = (np.arange(D) * 5 // D).astype(np.int64)
    arr3 = pool_sim(routes, prefix, quota12, sc_rand, 2, base12_pl)
    res3 = analyse(arr3, base12_pl, fold, cfg)
    cf_rows = [r for r in res3["rows"] if r["mode"] == "crossfit"]
    worst = max(r["net_ms"] for r in cf_rows)
    best_s = " ".join(f"{r['s_c']:g}/{r['s_w']:g}:{r['net_ms']:.4f}" for r in cf_rows)
    report("(iii) score=random uniform: crossfit net <= +0.3", worst <= 0.3,
           f"max crossfit net over pairs = {worst:.4f} ms/tok; per pair {best_s}")

    # (iv) hand-built tiny layer: quota 3, max-spec 1, tau 0.5, 4 tokens, experts A=0 B=1 C=2 D=3 E=4
    dem4 = [[0, 1], [2, 0], [1, 3], [0, 3]]  # j0 {A,B}, j1 {C,A}, j2 {B,D}, j3 {A,D}
    sc4 = np.zeros((4, N_E), np.float32)
    sc4[0, 2] = 0.9  # j0: C
    sc4[2, 3] = 0.9  # j2: D
    sc4[3, 4] = 0.9  # j3: E
    order4 = prep_layer(sc4)
    ngt4 = np.count_nonzero(sc4 >= np.float32(0.5), axis=1).tolist()
    got = run_layer(dem4, order4, ngt4, [], 3, 1)
    exp = ([2, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1])
    base4 = run_layer(dem4, order4, [0] * 4, [], 3, 1)
    exp_base = [2, 1, 1, 0]
    ok = all(list(g_) == list(e_) for g_, e_ in zip(got, exp)) and list(base4[0]) == exp_base
    report("(iv) hand-built tiny layer", ok,
           f"miss={list(got[0])} spec={list(got[1])} fresh={list(got[2])} kept={list(got[3])} "
           f"wasted={list(got[4])} (expected miss=[2, 0, 1, 1] spec=[1, 0, 1, 1] fresh=[0, 0, 1, 0] "
           f"kept=[0, 1, 0, 0] wasted=[0, 0, 0, 1]); baseline miss={list(base4[0])} expected {exp_base}")

    print("SELFTEST", "PASS" if ok_all else "FAIL", flush=True)
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", help="route trace CSV (turn,phase,step,layer,slot,expert)")
    ap.add_argument("--cache-gib", type=float, default=20.0)
    ap.add_argument("--quota", choices=["builtin", "equal"], default="builtin")
    ap.add_argument("--score", action="append", default=[], help="SCORE .npz (repeatable)")
    ap.add_argument("--max-spec", type=int, default=2, help="speculative inserts per token and layer")
    ap.add_argument("--decode-ms", type=float, default=90.0)
    ap.add_argument("--stall-ms", default="0.66,0.5", help="comma list of s_c (ms per covered miss)")
    ap.add_argument("--wasted-ms", default="0.02,0.10,0.20", help="comma list of s_w (ms per wasted read)")
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(selftest())
    if not a.routes or not a.score:
        ap.error("--routes and at least one --score are required (or --selftest)")

    t0 = time.time()
    routes, prefix, S = read_routes(a.routes)
    D = S - prefix
    quota = quotas(a.cache_gib, a.quota)
    print(f"routes: {a.routes} steps={S} prefix={prefix} D={D}", flush=True)
    print(f"quota={a.quota} cache_gib={a.cache_gib:g} pin=all policy=lfu quota min={int(quota.min())} "
          f"max={int(quota.max())} sum={int(quota.sum())} max_spec={a.max_spec}", flush=True)
    _, _, base = simulate(routes, prefix, "lfu", quota, "all")
    base_pl = base.sum(axis=2).astype(np.int64)
    base_tok = float(base_pl.sum()) / D
    print(f"baseline (no speculation) misses/tok = {base_tok:.3f}  ({time.time() - t0:.1f}s)", flush=True)
    del base
    cfg = {"taus": TAUS, "decode_ms": a.decode_ms,
           "stall": [float(x) for x in a.stall_ms.split(",") if x],
           "wasted": [float(x) for x in a.wasted_ms.split(",") if x]}
    out = {"routes": a.routes, "prefix": prefix, "D": D, "cache_gib": a.cache_gib, "quota": a.quota,
           "quota_per_layer": quota.tolist(), "max_spec": a.max_spec, "decode_ms": a.decode_ms,
           "stall": cfg["stall"], "wasted": cfg["wasted"], "baseline_misses_per_tok": base_tok,
           "taus": [None if math.isinf(t) else t for t in TAUS], "scores": []}
    for path in a.score:
        t1 = time.time()
        z = load_score(path)
        if tuple(z["score"].shape) != (D, N_L, N_E):
            raise SystemExit(f"{path}: score shape {z['score'].shape}, expected {(D, N_L, N_E)}")
        fold = np.asarray(z["fold"], dtype=np.int64)
        if fold.shape != (D,):
            raise SystemExit(f"{path}: fold shape {fold.shape}, expected ({D},)")
        arr = pool_sim(routes, prefix, quota, z["score"], a.max_spec, base_pl)
        res = analyse(arr, base_pl, fold, cfg)
        res.update({"name": z["name"], "file": path})
        print_score(z["name"], res, cfg, base_tok, D)
        print(f"  [{z['name']}] simulated in {time.time() - t1:.1f}s", flush=True)
        out["scores"].append(res)
    print(f"\n[la_pool] done in {time.time() - t0:.1f}s", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[la_pool] wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
