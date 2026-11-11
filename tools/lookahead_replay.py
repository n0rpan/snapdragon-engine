#!/usr/bin/env python3
"""Cache-aware recall replay for learned expert lookahead (numpy only; offline).

For every held-out decode step t+1 (the same split as lookahead_train.py):
  1. Snapshot the per-layer CLOCK pool residency (state before step t+1's demand accesses).
  2. Prefetch: per layer, the top-K predicted experts that are NOT resident (or every
     non-resident expert with probability >= tau in threshold mode). Prefetches go to a
     staging set, never into the pool.
  3. Demand: the step's accesses (layer order, slot order) go through the CLOCK pools.
     A miss that was staged is a covered miss; it is inserted exactly as a demand miss.
     A staged expert not demanded in that step is a wasted read.
Prefetching never touches the pools, so the demand miss stream (misses/token without
lookahead) is identical for every predictor and is computed once.

Pools: one CLOCK per layer, capacity floor(GiB * 2**30 * 512 / 50363105280) (21 GiB -> 229).
Warm-up: distinct prefill experts per layer in first-use order, then every decode step
before the held-out range, in order.

Predictors: MLP checkpoints (from lookahead_train.py), per-layer training frequency, and
previous token (experts of step t). Scores are probabilities per layer.

Usage:
  ~/.cache/local-ai/venv-lookahead/bin/python -I tools/lookahead_replay.py \
      --lae X.lae --routes X.routes.csv --model w64.npz --model w128.npz --model w256.npz
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookahead_train import (  # noqa: E402
    N_E, N_L, TOP_K, freq_logprobs_from_train, load_dataset, load_mlp,
    mlp_probs_for_samples, prev_logprobs, sample_prev_targets,
)

TOTAL_BYTES = 50363105280
EMPTY = frozenset()
GO_RECALL, GO_WASTED = 0.75, 15.0  # first criterion (Opus-B's proposal), kept for reference only
# NET-TIME criterion (orchestrator, 2026-11-10; PROGRESS-M3 "Learned lookahead v1"): net ms/token = covered x s_c -
# wasted x s_w; GO if net >= 3% of decode ms at the optimistic wasted cost AND >= 0 at the pessimistic one.


class Clock:
    def __init__(self, cap):
        self.n, self.slot, self.refb = cap, [-1] * cap, [False] * cap
        self.map, self.hand = {}, 0

    def access(self, e, pinned):
        s = self.map.get(e)
        if s is not None:
            self.refb[s] = True
            return True
        steps = 0
        while True:
            h = self.hand
            self.hand = (h + 1) % self.n
            steps += 1
            if steps > 3 * self.n + 3:
                raise RuntimeError("clock: no evictable slot")
            if self.slot[h] in pinned:
                continue
            if self.refb[h]:
                self.refb[h] = False
                continue
            break
        if self.slot[h] >= 0:
            del self.map[self.slot[h]]
        self.slot[h], self.map[e], self.refb[h] = e, h, True
        return False


def cache_capacity(gib):
    return int(math.floor(gib * 2**30 * N_E / TOTAL_BYTES))


def replay_demand(ds, cap, log=print):
    """Warm the pools, then replay held-out steps. Returns per-step residency, demand, miss."""
    pools = [Clock(cap) for _ in range(N_L)]
    for l in range(N_L):
        for e in ds.prefill_warm[l].tolist():
            pools[l].access(int(e), EMPTY)
    for s in range(ds.prefix, ds.held0):
        R = ds.routes[s]
        for l in range(N_L):
            g = [int(x) for x in R[l]]
            pin = frozenset(g)
            for e in g:
                pools[l].access(e, pin)
    H = ds.n_hold
    res = np.zeros((H, N_L, N_E), bool)
    dem = np.zeros((H, N_L, N_E), bool)
    miss = np.zeros((H, N_L, N_E), bool)
    for j in range(H):
        R = ds.routes[ds.held0 + j]
        for l in range(N_L):
            keys = list(pools[l].map.keys())
            res[j, l, np.fromiter(keys, dtype=np.int64, count=len(keys))] = True
            g = [int(x) for x in R[l]]
            pin = frozenset(g)
            for e in g:
                dem[j, l, e] = True
                if not pools[l].access(e, pin):
                    miss[j, l, e] = True
    return res, dem, miss


def score_predictor(P, res, dem, miss, Ks, taus):
    """P: (H, 48, 512) probabilities. Returns dict of per-config totals (per token)."""
    H = P.shape[0]
    misses = int(miss.sum())
    rows = np.arange(N_L)[:, None]
    out = {}
    for K in Ks:
        out[("K", K)] = [0, 0, 0]  # prefetches, covered, wasted
    for tau in taus:
        out[("tau", tau)] = [0, 0, 0]
    for j in range(H):
        nonres = ~res[j]
        masked = np.where(nonres, P[j], -1.0)
        order = np.argsort(-masked, axis=1)[:, :max(Ks)]
        mj, dj = miss[j], dem[j]
        for K in Ks:
            st = np.zeros((N_L, N_E), bool)
            st[rows, order[:, :K]] = True
            acc = out[("K", K)]
            acc[0] += int(st.sum())
            acc[1] += int((st & mj).sum())
            acc[2] += int((st & ~dj).sum())
        for tau in taus:
            st = nonres & (P[j] >= tau)
            acc = out[("tau", tau)]
            acc[0] += int(st.sum())
            acc[1] += int((st & mj).sum())
            acc[2] += int((st & ~dj).sum())
    rows_out = []
    for key, (pref, cov, wst) in out.items():
        rows_out.append({
            "mode": key[0], "value": key[1],
            "prefetch_per_tok": pref / H, "covered_per_tok": cov / H,
            "recall": (cov / misses) if misses else float("nan"),
            "wasted_per_tok": wst / H,
        })
    return misses / H, rows_out


def per_layer_counts(P, res, dem, miss, taus, js):
    """For tokens js: per (tau, layer) totals of prefetches, covered, wasted with prefetch = non-resident & P >= tau."""
    T = len(taus)
    pre = np.zeros((T, N_L)); cov = np.zeros((T, N_L)); wst = np.zeros((T, N_L))
    for j in js:
        nonres = ~res[j]
        for ti, tau in enumerate(taus):
            st = nonres & (P[j] >= tau)
            pre[ti] += st.sum(axis=1)
            cov[ti] += (st & miss[j]).sum(axis=1)
            wst[ti] += (st & ~dem[j]).sum(axis=1)
    n = max(1, len(js))
    return pre / n, cov / n, wst / n


def net_per_layer(P, res, dem, miss, s_c, s_w_opt, s_w_pes, decode_ms, cap_reads):
    """Per-layer thresholds tuned on the first half of the held-out tokens (optimistic cost), evaluated on the second
    half only. Returns the evaluation numbers and the GO decision."""
    H = P.shape[0]
    taus = [1.0] + list(np.geomspace(0.5, 1e-4, 24))   # 1.0 = never prefetch this layer
    tune, evl = np.arange(0, H // 2), np.arange(H // 2, H)
    _, c_t, w_t = per_layer_counts(P, res, dem, miss, taus, tune)
    best = np.argmax(c_t * s_c - w_t * s_w_opt, axis=0)          # per layer
    p_e, c_e, w_e = per_layer_counts(P, res, dem, miss, taus, evl)
    L = np.arange(N_L)
    pre, cov, wst = p_e[best, L].sum(), c_e[best, L].sum(), w_e[best, L].sum()
    misses_e = float(miss[evl].sum()) / max(1, len(evl))
    opt = cov * s_c - wst * s_w_opt
    pes = cov * s_c - wst * s_w_pes
    reads = misses_e + pre
    return {"tuned_taus_per_layer": [float(taus[b]) for b in best], "prefetch_per_tok": float(pre),
            "covered_per_tok": float(cov), "wasted_per_tok": float(wst), "misses_per_tok": float(misses_e),
            "recall": float(cov / misses_e) if misses_e else float("nan"), "net_ms_opt": float(opt), "net_ms_pes": float(pes),
            "reads_per_tok": float(reads), "within_bandwidth_cap": bool(reads <= cap_reads),
            "go": bool(opt >= 0.03 * decode_ms and pes >= 0 and reads <= cap_reads)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lae", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--model", action="append", default=[], help="MLP .npz (repeatable)")
    ap.add_argument("--cache-gib", type=float, default=21.0)
    ap.add_argument("--taus", default="0.1,0.05,0.02,0.01,0.005,0.002,0.001")
    ap.add_argument("--no-freq", action="store_true")
    ap.add_argument("--no-prev", action="store_true")
    ap.add_argument("--json", help="optional output path for all per-config numbers")
    ap.add_argument("--stall-ms-per-miss", type=float, default=1.7, help="s_c: critical-path stall saved per covered miss (036 post-mortem: 1.6-1.8 ms from the cache-size and LFU A/Bs)")
    ap.add_argument("--wasted-ms", type=float, default=0.20, help="s_w optimistic (staged: decrypt + bandwidth)")
    ap.add_argument("--wasted-ms-pes", type=float, default=0.82, help="s_w pessimistic (M0, with cache pollution)")
    ap.add_argument("--decode-ms", type=float, default=112.4, help="decode ms/token at the cache size (21 GiB: 112.4)")
    ap.add_argument("--max-reads", type=float, default=200.0, help="bandwidth cap: demand misses + prefetches per token")
    args = ap.parse_args()

    t0 = time.time()
    ds = load_dataset(args.lae, args.routes)
    cap = cache_capacity(args.cache_gib)
    print(f"[replay] cache {args.cache_gib} GiB -> {cap} slots per layer (of {N_E}); "
          f"held-out steps {ds.held0}..{ds.held0 + ds.n_hold - 1} ({ds.n_hold} tokens)")
    taus = [float(x) for x in args.taus.split(",") if x]
    Ks = list(range(1, TOP_K + 1))

    res, dem, miss = replay_demand(ds, cap)
    base_misses = float(miss.sum()) / ds.n_hold
    dem_per_tok = float(dem.sum()) / ds.n_hold
    print(f"[replay] without lookahead: {base_misses:.2f} misses/token "
          f"of {dem_per_tok:.0f} demand accesses/token (CLOCK, {cap} slots/layer)")

    held_samples = np.arange(ds.n_train, ds.n)
    predictors = []
    for path in args.model:
        m_name = os.path.splitext(os.path.basename(path))[0]
        model = load_mlp(path)
        P = mlp_probs_for_samples(model, ds, held_samples)
        predictors.append((f"MLP W={model.width} ({m_name})", P))
        del model
    if not args.no_freq:
        fl = freq_logprobs_from_train(ds)
        predictors.append(("frequency", np.broadcast_to(np.exp(fl), (ds.n_hold, N_L, N_E)).copy()))
    if not args.no_prev:
        pl = prev_logprobs(sample_prev_targets(ds, held_samples))
        predictors.append(("previous-token", np.exp(pl)))

    print("\nGO criterion: miss recall >= %.2f with <= %.0f wasted reads/token (any config)."
          % (GO_RECALL, GO_WASTED))
    summary = {"cache_gib": args.cache_gib, "slots_per_layer": cap,
               "misses_per_tok_no_lookahead": base_misses, "predictors": {}}
    for name, P in predictors:
        _, rows = score_predictor(P, res, dem, miss, Ks, taus)
        for r in rows:
            r["net_ms_opt"] = float(r["covered_per_tok"] * args.stall_ms_per_miss - r["wasted_per_tok"] * args.wasted_ms)
            r["net_ms_pes"] = float(r["covered_per_tok"] * args.stall_ms_per_miss - r["wasted_per_tok"] * args.wasted_ms_pes)
        pl = net_per_layer(P, res, dem, miss, args.stall_ms_per_miss, args.wasted_ms, args.wasted_ms_pes,
                           args.decode_ms, args.max_reads)
        summary["predictors"][name] = {"rows": rows, "per_layer_thresholds": pl}
        print(f"\n== {name} ==")
        print(f"{'config':>12s} {'prefetch/tok':>13s} {'covered/tok':>12s} {'recall':>8s} {'wasted/tok':>11s} "
              f"{'net ms opt':>11s} {'net ms pes':>11s}")
        for r in rows:
            label = f"K={r['value']}" if r["mode"] == "K" else f"tau={r['value']:g}"
            print(f"{label:>12s} {r['prefetch_per_tok']:13.2f} {r['covered_per_tok']:12.2f} "
                  f"{r['recall']:8.3f} {r['wasted_per_tok']:11.2f} {r['net_ms_opt']:11.2f} {r['net_ms_pes']:11.2f}")
        print(f"  per-layer thresholds (tuned on held-out half 1, evaluated on half 2): prefetch/tok {pl['prefetch_per_tok']:.2f}, "
              f"covered/tok {pl['covered_per_tok']:.2f}, wasted/tok {pl['wasted_per_tok']:.2f}, recall {pl['recall']:.3f}, "
              f"net ms opt {pl['net_ms_opt']:.2f} / pes {pl['net_ms_pes']:.2f}, reads/tok {pl['reads_per_tok']:.1f}")
        print(f"  NET-TIME {'GO' if pl['go'] else 'NO-GO'}: {name} (needs net opt >= {0.03 * args.decode_ms:.2f} ms, "
              f"net pes >= 0, reads/tok <= {args.max_reads:.0f})")
        passing = [r for r in rows if r["recall"] >= GO_RECALL and r["wasted_per_tok"] <= GO_WASTED]
        if passing:
            b = max(passing, key=lambda r: (r["recall"], -r["wasted_per_tok"]))
            label = f"K={b['value']}" if b["mode"] == "K" else f"tau={b['value']:g}"
            print(f"GO: {name}: {label} recall={b['recall']:.3f} wasted/tok={b['wasted_per_tok']:.2f}")
        else:
            under = [r for r in rows if r["wasted_per_tok"] <= GO_WASTED]
            best = max(under, key=lambda r: r["recall"]) if under else None
            extra = (f" best recall within the waste budget = {best['recall']:.3f}" if best else
                     " no config within the waste budget")
            print(f"NO-GO: {name}:{extra}")
    print(f"\n[replay] done in {time.time() - t0:.1f}s")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[replay] wrote {args.json}")


if __name__ == "__main__":
    main()
