#!/usr/bin/env python3
"""Offline MTP (multi-token prediction) cycle-cost simulator on the expert routing trace.

Stdlib only; no model, GPU, NPU or network. Run as: python3 -I tools/mtp_cycle_sim.py

Part A  Expert-union size and overlap of consecutive decode tokens, windows w = 1..4.
        Union per layer over sliding windows of w tokens; mean over layers and windows.
Part B  Per-layer expert cache replay (CLOCK and LFU, copied from eviction_replay.py) where
        a verify batch of w consecutive decode tokens is looked up as one union per layer:
        all union members are pinned while the group is processed. Warm-up uses the prefill's
        distinct experts of each layer in first-use order, as in eviction_replay.py.
        Groups are non-overlapping; a trailing partial group is counted as a group.
Part C  Cycle-cost model for verify width w = depth + 1 (depth 0 = plain decode).

Cache capacity: 20 GiB equal quota = 218 slots per layer (floor(20 GiB * 512 / total expert bytes)).
"""
import argparse
import collections
import csv
import json
import os

TRACE = os.path.join(os.environ.get("FLASH_PREREQ_DIR", "results/flash-prerequisites"),
                     "flash-fixed353-expert-ids.csv")
OUT = os.path.join(os.environ.get("FLASH_RESULTS_DIR", "results/m3"), "mtp-cycle-sim.json")
EVICTION_REF = os.path.join(os.environ.get("FLASH_RESULTS_DIR", "results/m3"), "eviction-replay.json")
N_LAYERS, N_EXP, TOP_K = 48, 512, 10
TOTAL_BYTES = 50363105280
EXPECT_RECORDS, EXPECT_DECODE_TOKENS = 242770, 353
MAX_W = 4
POLICIES = ["clock", "lfu"]
EMPTY = frozenset()
OD = collections.OrderedDict


class Clock:
    """Exact port of eviction_replay.py Clock (refbit on hit and insert, hand persists)."""

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


class LFU:
    """Copy of eviction_replay.py LFU (halving every 1000 layer accesses, ties by LRU)."""

    def __init__(self, cap):
        self.cap, self.res, self.cnt, self.n = cap, {}, collections.Counter(), 0

    def access(self, e, pinned):
        self.n += 1
        self.cnt[e] += 1
        if self.n % 1000 == 0:
            for k in list(self.cnt):
                self.cnt[k] >>= 1
        hit = e in self.res
        if not hit and len(self.res) >= self.cap:
            v = min((k for k in self.res if k not in pinned),
                    key=lambda k: (self.cnt[k], self.res[k]))
            del self.res[v]
        self.res[e] = self.n
        return hit


POLICY_CLS = {"clock": Clock, "lfu": LFU}


def load_trace():
    """Same parsing as eviction_replay.py load_trace()."""
    prefill, dec = [], [collections.defaultdict(list) for _ in range(N_LAYERS)]
    n_rec = 0
    with open(TRACE, newline="") as f:
        for r in csv.DictReader(f):
            n_rec += 1
            s, ph, l, sl, e = (int(r[k]) for k in ("step", "phase", "layer", "slot", "expert"))
            if ph == 0:
                prefill.append((s, sl, l, e))
            else:
                dec[l][s].append((sl, e))
    warm = [[] for _ in range(N_LAYERS)]
    seen = [set() for _ in range(N_LAYERS)]
    for s, sl, l, e in sorted(prefill):
        if e not in seen[l]:
            seen[l].add(e)
            warm[l].append(e)
    # groups[l][t] = the TOP_K experts of decode token t at layer l, in slot order
    groups = [[[e for _, e in sorted(d[s])] for s in sorted(d)] for d in dec]
    return n_rec, prefill, warm, groups


def part_a(groups):
    T = len(groups[0])
    windows = {}
    for w in range(1, MAX_W + 1):
        tot, cnt, mn, mx = 0, 0, None, None
        for l in range(N_LAYERS):
            gl = groups[l]
            for t in range(T - w + 1):
                u = set()
                for k in range(w):
                    u.update(gl[t + k])
                n = len(u)
                tot += n
                cnt += 1
                mn = n if mn is None else min(mn, n)
                mx = n if mx is None else max(mx, n)
        mean = tot / cnt
        windows[w] = {"windows_per_layer": T - w + 1, "mean_union": mean,
                      "ratio_union_over_10w": mean / (TOP_K * w),
                      "min_union": mn, "max_union": mx}
    per_layer_overlap = []
    for l in range(N_LAYERS):
        gl = groups[l]
        s = sum(len(set(gl[t]) & set(gl[t + 1])) for t in range(T - 1))
        per_layer_overlap.append(s / (T - 1))
    return {
        "windows": windows,
        "mean_overlap_consecutive": sum(per_layer_overlap) / N_LAYERS,
        "per_layer_overlap": per_layer_overlap,
    }


def part_b(policy, cap, warm, groups, w):
    T = len(groups[0])
    cls = POLICY_CLS[policy]
    pols = [cls(cap) for _ in range(N_LAYERS)]
    for l in range(N_LAYERS):
        for e in warm[l]:
            pols[l].access(e, EMPTY)
    misses, n_groups = 0, 0
    for start in range(0, T, w):
        n_groups += 1
        stop = min(start + w, T)
        for l in range(N_LAYERS):
            union, seen = [], set()
            for t in range(start, stop):
                for e in groups[l][t]:
                    if e not in seen:
                        seen.add(e)
                        union.append(e)
            pinned = frozenset(union)
            pol = pols[l]
            for e in union:
                if not pol.access(e, pinned):
                    misses += 1
    return {"policy": policy, "w": w, "groups": n_groups, "tokens": T, "misses": misses,
            "misses_per_group": misses / n_groups, "misses_per_token": misses / T}


def part_c(args, union_mean, misses_pg):
    p = [args.p1, args.p2, args.p3]
    rows = []
    for d in range(0, MAX_W):
        w = d + 1
        dense = args.dense_base + args.dense_step * (w - 1)
        experts = args.expert_ms * (union_mean[w] / TOP_K)
        wait = misses_pg[w] * args.ms_per_miss / args.qd_overlap
        draft = d * args.draft_ms
        cycle = dense + experts + args.overhead + wait + draft
        exp_tok = 1 + sum(p[:d])
        rows.append({"depth": d, "w": w, "dense_ms": dense, "experts_ms": experts,
                     "overhead_ms": args.overhead, "wait_ms": wait, "draft_ms": draft,
                     "cycle_ms": cycle, "expected_tokens": exp_tok,
                     "ms_per_token": cycle / exp_tok})
    base = rows[0]["ms_per_token"]
    for r in rows:
        r["speedup_vs_depth0"] = base / r["ms_per_token"]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget-gib", type=int, default=20)
    ap.add_argument("--dense-base", type=float, default=42.0, help="dense(1) ms")
    ap.add_argument("--dense-step", type=float, default=6.0, help="dense ms per extra row")
    ap.add_argument("--expert-ms", type=float, default=18.0,
                    help="expert ms per token at 10 experts/layer (scaled by union/10)")
    ap.add_argument("--overhead", type=float, default=30.0, help="graph overhead per verify (ms)")
    ap.add_argument("--ms-per-miss", type=float, default=0.5)
    ap.add_argument("--qd-overlap", type=float, default=4.0)
    ap.add_argument("--draft-ms", type=float, default=4.0, help="draft head per drafted token")
    ap.add_argument("--p1", type=float, default=0.945, help="P(1st drafted accepted)")
    ap.add_argument("--p2", type=float, default=0.86, help="P(first 2 accepted), prefix")
    ap.add_argument("--p3", type=float, default=0.75, help="P(first 3 accepted), prefix")
    args = ap.parse_args()

    n_rec, prefill, warm, groups = load_trace()
    T = len(groups[0])
    bad = sum(1 for gs in groups for g in gs if len(g) != TOP_K or len(set(g)) != TOP_K)
    checks = {"records": n_rec, "records_ok": n_rec == EXPECT_RECORDS,
              "decode_tokens": T, "decode_tokens_ok": T == EXPECT_DECODE_TOKENS,
              "decode_token_layers_not_10_distinct": bad,
              "decode_token_layers_ok": bad == 0 and all(len(gs) == T for gs in groups)}
    if not (checks["records_ok"] and checks["decode_tokens_ok"] and checks["decode_token_layers_ok"]):
        raise SystemExit(f"trace sanity failed: {checks}")

    cap = args.budget_gib * 2**30 * N_EXP // TOTAL_BYTES
    a = part_a(groups)
    b = {pol: {w: part_b(pol, cap, warm, groups, w) for w in range(1, MAX_W + 1)}
         for pol in POLICIES}
    union_mean = {w: a["windows"][w]["mean_union"] for w in range(1, MAX_W + 1)}
    misses_pg = {w: b["clock"][w]["misses_per_group"] for w in range(1, MAX_W + 1)}
    c = part_c(args, union_mean, misses_pg)

    # Sanity 1: Part A w=1 union must be exactly 10.
    w1 = a["windows"][1]
    sanity_a = w1["min_union"] == TOP_K and w1["max_union"] == TOP_K and w1["mean_union"] == TOP_K
    # Sanity 2: Part B w=1 CLOCK misses/token vs eviction_replay.py reference.
    ref_val, ref_src = None, "not found"
    if os.path.exists(EVICTION_REF):
        with open(EVICTION_REF) as f:
            for cfg in json.load(f)["configs"]:
                if cfg["policy"] == "clock" and cfg["budget_gib"] == args.budget_gib \
                        and cfg["quota"] == "equal":
                    ref_val, ref_src = cfg["misses_per_token"], EVICTION_REF
    mine = b["clock"][1]["misses_per_token"]
    sanity_b = ref_val is not None and abs(mine - ref_val) <= 0.01
    sanity = {"partA_w1_union_exactly_10": sanity_a,
              "partB_w1_clock_misses_per_token": mine,
              "eviction_replay_clock_equal_misses_per_token": ref_val,
              "eviction_replay_source": ref_src,
              "partB_w1_matches_within_0.01": sanity_b,
              "trace_checks": checks, "capacity_per_layer": cap}

    out = {"trace": TRACE, "params": vars(args), "sanity": sanity,
           "part_a": a, "part_b": b, "part_c": c}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # Compact markdown summary.
    print(f"cap/layer={cap} decode tokens={T} layers={N_LAYERS} experts/token={TOP_K}")
    print("\n### Part A (decode windows)\n")
    print("| w | mean union/layer | union/(10w) | min | max |")
    print("|---|---|---|---|---|")
    for w in range(1, MAX_W + 1):
        x = a["windows"][w]
        print(f"| {w} | {x['mean_union']:.3f} | {x['ratio_union_over_10w']:.4f} | "
              f"{x['min_union']} | {x['max_union']} |")
    print(f"\nmean |E_t & E_t+1| per layer: {a['mean_overlap_consecutive']:.3f}")
    print("\n### Part B (misses, cap 218/layer, groups of w tokens)\n")
    print("| w | CLOCK misses/group | CLOCK misses/token | LFU misses/group | LFU misses/token |")
    print("|---|---|---|---|---|")
    for w in range(1, MAX_W + 1):
        cl, lf = b["clock"][w], b["lfu"][w]
        print(f"| {w} | {cl['misses_per_group']:.2f} | {cl['misses_per_token']:.2f} | "
              f"{lf['misses_per_group']:.2f} | {lf['misses_per_token']:.2f} |")
    print("\n### Part C (cycle model, CLOCK misses)\n")
    print("| depth | w | dense | experts | OV | wait | draft | cycle ms | E[tok] | ms/tok | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in c:
        print(f"| {r['depth']} | {r['w']} | {r['dense_ms']:.2f} | {r['experts_ms']:.2f} | "
              f"{r['overhead_ms']:.1f} | {r['wait_ms']:.2f} | {r['draft_ms']:.1f} | "
              f"{r['cycle_ms']:.2f} | {r['expected_tokens']:.3f} | {r['ms_per_token']:.2f} | "
              f"{r['speedup_vs_depth0']:.3f} |")
    print("\n### Sanity")
    print(f"- Part A w=1 union exactly 10: {sanity_a}")
    print(f"- Part B w=1 CLOCK misses/token {mine:.4f} vs eviction_replay {ref_val} "
          f"(|diff| <= 0.01: {sanity_b})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
