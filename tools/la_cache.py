#!/usr/bin/env python3
"""Per-layer expert-cache simulator for the F-LOOKAHEAD pipeline (numpy only; offline).

Simulates the ENGINE's expert cache (crates/qwen4exp/src/cache.rs): one pool per layer, quota
slots per pool, policy clock (second-chance CLOCK) or lfu (port of Policy::Lfu: counts for every
expert of the layer, halved every 1000 demand accesses of the layer, victim = empty slot first,
else fewest counts, ties to the least recently used).

Quota schemes (slots per layer, C_l):
  equal:   floor(cache_gib * 2**30 * 512 / TOTAL_BYTES) for every layer (20 GiB -> 218).
  builtin: port of Cache::quotas with uniform slot bytes (2049280), weights = FLASH_QUOTA_WEIGHTS
           (stream.rs), floor 64 (stream.rs calls Cache::quotas(.., 64)), budget = cache_gib * 2**30.

Replay per layer: warm-up = the distinct prefill experts of the layer in first-use order (step,
then slot; -1 skipped), accessed through the policy with nothing pinned. Then decode tokens in
order; per (token, layer): the residency snapshot is recorded BEFORE the layer's accesses, then the
10 experts are accessed in slot order. A non-resident access is a miss (inserted, evicting a victim).
--pin all (default, the spec): all 10 experts of the token-layer are never evicted.
--pin touched (engine-faithful): only the experts already accessed in this token-layer are protected
(cache.rs protects slots stamped by the current FG_EXP_FETCH call; not-yet-accessed resident experts
of the same token-layer can still be evicted by an earlier miss of that call).

SIM file (np.savez_compressed): res, dem, miss bool [D,48,512]; quota int [48]; prefix, D int;
policy, quota_scheme str; cache_gib float.

Usage:
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 \
      ~/.cache/local-ai/venv-lookahead/bin/python -I tools/la_cache.py \
      --routes FIX.csv --cache-gib 20 --policy lfu --quota builtin --out results/la/sim.npz
  ... tools/la_cache.py --selftest           (fixed353 trace vs results/m3/eviction-replay.json)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookahead_train import N_E, N_L, TOP_K, read_routes  # noqa: E402

TOTAL_BYTES = 50363105280
SLOT_BYTES = TOTAL_BYTES // (N_L * N_E)  # 2049280, exact
ENGINE_FLOOR = 64  # stream.rs: Cache::quotas(&lay, cfg.cache_bytes, &w, 64)
LFU_AGE = 1000  # cache.rs: LFU_AGE
EMPTY = frozenset()
FIX = os.path.join(os.environ.get("FLASH_PREREQ_DIR", "results/flash-prerequisites"),
                   "flash-fixed353-expert-ids.csv")
REF_JSON = os.path.join(os.environ.get("FLASH_RESULTS_DIR", "results/m3"), "eviction-replay.json")
STREAM_RS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "qwen4exp",
                         "src", "stream.rs")
# stream.rs line 90 FLASH_QUOTA_WEIGHTS (checked against the source in --selftest)
FLASH_QUOTA_WEIGHTS = [
    342.0, 270.0, 249.0, 138.0, 211.0, 245.0, 266.0, 263.0, 186.0, 172.0, 232.0, 193.0, 246.0, 242.0,
    195.0, 121.0, 135.0, 198.0, 242.0, 200.0, 211.0, 236.0, 285.0, 183.0, 196.0, 176.0, 222.0, 196.0,
    242.0, 233.0, 203.0, 160.0, 169.0, 227.0, 235.0, 202.0, 228.0, 227.0, 277.0, 192.0, 191.0, 242.0,
    249.0, 202.0, 243.0, 226.0, 224.0, 218.0]


class Clock:
    """Exact port of the engine CLOCK (cache.rs), as in lookahead_replay.py / eviction_replay.py."""

    def __init__(self, cap):
        self.n, self.slot, self.refb = cap, [-1] * cap, [False] * cap
        self.map, self.hand = {}, 0

    def resident(self):
        return list(self.map.keys())

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


class Lfu:
    """Port of Policy::Lfu in cache.rs lookup_protect (demand accesses only; the sim has no prefetch).

    tick: per pool, +1 per access, counts of ALL experts of the layer incremented, then halved when
    tick % LFU_AGE == 0 (the halving happens after the increment, as in the engine). last[slot] = tick of
    the last access that hit or inserted the slot. Victim: first empty evictable slot (lowest index);
    otherwise the evictable slot minimising (cnt[owner], last), ties to the lowest slot index.
    """

    def __init__(self, cap):
        self.n = cap
        self.owner = [-1] * cap
        self.last = [0] * cap
        self.map = {}
        self.cnt = [0] * N_E
        self.tick = 0

    def resident(self):
        return list(self.map.keys())

    def access(self, e, pinned):
        self.tick += 1
        self.cnt[e] += 1
        if self.tick % LFU_AGE == 0:
            self.cnt = [c >> 1 for c in self.cnt]
        s = self.map.get(e)
        if s is not None:
            self.last[s] = self.tick
            return True
        best = -1
        best_key = None
        for h in range(self.n):
            o = self.owner[h]
            if o >= 0 and o in pinned:
                continue
            if o < 0:
                best = h
                break
            key = (self.cnt[o], self.last[h])
            if best_key is None or key < best_key:
                best, best_key = h, key
        if best < 0:
            raise RuntimeError("lfu: no evictable slot")
        v = best
        self.last[v] = self.tick
        o = self.owner[v]
        if o >= 0:
            del self.map[o]
        self.owner[v] = e
        self.map[e] = v
        return False


def quotas_equal(cache_gib):
    q = int(math.floor(cache_gib * 2**30 * N_E / TOTAL_BYTES))
    return np.full(N_L, q, dtype=np.int64)


def quotas_builtin(cache_gib, floor=ENGINE_FLOOR):
    """Port of Cache::quotas (cache.rs) with slot bytes uniform (layout.geom[l].slot == SLOT_BYTES)."""
    budget = int(cache_gib * (1 << 30))
    slot = [SLOT_BYTES] * N_L
    floor_bytes = sum(floor * slot[l] for l in range(N_L))
    if floor_bytes > budget:
        raise ValueError("budget below the per-layer floor")
    w = [FLASH_QUOTA_WEIGHTS[l] * slot[l] for l in range(N_L)]  # stream.rs: weight x slot
    wsum = 0.0
    for x in w:  # sequential sum, as Rust's iter().sum() (Python's sum() may be compensated)
        wsum += x
    spare = float(budget - floor_bytes)
    q = [min(N_E, floor + int(math.floor(spare * w[l] / wsum / slot[l]))) for l in range(N_L)]
    while sum(q[l] * slot[l] for l in range(N_L)) > budget:
        mx = max(q)
        l = max(i for i in range(N_L) if q[i] == mx)  # Rust max_by_key returns the LAST maximum
        q[l] -= 1
    return np.array(q, dtype=np.int64)


def quotas(cache_gib, scheme):
    if scheme == "equal":
        return quotas_equal(cache_gib)
    if scheme == "builtin":
        return quotas_builtin(cache_gib)
    raise ValueError(scheme)


def simulate(routes, prefix, policy, quota, pin="all"):
    """routes int16 [S,48,10] (-1 absent). Returns (res, dem, miss) bool [D,48,512], D = S - prefix."""
    S = routes.shape[0]
    D = S - prefix
    make = {"clock": Clock, "lfu": Lfu}[policy]
    pools = [make(int(quota[l])) for l in range(N_L)]
    for l in range(N_L):  # warm-up: distinct prefill experts, first-use order (step, slot)
        seen = set()
        pol = pools[l]
        for s in range(prefix):
            for x in routes[s, l].tolist():
                if x < 0 or x in seen:
                    continue
                seen.add(x)
                pol.access(x, EMPTY)
    res = np.zeros((D, N_L, N_E), dtype=bool)
    dem = np.zeros((D, N_L, N_E), dtype=bool)
    miss = np.zeros((D, N_L, N_E), dtype=bool)
    for j in range(D):
        R = routes[prefix + j]
        for l in range(N_L):
            pol = pools[l]
            keys = pol.resident()
            if keys:
                res[j, l, np.asarray(keys, dtype=np.int64)] = True
            g = [int(x) for x in R[l]]
            if len(g) != TOP_K or min(g) < 0 or len(set(g)) != TOP_K:
                raise ValueError(f"decode token {j} layer {l}: expected {TOP_K} distinct experts, got {g}")
            full = frozenset(g)
            touched = set()
            for e in g:
                dem[j, l, e] = True
                pinned = full if pin == "all" else frozenset(touched)
                if not pol.access(e, pinned):
                    miss[j, l, e] = True
                touched.add(e)
    return res, dem, miss


def run_config(routes, prefix, policy, scheme, cache_gib, pin="all"):
    quota = quotas(cache_gib, scheme)
    res, dem, miss = simulate(routes, prefix, policy, quota, pin)
    return quota, res, dem, miss


def stats(res, dem, miss):
    D = miss.shape[0]
    per_tok = miss.reshape(D, -1).sum(axis=1)
    blocks = [b for b in np.array_split(np.arange(D), 5)]
    per_fifth = [float(per_tok[b].mean()) for b in blocks]
    nonres = int((dem & ~res).sum())
    assert not np.any(dem & ~res & ~miss), "a non-resident demanded expert was not a miss"
    return {
        "misses_per_token": float(per_tok.sum()) / D,
        "decode_misses": int(per_tok.sum()),
        "per_fifth": per_fifth,
        "nonres_per_token_layer": nonres / (D * N_L),
        "miss_per_token_layer": float(miss.sum()) / (D * N_L),
    }


def save_sim(path, res, dem, miss, quota, prefix, policy, scheme, cache_gib):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, res=res, dem=dem, miss=miss, quota=np.asarray(quota, dtype=np.int64),
                        prefix=np.int64(prefix), D=np.int64(res.shape[0]), policy=np.str_(policy),
                        quota_scheme=np.str_(scheme), cache_gib=np.float64(cache_gib))


def selftest(routes, prefix, ref_path):
    with open(STREAM_RS) as f:
        src = f.read()
    line = next(x for x in src.splitlines() if "pub const FLASH_QUOTA_WEIGHTS" in x)
    rs_w = [float(x) for x in line.split("= [", 1)[1].split("];", 1)[0].split(",")]
    assert rs_w == FLASH_QUOTA_WEIGHTS, "FLASH_QUOTA_WEIGHTS differs from stream.rs"
    print("FLASH_QUOTA_WEIGHTS == stream.rs: ok")
    with open(ref_path) as f:
        ref = json.load(f)
    refc = {(c["policy"], c["budget_gib"], c["quota"]): c for c in ref["configs"]}
    tol = 0.05
    print(f"{'policy':6s} {'GiB':>4s} {'quota':6s} {'mine mpt':>10s} {'ref mpt':>10s} {'diff':>8s} "
          f"{'mine dm':>8s} {'ref dm':>7s}  verdict")
    ok_all = True
    for pol in ("clock", "lfu"):
        for gib in (16, 20):
            quota, res, dem, miss = run_config(routes, prefix, pol, "equal", gib)
            st = stats(res, dem, miss)
            r = refc[(pol, gib, "equal")]
            diff = st["misses_per_token"] - r["misses_per_token"]
            verdict = "PASS" if abs(diff) <= tol else "FAIL"
            if verdict == "FAIL":
                ok_all = False
            print(f"{pol:6s} {gib:4d} {'equal':6s} {st['misses_per_token']:10.3f} "
                  f"{r['misses_per_token']:10.3f} {diff:+8.3f} {st['decode_misses']:8d} "
                  f"{r['decode_misses']:7d}  {verdict}", flush=True)
            if pol == "lfu" and verdict == "FAIL":
                print("   (LFU reference differs; see report for the pin-rule explanation)")
    print("SELFTEST", "PASS" if ok_all else "FAIL (tolerance 0.05 misses/token)")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", default=FIX, help="route trace CSV (default: fixed353)")
    ap.add_argument("--cache-gib", type=float, default=20.0)
    ap.add_argument("--policy", choices=["lfu", "clock"], default="lfu")
    ap.add_argument("--quota", choices=["builtin", "equal"], default="builtin")
    ap.add_argument("--pin", choices=["all", "touched"], default="all",
                    help="all: the 10 experts of a token-layer are never evicted (spec); "
                         "touched: only already-accessed ones (engine cache.rs)")
    ap.add_argument("--out", help="SIM .npz output path")
    ap.add_argument("--ref", default=REF_JSON, help="eviction-replay.json for --selftest")
    ap.add_argument("--selftest", action="store_true", help="check clock/lfu + equal at 16 and 20 GiB")
    a = ap.parse_args()

    t0 = time.time()
    routes, prefix, S = read_routes(a.routes)
    print(f"routes: {a.routes}  steps={S} prefix={prefix} D={S - prefix}", flush=True)
    if a.selftest:
        ok = selftest(routes, prefix, a.ref)
        print(f"time {time.time() - t0:.1f}s")
        raise SystemExit(0 if ok else 1)
    if not a.out:
        ap.error("--out is required unless --selftest")
    quota, res, dem, miss = run_config(routes, prefix, a.policy, a.quota, a.cache_gib, a.pin)
    st = stats(res, dem, miss)
    print(f"policy={a.policy} quota={a.quota} cache_gib={a.cache_gib} pin={a.pin}")
    print(f"quota min={int(quota.min())} max={int(quota.max())} sum={int(quota.sum())}")
    print(f"misses/token (all {res.shape[0]} decode tokens) = {st['misses_per_token']:.3f} "
          f"(total {st['decode_misses']})")
    print("misses/token per fifth: " + " ".join(f"{x:.3f}" for x in st["per_fifth"]))
    print(f"mean non-resident demanded experts per (token,layer) = {st['nonres_per_token_layer']:.4f}; "
          f"mean misses per (token,layer) = {st['miss_per_token_layer']:.4f}")
    save_sim(a.out, res, dem, miss, quota, prefix, a.policy, a.quota, a.cache_gib)
    print(f"wrote {a.out}  time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
