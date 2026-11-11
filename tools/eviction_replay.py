#!/usr/bin/env python3
"""Offline expert-cache eviction replay (stdlib only; no model, GPU, NPU or network).

Run as: python3 -I tools/eviction_replay.py

Model
-----
Expert cache with one independent pool per layer (48 layers x 512 experts). Total expert
bytes of the trace's model = 50,363,105,280. Budgets are in GiB (2**30 bytes).

Quota schemes (C_l = slots of layer l):
  equal:   C_l = floor(budget_bytes * 512 / 50363105280) for every layer.
  prefill: C_l proportional to the number of distinct experts the prefill used in layer l,
           floor 64, cap 512, scaled so sum_l C_l = 48 * C_equal (largest-weight layers get
           the leftover units).

Replay per layer (layers are independent pools, so they are replayed one after another):
  1. Warm-up: the distinct prefill experts of the layer in first-use order (step, then slot)
     are accessed through the policy. Misses are counted as warm-up misses.
  2. Decode: tokens in step order; per token the 10 experts of the layer in slot order.
     A non-resident access is a decode miss (inserted, evicting a victim per policy).
     The 10 experts of the current token-layer are pinned: never chosen as victims.

Policies (per layer):
  clock  exact port of the Rust CLOCK (refbit on hit and on insert, hand persists, empty
         slots evictable immediately, pinned slots skipped without clearing their refbit).
  lru    least recently used.
  slru   probation 20% / protected 80%; probation hit promotes to protected MRU; protected
         overflow demotes its LRU to probation MRU; evict probation LRU.
  arc    Megiddo-Modha ARC with ghost lists B1/B2 and adaptive target p.
  lfu    access counts, all counts of the layer halved every 1000 layer accesses, ties by LRU.
  opt    Belady: evict the resident expert whose next use is farthest away (upper bound).
         Uses the full per-layer future (warm-up + decode).
"""
import collections
import csv
import json
import math
import os

TRACE = os.path.join(os.environ.get("FLASH_PREREQ_DIR", "results/flash-prerequisites"),
                     "flash-fixed353-expert-ids.csv")
OUT = os.path.join(os.environ.get("FLASH_RESULTS_DIR", "results/m3"), "eviction-replay.json")
N_LAYERS, N_EXP, TOP_K = 48, 512, 10
TOTAL_BYTES = 50363105280
EXPECT_RECORDS, EXPECT_DECODE_TOKENS, EXPECT_PREFILL_TOKENS = 242770, 353, 156
BUDGETS_GIB = [16, 18, 20]
SCHEMES = ["equal", "prefill"]
POLICIES = ["clock", "lru", "slru", "arc", "lfu", "opt"]
REF_LRU_PREFILL_16 = 58.9  # previous independent replay, for a plausibility check only
EMPTY = frozenset()
INF = float("inf")
OD = collections.OrderedDict


def lru_pick(od, pinned):
    """Least recently used key of an OrderedDict that is not pinned."""
    for k in od:
        if k not in pinned:
            return k
    raise RuntimeError("no evictable entry")


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


class LRU:
    def __init__(self, cap):
        self.cap, self.od = cap, OD()

    def access(self, e, pinned):
        if e in self.od:
            self.od.move_to_end(e)
            return True
        if len(self.od) >= self.cap:
            del self.od[lru_pick(self.od, pinned)]
        self.od[e] = None
        return False


class SLRU:
    def __init__(self, cap):
        self.cap = cap
        self.p_cap = max(1, int(0.2 * cap))
        self.q_cap = cap - self.p_cap
        self.prob, self.prot = OD(), OD()

    def access(self, e, pinned):
        if e in self.prot:
            self.prot.move_to_end(e)
            return True
        if e in self.prob:
            del self.prob[e]
            self.prot[e] = None
            if len(self.prot) > self.q_cap:
                d, _ = self.prot.popitem(last=False)
                self.prob[d] = None
            return True
        if len(self.prob) + len(self.prot) >= self.cap:
            if self.prob:
                del self.prob[lru_pick(self.prob, pinned)]
            else:
                del self.prot[lru_pick(self.prot, pinned)]
        self.prob[e] = None
        return False


class ARC:
    def __init__(self, cap):
        self.c, self.p = cap, 0.0
        self.t1, self.t2, self.b1, self.b2 = OD(), OD(), OD(), OD()

    def _replace(self, x, pinned):
        if self.t1 and (len(self.t1) > self.p or (x in self.b2 and len(self.t1) == self.p)):
            order = ((self.t1, self.b1), (self.t2, self.b2))
        else:
            order = ((self.t2, self.b2), (self.t1, self.b1))
        for src, ghost in order:
            for k in src:
                if k not in pinned:
                    del src[k]
                    ghost[k] = None
                    return
        raise RuntimeError("arc: no evictable entry")

    def access(self, x, pinned):
        c = self.c
        if x in self.t1:
            del self.t1[x]
            self.t2[x] = None
            return True
        if x in self.t2:
            self.t2.move_to_end(x)
            return True
        if x in self.b1:
            self.p = min(float(c), self.p + max(len(self.b2) / len(self.b1), 1.0))
            self._replace(x, pinned)
            del self.b1[x]
            self.t2[x] = None
            return False
        if x in self.b2:
            self.p = max(0.0, self.p - max(len(self.b1) / len(self.b2), 1.0))
            self._replace(x, pinned)
            del self.b2[x]
            self.t2[x] = None
            return False
        l1 = len(self.t1) + len(self.b1)
        if l1 == c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                self._replace(x, pinned)
            else:
                del self.t1[lru_pick(self.t1, pinned)]
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) == 2 * c:
                self.b2.popitem(last=False)
            self._replace(x, pinned)
        self.t1[x] = None
        return False


class LFU:
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


class OPT:
    def __init__(self, cap, stream):
        """stream: the layer's full access sequence (warm-up then decode), in replay order."""
        self.cap, self.res, self.i = cap, {}, 0
        self.nxt = [INF] * len(stream)
        last = {}
        for i in range(len(stream) - 1, -1, -1):
            self.nxt[i] = last.get(stream[i], INF)
            last[stream[i]] = i

    def access(self, e, pinned):
        nu = self.nxt[self.i]
        self.i += 1
        if e in self.res:
            self.res[e] = nu
            return True
        if len(self.res) >= self.cap:
            v = max((k for k in self.res if k not in pinned), key=self.res.__getitem__)
            del self.res[v]
        self.res[e] = nu
        return False


def make_policy(name, cap, stream):
    if name == "opt":
        return OPT(cap, stream)
    return {"clock": Clock, "lru": LRU, "slru": SLRU, "arc": ARC, "lfu": LFU}[name](cap)


def load_trace():
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
    groups = [[[e for _, e in sorted(d[s])] for s in sorted(d)] for d in dec]
    return n_rec, prefill, warm, groups


def quotas(budget, scheme, distinct):
    c_eq = budget * 2**30 * N_EXP // TOTAL_BYTES
    if scheme == "equal":
        return [c_eq] * N_LAYERS
    total = N_LAYERS * c_eq

    def alloc(k):
        return [min(N_EXP, max(64, int(math.floor(k * w)))) for w in distinct]

    lo, hi = 0.0, float(N_EXP)
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(alloc(mid)) <= total:
            lo = mid
        else:
            hi = mid
    caps = alloc(lo)
    order = sorted(range(N_LAYERS), key=lambda l: -distinct[l])
    rem, i = total - sum(caps), 0
    while rem > 0 and i < 100 * N_LAYERS:
        l = order[i % N_LAYERS]
        if caps[l] < N_EXP:
            caps[l] += 1
            rem -= 1
        i += 1
    return caps


def replay(policy, caps, warm, groups):
    wu_misses, dec_misses, per_layer = 0, 0, [0] * N_LAYERS
    for l in range(N_LAYERS):
        stream = warm[l] + [e for g in groups[l] for e in g] if policy == "opt" else None
        pol = make_policy(policy, caps[l], stream)
        for e in warm[l]:
            wu_misses += 0 if pol.access(e, EMPTY) else 1
        for g in groups[l]:
            pinned = frozenset(g)
            for e in g:
                if not pol.access(e, pinned):
                    per_layer[l] += 1
    dec_misses = sum(per_layer)
    dec_acc = sum(len(g) for gs in groups for g in gs)  # g already holds the TOP_K experts
    return wu_misses, dec_misses, per_layer, dec_acc


def main():
    n_rec, prefill, warm, groups = load_trace()
    dec_steps = [len(gs) for gs in groups]
    n_dec_tok = len(groups[0])
    bad = sum(1 for gs in groups for g in gs if len(g) != TOP_K or len(set(g)) != TOP_K)
    pre_groups = {(s, l) for s, _, l, _ in prefill}
    missing = collections.Counter(l for l in range(N_LAYERS)
                                  for s in range(EXPECT_PREFILL_TOKENS) if (s, l) not in pre_groups)
    distinct = [len(w) for w in warm]
    checks = {
        "records": n_rec, "records_expected": EXPECT_RECORDS, "records_ok": n_rec == EXPECT_RECORDS,
        "decode_tokens": n_dec_tok, "decode_tokens_expected": EXPECT_DECODE_TOKENS,
        "decode_tokens_ok": n_dec_tok == EXPECT_DECODE_TOKENS,
        "decode_token_layers_not_10_distinct": bad,
        "decode_token_layers_ok": bad == 0 and all(n == EXPECT_DECODE_TOKENS for n in dec_steps),
        "prefill_records": len(prefill), "prefill_token_layer_groups": len(pre_groups),
        "prefill_token_layer_groups_expected": EXPECT_PREFILL_TOKENS * N_LAYERS,
        "prefill_missing_groups_by_layer": dict(missing),
        "distinct_prefill_per_layer_min_max": [min(distinct), max(distinct)],
    }
    for k, v in checks.items():
        print(f"{k}: {v}")
    if not (checks["records_ok"] and checks["decode_tokens_ok"] and checks["decode_token_layers_ok"]):
        raise SystemExit("sanity check failed")

    caps_by = {}
    for b in BUDGETS_GIB:
        for sc in SCHEMES:
            caps_by[(b, sc)] = quotas(b, sc, distinct)
            caps = caps_by[(b, sc)]
            print(f"quota {b}GiB {sc}: sum={sum(caps)} min={min(caps)} max={max(caps)}")

    configs = []
    for pol in POLICIES:
        for b in BUDGETS_GIB:
            for sc in SCHEMES:
                caps = caps_by[(b, sc)]
                wu, dm, pl, dacc = replay(pol, caps, warm, groups)
                configs.append({
                    "policy": pol, "budget_gib": b, "quota": sc,
                    "capacity_min": min(caps), "capacity_max": max(caps),
                    "warmup_misses": wu, "decode_misses": dm,
                    "decode_accesses": dacc,
                    "misses_per_token": dm / EXPECT_DECODE_TOKENS,
                    "decode_hit_rate": 1 - dm / dacc,
                    "per_layer_decode_misses": pl,
                })
                print(f"  {pol:5s} {b}GiB {sc:8s} mpt={dm / EXPECT_DECODE_TOKENS:7.2f} "
                      f"hit={100 * (1 - dm / dacc):5.1f}% warm={wu}", flush=True)

    ref = next(c for c in configs if c["policy"] == "lru" and c["budget_gib"] == 16
               and c["quota"] == "prefill")
    ref_ok = 45 <= ref["misses_per_token"] <= 75
    print(f"reference LRU 16GiB prefill: {ref['misses_per_token']:.2f} misses/token "
          f"(previous replay {REF_LRU_PREFILL_16}; plausible 45-75: {ref_ok})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {
        "trace": TRACE, "sanity": checks,
        "quota_capacities": {f"{b}GiB_{sc}": caps for (b, sc), caps in caps_by.items()},
        "reference_lru_16_prefill_ok": ref_ok, "configs": configs,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")

    cols = [(b, sc) for b in BUDGETS_GIB for sc in SCHEMES]
    print("\n| policy | " + " | ".join(f"{b}G {sc}" for b, sc in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for pol in POLICIES:
        cells = []
        for b, sc in cols:
            c = next(x for x in configs if x["policy"] == pol and x["budget_gib"] == b
                     and x["quota"] == sc)
            cells.append(f"{c['misses_per_token']:.2f} ({100 * c['decode_hit_rate']:.1f}%)")
        print(f"| {pol} | " + " | ".join(cells) + " |")
    wu_set = sorted({c["warmup_misses"] for c in configs})
    print(f"\nCells: decode misses/token (decode hit rate). Warm-up misses: {wu_set}")


if __name__ == "__main__":
    main()
