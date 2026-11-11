#!/usr/bin/env python3
"""Static-plan decision rule for the JSON written by `flash-m3 run`.

Usage: python3 -I tools/flash_profile_rule.py RUN.json [--json]
Stdlib only. Reads one file; no model, cargo, git or lock files.
"""
import argparse
import json
import sys

# First matching bucket wins, in dict order. "shexp" must precede "experts"
# because "shexp" contains the substring "exp".
BUCKETS = {
    "shared_expert": ["shexp"],
    "experts": ["ffn_moe", "exp"],
    "attention": ["kq", "attn", "kqv"],
    "gdn_ssm": ["ssm", "gdn", "conv", "delta"],
    "hyper_conn": ["hc"],
    "ple": ["ple"],
    "indexer": ["idx", "index", "pool", "qsa"],
    "norms": ["norm"],
    "output_head": ["result_output", "output", "logits"],
}
OTHER = "other"
EXPERTS = "experts"
CALIB = "_empty_node"


def fail(msg):
    print(f"flash_profile_rule: {msg}", file=sys.stderr)
    sys.exit(2)


def num(v, what):
    try:
        return float(v)
    except (TypeError, ValueError):
        fail(f"{what} is missing or not a number: {v!r}")


def bucket_of(kind):
    for name, pats in BUCKETS.items():
        if any(p in kind for p in pats):
            return name
    return OTHER


def main():
    ap = argparse.ArgumentParser(description="Static-plan rule for flash-m3 profile output.")
    ap.add_argument("run_json")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    args = ap.parse_args()

    try:
        with open(args.run_json, encoding="utf-8") as f:
            run = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        fail(f"cannot read {args.run_json}: {e}")
    if not isinstance(run, dict):
        fail("top-level JSON is not an object")

    prof = run.get("profile_ms_per_token")
    if prof is None:
        fail("profile_ms_per_token is null or missing; rerun flash-m3 with profiling enabled")
    if not isinstance(prof, list):
        fail("profile_ms_per_token is not a list")

    mean_block = run.get("ms_per_token")
    mean = mean_block.get("mean") if isinstance(mean_block, dict) else None
    if mean is None:
        fail("ms_per_token.mean is missing")
    mean = num(mean, "ms_per_token.mean")
    wait = num(run.get("expert_wait_ms_per_token_thread0"), "expert_wait_ms_per_token_thread0")
    fetch = run.get("fetch_ms_per_token")

    calib = [r for r in prof if isinstance(r, dict) and r.get("kind") == CALIB]
    cal_calls = sum(num(r.get("calls_per_token"), "calibration calls_per_token") for r in calib)
    cal_ms = sum(num(r.get("ms_per_token"), "calibration ms_per_token") for r in calib)
    if not calib or cal_calls <= 0:
        fail(f"no usable {CALIB} calibration row in profile_ms_per_token")
    empty_ms = cal_ms / cal_calls

    rows = []
    for r in prof:
        if not isinstance(r, dict) or r.get("kind") == CALIB:
            continue
        kind = str(r.get("kind", "?"))
        calls = num(r.get("calls_per_token"), f"{kind} calls_per_token")
        raw = num(r.get("ms_per_token"), f"{kind} ms_per_token")
        net = max(0.0, raw - calls * empty_ms)
        rows.append({"kind": kind, "calls": calls, "raw_ms": raw, "net_ms": net, "bucket": bucket_of(kind)})
    # Node-by-node profiling cannot overlap the expert SSD reads with other nodes: the first expert compute node after
    # the fetch (ffn_moe_gate, else ffn_moe_up) absorbs the wait. Its own compute equals its sibling's (same shapes and
    # types), so the excess is IO, not compute (ticket 010: 25.7 ms of 32.5 ms). Move it out of the compute sum.
    by = {r["kind"]: r for r in rows}
    io_profile = 0.0
    g, u = by.get("ffn_moe_gate"), by.get("ffn_moe_up")
    if g and u:
        first, sib = (g, u) if g["net_ms"] >= u["net_ms"] else (u, g)
        io_profile = max(0.0, first["net_ms"] - sib["net_ms"])
        first["net_ms"] -= io_profile
        rows.append({"kind": "expert_io_absorbed (excluded)", "calls": 0.0, "raw_ms": io_profile, "net_ms": 0.0, "bucket": "other"})
    rows.sort(key=lambda r: -r["net_ms"])

    compute = sum(r["net_ms"] for r in rows)
    nodes = sum(r["calls"] for r in rows)
    buckets = {name: 0.0 for name in list(BUCKETS) + [OTHER]}
    for r in rows:
        buckets[r["bucket"]] += r["net_ms"]
    other_kinds = sorted({r["kind"] for r in rows if r["bucket"] == OTHER})

    overhead = (mean - wait) - compute
    per_node_us = overhead * 1000.0 / nodes if nodes > 0 else None
    non_exp = [(k, v) for k, v in buckets.items() if k != EXPERTS]
    non_expert_max_name, non_expert_max = max(non_exp, key=lambda kv: kv[1])

    if overhead >= 10:
        static, reason = True, f"overhead_ms {overhead:.2f} >= 10"
    elif mean > 105 and overhead > non_expert_max:
        static, reason = True, (f"ms/token {mean:.2f} > 105 and overhead_ms {overhead:.2f} exceeds "
                                f"every non-expert bucket (max {non_expert_max_name} {non_expert_max:.2f})")
    else:
        why = [f"overhead_ms {overhead:.2f} < 10"]
        if mean <= 105:
            why.append(f"ms/token {mean:.2f} <= 105")
        else:
            why.append(f"overhead_ms not above {non_expert_max_name} {non_expert_max:.2f}")
        static, reason = False, "; ".join(why)

    result = {
        "empty_ms_per_call": empty_ms,
        "nodes_per_token": nodes,
        "rows": rows,
        "compute_ms": compute,
        "expert_io_absorbed_ms": io_profile,
        "buckets_ms": buckets,
        "other_kinds": other_kinds,
        "ms_per_token_mean": mean,
        "expert_wait_ms_per_token_thread0": wait,
        "fetch_ms_per_token": fetch,
        "overhead_ms": overhead,
        "overhead_per_node_us": per_node_us,
        "static_plan": static,
        "reason": reason,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"empty_ms_per_call: {empty_ms:.6f}  (calibration calls/token {cal_calls:.1f})")
        print(f"nodes/token: {nodes:.1f}")
        print(f"{'kind':<36} {'calls/tok':>10} {'raw_ms':>10} {'net_ms':>10} {'%comp':>7}")
        for r in rows:
            pct = 100.0 * r["net_ms"] / compute if compute > 0 else 0.0
            print(f"{r['kind']:<36} {r['calls']:>10.2f} {r['raw_ms']:>10.3f} {r['net_ms']:>10.3f} {pct:>6.1f}%")
        print("buckets:")
        for name, v in buckets.items():
            pct = 100.0 * v / compute if compute > 0 else 0.0
            print(f"  {name:<16} {v:>10.3f} ms {pct:>6.1f}%")
        print(f"other kinds: {', '.join(other_kinds) if other_kinds else '(none)'}")
        print(f"compute_ms: {compute:.3f}")
        print(f"ms/token mean: {mean:.3f}  expert_wait_thread0: {wait:.3f}  fetch (info): {fetch}")
        per_node_s = f"{per_node_us:.3f}" if per_node_us is not None else "n/a"
        print(f"overhead_ms: {overhead:.3f}  overhead_per_node_us: {per_node_s}")
        print(f"STATIC PLAN: {'YES' if static else 'NO'} ({reason})")


if __name__ == "__main__":
    main()
