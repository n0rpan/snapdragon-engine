#!/usr/bin/env python3
"""Per-draft-position MTP acceptance breakdown from mtp-check-*.json (WP-D5a task 2, offline).

Input: JSON with prompts[i].depths[j].mtp = {verify_steps, drafted, accepted, accepted_at_depth,
tokens_out, tokens_per_verify, time_s{draft,verify,catchup,...}, ...}.

Definitions (checked against the files; see the consistency notes printed per model):
  accepted_at_depth[k-1] = S_k = number of verify steps whose accepted prefix is >= k drafts
                           (prefix survival; sum(accepted_at_depth) == accepted).
  S_0 = verify_steps. Every verify step proposes one draft per position, so
  drafted == verify_steps * depth.
  P(accept draft 1)                       = S_1 / S_0
  P(accept draft k | draft k-1 accepted)  = S_k / S_{k-1}      (k = 2..depth)
  tokens/verify                           = tokens_out / verify_steps (equals the file's field)
  draft share                             = draft / (draft + verify + catchup), from time_s
                                            (equals time_split.draft_pct / 100)
Pooled rows sum the raw counts over prompts and then take ratios (not means of per-prompt ratios).
Cells show "-" where the position is not drafted at that depth or the conditioning count is zero.

Usage
  tools/mtp_accept_breakdown.py [--dir DIR] [--files mtp-check-ornith.json mtp-check-qwen36.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_DIR = os.environ.get("MTP_CHECK_DIR", "results")
DEFAULT_FILES = ("mtp-check-ornith.json", "mtp-check-qwen36.json")


def _acc_list(mtp, depth):
    s = list(mtp["accepted_at_depth"])
    return s + [0] * (depth - len(s))


def _cond(num, den):
    return None if not den else num / den


def _pct(x):
    return "-" if x is None else f"{100 * x:6.1f}%"


def _num(x, spec):
    return "-" if x is None else format(x, spec)


def _check(label, mtp, depth):
    notes = []
    if sum(mtp["accepted_at_depth"]) != mtp["accepted"]:
        notes.append(f"{label}: sum(accepted_at_depth) != accepted")
    if mtp["drafted"] != mtp["verify_steps"] * depth:
        notes.append(f"{label}: drafted != verify_steps*depth")
    if len(mtp["accepted_at_depth"]) > depth:
        notes.append(f"{label}: accepted_at_depth longer than depth")
    return notes


def _draft_share(ts):
    tot = ts.get("draft", 0.0) + ts.get("verify", 0.0) + ts.get("catchup", 0.0)
    return None if not tot else ts.get("draft", 0.0) / tot


def load(path):
    with open(path) as f:
        return json.load(f)


def rows_for(data):
    """Yield (prompt_label, depth, mtp_dict) for every prompt and depth."""
    for p in data["prompts"]:
        label = os.path.basename(p["ref"].rstrip("/"))
        for d in p["depths"]:
            yield label, int(d["depth"]), d["mtp"]


def summarise_row(mtp, depth, max_depth):
    S = _acc_list(mtp, depth)
    S0 = mtp["verify_steps"]
    chain = [S0] + S[:depth]  # chain[k] = S_k
    p = [None] * (max_depth + 1)  # p[k] for k = 1..max_depth
    for k in range(1, max_depth + 1):
        if k > depth:
            continue
        if k == 1:
            p[k] = _cond(chain[1], chain[0])
        else:
            p[k] = _cond(chain[k], chain[k - 1])
    tpv = mtp["tokens_out"] / mtp["verify_steps"] if mtp["verify_steps"] else None
    return {
        "verify_steps": mtp["verify_steps"],
        "tok_per_verify": tpv,
        "draft_share": _draft_share(mtp["time_s"]),
        "p": p,
        "S": chain,
    }


def pooled(entries, depth, max_depth):
    """Sum raw counts across prompts for one depth, then form the ratios."""
    verify = sum(m["verify_steps"] for _, d, m in entries if d == depth)
    tokens = sum(m["tokens_out"] for _, d, m in entries if d == depth)
    ts = {"draft": 0.0, "verify": 0.0, "catchup": 0.0}
    S = [0] * (depth + 1)
    S[0] = verify
    for _, d, m in entries:
        if d != depth:
            continue
        for k, v in enumerate(_acc_list(m, depth), start=1):
            S[k] += v
        for key in ts:
            ts[key] += m["time_s"].get(key, 0.0)
    p = [None] * (max_depth + 1)
    for k in range(1, max_depth + 1):
        if k > depth:
            continue
        p[k] = _cond(S[k], S[k - 1])
    return {
        "verify_steps": verify,
        "tok_per_verify": tokens / verify if verify else None,
        "draft_share": _draft_share(ts),
        "p": p,
        "S": S,
    }


def print_model(name, data):
    entries = list(rows_for(data))
    depths = sorted({d for _, d, _ in entries})
    max_depth = max(depths)
    print(f"== {name}  ({data.get('model', '?')})")
    head = f"{'prompt':22} {'depth':>5} {'verify':>6} {'tok/vfy':>7} {'draft%':>7} " + " ".join(
        f"{'P(d' + str(k) + (')' if k == 1 else '|d' + str(k - 1) + ')'):>8}" for k in range(1, max_depth + 1))
    print(head)
    print("-" * len(head))
    notes = []
    for label, d, m in entries:
        r = summarise_row(m, d, max_depth)
        cells = " ".join(f"{_pct(r['p'][k]):>8}" for k in range(1, max_depth + 1))
        print(f"{label:22} {d:>5} {r['verify_steps']:>6} {_num(r['tok_per_verify'], '7.3f'):>7} "
              f"{_pct(r['draft_share']):>7} {cells}")
        notes += _check(f"{label} d{d}", m, d)
    print("-" * len(head))
    for d in depths:
        r = pooled(entries, d, max_depth)
        cells = " ".join(f"{_pct(r['p'][k]):>8}" for k in range(1, max_depth + 1))
        print(f"{'POOLED':22} {d:>5} {r['verify_steps']:>6} {_num(r['tok_per_verify'], '7.3f'):>7} "
              f"{_pct(r['draft_share']):>7} {cells}")
    # Flag depth rows that match the previous depth exactly on verify/accepted (suspicious for MTP chains).
    by_label = {}
    for label, d, m in entries:
        by_label.setdefault(label, {})[d] = m
    for label, dd in by_label.items():
        for d in sorted(dd):
            if d - 1 in dd and dd[d]["verify_steps"] == dd[d - 1]["verify_steps"] and \
                    dd[d]["accepted"] == dd[d - 1]["accepted"] and dd[d]["drafted"] != dd[d - 1]["drafted"]:
                notes.append(f"{label}: depth {d} has identical verify_steps/accepted to depth {d - 1} "
                             f"(drafted differs); the extra draft position never accepted")
    if notes:
        print("notes:")
        for n in notes:
            print("  " + n)
    else:
        print("notes: counts consistent (accepted_at_depth sums, drafted = verify_steps*depth)")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    args = ap.parse_args(argv)
    for fn in args.files:
        path = fn if os.path.isabs(fn) else os.path.join(args.dir, fn)
        print_model(os.path.basename(path), load(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
