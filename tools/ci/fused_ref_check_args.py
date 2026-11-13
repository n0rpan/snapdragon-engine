#!/usr/bin/env python3
"""Subprocess tests of `oxe fused-ref ... --check-args` (no tokenizer, no model, no hardware).

Runs the three documented O-117 commands of FUSED-REF.md (prefill-fa2, prefill-fa2x and
fused-decode, at 32K and 64K) through the real CLI parser, plus the inputs it must refuse.
Usage: fused_ref_check_args.py <path to oxe built with --features f32-reference>
"""
import json
import subprocess
import sys

OXE = sys.argv[1]
BASE = ["--model", "/nonexistent/m.gguf", "--cpu-list", "0-7", "--prefix-ids-file", "/nonexistent/ids.txt",
        "--n", "1000", "--ref-samples", "200"]


def run(args):
    p = subprocess.run([OXE, "fused-ref", *args, "--check-args"], capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr


def resolved(out):
    for line in out.splitlines():
        if line.startswith("{") and "fused_ref_resolved" in line:
            return json.loads(line)["fused_ref_resolved"]
    raise AssertionError(f"no fused_ref_resolved line in {out!r}")


fails = 0


def check(name, ok, detail):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += not ok


# the documented commands
for c in (32768, 65536):
    for k in ("prefill-fa2", "prefill-fa2x"):
        args = ["--candidate", k, *BASE, "--prefix-len", str(c), "--state-file", f"/tmp/st/pf-f16-{c}.state",
                "--ubatch", "256", "--control", "decode", "--out", f"/tmp/o117-{k}-{c}.json"]
        rc, out, err = run(args)
        ok = rc == 0
        if ok:
            r = resolved(out)
            ok = (r["candidate"], r["control"], r["prefix_len"], r["n"], r["ref_samples"], r["ubatch"]) == (k, "decode", c, 1000, 200, 256)
            ok = ok and (r["kld_factor"], r["top1_slack"], r["ref_ratio_max"], r["min_positions"], r["min_ref_samples"]) == (1.0, 0.005, 1.1, 1000, 200)
        check(f"documented {k} {c}", ok, (rc, out, err))
    args = ["--candidate", "fused-decode", *BASE, "--prefix-len", str(c), "--state-file", f"/tmp/st/pf-f16-{c}.state",
            "--control", "ubatch:256", "--out", f"/tmp/o117-fused-decode-{c}.json"]
    rc, out, err = run(args)
    ok = rc == 0 and resolved(out)["candidate"] == "fused-decode" and resolved(out)["control"] == "ubatch:256"
    check(f"documented fused-decode {c}", ok, (rc, out, err))

# defaults: prefill -> control decode, fused-decode -> ubatch:256
good = [*BASE, "--prefix-len", "32768", "--state-file", "/tmp/s", "--out", "/tmp/o.json"]
rc, out, err = run(["--candidate", "prefill-fa2", *good])
check("default control prefill", rc == 0 and resolved(out)["control"] == "decode", (rc, out, err))
rc, out, err = run(["--candidate", "fused-decode", *good])
check("default control fused-decode", rc == 0 and resolved(out)["control"] == "ubatch:256", (rc, out, err))
rc, out, err = run(["--candidate", "prefill-fa2", *good, "--top1-slack-pp", "0.25", "--kld-factor", "2"])
check("gate numbers", rc == 0 and resolved(out)["top1_slack"] == 0.0025 and resolved(out)["kld_factor"] == 2.0, (rc, out, err))

# refused
refused = {
    "no candidate": good,
    "unknown candidate": ["--candidate", "prefill-fa3", *good],
    "bad control": ["--candidate", "prefill-fa2", *good, "--control", "ubatch:0"],
    "old boolean --control": ["--candidate", "prefill-fa2", "--control", *good],
    "fused-decode with decode control": ["--candidate", "fused-decode", *good, "--control", "decode"],
    "removed --noise-floor-kld": ["--candidate", "prefill-fa2", *good, "--noise-floor-kld", "0.01"],
    "removed --control-ubatch": ["--candidate", "prefill-fa2", *good, "--control-ubatch", "4"],
    "nan kld-factor": ["--candidate", "prefill-fa2", *good, "--kld-factor", "nan"],
    "no state file": ["--candidate", "prefill-fa2", *BASE, "--prefix-len", "32768", "--out", "/tmp/o.json"],
    "unknown option": ["--candidate", "prefill-fa2", *good, "--ref-sample", "200"],
}
for name, args in refused.items():
    rc, out, err = run(args)
    check(f"refuses {name}", rc != 0, (rc, out, err))

sys.exit(1 if fails else 0)
