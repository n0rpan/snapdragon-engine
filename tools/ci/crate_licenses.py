#!/usr/bin/env python3
"""Collect the licence texts of the third-party crates linked into one binary.

    tools/ci/crate_licenses.py PACKAGE OUT_DIR [--features F]

Lists the package's normal (runtime) dependency closure with `cargo tree`, looks each crate up
in `cargo metadata` and copies its LICENSE*/COPYING*/NOTICE* files to OUT_DIR/<name>-<version>/.
Writes OUT_DIR/INDEX.txt (name, version, licence expression, repository). Workspace crates are
skipped: they are covered by the repository's LICENSE / NOTICE / THIRD_PARTY_NOTICES.
Fails if a crate has neither a licence file nor a licence expression.
"""
import json
import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    package, out = args[0], pathlib.Path(args[1])
    features = []
    if "--features" in args:
        features = ["--features", args[args.index("--features") + 1]]
    meta = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--locked"], text=True))
    members = set(meta["workspace_members"])
    by_key = {(p["name"], p["version"]): p for p in meta["packages"] if p["id"] not in members}
    tree = subprocess.check_output(
        ["cargo", "tree", "--locked", "-p", package, *features, "-e", "normal",
         "--prefix", "none", "--format", "{p}"], text=True)
    used = set()
    for line in tree.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("v"):
            used.add((parts[0], parts[1][1:]))
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for key in sorted(used):
        pkg = by_key.get(key)
        if pkg is None:  # a workspace crate
            continue
        src = pathlib.Path(pkg["manifest_path"]).parent
        files = sorted(f for f in src.iterdir() if f.is_file() and
                       f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE")))
        dest = out / f"{key[0]}-{key[1]}"
        dest.mkdir(exist_ok=True)
        for f in files:
            shutil.copyfile(f, dest / f.name)
        lic = pkg.get("license") or ""
        if not files and not lic:
            missing.append(f"{key[0]} {key[1]}")
        rows.append(f"{key[0]} {key[1]}\t{lic or '(see files)'}\t{pkg.get('repository') or ''}")
    (out / "INDEX.txt").write_text(
        "Third-party Rust crates linked into this binary (name version, licence, repository).\n"
        "Each crate's licence files are in the directory of the same name.\n\n"
        + "\n".join(rows) + "\n")
    if missing:
        print("no licence information for: " + ", ".join(missing), file=sys.stderr)
        return 1
    print(f"{len(rows)} crates -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
