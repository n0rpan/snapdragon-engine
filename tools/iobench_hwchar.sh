#!/usr/bin/env bash
# F-IO hardware characterisation of the expert read path, for a QUIET window (the caller holds the build slots and the
# inference lease; this script takes no lock). Model-free: reads the expert sidecar with O_DIRECT through the engine's
# own IO code. ~3.5 min, < 400 MiB, IO workers on cores 8-10(-11), control thread on core 7.
#   tools/iobench_hwchar.sh OUT_DIR        -> OUT_DIR/hwchar.jsonl, OUT_DIR/hwchar.txt (the table, also on stdout)
# Part 1 (closed loop, whole 1.5-2.6 MiB records of a real miss sequence): depth 1 / 4 / 16 x IO threads 1 / 2 / 3
#         (striped) and 3 unstriped, on the encrypted original and on the 041 nodatasum copy.
# Part 2 (decode replay: fixed353 routes, engine LFU + quotas, 1.5 ms of compute between layers, 120 tokens): today's
#         IO (1 thread, parks), 1 thread spinning, 3 threads, 3 threads striped; both files; and 18 GiB for two arms.
set -uo pipefail
OUT=${1:?usage: iobench_hwchar.sh OUT_DIR}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/flash-iobench"
F=~/.cache/local-ai/snapdragon-engine/flash/experts-r4.bin
N=~/.cache/local-ai/snapdragon-engine/flash/experts-r4-nodatasum.bin
mkdir -p "$OUT"; J="$OUT/hwchar.jsonl"; : > "$J"
sha256sum "$BIN" | tee "$OUT/binary.sha256"
SP="spins=4000000000"
run() { timeout --kill-after=5 120 systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 "$BIN" "$@" --json "$J" > /dev/null 2>> "$OUT/stderr.log" || echo "ARM FAILED: $*" | tee -a "$OUT/stderr.log"; }
for pair in "orig:$F" "nods:$N"; do
  tag=${pair%%:*}; file=${pair#*:}
  [ -e "$file" ] || { echo "missing $file, skipping $tag"; continue; }
  for d in 1 4 16; do
    run --mode closed --depth $d --reads 1500 --file "$file" --io "$SP" --label "c-$tag-t1-d$d"
    run --mode closed --depth $d --reads 1500 --file "$file" --io "cpus=8,9;split=2;$SP" --label "c-$tag-t2s-d$d"
    run --mode closed --depth $d --reads 1500 --file "$file" --io "cpus=8,9,10;split=3;$SP" --label "c-$tag-t3s-d$d"
    run --mode closed --depth $d --reads 1500 --file "$file" --io "cpus=8,9,10;$SP" --label "c-$tag-t3-d$d"
  done
  L="--mode layers --tokens 120 --think-us 1500 --file $file"
  run $L --label "l-$tag-today"
  run $L --io "spins=1000000" --label "l-$tag-t1-spin"
  run $L --io "cpus=8,9,10;spins=1000000" --label "l-$tag-t3-spin"
  run $L --io "cpus=8,9,10;split=3;spins=1000000" --label "l-$tag-t3s-spin"
done
run --mode layers --tokens 120 --think-us 1500 --cache-gib 18 --label "l-orig-18g-today"
run --mode layers --tokens 120 --think-us 1500 --cache-gib 18 --io "cpus=8,9,10;split=3;spins=1000000" --label "l-orig-18g-t3s-spin"
python3 -I "$ROOT/tools/iobench_table.py" "$J" | tee "$OUT/hwchar.txt"
