#!/usr/bin/env bash
# moe-prefill: the ordered DEV-window build + model-free check (shortest first). No model, no lease.
#   tools/moepf_dev_check.sh [bench]     bench: also the single-thread kernel timings (core 4)
# Every step goes through tools/moepf_locked.sh (flock -n, WINDOW re-check, bounded memory) and the
# script stops at the first step that is refused, so nothing ever starts outside a DEV / QUALITY window.
set -euo pipefail
cd "$(dirname "$0")/.."
R=tools/moepf_locked.sh
mkdir -p results
$R cargo test --release -j 2 -p kernels-sys > results/moepf-test-kernels.log 2>&1; grep -E 'test result|FAILED|panicked|^error' results/moepf-test-kernels.log
$R --need-min 2 --mem 8G cargo build --release -j 2 > results/moepf-build.log 2>&1; grep -E '^(warning|error)|Finished' results/moepf-build.log
$R --need-min 6 cargo test --release -j 2 -p model -p sched > results/moepf-test-model.log 2>&1; grep -E 'test result|FAILED|panicked|^error' results/moepf-test-model.log
sha256sum target/release/oxe
if [ "${1:-}" = bench ]; then
  $R taskset -c 4 cargo test --release -j 2 -p kernels-sys -- --ignored kbench --nocapture --test-threads 1 2>&1 | grep -vE '^(running|test |$)' | tee results/moepf-kbench.txt
  KB_NCOL=64 $R taskset -c 4 cargo test --release -j 2 -p kernels-sys -- --ignored kbench_attn --nocapture --test-threads 1 2>&1 | grep -vE '^(running|test |$)' | tee results/moepf-kbench-attn64.txt
fi
