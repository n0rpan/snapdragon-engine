#!/usr/bin/env bash
# Build and run tools/vl_check (model-free): bitwise check + one-thread timing of the lane GEMVs.
#   tools/vl_check/run.sh [type] [notime]     type: 3, 4, 5, 6, 80 (compiles only that kernel); 0 or none = all present
# Compiles with the flags of crates/kernels-sys/build.rs (the numerics depend on them). Takes a build lock through
# tools/vl_locked.sh (exit 75 / 76 / 77 = not started, try again later; --retry polls for ~2 min).
set -uo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
T=${1:-0}
OUT=${VL_OUT:-${TMPDIR:-/tmp}/vlcheck-$T}
mkdir -p "$OUT"
ARCH="-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme"
BASE="-O3 -DNDEBUG -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -w -fno-fast-math"
INC="-I$R/vendor/ggml-subset -I$R/vendor/ggml-subset/generated -I$R/crates/kernels-sys/csrc"
ONLY=""; [ "$T" != 0 ] && ONLY="-DOXE_VL_ONLY=$T"
PIN=${VL_CPU:-4}
"$R/tools/vl_locked.sh" --retry 24 --timeout 240 bash -c "
  set -e
  cc $BASE -std=gnu11 $INC -c $R/vendor/ggml-subset/generated/quants_base.c -o $OUT/qb.o
  cc $BASE $ARCH -std=gnu11 $INC -c $R/vendor/ggml-subset/generated/quants_subset.c -o $OUT/qs.o
  c++ $BASE $ARCH -std=gnu++17 $INC $ONLY -c $R/tools/vl_check/vl_check.cpp -o $OUT/vl.o
  c++ $OUT/vl.o $OUT/qb.o $OUT/qs.o -lm -o $OUT/vl_check
  taskset -c $PIN $OUT/vl_check $T ${2:-}
"
