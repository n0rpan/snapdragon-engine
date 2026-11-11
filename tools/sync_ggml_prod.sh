#!/usr/bin/env bash
# Copy the PRODUCTION ggml objects (libggml-base.a, libggml-cpu.a) and the matching public headers
# into the engine's cache, so the Flash graph port computes with the very objects production runs
# (BigMoeOnEdge 930b2c9 + llama.cpp 04ff7df, built Release, -mcpu=oryon-1+..., OpenMP, LLAMAFILE on,
# CPU_REPACK/KleidiAI off; the build that reproduced the fixed353 SHA). No compilation here.
#   tools/sync_ggml_prod.sh [BIGMOE_TREE]   (default: $BIGMOE_TREE)
set -euo pipefail
SRC=${1:-${BIGMOE_TREE:?set BIGMOE_TREE to your BigMoeOnEdge checkout}}
DST=${OXF_GGML_PROD:-$HOME/.cache/local-ai/snapdragon-engine/ggml-prod}
BUILD=${BIGMOE_BUILD:-$SRC/build-flash}   # the CMake build directory of that tree
B=$BUILD/third_party/llama.cpp/ggml/src
mkdir -p "$DST/include" "$DST/lib"
cp "$B/libggml-base.a" "$B/libggml-cpu.a" "$DST/lib/"
cp "$SRC"/third_party/llama.cpp/ggml/include/*.h "$DST/include/"
{
  echo "bigmoe_commit $(git -C "$SRC" rev-parse HEAD)"
  echo "llama_cpp_commit $(git -C "$SRC" ls-tree HEAD third_party/llama.cpp | awk '{print $3}')"
  grep -A6 "ggml-cpu.dir/ggml-cpu/ops.cpp.o:" "$BUILD/build.ninja" | grep -E "^\s+FLAGS" | sed 's/^ */cpu_cxx_flags /'
  (cd "$DST" && sha256sum lib/*.a include/*.h)
} > "$DST/MANIFEST"
cat "$DST/MANIFEST" | head -8
