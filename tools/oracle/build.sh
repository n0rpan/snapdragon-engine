#!/usr/bin/env bash
# Build the llama.cpp oracle (dump_ref) against the pinned llama.cpp with KleidiAI OFF.
#
#   LLAMA=<llama.cpp checkout>  (branch claude/q3k-moe-gemm @ ee6b9a1)
#   build dir: $LLAMA/build-oracle  (configured once by this script if missing)
#
# KleidiAI must be OFF: with it ON, llama.cpp re-expands Q6_K/Q8_0 (and Q4_K dense) weights into
# KleidiAI int8 layouts, which is a different computation from the native repack kernels that
# oxe vendors. The oracle is the same commit with the native repack path.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA="${LLAMA:?set LLAMA to your llama.cpp checkout}"
B="$LLAMA/build-oracle"
CMAKE="${CMAKE:-$HOME/.local/share/ai-stack/buildenv/lib/python3.14/site-packages/cmake/data/bin/cmake}"
NINJA="${NINJA:-$HOME/.local/share/ai-stack/buildenv/bin/ninja}"
JOBS="${JOBS:-2}"

commit="$(git -C "$LLAMA" rev-parse HEAD)"
if [ "$commit" != "ee6b9a1caf38138cafcfa3d6104f93d8a32db19a" ]; then
    echo "warning: $LLAMA is at $commit, not the pinned ee6b9a1" >&2
fi
if [ ! -f "$B/build.ninja" ]; then
    nice -n 19 "$CMAKE" -S "$LLAMA" -B "$B" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF -DGGML_CPU_REPACK=ON -DGGML_CPU_KLEIDIAI=OFF \
        -DGGML_OPENMP=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF \
        -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
fi
nice -n 19 "$NINJA" -C "$B" -j"$JOBS" llama
mkdir -p "$HERE/bin"
nice -n 19 g++ -O2 -std=gnu++17 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$HERE/dump_ref.cpp" \
    "$B/src/libllama.a" "$B/ggml/src/libggml.a" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/libggml-base.a" \
    -fopenmp -lpthread -ldl -o "$HERE/bin/dump_ref"
echo "built $HERE/bin/dump_ref (llama.cpp $commit, KleidiAI OFF)"
