#!/usr/bin/env bash
# Build the production ggml static libraries that `oxe-server --features flash` links
# (README "Flash build prerequisite"), from the pinned public sources only:
#   Helldez/BigMoeOnEdge 930b2c9ebd8c2cface6566bd9f11075ac07adfe0 (Apache-2.0)
#   Helldez/llama.cpp    7ac98f4a8089a36c0581b7c6e8973417a8fa175c (MIT)
#   + patches/flash-demand-mapping.patch (patched llama tree 4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef)
# then copies them with tools/sync_ggml_prod.sh into $OXF_GGML_PROD.
# Used by .github/workflows/release.yml; also runnable by hand.
#   tools/ci/build_ggml_prod.sh WORK_DIR
# Env: OXF_GGML_PROD (destination), GGML_PROD_JOBS (default 4).
set -euo pipefail

BIGMOE_REPO=https://github.com/Helldez/BigMoeOnEdge.git
BIGMOE_REV=930b2c9ebd8c2cface6566bd9f11075ac07adfe0
LLAMA_REPO=https://github.com/Helldez/llama.cpp.git
LLAMA_REV=7ac98f4a8089a36c0581b7c6e8973417a8fa175c
PATCHED_TREE=4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef
PATCH_SHA256=50eab1d270721d5e0d1215ad858324502308bb6be855af6b73108457b3d539cc

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${1:?usage: build_ggml_prod.sh WORK_DIR}
JOBS=${GGML_PROD_JOBS:-4}
PATCH=$ROOT/patches/flash-demand-mapping.patch
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)

echo "$PATCH_SHA256  $PATCH" | sha256sum -c -

# Public HTTPS only: no inherited credential helper, headers, hooks or identity (lab docs/FLASH-REBUILD.md).
pgit() {
  env -u GIT_ASKPASS -u SSH_ASKPASS -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= -c core.askPass= -c http.extraHeader= \
        -c core.hooksPath=/dev/null -c init.defaultBranch=main -c advice.detachedHead=false "$@"
}
fetch_pinned() { # dir repo rev
  pgit init -q "$1"
  pgit -C "$1" remote add origin "$2"
  pgit -C "$1" fetch -q --depth 1 origin "$3"
  pgit -C "$1" checkout -q --detach FETCH_HEAD
  test "$(pgit -C "$1" rev-parse HEAD)" = "$3"
}

fetch_pinned "$WORK/parent" "$BIGMOE_REPO" "$BIGMOE_REV"
fetch_pinned "$WORK/llama-fork" "$LLAMA_REPO" "$LLAMA_REV"
pgit -C "$WORK/llama-fork" apply --index "$PATCH"
tree=$(pgit -C "$WORK/llama-fork" write-tree)
if [ "$tree" != "$PATCHED_TREE" ]; then
  echo "patched llama.cpp tree is $tree, expected $PATCHED_TREE" >&2
  exit 1
fi
pgit -C "$WORK/llama-fork" -c user.name='Local source reconstruction' \
  -c user.email='source-reconstruction@invalid' -c commit.gpgsign=false \
  commit -q --no-gpg-sign -m 'build: reconstruct qualified demand mapping'
rm -rf "$WORK/parent/third_party/llama.cpp"
pgit clone -q --no-hardlinks "$WORK/llama-fork" "$WORK/parent/third_party/llama.cpp"
pgit -C "$WORK/parent" add third_party/llama.cpp

# The qualified CPU configuration (lab docs/FLASH-REBUILD.md, "Build with the qualified CPU configuration").
FLASH_FLAGS='-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME'
# The qualified build used a GCC configured with --enable-default-pie (Arch), so its objects are
# PIE and link into Rust's PIE executables. Fedora's GCC is not default-PIE: ask for it.
if ! cc -v 2>&1 | grep -q -- --enable-default-pie; then
  FLASH_FLAGS="$FLASH_FLAGS -fPIE"
fi
cmake -S "$WORK/parent" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=c++ \
  -DCMAKE_C_FLAGS="$FLASH_FLAGS" -DCMAKE_CXX_FLAGS="$FLASH_FLAGS" \
  -DBMOE_BUILD_CLI=ON -DBMOE_BUILD_SERVER=ON -DBMOE_BUILD_DESKTOP=OFF \
  -DBMOE_BUILD_TESTS=ON -DBMOE_BUILD_TOOLS=OFF -DBMOE_BACKEND_MODULES=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_RPC=ON \
  -DGGML_VULKAN=OFF -DGGML_HEXAGON=OFF -DGGML_OPENCL=OFF \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_LLGUIDANCE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$WORK/build" --parallel "$JOBS" --target ggml-base ggml-cpu

BIGMOE_BUILD="$WORK/build" "$ROOT/tools/sync_ggml_prod.sh" "$WORK/parent"
DST=${OXF_GGML_PROD:-$HOME/.cache/local-ai/snapdragon-engine/ggml-prod}
# Keep both pinned build-source licences, with explicit NOTICE absence and hashes.
python3 -I "$ROOT/tools/ci/flash_licenses.py" collect \
  "$WORK/parent" "$WORK/parent/third_party/llama.cpp" "$DST"
{
  echo "llama_cpp_patched_tree $tree"
  echo "toolchain $(cc --version | head -n1) | $(cmake --version | head -n1) | ninja $(ninja --version)"
} >> "$DST/MANIFEST"
cat "$DST/MANIFEST"
