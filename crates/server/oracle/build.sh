#!/bin/sh
# Build the llama.cpp differential oracle against an existing llama.cpp build tree
# (shared libs). Usage: build.sh [LLAMA_SRC] [OUT]
set -eu
L=${1:-$HOME/.local/share/ai-stack/src/llama.cpp}
OUT=${2:-$(dirname "$0")/llama-oracle}
exec nice -n 19 g++ -std=c++17 -O1 -o "$OUT" "$(dirname "$0")/llama_oracle.cpp" \
  -I"$L/include" -I"$L/ggml/include" -I"$L/common" -I"$L/tools/server" -I"$L/tools/mtmd" -I"$L/vendor" \
  -L"$L/build/bin" -Wl,-rpath,"$L/build/bin" -lllama-server-impl -lllama-common -lllama -lggml -lggml-base
