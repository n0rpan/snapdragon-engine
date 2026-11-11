#!/usr/bin/env bash
# One-command decode benchmark with REAL math (Codex lane; replaces M0's run_m0.sh).
# Builds if needed, runs `oxe bench` on Ornith with the M0 gate settings, writes
# results/bench-<timestamp>.json and prints a one-line summary. Extra arguments are passed
# through (later flags override earlier ones), e.g.:
#   tools/run_bench.sh --cpu-list 0,1,2,4,5,6,8,9          # spread over the 3 clusters
#   tools/run_bench.sh --weights anon                      # anonymous THP arena (M0 layout)
#   tools/run_bench.sh --model <Qwen3.6 gguf>
#   tools/run_bench.sh --profile                           # per-phase work/wait table
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x target/release/oxe ] || [ -n "$(find crates vendor -newer target/release/oxe -type f | head -1)" ]; then
  CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}" cargo build --release >&2
fi
mkdir -p results
out="results/bench-$(date +%Y%m%d-%H%M%S).json"
MODEL="${MODEL:-${OXE_MODEL_DIR:-$HOME/.local/share/ai-stack/models/llm}/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf}"
./target/release/oxe bench --model "$MODEL" --cpu-list 0-7 --ctx 2048 --prompt-len 16 --tokens 128 --warmup 16 "$@" --out "$out" >/dev/null
echo "$out"
