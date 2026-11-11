#!/usr/bin/env bash
# M3-ATTN long-context oracle fixtures {qwen36,ornith}-lc8k: llama.cpp (native-layout dump_ref,
# KleidiAI off, FA off, f16 KV) batched prefill of an 8180-token prompt (ubatch 256, logits of the
# last token, full state) + 32 greedy decode tokens at ctx 8448 (n_kv crosses 8192 -> 8448 = size).
# Prompt: quality-corpus.txt + a frozen copy of engine/prefill.rs (kept in the fixture dir).
# Check with: tools/run_prefill_verify.sh qwen36-lc8k ornith-lc8k
set -uo pipefail
MODELS="${OXE_MODEL_DIR:-$HOME/.local/share/ai-stack/models/llm}"
O="${OXE_ORACLE_DIR:-$HOME/.cache/local-ai/snapdragon-engine/oracle}"
L="${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock}"
D=${DUMP_REF:-$(dirname "$0")/oracle/bin/dump_ref}
SRC=${PROMPT:-$O/lc8k-src/prompt.txt}
for f in qwen36 ornith; do
  case $f in
    ornith) M="$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf" ;;
    qwen36) M="$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf" ;;
  esac
  F=$O/$f-lc8k; mkdir -p "$F"; cp "$SRC" "$F/prompt.txt"
  while true; do
    flock -n -E 75 "$L" nice -n 19 "$D" --model "$M" --out "$F" --prompt-file "$F/prompt.txt" --n-tokens 8180 --batch \
       --ubatch 256 --last-only --state "$F/state.bin" --logits --gen 32 --ctx 8448 --threads 6 > "$F/log.txt" 2>&1
    rc=$?; [ $rc -ne 75 ] && break
    echo "lease busy, retrying in 30 s" >&2; sleep 30
  done
  echo "$f-lc8k rc=$rc"
done
