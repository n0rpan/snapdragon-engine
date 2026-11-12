#!/usr/bin/env bash
# M3.2 exactness gate: oxe batched prefill vs llama.cpp's batched prefill (native-layout oracle,
# KleidiAI off, ubatch 256, logits of the last token only). Holds the inference lease for each run.
#
#   tools/run_prefill_verify.sh [--cpu-list 0-7] [fixture ...]
# fixtures default to the six under ~/.cache/local-ai/snapdragon-engine/oracle/{ornith,qwen36}-pf{64,512,2048}
# (made by tools/oracle/bin/dump_ref --batch --ubatch 256 --last-only --state ... --gen 8; see docs/lab-notes/M3-PREFILL.md).
# Long-context fixtures {ornith,qwen36}-lc8k (8180 tokens + 32 decode rows): tools/make_lc8k_fixtures.sh (docs/lab-notes/M3-ATTN.md).
set -uo pipefail
MODELS="${OXE_MODEL_DIR:-$HOME/.local/share/ai-stack/models/llm}"
cd "$(dirname "$0")/.."
CPUS=0-7
if [ "${1:-}" = "--cpu-list" ]; then CPUS=$2; shift 2; fi
O="${OXE_ORACLE_DIR:-$HOME/.cache/local-ai/snapdragon-engine/oracle}"
LOCK="${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock}"
FIX=("$@")
[ ${#FIX[@]} -eq 0 ] && FIX=(ornith-pf64 ornith-pf512 ornith-pf2048 qwen36-pf64 qwen36-pf512 qwen36-pf2048)
mkdir -p results
for f in "${FIX[@]}"; do
  case $f in
    ornith*) M="$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf" ;;
    qwen36*) M="$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf" ;;
    *) echo "unknown fixture $f" >&2; exit 2 ;;
  esac
  out=results/prefill-verify-$f.json
  while true; do
    flock -n -E 75 "$LOCK" nice -n 19 ./target/release/oxe prefill-verify --model "$M" --ref "$O/$f" --cpu-list "$CPUS" > "$out" 2> "$out.log"
    rc=$?
    [ $rc -ne 75 ] && break
    echo "lease held (Codex measuring), retrying in 3 min" >&2
    sleep 180
  done
  python3 - "$out" "$f" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
s = r.get("state") or {}
print(f'{sys.argv[2]:15s} tokens {r["prompt_tokens"]:5d}  last logits exact {r["last_logits_bit_exact"]}  '
      f'state identical {s.get("identical")}  decode rows exact {r["decode_rows_bit_exact"]}  greedy {r["greedy_identical"]}')
EOF
done
