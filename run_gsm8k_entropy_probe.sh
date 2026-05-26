#!/bin/bash
# GSM8K entropy probe — generates the CoT-side data for the central figure.
#
# Mirrors the existing MBPP entropy data recording protocol exactly,
# adapted for GSM8K's task-native conventions (Wei 2022 8-shot CoT
# prompt). Once this lands, regenerate the central figure (now 2-panel:
# MBPP + GSM8K per model) via:
#
#   uv run python -m bench.eval.entropy_survival_curves \\
#     --models Qwen--Qwen2.5-Coder-7B-Instruct codellama--CodeLlama-7b-Instruct-hf \\
#     --output-dir results/entropy_probe/_central_figure_v2
#
# Locked parameters (decided with user 2026-05-26):
# - Models: Qwen2.5-Coder + CodeLlama (same as MBPP entropy data)
# - Method: pless @ T=1.0 (= α=2; matches MBPP entropy)
# - Backend: hf (vLLM doesn't support entropy_log + breaks numerical parity)
# - dtype: bfloat16 (runner default; matches MBPP entropy)
# - Problems: 500 (seed=0; matches GSM8K α-sweep)
# - Samples/problem: 10 (matches MBPP entropy data)
# - max_new_tokens: 512 (matches MBPP entropy recording protocol)
# - Prompt: Wei 2022 8-shot CoT (matches GSM8K α-sweep;
#           empirically verified 0% code emission on Qwen2.5-Coder)
#
# Outputs (one set per model):
#   results/pless_alpha_entropy/<model>/gsm8k/pless_t1.0.jsonl              (regular samples)
#   results/pless_alpha_entropy/<model>/gsm8k/pless_t1.0.jsonl.entropy.jsonl (sidecar, ~1.5GB)
#
# Pre-flight: smoke CodeLlama for code emission BEFORE full run
# (it's the only model we haven't tested on GSM8K Wei 2022 prompt).
# If smoke shows code emission, abort.
#
# Estimated GPU cost (H100):
#   Qwen2.5-Coder: ~1-2 hr
#   CodeLlama:     ~1-2 hr
#   Total:         ~2-4 hr
#
# Usage:
#   ./run_gsm8k_entropy_probe.sh             # full run, both models
#   SMOKE=1 ./run_gsm8k_entropy_probe.sh     # 5 problems × 1 sample × CodeLlama only (smoke)
#   MODELS="Qwen/Qwen2.5-Coder-7B-Instruct" ./run_gsm8k_entropy_probe.sh   # one model

set -euo pipefail

MODELS_DEFAULT="Qwen/Qwen2.5-Coder-7B-Instruct codellama/CodeLlama-7b-Instruct-hf"
MODELS="${MODELS:-$MODELS_DEFAULT}"
N_PROBLEMS="${N_PROBLEMS:-500}"
N_SAMPLES="${N_SAMPLES:-10}"
SEED="${SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DTYPE="${DTYPE:-bfloat16}"
# Default is .../pless_alpha_entropy/gsm8k so the runner's <slug>/ subdir
# lands at .../gsm8k/<slug>/, mirroring the MBPP layout exactly.
RESULTS_DIR_BASE="${RESULTS_DIR_BASE:-results/pless_alpha_entropy/gsm8k}"
LOG_DIR="${LOG_DIR:-/tmp/gsm8k_entropy_logs}"
mkdir -p "$LOG_DIR"

# Smoke mode: 5 problems × 1 sample × CodeLlama only
if [ "${SMOKE:-0}" = "1" ]; then
  MODELS="codellama/CodeLlama-7b-Instruct-hf"
  N_PROBLEMS=5
  N_SAMPLES=1
  RESULTS_DIR_BASE="${RESULTS_DIR_BASE}_smoke"
  echo "[SMOKE MODE] 5 problems × 1 sample × CodeLlama only"
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "GSM8K entropy probe"
echo "  Models:         $MODELS"
echo "  Method:         pless @ T=$TEMPERATURE  (= α=2; matches MBPP entropy)"
echo "  Backend:        hf  (vLLM unsupported for entropy_log)"
echo "  dtype:          $DTYPE"
echo "  Problems:       $N_PROBLEMS (seed=$SEED)"
echo "  Samples/task:   $N_SAMPLES"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Results base:   $RESULTS_DIR_BASE"
echo "  Log dir:        $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════════════"

for MODEL in $MODELS; do
  # The runner uses model.replace("/", "--") for its output subdir
  # — match that here so we can pre-check for an existing sidecar.
  MODEL_SLUG_DBL=$(echo "$MODEL" | sed 's|/|--|g')
  MODEL_SLUG_SH=$(echo "$MODEL" | tr '/' '-')  # for log file name only
  OUT_DIR="$RESULTS_DIR_BASE/$MODEL_SLUG_DBL"
  LOG="$LOG_DIR/${MODEL_SLUG_SH}_entropy.log"
  EXPECTED_SIDECAR="$OUT_DIR/pless_t${TEMPERATURE}.jsonl.entropy.jsonl"

  if [ -f "$EXPECTED_SIDECAR" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $MODEL_SLUG_DBL entropy sidecar already exists at $EXPECTED_SIDECAR"
    echo "       (set FORCE=1 to re-run)"
    continue
  fi

  echo
  echo "───── $MODEL → $OUT_DIR (log: $LOG) ─────"
  # Pass --results-dir as the parent dir; the runner appends <double-hyphen-slug>/
  # under it. Net layout: $RESULTS_DIR_BASE/<slug>/pless_t1.0.jsonl[.entropy.jsonl]
  # (matches results/pless_alpha_entropy/mbpp/<slug>/...).
  uv run python -m bench.gsm8k \
    --model "$MODEL" \
    --method pless \
    --temperature "$TEMPERATURE" \
    --n-samples "$N_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --n-problems "$N_PROBLEMS" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --log-entropy \
    --results-dir "$RESULTS_DIR_BASE" \
    2>&1 | tee "$LOG"

  # Sanity-check the sidecar was written
  if [ -f "$EXPECTED_SIDECAR" ]; then
    N_ROWS=$(wc -l < "$EXPECTED_SIDECAR" | tr -d ' ')
    echo "  ✓ sidecar written: $EXPECTED_SIDECAR ($N_ROWS rows)"
  else
    echo "  ⚠ WARN: expected sidecar not found at $EXPECTED_SIDECAR"
  fi
done

if [ "${SMOKE:-0}" = "1" ]; then
  echo
  echo "── SMOKE: verify zero code emission in CodeLlama output ──"
  SMOKE_JSONL="$RESULTS_DIR_BASE/codellama--CodeLlama-7b-Instruct-hf/pless_t${TEMPERATURE}.jsonl"
  if [ -f "$SMOKE_JSONL" ]; then
    uv run python -c "
import json
with open('$SMOKE_JSONL') as f:
    recs = [json.loads(l) for l in f]
total = sum(len(r['samples']) for r in recs)
code_count = sum(1 for r in recs for s in r['samples'] if '\`\`\`' in s or 'def ' in s[:200])
verbal = sum(1 for r in recs for s in r['samples'] if 'answer is' in s.lower())
print(f'  Total samples: {total}')
print(f'  Contain code fences or def: {code_count} ({100*code_count/total:.1f}%)')
print(f'  Contain \"The answer is\": {verbal} ({100*verbal/total:.1f}%)')
if code_count / total > 0.05:
    print('  ❌ FAIL: > 5%% code emission. Abort full run; need stronger anti-code prompt.')
else:
    print('  ✓ PASS: < 5%% code emission. CodeLlama follows verbal CoT.')
"
  fi
fi

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "Done. Output layout (mirrors results/pless_alpha_entropy/mbpp/<slug>/):"
echo "  $RESULTS_DIR_BASE/<slug>/pless_t1.0.jsonl"
echo "  $RESULTS_DIR_BASE/<slug>/pless_t1.0.jsonl.entropy.jsonl"
echo
echo "Next step: regenerate the central figure (2-panel MBPP + GSM8K)"
echo "  uv run python -m bench.eval.entropy_survival_curves \\"
echo "    --models Qwen--Qwen2.5-Coder-7B-Instruct codellama--CodeLlama-7b-Instruct-hf \\"
echo "    --datasets mbpp gsm8k \\"
echo "    --output-dir results/entropy_probe/_central_figure_v2"
echo "(Note: --datasets is a planned addition to the survival_curves module;"
echo " requires a small patch to handle the gsm8k/ subdir.)"
echo "═══════════════════════════════════════════════════════════════════════"
