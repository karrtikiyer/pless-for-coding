#!/bin/bash
# Chop-and-continue RESCUE fair test (HF token-by-token, CUDA) — Phase 1.
#
# THE QUESTION: while running pless @ alpha=2 (the confident-case default, best pass@1),
# can we DETECT a thinking-phase loop and ESCAPE it — recovering a task that would
# otherwise truncate-and-fail — better than the current best rescue (force </think> and
# extract whatever solution already exists)?
#
# For each task: chop its REAL saved pless ramble at the loop onset (find_loop), then run
# four arms on the IDENTICAL chopped prefix (only the post-chop ACTION differs):
#   A_force      force </think> + ```python, extract the existing solution (baseline).
#   chop_only    continue thinking at pless_alpha(5), NO nudge   (control: chop alone).
#   chop_pivot   nudge "step back, try a different approach" + continue at alpha=5.
#   chop_restart nudge "discard it, reconsider from scratch"  + continue at alpha=5.
# All three chop arms re-detect loops live (30-gram) and re-chop (cap 3).
#
# WHY HF, not vLLM: the mechanism is mid-stream context surgery (chop + re-chop). HF
# token-by-token gives that directly; vLLM would need abort/resubmit orchestration we'd
# then discard. Throughput is not the bottleneck here (~224 short continuations).
#
# WHY THESE 14 TASKS (Phase 1): they are the "signal-before-loop" subset for which A30
# computed a passing-config reference depth (analysis/proxy_reasoning_depth.md), so
# MAX_CONT=16384 is KNOWN-adequate (covers all 14 — the deepest needs ~15k cont tokens).
# They are also the optimistic subset: if chop-continue can't beat A_force HERE, it won't
# on the 26 pure-flailing tasks. Phase 2 (all 40 + an alpha=2 mechanism control) only if
# an alpha=5 chop arm clears the A_force bar.
#
# Honest prior (2506.10979, EMNLP 2025): instructing a model to reconsider with the bad
# thought STILL IN CONTEXT is weak (inverse scaling). Our chop REMOVES the thought, so
# chop_only is the control that tests whether the chop — not the nudge — does the work.
#
# RUN ON A CUDA GPU. Uses the default uv venv (HF/transformers, NOT the vLLM venv).
# Usage:   ./run_chop_restart_apps_qwen3.sh
# Env overrides: TASK_IDS, N (4), MAX_CONT (16384), ALPHA (5), MAX_CHOPS (3),
#                MAX_CTX (32768), OUT, CUDA_VISIBLE_DEVICES (pin a GPU).

set -euo pipefail

# 14 anchored "signal-before-loop" tasks (A30 / proxy_reasoning_depth.md).
TASK_IDS="${TASK_IDS:-417 558 616 927 990 1085 1086 1125 1126 1171 1178 1224 1226 1328}"
N="${N:-4}"
MAX_CONT="${MAX_CONT:-16384}"
ALPHA="${ALPHA:-5}"
MAX_CHOPS="${MAX_CHOPS:-3}"
MAX_CTX="${MAX_CTX:-32768}"
OUT="${OUT:-results/_chop_restart_probe/qwen3_chop_restart_phase1_n${N}.json}"

mkdir -p "$(dirname "$OUT")"

echo "Phase-1 chop-restart fair test"
echo "  tasks : $TASK_IDS"
echo "  N=$N MAX_CONT=$MAX_CONT ALPHA=$ALPHA MAX_CHOPS=$MAX_CHOPS MAX_CTX=$MAX_CTX"
echo "  out   : $OUT"

TASK_IDS="$TASK_IDS" N="$N" MAX_CONT="$MAX_CONT" ALPHA="$ALPHA" \
MAX_CHOPS="$MAX_CHOPS" MAX_CTX="$MAX_CTX" OUT="$OUT" \
PYTHONPATH=. \
  uv run python scripts/chop_restart_alpha_compare.py 2>&1 | tee "${OUT%.json}.log"
