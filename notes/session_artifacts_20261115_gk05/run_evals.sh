#!/bin/bash
cd /Users/karrtikiyer/projects/airesearch/pless-for-coding || exit 99
R=results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046
for p in atcoder leetcode; do
  stem=gk_0_5_${p}_think_t1.0
  log=/tmp/eval_gk05/logs/${stem}.log
  echo "START $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST') epoch=$(date +%s)" > $log
  uv run python -m bench.eval --results-file $R/${stem}.jsonl --dataset livecodebench --workers 4 --skip-structural-diversity >> $log 2>&1
  rc=$?
  echo "END $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST') epoch=$(date +%s) rc=$rc" >> $log
done
echo ALLDONE > /tmp/eval_gk05/logs/ALLDONE
