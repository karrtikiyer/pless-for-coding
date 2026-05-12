#!/bin/bash
# Stage 1 of the algosim Qwen3-8B split-decoding diversity eval (CPU-side).
#
# Builds algosim-compatible parquet requests from our existing JSONL results.
# Run this anywhere (no GPU required). After it finishes, rsync the
# algosim_data/ directory to the GPU machine and run run_algosim_judge_qwen3.sh
# there.
set -euo pipefail

CONFIGS="${CONFIGS:-A,C,P15,T15P,H7P,H8P,H9P,H10P,H11P,H12P}"
OUT_DIR="${OUT_DIR:-algosim_data/requests}"

uv run python -m bench.eval.algosim_export \
  --configs "$CONFIGS" \
  --output-dir "$OUT_DIR"

cat <<EOF

Export complete.
Next: rsync the algosim_data/ directory to the GPU machine, e.g.
  rsync -avz algosim_data/ <gpu-host>:<repo>/algosim_data/
Then run run_algosim_judge_qwen3.sh on the GPU box.
EOF
