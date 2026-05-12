#!/bin/bash
# Stage 2 of the algosim Qwen3-8B split-decoding diversity eval (GPU-side).
#
# Bootstraps an isolated .venv-algosim (Python 3.10 + algosim's pinned vLLM /
# transformers stack), then runs the AlgoSim clustering with Llama-3.1-8B-Instruct
# and computes DA@K / EA / NAUADC.
#
# Run on a CUDA-capable host with >= 18 GB free VRAM and HF_TOKEN set
# (Llama-3.1-8B-Instruct is a gated model).
#
# Inputs (default paths under the repo root):
#   algosim_data/requests/   — parquet files produced by run_algosim_export_qwen3.sh
#
# Outputs:
#   algosim_data/responses/  — per-config clustering parquet files
#   algosim_data/algosim_metrics.json — DA@K / EA / NAUADC under the "ATCODER" bucket
set -euo pipefail

REQUESTS_DIR="${REQUESTS_DIR:-algosim_data/requests}"
RESPONSES_DIR="${RESPONSES_DIR:-algosim_data/responses}"
METRICS_PATH="${METRICS_PATH:-algosim_data/algosim_metrics.json}"
ALGOSIM_VENV="${ALGOSIM_VENV:-.venv-algosim}"

# ── Preflight ─────────────────────────────────────────────────────────────────
echo "[preflight] checking environment..."

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/." >&2
  exit 1
fi

if [ ! -d algosim ]; then
  echo "ERROR: algosim submodule missing. Run 'git submodule update --init'." >&2
  exit 1
fi

if [ ! -d "$REQUESTS_DIR" ] || [ -z "$(ls -A "$REQUESTS_DIR"/*.parquet 2>/dev/null || true)" ]; then
  echo "ERROR: no parquet files in $REQUESTS_DIR. Run run_algosim_export_qwen3.sh first and rsync the output here." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set." >&2
  echo "       Llama-3.1-8B-Instruct is gated; export your HF token before running." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script requires a CUDA GPU." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader

# ── Patch upstream algosim bugs ───────────────────────────────────────────────
# infer_algosim.py:62 returns an undefined `completions` variable. See
# algosim_patches/. Apply idempotently — `git apply --check` skips if already applied.
for patch in algosim_patches/*.patch; do
  [ -e "$patch" ] || continue
  if (cd algosim && git apply --check "../$patch" 2>/dev/null); then
    echo "[patch] applying $patch"
    (cd algosim && git apply "../$patch")
  else
    echo "[patch] $patch already applied (or no longer applicable); skipping"
  fi
done

# ── Bootstrap algosim env ─────────────────────────────────────────────────────
if [ ! -d "$ALGOSIM_VENV" ]; then
  echo "[bootstrap] creating $ALGOSIM_VENV (Python 3.10 + algosim requirements)..."
  # `uv venv` does not install pip by default; use `uv pip install --python <venv>`
  # to install into the venv from outside.
  uv venv "$ALGOSIM_VENV" --python 3.10
  uv pip install --python "$ALGOSIM_VENV/bin/python" -r algosim/requirements.txt
else
  echo "[bootstrap] reusing existing $ALGOSIM_VENV"
fi

PY="$ALGOSIM_VENV/bin/python"

"$PY" - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"[preflight] torch={torch.__version__}, "
      f"device={torch.cuda.get_device_name(0)}, "
      f"free_gb={torch.cuda.mem_get_info()[0] / 1e9:.1f}")
PYEOF

# vLLM defaults (override in your shell if needed)
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$RESPONSES_DIR" "$(dirname "$METRICS_PATH")"

# ── Clustering (vLLM judge: Llama-3.1-8B-Instruct) ────────────────────────────
echo "[clustering] running algosim/clustering_solutions.py ..."
"$PY" algosim/clustering_solutions.py \
  --input_dir  "$REQUESTS_DIR" \
  --output_dir "$RESPONSES_DIR"

# ── Metrics (DA@K / EA / NAUADC) ──────────────────────────────────────────────
echo "[metrics] running algosim/compute_metrics.py ..."
"$PY" algosim/compute_metrics.py \
  --clustering_response_dir "$RESPONSES_DIR" \
  --metrics_path "$METRICS_PATH"

cat <<EOF

Judge + metrics complete.
  responses → $RESPONSES_DIR
  metrics   → $METRICS_PATH

Next: scp $METRICS_PATH back to the Mac and run
  uv run python -m bench.eval.algosim_report
EOF
