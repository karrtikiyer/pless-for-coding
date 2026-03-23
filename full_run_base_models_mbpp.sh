#!/usr/bin/env bash
source /workspace/env
cd /workspace/pless-for-coding

LOG=full_run_base_models.log
MARKERS=sync_markers_base_models.log

echo "[$(date)] Installing transformers>=5 for modern base models" >> $LOG
uv add 'transformers>=5' >> $LOG 2>&1

MODELS=("codellama/CodeLlama-7b-hf" "meta-llama/Llama-2-7b-hf")
CONFIGS=("temp:0.7" "pless:0.6" "pless_norm:0.6" "pless:1.0" "pless_norm:1.0" "top_p:0.9")

for model in "${MODELS[@]}"; do
  model_short="${model##*/}"
  echo "" >> $LOG
  echo "[$(date)] === STARTING MODEL: $model ===" | tee -a $LOG

  for cfg in "${CONFIGS[@]}"; do
    method="${cfg%%:*}"; val="${cfg##*:}"
    echo "" >> $LOG
    echo "[$(date)] STARTING: $model_short $method @ $val" | tee -a $LOG

    if [ "$method" = "top_p" ]; then
      uv run python -m bench --model "$model" --method top_p \
        --top-p "$val" --temperature 1.0 --mbpp-config full >> $LOG 2>&1
    else
      uv run python -m bench --model "$model" --method "$method" \
        --temperature "$val" --mbpp-config full >> $LOG 2>&1
    fi

    echo "[$(date)] COMPLETED: $model_short $method @ $val" | tee -a $LOG
    echo "DONE:$model_short:$method:$val:$(date -Iseconds)" >> $MARKERS
  done

  echo "[$(date)] === MODEL COMPLETE: $model ===" | tee -a $LOG
  echo "MODEL_DONE:$model_short:$(date -Iseconds)" >> $MARKERS
done

echo "ALL_DONE:$(date -Iseconds)" >> $MARKERS
echo "[$(date)] === ALL MODELS AND CONFIGS COMPLETE ===" | tee -a $LOG
