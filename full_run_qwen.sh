#!/usr/bin/env bash
source /workspace/env
cd /workspace/pless-for-coding

LOG=full_run.log
MARKERS=sync_markers.log

echo "[$(date)] Installing transformers<5 for Qwen-7B" >> $LOG
uv add 'transformers<5,>=4.37' >> $LOG 2>&1

CONFIGS=("temp:0.7" "pless:0.6" "pless_norm:0.6" "pless:1.0" "pless_norm:1.0" "top_p:0.9")

for cfg in "${CONFIGS[@]}"; do
  method="${cfg%%:*}"; val="${cfg##*:}"
  echo "" >> $LOG
  echo "[$(date)] STARTING: $method @ $val" | tee -a $LOG

  if [ "$method" = "top_p" ]; then
    uv run python -m bench --model Qwen/Qwen-7B --method top_p \
      --top-p "$val" --temperature 1.0 --mbpp-config full >> $LOG 2>&1
  else
    uv run python -m bench --model Qwen/Qwen-7B --method "$method" \
      --temperature "$val" --mbpp-config full >> $LOG 2>&1
  fi

  echo "[$(date)] COMPLETED: $method @ $val" | tee -a $LOG
  echo "DONE:$method:$val:$(date -Iseconds)" >> $MARKERS
done

echo "ALL_DONE:$(date -Iseconds)" >> $MARKERS
echo "[$(date)] === ALL 6 CONFIGS COMPLETE ===" | tee -a $LOG
