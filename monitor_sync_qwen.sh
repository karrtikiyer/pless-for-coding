#!/usr/bin/env bash
# Polls remote every 5 min, syncs after each config completes
SSH="ssh -p 20333 -i ~/.ssh/id_ed25519 root@216.81.245.26"
REMOTE_DIR="/workspace/pless-for-coding"
LOCAL_DIR="./results/pless_full_mbpp_results/Qwen--Qwen-7B"
LAST_COUNT=0

mkdir -p "$LOCAL_DIR/metrics"

while true; do
  COUNT=$($SSH "wc -l < $REMOTE_DIR/sync_markers.log 2>/dev/null || echo 0" 2>/dev/null)
  if [ "$COUNT" -gt "$LAST_COUNT" ]; then
    echo "[$(date)] New completion detected (markers: $COUNT). Syncing..."
    scp -P 20333 -i ~/.ssh/id_ed25519 -r \
      "root@216.81.245.26:$REMOTE_DIR/results/Qwen--Qwen-7B/" "$LOCAL_DIR/"
    $SSH "tail -1 $REMOTE_DIR/sync_markers.log"
    LAST_COUNT=$COUNT
    [ "$COUNT" -ge 6 ] && echo "[$(date)] All 6 configs done! Tell user to shut down pod." && break
  fi
  sleep 300
done
