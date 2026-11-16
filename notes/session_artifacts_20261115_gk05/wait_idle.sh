#!/bin/bash
# poll every 20 s; idle = no pytest/bench.eval and 1-min load < 3; max 15 min
log=/tmp/eval_gk05/logs/wait_idle.log
start=$(date +%s)
echo "WAIT START $(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M:%S IST')" > $log
until false; do
  procs=$(pgrep -fl "pytest|uv run python -m bench.eval" | grep -v wait_idle)
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  now=$(date +%s); el=$((now-start))
  echo "$(TZ=Asia/Kolkata date '+%H:%M:%S IST') elapsed=${el}s load1=$load procs=[$(echo $procs | tr '\n' ';')]" >> $log
  if [ -z "$procs" ] && awk -v l=$load 'BEGIN{exit !(l<3)}'; then echo "IDLE_OK elapsed=${el}s" >> $log; break; fi
  if [ $el -ge 900 ]; then echo "MAX_WAIT_HIT elapsed=${el}s" >> $log; break; fi
  sleep 20
done
echo WAITDONE > /tmp/eval_gk05/logs/WAITDONE
