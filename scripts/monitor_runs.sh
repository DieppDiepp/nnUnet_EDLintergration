#!/bin/bash
# monitor_runs.sh — xem nhanh trạng thái các run train/inference trên box GPU.
#   bash monitor_runs.sh          # tổng quan (tmux, GPU, epoch mới nhất, [EDLLoss] mỗi log)
#   bash monitor_runs.sh <exp>    # tail -f log của 1 thí nghiệm (theo tên file log, không cần .log)
# Env mặc định (đổi nếu khác): LOGDIR, RESULTS.
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
LOGDIR=${LOGDIR:-/content/logs}
RESULTS=${RESULTS:-/content/smoke_results}

if [ -n "$1" ]; then
  f="$LOGDIR/$1"; [ -f "$f" ] || f="$LOGDIR/$1.log"
  echo "== tail $f =="; tail -f "$f"; exit 0
fi

echo "========== $(date '+%F %T') =========="
echo "== tmux sessions =="; tmux ls 2>/dev/null || echo "(no tmux)"
echo
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi lỗi env?)"
echo
echo "== Logs ($LOGDIR) =="
for L in "$LOGDIR"/*.log; do
  [ -f "$L" ] || continue
  name=$(basename "$L")
  edl=$(grep -h "\[EDLLoss\]" "$L" 2>/dev/null | tail -1 | sed 's/.*\[EDLLoss\]/[EDLLoss]/')
  ep=$(grep -hE "Epoch [0-9]+$|Epoch time" "$L" 2>/dev/null | tail -2 | tr '\n' ' ')
  err=$(grep -hE "Error|Traceback|error" "$L" 2>/dev/null | tail -1)
  printf -- "-- %s\n   %s\n   %s\n" "$name" "${edl:-(chưa build loss)}" "${ep:-(chưa có epoch)}"
  [ -n "$err" ] && printf "   !! %s\n" "$err"
done
echo
echo "== Metrics summary có sẵn ($RESULTS) =="
find "$RESULTS" -name "metrics_summary_fold*.csv" 2>/dev/null | head -20 || true
