#!/bin/bash
export PATH=/root/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
# Kill all training processes
pids=$(ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null
    echo "Killed PIDs: $pids"
else
    echo "No training processes found"
fi
sleep 3
# Verify
remaining=$(ps aux | grep "train/train.py" | grep -v grep | wc -l)
echo "Remaining: $remaining"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
