#!/bin/bash
set -euo pipefail

mkdir -p /app/logs
LOGFILE="/app/logs/run_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "Starting at $(date)" | tee -a "$LOGFILE"
python main.py 2>&1 | tee -a "$LOGFILE"
echo "Finished at $(date)" | tee -a "$LOGFILE"