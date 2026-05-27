#!/bin/bash
# Monitor rhypedb server memory and disk usage
PID=$1
DATA_DIR=$2

if [ -z "$PID" ] || [ -z "$DATA_DIR" ]; then
    echo "Usage: monitor.sh <pid> <data_dir>"
    exit 1
fi

while kill -0 $PID 2>/dev/null; do
    RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "$RSS" ]; then
        RSS_MB=$((RSS / 1024))
        DISK=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        HEALTH=$(curl -s http://127.0.0.1:4220/health 2>/dev/null || echo "down")
        echo "$(date +%H:%M:%S) | RSS: ${RSS_MB}MB | Disk: ${DISK} | ${HEALTH}"
    fi
    sleep 10
done
echo "Process $PID exited"
