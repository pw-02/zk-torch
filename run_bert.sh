#!/bin/bash

./clean.sh
cargo run --release --bin zk_torch --features fold, mock_prove -- bert.yaml &
PID=$!  

LOGFILE="memory_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Logging memory usage every 3 seconds into $LOGFILE"
echo "Timestamp, Total(MB), Used(MB), Free(MB), Shared(MB), Buff/Cache(MB), Available(MB)" > "$LOGFILE"

# Loop while program is running
while kill -0 $PID 2>/dev/null; do
    free -m | awk 'NR==2{
        printf "%s, %s, %s, %s, %s, %s, %s\n",
        strftime("%Y-%m-%d %H:%M:%S"), $1, $2, $3, $4, $5, $7
    }' >> "$LOGFILE"
    sleep 3
done

echo "Program finished, stopped logging."
