#!/usr/bin/env bash
# sglang_70b_model.sh — start/stop meta-llama/Meta-Llama-3-70B-Instruct
#
# Usage:
#   ./sglang_70b_model.sh start [--device <devices>] [--port <port>]
#   ./sglang_70b_model.sh stop  [--port <port>]
#   ./sglang_70b_model.sh stop  --all
#
# --device  Comma-separated GPU ids (default: 0).
#           All GPUs go to one instance via TP when a single --port is given.
#             1 instance TP=4 : --device 0,1,2,3 --port 30000
#             2 instances TP=2: --device 0,1,2,3 --port 30000,30001
# --port    Port(s), comma-separated (default: 30000).
# --all     (stop only) Stop all instances of this model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="meta-llama/Meta-Llama-3-70B-Instruct"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 start [--device <devices>] [--port <port>]"
    echo "       $0 stop  [--port <port>]"
    echo "       $0 stop  --all"
    exit 1
fi

COMMAND="$1"; shift

DEVICE="0"
PORT="30000"
STOP_ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        --all)    STOP_ALL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$COMMAND" in
    start)
        exec "$SCRIPT_DIR/sglang.sh" start \
            --model  "$MODEL" \
            --device "$DEVICE" \
            --port   "$PORT" \
            --no-embedding
        ;;
    stop)
        if $STOP_ALL; then
            exec "$SCRIPT_DIR/sglang.sh" stop --model "$MODEL"
        else
            exec "$SCRIPT_DIR/sglang.sh" stop --model "$MODEL" --port "$PORT"
        fi
        ;;
    *) echo "Unknown command: $COMMAND (use start or stop)"; exit 1 ;;
esac
