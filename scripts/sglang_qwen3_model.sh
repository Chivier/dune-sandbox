#!/usr/bin/env bash
# sglang_qwen3_model.sh — start/stop Qwen/Qwen3-8B
#
# Usage:
#   ./sglang_qwen3_model.sh start [--device <gpu_ids>] [--port <base_port>] [--instances <n>]
#   ./sglang_qwen3_model.sh stop  [--port <base_port>]
#   ./sglang_qwen3_model.sh stop  --all
#
# --device     Comma-separated GPU ids. Instances are distributed round-robin.
#              Examples:  --device 1        → all instances on GPU 1
#                         --device 1,2      → alternate between GPU 1 and GPU 2
# --port       Base port; instances get base, base+1, ..., base+N-1 (default: 30000)
# --instances  Number of model copies (default: 4). Each needs ~20 GB VRAM.
# --all        (stop only) Stop all instances of this model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="Qwen/Qwen3-8B"
DEFAULT_INSTANCES=4

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 start [--device <gpu_ids>] [--port <base_port>] [--instances <n>]"
    echo "       $0 stop  [--port <base_port>]"
    echo "       $0 stop  --all"
    echo ""
    echo "  --device     Comma-separated GPU ids (default: 0). Round-robin across instances."
    echo "  --instances  Number of model copies (default: $DEFAULT_INSTANCES)."
    echo "               Each 8B instance needs ~20 GB VRAM (weights + KV cache)."
    exit 1
fi

COMMAND="$1"; shift

DEVICE="0"
BASE_PORT="30000"
NUM_INSTANCES="$DEFAULT_INSTANCES"
STOP_ALL=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)    DEVICE="$2";        shift 2 ;;
        --port)      BASE_PORT="$2";     shift 2 ;;
        --instances) NUM_INSTANCES="$2"; shift 2 ;;
        --all)       STOP_ALL=true;      shift   ;;
        *)           EXTRA_ARGS+=("$1"); shift   ;;
    esac
done

# Parse device list (may be single "1" or multi "1,2")
IFS=',' read -ra GPUS <<< "$DEVICE"
NUM_GPUS=${#GPUS[@]}

# Build comma-separated device and port lists.
# Instances are distributed round-robin across the provided GPUs.
build_lists() {
    local dev_list="" port_list=""
    for (( i=0; i<NUM_INSTANCES; i++ )); do
        dev_list+="${GPUS[$(( i % NUM_GPUS ))]}"
        port_list+="$((BASE_PORT + i))"
        (( i < NUM_INSTANCES - 1 )) && { dev_list+=","; port_list+=","; }
    done
    echo "$dev_list $port_list"
}

case "$COMMAND" in
    start)
        read -r DEVICE_LIST PORT_LIST < <(build_lists)
        echo "Launching $NUM_INSTANCES instances of $MODEL across GPU(s) ${GPUS[*]}"
        echo "Ports: $PORT_LIST"
        echo "Device map: $DEVICE_LIST"
        exec "$SCRIPT_DIR/sglang.sh" start \
            --model  "$MODEL" \
            --device "$DEVICE_LIST" \
            --port   "$PORT_LIST" \
            --no-embedding \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;
    stop)
        if $STOP_ALL; then
            exec "$SCRIPT_DIR/sglang.sh" stop --model "$MODEL"
        else
            read -r _ PORT_LIST < <(build_lists)
            exec "$SCRIPT_DIR/sglang.sh" stop --model "$MODEL" --port "$PORT_LIST"
        fi
        ;;
    *) echo "Unknown command: $COMMAND (use start or stop)"; exit 1 ;;
esac
