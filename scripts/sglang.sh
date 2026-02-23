#!/usr/bin/env bash
# sglang.sh — start/stop sglang model servers (supports parallel multi-instance + TP)
#
# DEVICE / PORT RULES:
#   Number of ports  = number of instances to launch.
#   Number of devices / number of ports = tp_size per instance.
#
#   Examples:
#     --device 0,1,2,3  --port 30000              → 1 instance,  tp_size=4
#     --device 0,1,2,3  --port 30000,30001         → 2 instances, tp_size=2 each
#     --device 0,1,2,3  --port 30000,30001,30002,30003 → 4 instances, tp_size=1 each
#     --device 0,0,0,0,0,0,0,0 --port 30000,...,30007  → 8 instances on same GPU
#
#   If only one --port is given with multiple devices, it is treated as TP (single instance).
#   If --port is omitted, defaults to 30000 (single instance).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PLUGINS_DIR="$PROJECT_DIR/sglang_plugins"
PID_DIR="/tmp/sglang_pids"

usage() {
    cat <<EOF
Usage:
  $0 start --model <model_id> [--device <devices>] [--port <ports>] [extra sglang args...]
  $0 stop  [--model <model_id>] [--port <ports>]
  $0 stop  --all

Options:
  --model    HuggingFace model ID (required for start)
  --device   Comma-separated GPU ids. Divided evenly across instances.
               1 instance TP=4 : --device 0,1,2,3 --port 30000
               4 instances TP=1: --device 0,1,2,3 --port 30000,30001,30002,30003
               8 instances same GPU: --device 0,0,0,0,0,0,0,0 --port 30000,...,30007
             Default: CUDA_VISIBLE_DEVICES or "0"
  --port     Comma-separated ports. Count determines number of instances.
             Default: 30000
  --no-embedding  Skip --is-embedding flags (use for generative models)
  --all      (stop only) Stop every running sglang server
  Extra args are forwarded verbatim to sglang.launch_server.

Examples:
  $0 start --model meta-llama/Meta-Llama-3-70B-Instruct --device 0,1,2,3 --port 30000
  $0 start --model meta-llama/Prompt-Guard-86M --device 1,1,2 --port 30000,30001,30002
  $0 stop --model meta-llama/Prompt-Guard-86M
  $0 stop --port 30000,30001
  $0 stop --all
EOF
}

model_to_pid_file() {
    local model="$1" port="$2" safe_name
    safe_name="$(echo "$model" | tr '/' '_' | tr -dc 'a-zA-Z0-9_-')"
    echo "$PID_DIR/${safe_name}_${port}.pid"
}

# Launch one server instance
# Args: model  cuda_devs  tp_size  port  embedding  [extra...]
start_one() {
    local model="$1" cuda_devs="$2" tp_size="$3" port="$4" embedding="$5"
    shift 5
    local extra_args=("$@")

    mkdir -p "$PID_DIR"
    local pid_file
    pid_file="$(model_to_pid_file "$model" "$port")"

    if [[ -f "$pid_file" ]]; then
        local existing_pid
        existing_pid="$(cat "$pid_file")"
        if kill -0 "$existing_pid" 2>/dev/null; then
            echo "  [port $port] Already running (PID $existing_pid), skipping."
            return 0
        fi
        rm -f "$pid_file"
    fi

    local log_file="/tmp/sglang_${port}.log"
    echo "  [port $port] CUDA_VISIBLE_DEVICES=$cuda_devs  tp_size=$tp_size  log=$log_file"

    # triton backend avoids flashinfer JIT compilation (requires CUDA headers)
    local base_flags=(--attention-backend triton --sampling-backend pytorch --disable-cuda-graph)
    local embed_flags=()
    if [[ "$embedding" == "true" ]]; then
        embed_flags=(--is-embedding "${base_flags[@]}" --context-length 512 --mem-fraction-static 0.1)
    else
        embed_flags=("${base_flags[@]}")
    fi

    CUDA_VISIBLE_DEVICES="$cuda_devs" \
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    PYTHONPATH="$PLUGINS_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    SGLANG_EXTERNAL_MODEL_PACKAGE="deberta_plugin" \
        "$VENV_PYTHON" -m sglang.launch_server \
            --model-path "$model" \
            --port "$port" \
            --host 0.0.0.0 \
            --tp-size "$tp_size" \
            "${embed_flags[@]+"${embed_flags[@]}"}" \
            "${extra_args[@]+"${extra_args[@]}"}" \
        > "$log_file" 2>&1 &

    echo "$!" > "$pid_file"
}

cmd_start() {
    local model="" raw_devices="${CUDA_VISIBLE_DEVICES:-0}" raw_ports="30000"
    local embedding="true"
    local extra_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)        model="$2";        shift 2 ;;
            --device)       raw_devices="$2";  shift 2 ;;
            --port)         raw_ports="$2";    shift 2 ;;
            --no-embedding) embedding="false"; shift   ;;
            *)              extra_args+=("$1"); shift  ;;
        esac
    done

    [[ -z "$model" ]] && { echo "Error: --model is required"; usage; exit 1; }

    IFS=',' read -ra all_devices <<< "$raw_devices"
    IFS=',' read -ra ports       <<< "$raw_ports"

    local n_instances=${#ports[@]}
    local n_devices=${#all_devices[@]}

    if (( n_devices % n_instances != 0 )); then
        echo "Error: device count ($n_devices) must be divisible by port count ($n_instances)"
        exit 1
    fi
    local tp_size=$(( n_devices / n_instances ))

    echo "Starting $n_instances sglang instance(s) for model: $model  (tp_size=$tp_size)"

    # Detect whether multiple instances share a GPU → launch sequentially to avoid
    # memory races (each sglang process profiles free VRAM at startup; concurrent
    # launches cause over-allocation and OOM).
    local unique_devs
    unique_devs="$(printf '%s\n' "${all_devices[@]}" | sort -u | wc -l)"
    local sequential=false
    (( n_instances > unique_devs )) && sequential=true

    local pids=()
    local timeout_per=180

    for (( i=0; i<n_instances; i++ )); do
        local cuda_devs port pid
        cuda_devs="$(IFS=','; echo "${all_devices[*]:$(( i * tp_size )):$tp_size}")"
        port="${ports[$i]}"
        start_one "$model" "$cuda_devs" "$tp_size" "$port" "$embedding" \
                  "${extra_args[@]+"${extra_args[@]}"}"
        pid="$(cat "$(model_to_pid_file "$model" "$port")")"
        pids+=("$pid")

        # When sharing a GPU, wait for this instance to be ready before starting the next
        # so that memory profiling sees accurate free VRAM.
        if $sequential && (( i < n_instances - 1 )); then
            echo "  Waiting for port $port to be ready before starting next instance..."
            local elapsed=0
            while true; do
                if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
                    echo "  [port $port] Ready."
                    break
                fi
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "  [port $port] Process exited unexpectedly. Check /tmp/sglang_${port}.log"
                    exit 1
                fi
                (( elapsed >= timeout_per )) && { echo "Timeout waiting for port $port."; exit 1; }
                sleep 2; elapsed=$(( elapsed + 2 ))
            done
        fi
    done

    echo "Waiting for all instances to be ready (timeout ${timeout_per}s)..."
    local elapsed=0

    while true; do
        local all_ready=true
        for (( i=0; i<n_instances; i++ )); do
            local port="${ports[$i]}" pid="${pids[$i]}"
            if ! curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "  [port $port] Process exited unexpectedly. Check /tmp/sglang_${port}.log"
                    exit 1
                fi
                all_ready=false
            fi
        done
        $all_ready && break
        (( elapsed >= timeout_per )) && { echo "Timeout. Check /tmp/sglang_<port>.log"; exit 1; }
        sleep 2; elapsed=$(( elapsed + 2 ))
    done

    echo "All instances ready:"
    for port in "${ports[@]}"; do echo "  http://localhost:$port"; done
}

stop_pids() {
    local pid_files=("$@")
    [[ ${#pid_files[@]} -eq 0 ]] && { echo "No running sglang servers found."; return; }

    for pid_file in "${pid_files[@]}"; do
        [[ -f "$pid_file" ]] || continue
        local pid label
        pid="$(cat "$pid_file")"
        label="$(basename "$pid_file" .pid)"
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $label (PID $pid)..."
            kill "$pid"
            local i=0
            while kill -0 "$pid" 2>/dev/null && (( i < 10 )); do sleep 1; i=$(( i + 1 )); done
            kill -0 "$pid" 2>/dev/null && { echo "Force killing PID $pid"; kill -9 "$pid"; }
            echo "  Stopped."
        else
            echo "$label: not running (stale PID)."
        fi
        rm -f "$pid_file"
    done
}

cmd_stop() {
    local model="" raw_ports="" stop_all=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --port)  raw_ports="$2"; shift 2 ;;
            --all)   stop_all=true; shift ;;
            *) echo "Unknown option: $1"; usage; exit 1 ;;
        esac
    done

    mkdir -p "$PID_DIR"
    local pid_files=()

    if $stop_all || [[ -z "$model" && -z "$raw_ports" ]]; then
        # Stop everything
        mapfile -t pid_files < <(find "$PID_DIR" -name "*.pid" 2>/dev/null | sort)
    elif [[ -n "$raw_ports" ]]; then
        IFS=',' read -ra ports <<< "$raw_ports"
        for port in "${ports[@]}"; do
            if [[ -n "$model" ]]; then
                pid_files+=("$(model_to_pid_file "$model" "$port")")
            else
                mapfile -t -O "${#pid_files[@]}" pid_files < \
                    <(find "$PID_DIR" -name "*_${port}.pid" 2>/dev/null)
            fi
        done
    elif [[ -n "$model" ]]; then
        local safe_name
        safe_name="$(echo "$model" | tr '/' '_' | tr -dc 'a-zA-Z0-9_-')"
        mapfile -t pid_files < <(find "$PID_DIR" -name "${safe_name}_*.pid" 2>/dev/null | sort)
    fi

    stop_pids "${pid_files[@]+"${pid_files[@]}"}"
}

# ── main ──────────────────────────────────────────────────────────────────────

[[ $# -eq 0 ]] && { usage; exit 1; }
COMMAND="$1"; shift

case "$COMMAND" in
    start) cmd_start "$@" ;;
    stop)  cmd_stop  "$@" ;;
    *)     echo "Unknown command: $COMMAND"; usage; exit 1 ;;
esac
