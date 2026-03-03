#!/usr/bin/env bash
# vllm.sh — start/stop vLLM OpenAI API servers (supports parallel multi-instance + TP)
#
# DEVICE / PORT RULES:
#   Number of ports  = number of instances to launch.
#   Number of devices / number of ports = tensor_parallel_size per instance.
#
#   Examples:
#     --device 0,1,2,3  --port 30000              -> 1 instance,  TP=4
#     --device 0,1,2,3  --port 30000,30001         -> 2 instances, TP=2 each
#     --device 0,1,2,3  --port 30000,30001,30002,30003 -> 4 instances, TP=1 each
#     --device 0,0,0,0,0,0,0,0 --port 30000,...,30007  -> 8 instances on same GPU
#
#   If only one --port is given with multiple devices, it is treated as TP (single instance).
#   If --port is omitted, defaults to 30000 (single instance).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PID_DIR="/tmp/vllm_pids"

if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON="${PYTHON:-python3}"
fi

usage() {
    cat <<EOF
Usage:
  $0 start --model <model_id> [--device <devices>] [--port <ports>] [extra vllm args...]
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
  --all      (stop only) Stop every running vLLM server
  Extra args are forwarded verbatim to vllm.entrypoints.openai.api_server.

  Stability defaults:
    For TP>1, this script auto-adds:
      --disable-custom-all-reduce
    unless already provided in extra args.
    It does NOT force --enforce-eager by default, because eager mode can
    destabilize some models (including Qwen3 TP launches).
    Controls:
      VLLM_SAFE_TP_MODE=auto|off|aggressive (default: auto)
        auto: add --disable-custom-all-reduce only
        aggressive: also add --enforce-eager
      VLLM_FORCE_EAGER=1 to force --enforce-eager regardless of mode

  Memory defaults:
    If --gpu-memory-utilization/--kv-cache-memory-bytes is not provided,
    this script can auto-add --gpu-memory-utilization per instance at launch
    time, based on live nvidia-smi free memory and remaining launches per GPU.
    Controls:
      VLLM_AUTO_GPU_MEM_UTIL=auto|off   (default: auto)
      VLLM_AUTO_MEM_HEADROOM_MB         (default: 2048)
      VLLM_AUTO_MEM_MIN_FRACTION        (default: 0.05)
      VLLM_AUTO_MEM_MAX_FRACTION        (default: 0.90)

Examples:
  $0 start --model meta-llama/Llama-3.1-8B-Instruct --device 0 --port 30000
  $0 start --model meta-llama/Llama-3.1-70B-Instruct --device 0,1,2,3 --port 30000
  $0 stop --model meta-llama/Llama-3.1-8B-Instruct
  $0 stop --port 30000,30001
  $0 stop --all
EOF
}

model_to_pid_file() {
    local model="$1" port="$2" safe_name
    safe_name="$(echo "$model" | tr '/' '_' | tr -dc 'a-zA-Z0-9_-')"
    echo "$PID_DIR/${safe_name}_${port}.pid"
}

has_cli_flag() {
    local flag="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
            return 0
        fi
    done
    return 1
}

# Launch one server instance
# Args: model  cuda_devs  tp_size  port  [extra...]
start_one() {
    local model="$1" cuda_devs="$2" tp_size="$3" port="$4"
    shift 4
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

    local log_file="/tmp/vllm_${port}.log"
    echo "  [port $port] CUDA_VISIBLE_DEVICES=$cuda_devs  tp_size=$tp_size  log=$log_file"

    CUDA_VISIBLE_DEVICES="$cuda_devs" \
        "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --port "$port" \
            --host 0.0.0.0 \
            --tensor-parallel-size "$tp_size" \
            "${extra_args[@]+"${extra_args[@]}"}" \
        > "$log_file" 2>&1 &

    echo "$!" > "$pid_file"
}

cmd_start() {
    local model="" raw_devices="${CUDA_VISIBLE_DEVICES:-0}" raw_ports="30000"
    local extra_args=()
    local safe_tp_mode="${VLLM_SAFE_TP_MODE:-auto}"
    local force_eager="${VLLM_FORCE_EAGER:-0}"
    local auto_gpu_mem_mode="${VLLM_AUTO_GPU_MEM_UTIL:-auto}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)  model="$2";       shift 2 ;;
            --device) raw_devices="$2"; shift 2 ;;
            --port)   raw_ports="$2";   shift 2 ;;
            *)        extra_args+=("$1"); shift ;;
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
    local auto_mem_headroom_mb="${VLLM_AUTO_MEM_HEADROOM_MB:-2048}"
    local auto_mem_min_fraction="${VLLM_AUTO_MEM_MIN_FRACTION:-0.05}"
    local auto_mem_max_fraction="${VLLM_AUTO_MEM_MAX_FRACTION:-0.90}"

    case "$safe_tp_mode" in
        auto|off|aggressive) ;;
        *)
            echo "Warning: invalid VLLM_SAFE_TP_MODE='$safe_tp_mode' (expected: auto|off|aggressive). Using auto."
            safe_tp_mode="auto"
            ;;
    esac

    case "$auto_gpu_mem_mode" in
        auto|off) ;;
        *)
            echo "Warning: invalid VLLM_AUTO_GPU_MEM_UTIL='$auto_gpu_mem_mode' (expected: auto|off). Using auto."
            auto_gpu_mem_mode="auto"
            ;;
    esac

    local safe_flags_added=()
    if (( tp_size > 1 )) && [[ "$safe_tp_mode" != "off" ]]; then
        if ! has_cli_flag "--disable-custom-all-reduce" "${extra_args[@]}"; then
            extra_args+=(--disable-custom-all-reduce)
            safe_flags_added+=(--disable-custom-all-reduce)
        fi
        if [[ "$safe_tp_mode" == "aggressive" ]] && ! has_cli_flag "--enforce-eager" "${extra_args[@]}"; then
            extra_args+=(--enforce-eager)
            safe_flags_added+=(--enforce-eager)
        fi
    fi
    if [[ "$force_eager" == "1" ]] && ! has_cli_flag "--enforce-eager" "${extra_args[@]}"; then
        extra_args+=(--enforce-eager)
        safe_flags_added+=(--enforce-eager)
    fi

    local has_gpu_mem_arg=false
    if has_cli_flag "--gpu-memory-utilization" "${extra_args[@]}"; then
        has_gpu_mem_arg=true
    fi
    if has_cli_flag "--kv-cache-memory-bytes" "${extra_args[@]}"; then
        has_gpu_mem_arg=true
    fi

    local -a instance_cuda_devs=()
    declare -A remaining_uses=()
    for (( i=0; i<n_instances; i++ )); do
        local cuda_group
        cuda_group="$(IFS=','; echo "${all_devices[*]:$(( i * tp_size )):$tp_size}")"
        instance_cuda_devs+=("$cuda_group")
        IFS=',' read -ra group_devs <<< "$cuda_group"
        for dev in "${group_devs[@]}"; do
            remaining_uses["$dev"]=$(( ${remaining_uses["$dev"]:-0} + 1 ))
        done
    done

    local can_auto_mem=true
    local auto_mem_warning_emitted=false
    if [[ "$auto_gpu_mem_mode" == "auto" ]] && ! $has_gpu_mem_arg; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            can_auto_mem=false
            echo "Warning: nvidia-smi not found; auto GPU memory tuning disabled."
            auto_mem_warning_emitted=true
        fi
    fi

    echo "Starting $n_instances vLLM instance(s) for model: $model  (tp_size=$tp_size)"
    if (( ${#safe_flags_added[@]} > 0 )); then
        echo "TP>1 safe defaults enabled: ${safe_flags_added[*]} (set VLLM_SAFE_TP_MODE=off to disable)"
    fi

    # If any GPU is used by more than one instance, launch sequentially to avoid
    # startup memory races and OOM.
    local sequential=false
    for dev in "${!remaining_uses[@]}"; do
        if (( ${remaining_uses["$dev"]} > 1 )); then
            sequential=true
            break
        fi
    done

    local pids=()
    local timeout_per=180

    for (( i=0; i<n_instances; i++ )); do
        local cuda_devs port pid
        local instance_extra_args=("${extra_args[@]+"${extra_args[@]}"}")
        cuda_devs="${instance_cuda_devs[$i]}"
        port="${ports[$i]}"

        if [[ "$auto_gpu_mem_mode" == "auto" ]] && ! $has_gpu_mem_arg && $can_auto_mem; then
            local smi_out
            if smi_out="$(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
                declare -A gpu_total_mb=()
                declare -A gpu_free_mb=()
                local gpu_idx total_mb free_mb
                while IFS=',' read -r gpu_idx total_mb free_mb; do
                    gpu_idx="${gpu_idx//[[:space:]]/}"
                    total_mb="${total_mb//[[:space:]]/}"
                    free_mb="${free_mb//[[:space:]]/}"
                    [[ -z "$gpu_idx" ]] && continue
                    gpu_total_mb["$gpu_idx"]="$total_mb"
                    gpu_free_mb["$gpu_idx"]="$free_mb"
                done <<< "$smi_out"

                local computed_fraction=""
                local auto_mem_ok=true
                local notes=()
                IFS=',' read -ra launch_devs <<< "$cuda_devs"
                for dev in "${launch_devs[@]}"; do
                    total_mb="${gpu_total_mb[$dev]:-0}"
                    free_mb="${gpu_free_mb[$dev]:-0}"
                    local launches_left="${remaining_uses[$dev]:-0}"
                    if (( total_mb <= 0 || free_mb <= 0 || launches_left <= 0 )); then
                        auto_mem_ok=false
                        break
                    fi
                    local usable_mb=$(( free_mb - auto_mem_headroom_mb ))
                    (( usable_mb < 256 )) && usable_mb=256
                    local per_launch_mb=$(( usable_mb / launches_left ))
                    local frac
                    frac="$(awk -v b="$per_launch_mb" -v t="$total_mb" -v min="$auto_mem_min_fraction" -v max="$auto_mem_max_fraction" \
                        'BEGIN { f=b/t; if (f<min) f=min; if (f>max) f=max; printf "%.3f", f }')"
                    if [[ -z "$computed_fraction" ]]; then
                        computed_fraction="$frac"
                    else
                        computed_fraction="$(awk -v a="$computed_fraction" -v b="$frac" 'BEGIN { if (a < b) printf "%.3f", a; else printf "%.3f", b }')"
                    fi
                    notes+=("gpu$dev free=${free_mb}MiB left=$launches_left")
                done

                if $auto_mem_ok && [[ -n "$computed_fraction" ]]; then
                    instance_extra_args+=(--gpu-memory-utilization "$computed_fraction")
                    echo "  [port $port] Auto --gpu-memory-utilization=$computed_fraction (${notes[*]})"
                    if awk -v f="$computed_fraction" 'BEGIN { exit !(f <= 0.10) }'; then
                        echo "  [port $port] Warning: very low available VRAM detected; model loading may still fail."
                    fi
                else
                    if ! $auto_mem_warning_emitted; then
                        echo "Warning: auto GPU memory tuning failed; using vLLM defaults."
                        auto_mem_warning_emitted=true
                    fi
                    can_auto_mem=false
                fi
            else
                if ! $auto_mem_warning_emitted; then
                    echo "Warning: nvidia-smi query failed; using vLLM defaults for GPU memory utilization."
                    auto_mem_warning_emitted=true
                fi
                can_auto_mem=false
            fi
        fi

        start_one "$model" "$cuda_devs" "$tp_size" "$port" \
                  "${instance_extra_args[@]+"${instance_extra_args[@]}"}"
        pid="$(cat "$(model_to_pid_file "$model" "$port")")"
        pids+=("$pid")

        IFS=',' read -ra used_devs <<< "$cuda_devs"
        for dev in "${used_devs[@]}"; do
            remaining_uses["$dev"]=$(( ${remaining_uses["$dev"]:-1} - 1 ))
        done

        if $sequential && (( i < n_instances - 1 )); then
            echo "  Waiting for port $port to be ready before starting next instance..."
            local elapsed=0
            while true; do
                if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
                    echo "  [port $port] Ready."
                    break
                fi
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "  [port $port] Process exited unexpectedly. Check /tmp/vllm_${port}.log"
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
                    echo "  [port $port] Process exited unexpectedly. Check /tmp/vllm_${port}.log"
                    exit 1
                fi
                all_ready=false
            fi
        done
        $all_ready && break
        (( elapsed >= timeout_per )) && { echo "Timeout. Check /tmp/vllm_<port>.log"; exit 1; }
        sleep 2; elapsed=$(( elapsed + 2 ))
    done

    echo "All instances ready:"
    for port in "${ports[@]}"; do echo "  http://localhost:$port"; done
}

stop_pids() {
    local pid_files=("$@")
    [[ ${#pid_files[@]} -eq 0 ]] && { echo "No running vLLM servers found."; return; }

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

# -- main ---------------------------------------------------------------------

[[ $# -eq 0 ]] && { usage; exit 1; }
COMMAND="$1"; shift

case "$COMMAND" in
    start) cmd_start "$@" ;;
    stop)  cmd_stop  "$@" ;;
    *)     echo "Unknown command: $COMMAND"; usage; exit 1 ;;
esac
