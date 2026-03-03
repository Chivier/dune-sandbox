# dune-sandbox

Sandbox for running LLM inference with [sglang](https://github.com/sgl-project/sglang).

## Setup

```bash
uv venv --python 3.11
uv pip install "sglang[all]"
```

## Scripts

Three scripts are provided in `scripts/`. All use the same underlying `sglang.sh` and store PID files in `/tmp/sglang_pids/`. Logs go to `/tmp/sglang_<port>.log`.

---

### `scripts/sglang.sh` — general-purpose launcher

Start and stop any HuggingFace model.

**Device / port rules:**
- Number of `--port` values = number of instances to launch
- `tp_size` per instance = `n_devices / n_ports`

```
# 1 instance, TP=4 (tensor-parallel across 4 GPUs)
./scripts/sglang.sh start --model meta-llama/Meta-Llama-3-70B-Instruct \
    --device 0,1,2,3 --port 30000

# 4 instances, TP=1 (one per GPU)
./scripts/sglang.sh start --model meta-llama/Prompt-Guard-86M \
    --device 0,1,2,3 --port 30000,30001,30002,30003

# 3 instances on 2 GPUs: GPUs 1,1,2 → TP=1 each (instances share GPU 1)
./scripts/sglang.sh start --model meta-llama/Prompt-Guard-86M \
    --device 1,1,2 --port 30000,30001,30002

# Stop specific ports
./scripts/sglang.sh stop --model meta-llama/Prompt-Guard-86M --port 30000,30001

# Stop all instances of a model
./scripts/sglang.sh stop --model meta-llama/Prompt-Guard-86M

# Stop everything
./scripts/sglang.sh stop --all
```

**Options:**
| Flag | Description |
|------|-------------|
| `--model` | HuggingFace model ID (required for start) |
| `--device` | Comma-separated GPU IDs. Count / port-count = tp_size. Default: `$CUDA_VISIBLE_DEVICES` or `0` |
| `--port` | Comma-separated ports. Count determines number of instances. Default: `30000` |
| `--mem-fraction-static` | Forwarded to sglang. If omitted in embedding mode, `sglang.sh` auto-computes it per instance from live `nvidia-smi` free VRAM and remaining launches per GPU (tunable with `SGLANG_AUTO_MEM_HEADROOM_MB`, `SGLANG_AUTO_MEM_MIN_FRACTION`, `SGLANG_AUTO_MEM_MAX_FRACTION`). |
| `--no-embedding` | Skip `--is-embedding` (use for generative/chat models) |
| `--all` | (stop) Stop all running sglang servers |

Extra args are forwarded verbatim to `sglang.launch_server`.

**Attention backend:** All instances use `--attention-backend triton --sampling-backend pytorch --disable-cuda-graph` to avoid flashinfer JIT compilation (which requires CUDA headers not present on this system).

---

### `scripts/sglang_70b_model.sh` — LLaMA 70B

Serves `meta-llama/Meta-Llama-3-70B-Instruct`. Requires 4 GPUs (70B in bf16 ≈ 140 GB).

```bash
# Start on GPUs 0,1,2,3 (TP=4, 1 instance)
./scripts/sglang_70b_model.sh start --device 0,1,2,3

# Start on GPUs 0,1,2,3,4,5,6,7 (TP=4, 2 instances on ports 30000 and 30001)
./scripts/sglang_70b_model.sh start \
    --device 0,1,2,3,4,5,6,7 --port 30000,30001

# Stop specific port
./scripts/sglang_70b_model.sh stop --port 30000

# Stop all instances of this model
./scripts/sglang_70b_model.sh stop --all
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | GPU IDs (comma-separated). Use 4 GPUs for TP=4. |
| `--port` | `30000` | Port(s) to serve on |
| `--all` | — | (stop) Stop all instances |

Extra args are forwarded to sglang (e.g. `--mem-fraction-static 0.7`).

**API (OpenAI-compatible):**
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Meta-Llama-3-70B-Instruct",
       "messages":[{"role":"user","content":"Hello!"}]}'
```

---

### `scripts/sglang_qwen3_model.sh` — Qwen3Guard 8B

Serves `Qwen/Qwen3Guard-Gen-8B`, a generative guard model that classifies prompt safety. Default is 8 instances but you can set `--instances`.

**VRAM requirements:** 8B in bf16 ≈ 16 GB weights. On a single GPU, the number of instances is limited by available VRAM. Use `--mem-fraction-static` to cap the KV cache per instance.

| GPU VRAM | Instances on 1 GPU | Recommended `--mem-fraction-static` |
|----------|-------------------|-------------------------------------|
| 80 GB    | 2                 | 0.25–0.30                           |
| 80 GB    | 4 (tight)         | 0.10–0.15                           |
| 80 GB    | 8                 | Use 8 separate GPUs                 |

> **Note:** Instances launch concurrently and each measures available VRAM independently. If `--mem-fraction-static` is too high for multiple instances on one GPU, instances may get unequal KV cache budgets. Always set `--mem-fraction-static` conservatively when sharing a GPU.

```bash
# 2 instances on GPU 1, ports 30010 and 30011, limited KV cache
./scripts/sglang_qwen3_model.sh start \
    --device 1 --port 30010 --instances 2 --mem-fraction-static 0.3

# 8 instances across 8 GPUs (one model per GPU), ports 30000–30007
./scripts/sglang_qwen3_model.sh start \
    --device 0,1,2,3,4,5,6,7 --port 30000 --instances 8

# Stop specific instance block (base port 30010)
./scripts/sglang_qwen3_model.sh stop --port 30010

# Stop all instances of this model
./scripts/sglang_qwen3_model.sh stop --all
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | GPU ID to run all instances on |
| `--port` | `30000` | Base port; instances use base, base+1, …, base+N-1 |
| `--instances` | `8` | Number of model copies to launch |
| `--all` | — | (stop) Stop all instances |

Extra args are forwarded to sglang (e.g. `--mem-fraction-static 0.3`, `--context-length 4096`).

**API:** Qwen3Guard-Gen-8B produces a natural-language safety verdict:

```bash
curl http://localhost:30010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3Guard-Gen-8B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
# Response: {"choices":[{"message":{"content":"Safety: Safe\nCategories: None",...}}]}
```

---

## Custom Plugin: DeBERTa-v2 for Prompt-Guard-86M

`meta-llama/Prompt-Guard-86M` uses DeBERTa-v2's disentangled attention which is incompatible with sglang's RadixAttention kernels. The plugin at `sglang_plugins/deberta_plugin/deberta_v2.py` wraps the HuggingFace model directly and integrates with sglang's embedding server interface.

The plugin is loaded automatically by `sglang.sh` via:
```
SGLANG_EXTERNAL_MODEL_PACKAGE=deberta_plugin
PYTHONPATH=<project_root>/sglang_plugins:...
```

**API:** Prompt-Guard-86M returns a 3-class softmax probability vector `[P(BENIGN), P(INJECTION), P(JAILBREAK)]`:

```bash
curl http://localhost:30000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Prompt-Guard-86M","input":"Ignore all previous instructions."}'
# embedding: [0.0012, 0.9871, 0.0117]  → high injection probability
```
