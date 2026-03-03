#!/usr/bin/env bash
# install_deberta_linux.sh — install DeBERTa dependencies in the project .venv (Linux)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: virtualenv python not found at $VENV_PYTHON"
    echo "Create the env first, e.g.: uv venv --python 3.11"
    exit 1
fi

if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "[bootstrap] pip not found in .venv; running ensurepip..."
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

PIP_CMD=("$VENV_PYTHON" -m pip)

echo "[1/4] Upgrading packaging tools..."
"${PIP_CMD[@]}" install --upgrade pip setuptools wheel

echo "[2/4] Installing HuggingFace DeBERTa runtime deps..."
"${PIP_CMD[@]}" install --upgrade \
    "transformers>=4.45.0" \
    "tokenizers>=0.20.0" \
    "sentencepiece>=0.1.99" \
    "safetensors>=0.4.3" \
    "protobuf>=4.25.0"

echo "[3/4] Installing optional Microsoft DeBERTa package..."
if ! SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    "${PIP_CMD[@]}" install --upgrade DeBERTa; then
    echo "Warning: optional package 'DeBERTa' failed to install."
    echo "         HuggingFace DeBERTa still works via transformers."
fi

echo "[4/4] Verifying imports..."
"$VENV_PYTHON" - <<'PY'
checks = [
    ("transformers", None),
    ("torch", None),
    ("tokenizers", None),
    ("sentencepiece", None),
    ("transformers", "DebertaV2Model"),
    ("transformers", "DebertaV2Tokenizer"),
]

for mod, symbol in checks:
    try:
        module = __import__(mod, fromlist=[symbol] if symbol else [])
        if symbol:
            getattr(module, symbol)
            print(f"OK {mod}.{symbol}")
        else:
            print(f"OK {mod}")
    except Exception as exc:
        target = f"{mod}.{symbol}" if symbol else mod
        print(f"MISSING {target}: {type(exc).__name__}: {exc}")
        raise

try:
    import DeBERTa  # noqa: F401
    print("OK DeBERTa")
except Exception as exc:
    print(f"OPTIONAL missing DeBERTa: {type(exc).__name__}: {exc}")
PY

if [[ ! -d "$PROJECT_DIR/sglang_plugins/deberta_plugin" ]]; then
    cat <<EOF
Note:
  README mentions sglang plugin path:
    $PROJECT_DIR/sglang_plugins/deberta_plugin
  That directory does not exist in this workspace right now.
  If you launch Prompt-Guard through sglang.sh, create/install that plugin first.
EOF
fi

echo "Done. DeBERTa dependencies are installed in $PROJECT_DIR/.venv"
