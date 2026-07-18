#!/bin/bash
set -euo pipefail

echo "=============================================="
echo "Video Translation Container Starting..."
echo "=============================================="

REPO_URL="${REPO_URL:-https://github.com/calebtt/video_translate_clone.git}"
CODE_DIR="/workspace/video-translate-clone"
SAEX_DIR="${VTCLONE_REPO_PATH:-/workspace/Step-Audio-EditX}"
MODELS_DIR="/workspace/models"
JOBS_DIR="${VTCLONE_JOBS_DIR:-/workspace/jobs}"
API_PORT="${VTCLONE_API_PORT:-8000}"
ENABLE_JUPYTER="${ENABLE_JUPYTER:-1}"
ENABLE_API="${ENABLE_API:-1}"

# ------------------------------------------------
# 1. Sync pipeline code from GitHub (fallback: image copy)
# ------------------------------------------------
echo "[1/5] Syncing code from GitHub..."
if [ -d "$CODE_DIR/.git" ]; then
    echo "  Pulling latest changes..."
    if ! (cd "$CODE_DIR" && git pull --ff-only); then
        echo "  WARNING: git pull failed; using existing checkout"
    fi
else
    echo "  Cloning repository..."
    rm -rf "$CODE_DIR"
    if git clone --depth 1 "$REPO_URL" "$CODE_DIR"; then
        echo "  Clone OK"
    else
        echo "  WARNING: clone failed; copying bundled /opt/video_translate_clone"
        mkdir -p "$CODE_DIR"
        cp -a /opt/video_translate_clone/. "$CODE_DIR/" 2>/dev/null || true
    fi
fi

# Ensure bundled copy exists if git path empty
if [ ! -f "$CODE_DIR/video_translate_clone_perf.py" ] && [ -d /opt/video_translate_clone ]; then
    echo "  Seeding code from image..."
    mkdir -p "$CODE_DIR"
    cp -a /opt/video_translate_clone/. "$CODE_DIR/"
fi

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# ------------------------------------------------
# 2. Step-Audio-EditX source repo
# ------------------------------------------------
echo "[2/5] Checking Step-Audio-EditX repo..."
if [ ! -d "$SAEX_DIR" ]; then
    echo "  Cloning Step-Audio-EditX..."
    git clone --depth 1 https://github.com/stepfun-ai/Step-Audio-EditX.git "$SAEX_DIR"
else
    echo "  Step-Audio-EditX already present."
fi

# Install SAEX package deps if pyproject/requirements present (best-effort)
if [ -f "$SAEX_DIR/pyproject.toml" ] || [ -f "$SAEX_DIR/requirements.txt" ]; then
    echo "  Installing Step-Audio-EditX Python deps (best-effort)..."
    pip install --no-cache-dir -e "$SAEX_DIR" 2>/dev/null \
        || pip install --no-cache-dir -r "$SAEX_DIR/requirements.txt" 2>/dev/null \
        || echo "  WARNING: SAEX pip install skipped/failed — install vLLM/deps manually if needed"
fi

# ------------------------------------------------
# 3. Download models onto network volume
# ------------------------------------------------
mkdir -p "$MODELS_DIR" "$JOBS_DIR" /workspace/videos

download_hf_model() {
    local repo_id="$1"
    local dest="$2"
    local marker="$dest/.download_complete"

    if [ -f "$marker" ]; then
        echo "  $dest already complete."
        return 0
    fi

    echo "  Downloading $repo_id -> $dest ..."
    rm -rf "$dest"
    mkdir -p "$dest"

    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo_id" --local-dir "$dest" --local-dir-use-symlinks False
    elif python -c "import huggingface_hub" 2>/dev/null; then
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo_id}", local_dir="${dest}", local_dir_use_symlinks=False)
PY
    else
        # Fallback: git LFS
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "https://huggingface.co/${repo_id}" "$dest"
        (cd "$dest" && git lfs pull && rm -rf .git)
    fi

    # Basic non-empty check
    if [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  ERROR: download appears empty: $dest"
        return 1
    fi
    touch "$marker"
    echo "  Done: $dest"
}

echo "[3/5] Checking models..."
download_hf_model "stepfun-ai/Step-Audio-Tokenizer" "$MODELS_DIR/Step-Audio-Tokenizer"
download_hf_model "stepfun-ai/Step-Audio-EditX" "$MODELS_DIR/Step-Audio-EditX"

# ------------------------------------------------
# 4. Convenience symlinks / env
# ------------------------------------------------
echo "[4/5] Setting up workspace..."
ln -sfn "$CODE_DIR/video_translate_clone_perf.py" /workspace/translate.py
ln -sfn "$CODE_DIR/api_server.py" /workspace/api_server.py

export VTCLONE_REPO_PATH="$SAEX_DIR"
export VTCLONE_MODEL_PATH="$MODELS_DIR/Step-Audio-EditX"
export VTCLONE_TOKENIZER_PATH="$MODELS_DIR/Step-Audio-Tokenizer"
export VTCLONE_JOBS_DIR="$JOBS_DIR"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Free disk / GPU hints
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi
df -h /workspace 2>/dev/null | tail -1 || true

# ------------------------------------------------
# 4b. Public API key (required for exposed /v1 routes)
# ------------------------------------------------
API_KEY_FILE="${VTCLONE_API_KEY_FILE:-/workspace/.vtclone_api_key}"
export VTCLONE_REQUIRE_API_KEY="${VTCLONE_REQUIRE_API_KEY:-1}"
export VTCLONE_API_KEY_FILE="$API_KEY_FILE"

if [ -z "${VTCLONE_API_KEY:-}" ]; then
    if [ -f "$API_KEY_FILE" ] && [ -s "$API_KEY_FILE" ]; then
        export VTCLONE_API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
        echo "[auth] Loaded API key from $API_KEY_FILE"
    else
        export VTCLONE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        umask 077
        printf '%s\n' "$VTCLONE_API_KEY" > "$API_KEY_FILE"
        chmod 600 "$API_KEY_FILE" 2>/dev/null || true
        echo "[auth] Generated new API key -> $API_KEY_FILE"
    fi
else
    # Persist env-provided key so restarts stay consistent
    umask 077
    printf '%s\n' "$VTCLONE_API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE" 2>/dev/null || true
    echo "[auth] Using VTCLONE_API_KEY from environment (saved to $API_KEY_FILE)"
fi

echo ""
echo "=============================================="
echo "Ready."
echo "=============================================="
echo ""
echo "CLI:"
echo "  python /workspace/translate.py \\"
echo "    --video /workspace/videos/input.mp4 \\"
echo "    --src_lang de \\"
echo "    --mt_model Helsinki-NLP/opus-mt-de-en \\"
echo "    --repo_path $SAEX_DIR \\"
echo "    --model_path $MODELS_DIR/Step-Audio-EditX \\"
echo "    --tokenizer_path $MODELS_DIR/Step-Audio-Tokenizer"
echo ""
echo "PUBLIC API (bind 0.0.0.0:${API_PORT}):"
echo "  Auth header:  X-API-Key: ${VTCLONE_API_KEY}"
echo "  Or:           Authorization: Bearer ${VTCLONE_API_KEY}"
echo "  Key file:     ${API_KEY_FILE}"
echo "  Health:       curl http://127.0.0.1:${API_PORT}/health"
echo "  RunPod proxy: https://<POD_ID>-${API_PORT}.proxy.runpod.net"
echo "  Docs:         https://<POD_ID>-${API_PORT}.proxy.runpod.net/docs"
echo "  Example job:"
echo "    curl -X POST \"https://<POD_ID>-${API_PORT}.proxy.runpod.net/v1/jobs\" \\"
echo "      -H \"X-API-Key: ${VTCLONE_API_KEY}\" \\"
echo "      -F \"video=@./clip.mp4\" -F \"src_lang=de\" \\"
echo "      -F \"mt_model=Helsinki-NLP/opus-mt-de-en\""
echo "=============================================="

# ------------------------------------------------
# 5. Start API (+ optional Jupyter)
# ------------------------------------------------
echo "[5/5] Starting services..."

API_PID=""
if [ "$ENABLE_API" = "1" ]; then
    echo "  Starting public API on 0.0.0.0:${API_PORT}..."
    cd "$CODE_DIR"
    # nohup so the API survives shell exec → jupyter
    # --proxy-headers: trust RunPod/edge X-Forwarded-* when behind HTTPS proxy
    nohup env \
        VTCLONE_API_KEY="$VTCLONE_API_KEY" \
        VTCLONE_REQUIRE_API_KEY="$VTCLONE_REQUIRE_API_KEY" \
        VTCLONE_API_KEY_FILE="$API_KEY_FILE" \
        VTCLONE_JOBS_DIR="$JOBS_DIR" \
        VTCLONE_REPO_PATH="$SAEX_DIR" \
        VTCLONE_MODEL_PATH="$MODELS_DIR/Step-Audio-EditX" \
        VTCLONE_TOKENIZER_PATH="$MODELS_DIR/Step-Audio-Tokenizer" \
        python -m uvicorn vtclone.api:app \
            --host 0.0.0.0 \
            --port "$API_PORT" \
            --proxy-headers \
            --forwarded-allow-ips='*' \
            --log-level info \
        > /workspace/api.log 2>&1 &
    API_PID=$!
    disown "$API_PID" 2>/dev/null || true
    echo "  API pid=$API_PID (log: /workspace/api.log)"
fi

if [ "$ENABLE_JUPYTER" = "1" ]; then
    JUPYTER_ARGS=(--ip=0.0.0.0 --port=8888 --no-browser --allow-root)
    if [ -n "${JUPYTER_TOKEN:-}" ]; then
        JUPYTER_ARGS+=(--NotebookApp.token="$JUPYTER_TOKEN" --ServerApp.token="$JUPYTER_TOKEN")
        echo "  Jupyter token auth enabled"
    else
        # Generate a random token if none provided (safer than empty)
        GEN_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
        JUPYTER_ARGS+=(--NotebookApp.token="$GEN_TOKEN" --ServerApp.token="$GEN_TOKEN")
        echo "  Jupyter token (save this): $GEN_TOKEN"
    fi
    if [ -f /start.sh.runpod ]; then
        # Prefer keeping API up; still exec jupyter in foreground
        exec jupyter lab "${JUPYTER_ARGS[@]}"
    else
        exec jupyter lab "${JUPYTER_ARGS[@]}"
    fi
else
    if [ -n "$API_PID" ]; then
        wait "$API_PID"
    else
        echo "Nothing to run (ENABLE_API=0 ENABLE_JUPYTER=0); sleeping"
        exec sleep infinity
    fi
fi
