#!/bin/bash
set -e

echo "=============================================="
echo "Video Translation Container Starting..."
echo "=============================================="

# ------------------------------------------------
# 1. Pull/update code from GitHub
# ------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/calebtt/video_translate_clone.git}"
CODE_DIR="/workspace/video-translate-clone"

echo "[1/4] Syncing code from GitHub..."
if [ -d "$CODE_DIR/.git" ]; then
    echo "  Pulling latest changes..."
    cd "$CODE_DIR" && git pull --ff-only || git pull --rebase || true
    cd /workspace
else
    echo "  Cloning repository..."
    rm -rf "$CODE_DIR"
    git clone --depth 1 "$REPO_URL" "$CODE_DIR"
fi

# ------------------------------------------------
# 2. Clone Step-Audio-EditX repo if needed
# ------------------------------------------------
SAEX_DIR="/workspace/Step-Audio-EditX"

echo "[2/4] Checking Step-Audio-EditX repo..."
if [ ! -d "$SAEX_DIR" ]; then
    echo "  Cloning Step-Audio-EditX..."
    git clone --depth 1 https://github.com/stepfun-ai/Step-Audio-EditX.git "$SAEX_DIR"
else
    echo "  Step-Audio-EditX already present."
fi

# ------------------------------------------------
# 3. Download models if not present
# ------------------------------------------------
MODELS_DIR="/workspace/models"
mkdir -p "$MODELS_DIR"

echo "[3/4] Checking models..."

# Step-Audio-Tokenizer
if [ ! -d "$MODELS_DIR/Step-Audio-Tokenizer" ] || [ -z "$(ls -A $MODELS_DIR/Step-Audio-Tokenizer 2>/dev/null)" ]; then
    echo "  Downloading Step-Audio-Tokenizer (~2GB)..."
    rm -rf "$MODELS_DIR/Step-Audio-Tokenizer"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer "$MODELS_DIR/Step-Audio-Tokenizer"
    cd "$MODELS_DIR/Step-Audio-Tokenizer" && git lfs pull && rm -rf .git
    cd /workspace
else
    echo "  Step-Audio-Tokenizer already present."
fi

# Step-Audio-EditX model
if [ ! -d "$MODELS_DIR/Step-Audio-EditX" ] || [ -z "$(ls -A $MODELS_DIR/Step-Audio-EditX 2>/dev/null)" ]; then
    echo "  Downloading Step-Audio-EditX model (~5GB)..."
    rm -rf "$MODELS_DIR/Step-Audio-EditX"
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/stepfun-ai/Step-Audio-EditX "$MODELS_DIR/Step-Audio-EditX"
    cd "$MODELS_DIR/Step-Audio-EditX" && git lfs pull && rm -rf .git
    cd /workspace
else
    echo "  Step-Audio-EditX model already present."
fi

# ------------------------------------------------
# 4. Create convenience symlink and videos dir
# ------------------------------------------------
echo "[4/4] Setting up workspace..."
mkdir -p /workspace/videos

# Symlink the main script to /workspace for convenience
ln -sf "$CODE_DIR/video_translate_clone_perf.py" /workspace/translate.py

echo ""
echo "=============================================="
echo "Ready! Access JupyterLab via RunPod Connect."
echo "=============================================="
echo ""
echo "Quick start:"
echo "  python /workspace/translate.py \\"
echo "    --video /workspace/videos/input.mp4 \\"
echo "    --src_lang de \\"
echo "    --mt_model Helsinki-NLP/opus-mt-de-en \\"
echo "    --repo_path /workspace/Step-Audio-EditX \\"
echo "    --model_path /workspace/models"
echo ""
echo "=============================================="

# ------------------------------------------------
# Start Jupyter (RunPod's default behavior)
# ------------------------------------------------
if [ -f /start.sh.runpod ]; then
    # If there's a RunPod default start script, use it
    exec /start.sh.runpod
else
    # Otherwise start Jupyter manually
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
fi
