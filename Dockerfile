# =============================================================================
# Video Translation Container - Slim image, models on Network Volume
# =============================================================================
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for Step-Audio-EditX (filtered)
# Exclude torch/torchaudio (in base), gradio/spaces (not needed for CLI)
RUN pip install --no-cache-dir \
    torchaudio==2.4.0 \
    numpy \
    einops \
    transformers \
    accelerate \
    sentencepiece \
    safetensors \
    soundfile \
    librosa \
    scipy \
    onnxruntime \
    vector-quantize-pytorch \
    vocos

# Install translation pipeline dependencies
RUN pip install --no-cache-dir \
    faster-whisper>=1.0.0 \
    tqdm>=4.66.0 \
    httpx>=0.27.0

# Clean up
RUN pip cache purge && rm -rf /root/.cache/pip/*

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV REPO_URL=https://github.com/ctdontfollowme/video-translate-clone.git

WORKDIR /workspace

EXPOSE 8888 22

CMD ["/start.sh"]
