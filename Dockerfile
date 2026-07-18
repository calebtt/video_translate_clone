# =============================================================================
# Video Translation Container - Slim image, models on Network Volume
# =============================================================================
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsox-dev \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Python deps for pipeline + API (torch is in base image)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip cache purge \
    && rm -rf /root/.cache/pip/* /tmp/requirements.txt

# Bundle code as fallback; start.sh still git-pulls latest when network allows
COPY . /opt/video_translate_clone
RUN chmod +x /opt/video_translate_clone/start.sh \
    && ln -sf /opt/video_translate_clone/start.sh /start.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/video-translate-clone:/opt/video_translate_clone
ENV HF_HOME=/workspace/.cache/huggingface
ENV REPO_URL=https://github.com/calebtt/video_translate_clone.git
ENV VTCLONE_JOBS_DIR=/workspace/jobs
ENV VTCLONE_REPO_PATH=/workspace/Step-Audio-EditX
ENV VTCLONE_MODEL_PATH=/workspace/models/Step-Audio-EditX
ENV VTCLONE_TOKENIZER_PATH=/workspace/models/Step-Audio-Tokenizer
ENV VTCLONE_API_PORT=8000
ENV JUPYTER_TOKEN=

WORKDIR /workspace

EXPOSE 8888 8000 22

CMD ["/start.sh"]
