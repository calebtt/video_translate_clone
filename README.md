# Video Translation with Voice Cloning

Translate videos from any language to English with voice cloning.

- **Faster-Whisper** - Speech-to-text
- **HuggingFace Transformers** - Translation  
- **Step-Audio-EditX** - Zero-shot voice cloning TTS
- **FFmpeg** - Audio/video processing

## RunPod Setup (One-Time)

### 1. Create a Network Volume

Go to [RunPod Storage](https://runpod.io/console/user/storage) → **New Network Volume**:

| Setting | Value |
|---------|-------|
| Name | `video-translate` |
| Size | 50 GB |
| Region | Same as your pods |

### 2. Build & Push Docker Image

```bash
# Clone this repo locally
git clone https://github.com/ctdontfollowme/video-translate-clone.git
cd video-translate-clone

# Build
docker build --platform linux/amd64 -t ctdontfollowme/video-translate-clone:v1.0 .

# Push
docker login
docker push ctdontfollowme/video-translate-clone:v1.0
```

### 3. Create RunPod Template

Go to [RunPod Templates](https://runpod.io/console/user/templates) → **New Template**:

| Setting | Value |
|---------|-------|
| Template Name | `Video Translate Clone` |
| Container Image | `ctdontfollowme/video-translate-clone:v1.0` |
| Container Disk | 5 GB |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8888` |
| Expose TCP Ports | `22` |

### 4. First Launch

1. Deploy a pod with your template + Network Volume attached
2. First boot will download models (~7GB) - takes ~10 min
3. Models persist on volume - subsequent boots are fast (~30 sec)

---

## Usage

### Access JupyterLab
Click **Connect** → **HTTP Service [Port 8888]**

### Translate a Video

```bash
# Upload your video to /workspace/videos/ via Jupyter

python /workspace/translate.py \
  --video /workspace/videos/input.mp4 \
  --src_lang de \
  --mt_model Helsinki-NLP/opus-mt-de-en \
  --repo_path /workspace/Step-Audio-EditX \
  --model_path /workspace/models
```

Output: `/workspace/videos/input/translated.mp4`

---

## Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--video` | Input video | `/workspace/videos/input.mp4` |
| `--src_lang` | Source language | `de`, `ru`, `fr`, `es`, `zh` |
| `--mt_model` | Translation model | `Helsinki-NLP/opus-mt-de-en` |
| `--repo_path` | Step-Audio-EditX | `/workspace/Step-Audio-EditX` |
| `--model_path` | Models directory | `/workspace/models` |
| `--stage` | Run specific stage | `stt`, `tts`, `overlay`, `all` |
| `--skip_existing` | Skip generated TTS | flag |
| `--duck_gain` | Original audio vol | `0.15` (0-1) |

## Translation Models

| Language | Model |
|----------|-------|
| German → English | `Helsinki-NLP/opus-mt-de-en` |
| Russian → English | `Helsinki-NLP/opus-mt-ru-en` |
| French → English | `Helsinki-NLP/opus-mt-fr-en` |
| Spanish → English | `Helsinki-NLP/opus-mt-es-en` |
| Chinese → English | `Helsinki-NLP/opus-mt-zh-en` |

## Requirements

- GPU: 16GB+ VRAM (RTX 4090, A100, etc.)
- Network Volume: 50GB recommended

## Updating Code

Just push to GitHub - the container pulls latest code on every start.

```bash
git add -A && git commit -m "update" && git push
```

Then restart your pod.
