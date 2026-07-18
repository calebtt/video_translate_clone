# Video Translation with Voice Cloning

Translate videos from a source language to English with **zero-shot voice cloning**.

| Stage | Tool |
|-------|------|
| STT | Faster-Whisper |
| Translation | HuggingFace seq2seq (e.g. Opus-MT) |
| TTS / clone | Step-Audio-EditX (load-once in-process when possible) |
| Mix | FFmpeg (+ optional `atempo` duration match) |

Includes:

- **CLI** pipeline with staged runs (`stt` / `tts` / `overlay` / `all`)
- **HTTP job API** (upload video → poll → download result)
- RunPod-friendly Docker image (models on a network volume)

---

## RunPod setup (one-time)

### 1. Network volume

[RunPod Storage](https://runpod.io/console/user/storage) → **New Network Volume**:

| Setting | Value |
|---------|-------|
| Name | `video-translate` |
| Size | 50 GB+ |
| Region | Same as pods |

### 2. Build & push image

```bash
git clone https://github.com/calebtt/video_translate_clone.git
cd video_translate_clone

docker build --platform linux/amd64 -t calebtt/video-translate-clone:v1.1 .
docker login
docker push calebtt/video-translate-clone:v1.1
```

### 3. Template

| Setting | Value |
|---------|-------|
| Container Image | `calebtt/video-translate-clone:v1.1` |
| Container Disk | 10 GB |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8000,8888` |
| Expose TCP Ports | `22` |

**Public access (RunPod):** HTTP ports on the template are published via RunPod’s proxy. After the pod is running, open:

```text
https://<POD_ID>-8000.proxy.runpod.net
```

You can also use **Connect → HTTP Service [Port 8000]** in the RunPod UI. All `/v1/*` routes require an API key (see below). `/health` is public for probes.

Env vars:

| Env | Default | Purpose |
|-----|---------|---------|
| `VTCLONE_API_KEY` | auto-generated | Public API key (also written to volume) |
| `VTCLONE_API_KEY_FILE` | `/workspace/.vtclone_api_key` | Persist key across restarts |
| `VTCLONE_REQUIRE_API_KEY` | `1` | Enforce key on `/v1/*` (set `0` only for private nets) |
| `VTCLONE_CORS_ORIGINS` | `*` | CORS allowlist (comma-separated) |
| `ENABLE_API` | `1` | Start FastAPI on boot (public bind `0.0.0.0`) |
| `ENABLE_JUPYTER` | `1` | Start JupyterLab |
| `JUPYTER_TOKEN` | random | Jupyter auth token |
| `VTCLONE_API_PORT` | `8000` | API port (must match exposed HTTP port) |
| `VTCLONE_MAX_WORKERS` | `1` | Parallel jobs (usually 1 on one GPU) |

On first boot the container prints the API key and saves it under `/workspace/.vtclone_api_key` on the network volume. Set `VTCLONE_API_KEY` in the template for a stable production key.

### 4. First launch

1. Deploy pod with template + network volume.
2. First boot downloads models (~7 GB) onto the volume (~10 min).
3. Later boots reuse models (~30 s).

---

## HTTP API (public)

| | |
|--|--|
| Local | `http://127.0.0.1:8000` |
| RunPod public | `https://<POD_ID>-8000.proxy.runpod.net` |
| OpenAPI | `…/docs` |

### Auth

All **`/v1/*`** routes require a key (timing-safe compare). Prefer headers:

```http
X-API-Key: <your-key>
```

or:

```http
Authorization: Bearer <your-key>
```

(`Authorization: ApiKey <key>` and `?api_key=` also work.)

`/health` is **unauthenticated** so load balancers and RunPod checks stay simple.

### Health

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

### Create a job

```bash
export API="https://<POD_ID>-8000.proxy.runpod.net"
export VTCLONE_API_KEY="..."   # from pod logs or /workspace/.vtclone_api_key

curl -X POST "$API/v1/jobs" \
  -H "X-API-Key: $VTCLONE_API_KEY" \
  -F "video=@./my_video.mp4" \
  -F "src_lang=de" \
  -F "mt_model=Helsinki-NLP/opus-mt-de-en"
```

Response (`202`):

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "queued",
  "src_lang": "de",
  ...
}
```

### Poll status

```bash
curl "$API/v1/jobs/a1b2c3d4e5f6" -H "X-API-Key: $VTCLONE_API_KEY"
# or: -H "Authorization: Bearer $VTCLONE_API_KEY"
```

Statuses: `queued` → `running` → `completed` | `failed`.

### Download result

```bash
curl -L -o translated.mp4 \
  "$API/v1/jobs/a1b2c3d4e5f6/result" \
  -H "X-API-Key: $VTCLONE_API_KEY"
```

Segments JSON:

```bash
curl -L -o segments.json \
  "$API/v1/jobs/a1b2c3d4e5f6/segments" \
  -H "X-API-Key: $VTCLONE_API_KEY"
```

### List / delete

```bash
curl "$API/v1/jobs" -H "X-API-Key: $VTCLONE_API_KEY"
curl -X DELETE "$API/v1/jobs/a1b2c3d4e5f6" -H "X-API-Key: $VTCLONE_API_KEY"
```

### Local API process

```bash
export PYTHONPATH=.
export VTCLONE_JOBS_DIR=./jobs
export VTCLONE_API_KEY=dev-secret-change-me
export VTCLONE_REQUIRE_API_KEY=1
export VTCLONE_REPO_PATH=/path/to/Step-Audio-EditX
export VTCLONE_MODEL_PATH=/path/to/models/Step-Audio-EditX
export VTCLONE_TOKENIZER_PATH=/path/to/models/Step-Audio-Tokenizer

pip install -r requirements.txt
python api_server.py
# binds 0.0.0.0:8000 — ready for reverse proxy / RunPod HTTP port
```

---

## CLI usage

```bash
python /workspace/translate.py \
  --video /workspace/videos/input.mp4 \
  --src_lang de \
  --mt_model Helsinki-NLP/opus-mt-de-en \
  --repo_path /workspace/Step-Audio-EditX \
  --model_path /workspace/models/Step-Audio-EditX \
  --tokenizer_path /workspace/models/Step-Audio-Tokenizer
```

Output defaults to:

```
/workspace/videos/input/translated.mp4
/workspace/videos/input/segments.json
/workspace/videos/input/run_manifest.json
```

(project directory is next to the video: `<parent>/<stem>/`).

### Important flags

| Flag | Default | Description |
|------|---------|-------------|
| `--stage` | `all` | `stt` \| `tts` \| `overlay` \| `all` |
| `--skip_existing` | off | Reuse existing TTS wavs |
| `--duck_gain` | `0.15` | Original audio volume (0–1) |
| `--fix-overlaps` / `--no-fix-overlaps` | on | Repair overlapping segments |
| `--extend-video` / `--no-extend-video` | on | Freeze-frame if audio is longer |
| `--match-duration` / `--no-match-duration` | on | `atempo` stretch TTS into slots |
| `--vad-filter` / `--no-vad-filter` | on | Faster-Whisper VAD |
| `--device` | `cuda` | STT/MT device (`cuda` falls back to CPU) |
| `--tts_fail_on_error` | off | Abort if any TTS chunk fails |

---

## Translation models

| Language | Model |
|----------|-------|
| German → English | `Helsinki-NLP/opus-mt-de-en` |
| Russian → English | `Helsinki-NLP/opus-mt-ru-en` |
| French → English | `Helsinki-NLP/opus-mt-fr-en` |
| Spanish → English | `Helsinki-NLP/opus-mt-es-en` |
| Chinese → English | `Helsinki-NLP/opus-mt-zh-en` |

---

## Requirements

- GPU: **16 GB+ VRAM** (RTX 4090, A100, etc.) for Step-Audio-EditX
- Network volume: **50 GB** recommended
- Step-Audio-EditX also needs its own deps (notably **vLLM**) installed on the pod — `start.sh` attempts `pip install -e` on the SAEX checkout

---

## Layout

```
video_translate_clone/
  vtclone/
    pipeline.py      # orchestrator
    stt.py           # Faster-Whisper
    mt.py            # transformers MT
    tts_saex.py      # load-once SAEX (+ subprocess fallback)
    overlay.py       # FFmpeg mix + atempo
    api.py           # FastAPI jobs
    utils.py
  video_translate_clone_perf.py   # CLI entry
  api_server.py
  start.sh
  Dockerfile
  requirements.txt
```

## Updating code on RunPod

Push to GitHub; the container pulls on each start (`REPO_URL`).

```bash
git add -A && git commit -m "update" && git push
```

Then restart the pod (or re-pull inside `/workspace/video-translate-clone`).

---

## Changelog (v1.1)

- Job-based **HTTP API** (`/v1/jobs`)
- Modular package (`vtclone/`)
- **Load-once** SAEX TTS (in-process) with subprocess fallback
- **atempo** duration matching for better A/V alignment
- Boolean flags fixed (`--no-fix-overlaps`, `--no-extend-video`, …)
- Project dirs default **next to the input video**
- Whisper **tries CUDA**, falls back to CPU
- Safer Jupyter (token required)
- Model download markers + huggingface-cli preferred
- `run_manifest.json` per run
- Correct `calebtt` image/repo names in docs
