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
- **WebSocket API** (`WS /v1/ws`) — preferred client path: live progress + optional in-socket upload
- **HTTP job API** — multipart/binary upload, poll, download result (still fully supported)
- **API-key auth** for RunPod / internet exposure (`X-API-Key` / Bearer / WS `api_key`)
- RunPod-friendly Docker image (models on a network volume, code pull on start)

---

## RunPod requirements (what to rent)

This is a **GPU offline** pipeline (not real-time). Rent something that matches the table below or jobs will OOM / fill the disk.

### GPU

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| **VRAM** | **≥ 16 GB** (24 GB comfortable) | Step-Audio-EditX (vLLM) + Whisper + MT share the GPU |
| **Examples that work** | RTX 4090 24 GB, A5000 24 GB, A40 48 GB, A100 40/80 GB | Common RunPod community / secure cloud SKUs |
| **Usually too small** | 8–12 GB consumer cards | TTS alone tends to OOM |
| **CUDA** | Image is CUDA **12.4** + PyTorch 2.4 | Prefer recent NVIDIA drivers on the host |
| **Concurrency** | **1 job at a time** default (`VTCLONE_MAX_WORKERS=1`) | One heavy pipeline per GPU; don’t oversubscribe |

CPU-only can run STT/MT for tiny tests; **voice clone TTS effectively needs a GPU**.

### Storage (network volume — most important)

Put models + jobs on a **network volume** mounted at `/workspace`. Container disk alone is not enough for models + uploads.

| Use | Approx. size | Notes |
|-----|--------------|--------|
| Step-Audio-EditX weights | ~5 GB | On volume under `models/` |
| Step-Audio-Tokenizer | ~2 GB | Same |
| HuggingFace / Whisper / MT caches | ~3–8 GB | Grows with languages/models used |
| Code + SAEX git checkout | ~1–2 GB | |
| **Per job (peak)** | **~0.5–1.5 GB** for a max-size upload | Input ≤ 300 MB + extracted audio + TTS chunks + output |
| **Retained jobs** | input + `translated.mp4` until deleted | Clients should `DELETE` after download |

| Volume size | Fit for |
|-------------|---------|
| **50 GB** | **Minimum recommended** — models + a handful of jobs; delete results regularly |
| **80–100 GB** | Comfortable for light public use / several kept outputs |
| **&lt; 30 GB** | Tight; models alone can crowd you out |

**Upload cap:** API rejects files larger than **`VTCLONE_MAX_UPLOAD_MB` (default 300)** with HTTP **413** (or a WebSocket `error`). Tune via env if you have a larger volume and want bigger clips.

### Container disk vs network volume

| Mount | Suggested size | Holds |
|-------|----------------|--------|
| **Network volume** → `/workspace` | **50 GB+** | Models, jobs, caches, API key file |
| **Container disk** | **10–20 GB** | OS/image scratch only — not job storage |

Region of the volume must match the pod region.

### Network / ports

| Port | Purpose |
|------|---------|
| **8000** | Public API (HTTP + WebSocket upgrade) — **expose this** |
| **8888** | Optional JupyterLab |
| **22** | Optional SSH |

Public URLs after the pod is up:

```text
https://<POD_ID>-8000.proxy.runpod.net          # HTTP
wss://<POD_ID>-8000.proxy.runpod.net/v1/ws      # WebSocket
```

### RAM / CPU (host)

| Resource | Suggestion |
|----------|------------|
| **System RAM** | **≥ 32 GB** preferred (16 GB absolute floor) | Model load + FFmpeg + Python peaks |
| **vCPUs** | **4+** | Faster-Whisper CPU fallback and FFmpeg benefit |

### Cost / ops tips

- Prefer a **persistent network volume** so you don’t re-download ~7–15 GB of models every pod.
- Keep **`VTCLONE_MAX_WORKERS=1`** unless you have multi-GPU and know the VRAM math.
- After clients download results, call **`DELETE /v1/jobs/{id}`** so the volume does not fill with old MP4s.
- First boot: model download **~10+ minutes**; later boots with warm volume: **~30–60 s** to API ready.

---

## RunPod setup (one-time)

### 1. Network volume

[RunPod Storage](https://runpod.io/console/user/storage) → **New Network Volume**:

| Setting | Value |
|---------|-------|
| Name | `video-translate` |
| Size | **50 GB+** (see requirements above) |
| Region | Same as pods |

### 2. Build & push image

```bash
git clone https://github.com/calebtt/video_translate_clone.git
cd video_translate_clone

docker build --platform linux/amd64 -t calebtt/video-translate-clone:v1.2 .
docker login
docker push calebtt/video-translate-clone:v1.2
```

### 3. Template

| Setting | Value |
|---------|-------|
| Container Image | `calebtt/video-translate-clone:v1.2` |
| GPU | **16 GB+ VRAM** (24 GB recommended) |
| Container Disk | 10–20 GB |
| Volume Mount Path | `/workspace` |
| Network volume | 50 GB+ attached |
| Expose HTTP Ports | `8000,8888` |
| Expose TCP Ports | `22` |

**Public access:** Connect → **HTTP Service [Port 8000]** (HTTPS + `wss://` on the same host). All `/v1/*` routes require an API key. `/health` is public for probes.

Env vars:

| Env | Default | Purpose |
|-----|---------|---------|
| `VTCLONE_API_KEY` | auto-generated | Public API key (also written to volume) |
| `VTCLONE_API_KEY_FILE` | `/workspace/.vtclone_api_key` | Persist key across restarts |
| `VTCLONE_REQUIRE_API_KEY` | `1` | Enforce key on `/v1/*` |
| `VTCLONE_MAX_UPLOAD_MB` | **`300`** | Max video upload size (HTTP + WebSocket) |
| `VTCLONE_MAX_WORKERS` | `1` | Parallel jobs (keep 1 on a single GPU) |
| `VTCLONE_CORS_ORIGINS` | `*` | CORS allowlist (comma-separated) |
| `ENABLE_API` | `1` | Start FastAPI on boot (`0.0.0.0`) |
| `ENABLE_JUPYTER` | `1` | Start JupyterLab |
| `JUPYTER_TOKEN` | random | Jupyter auth token |
| `VTCLONE_API_PORT` | `8000` | Must match exposed HTTP port |

On first boot the container prints the API key and saves it under `/workspace/.vtclone_api_key`. Set `VTCLONE_API_KEY` in the template for a stable production key.

### 4. First launch

1. Deploy a pod that meets **GPU + volume** requirements above, with the template + network volume.
2. First boot downloads models onto the volume (~7 GB+ weights, often **~10–15 min**).
3. Confirm `GET /health` → `"status":"ok"` and `"max_upload_mb":300`.
4. Later boots reuse the volume (API ready in roughly **30–60 s**).

---

## API (public HTTP + WebSocket)

| | HTTP | WebSocket |
|--|------|-----------|
| Local | `http://127.0.0.1:8000` | `ws://127.0.0.1:8000/v1/ws` |
| RunPod public | `https://<POD_ID>-8000.proxy.runpod.net` | `wss://<POD_ID>-8000.proxy.runpod.net/v1/ws` |
| OpenAPI | `…/docs` | (protocol described below) |

**Prefer WebSockets** for interactive clients (live stage events, push on complete).  
HTTP remains available for simple `curl`, OpenAPI “try it”, and downloading the result file.

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `WS` | `/v1/ws` | yes | **Preferred** — live progress, optional video upload, push events |
| `GET` | `/health` | no | Liveness / GPU flag |
| `GET` | `/v1/models/defaults` | yes | Default model paths & MT examples |
| `POST` | `/v1/jobs` | yes | Create job — multipart video upload in the request |
| `POST` | `/v1/jobs/binary` | yes | Create job — raw video bytes in the body |
| `GET` | `/v1/jobs` | yes | List jobs |
| `GET` | `/v1/jobs/{id}` | yes | Job status / progress (poll fallback) |
| `GET` | `/v1/jobs/{id}/result` | yes | Download translated MP4 |
| `GET` | `/v1/jobs/{id}/segments` | yes | Download segments JSON |
| `DELETE` | `/v1/jobs/{id}` | yes | Delete job + artifacts |

Job lifecycle: `queued` → `running` → `completed` | `failed`.

### Auth

All **`/v1/*`** routes require a key (timing-safe compare). Prefer headers:

```http
X-API-Key: <your-key>
```

or:

```http
Authorization: Bearer <your-key>
```

Also accepted: `Authorization: ApiKey <key>` and query `?api_key=` (headers preferred).

WebSocket: `wss://…/v1/ws?api_key=<key>` or first message  
`{"type":"auth","api_key":"<key>"}`.

`/health` is **unauthenticated** so load balancers and RunPod checks stay simple.

### Health

```bash
export API="https://<POD_ID>-8000.proxy.runpod.net"
export WS_API="wss://<POD_ID>-8000.proxy.runpod.net/v1/ws"
export VTCLONE_API_KEY="..."   # from pod logs or /workspace/.vtclone_api_key

curl "$API/health"
```

### WebSocket (preferred)

One connection: authenticate → submit video → receive live progress → download result over HTTP.

**Client → server**

| Message | Meaning |
|---------|---------|
| `{"type":"auth","api_key":"…"}` | Auth if not using `?api_key=` |
| `{"type":"ping"}` | Keepalive → `pong` |
| `{"type":"subscribe","job_id":"…"}` | Live updates for one job (omit `job_id` = all jobs) |
| `{"type":"submit","src_lang":"de","mt_model":"…","filename":"clip.mp4"}` | Start upload; then send **binary** frames |
| *binary frames* | Video bytes (chunked) |
| `{"type":"upload_complete"}` | Finish upload → job queued (auto-subscribe) |
| `{"type":"get_job","job_id":"…"}` / `{"type":"list_jobs"}` | Snapshots |

**Server → client**

| `type` | Meaning |
|--------|---------|
| `hello` | Protocol banner |
| `ready_for_upload` | Send binary chunks now |
| `upload_progress` | Bytes received so far |
| `job_queued` / `progress` / `job_update` | Live pipeline stages |
| `completed` | Done — includes `result_path` / `segments_path` |
| `failed` / `error` | Failure detail |

**Python example** (`pip install websockets`):

```python
import asyncio, json
from pathlib import Path
import websockets

API_KEY = "..."
WS = "wss://<POD_ID>-8000.proxy.runpod.net/v1/ws"
VIDEO = Path("clip.mp4")

async def main():
    async with websockets.connect(f"{WS}?api_key={API_KEY}", max_size=None) as ws:
        print(await ws.recv())  # hello
        await ws.send(json.dumps({
            "type": "submit",
            "src_lang": "de",
            "mt_model": "Helsinki-NLP/opus-mt-de-en",
            "filename": VIDEO.name,
        }))
        assert json.loads(await ws.recv())["type"] == "ready_for_upload"
        data = VIDEO.read_bytes()
        step = 256 * 1024
        for i in range(0, len(data), step):
            await ws.send(data[i : i + step])
        await ws.send(json.dumps({"type": "upload_complete"}))
        while True:
            msg = json.loads(await ws.recv())
            print(msg["type"], msg.get("stage") or msg.get("job", {}).get("status"))
            if msg["type"] in ("completed", "failed"):
                break
        # Download: GET {API}{msg['result_path']} with X-API-Key

asyncio.run(main())
```

Hybrid pattern: **HTTP multipart upload**, then **WS subscribe** to that `job_id` for progress only.

### Create a job — HTTP upload video **with** the request

The API does **not** require a pre-staged path on the pod. Send the file in the same HTTP call.

#### Multipart

Form field for the file: **`video`** (alias: **`file`**). Required text field: **`src_lang`**.

```bash
curl -X POST "$API/v1/jobs" \
  -H "X-API-Key: $VTCLONE_API_KEY" \
  -F "video=@./my_video.mp4;type=video/mp4" \
  -F "src_lang=de" \
  -F "mt_model=Helsinki-NLP/opus-mt-de-en"
```

Optional form fields include `stage`, `whisper_model`, `device`, `duck_gain`, `ref_seconds`,
`prompt_text`, `skip_existing`, `fix_overlaps`, `extend_video`, `match_duration`, `vad_filter`,
`repo_path`, `model_path`, `tokenizer_path`.

Missing file → **400** with a clear error (not a silent failure).

#### Raw binary body

```bash
curl -X POST "$API/v1/jobs/binary?src_lang=de&mt_model=Helsinki-NLP/opus-mt-de-en" \
  -H "X-API-Key: $VTCLONE_API_KEY" \
  -H "Content-Type: video/mp4" \
  --data-binary @./my_video.mp4
```

Uploads are streamed to disk with a hard size cap: **`VTCLONE_MAX_UPLOAD_MB` (default 300)**.  
Over limit → **HTTP 413** (or WebSocket `error`); partial data is discarded.

Response (`202 Accepted`):

```json
{
  "id": "a1b2c3d4e5f6",
  "status": "queued",
  "src_lang": "de",
  "mt_model": "Helsinki-NLP/opus-mt-de-en",
  "input_video": "/workspace/jobs/.../input/input.mp4",
  ...
}
```

### Poll status

```bash
curl "$API/v1/jobs/a1b2c3d4e5f6" -H "X-API-Key: $VTCLONE_API_KEY"
# or: -H "Authorization: Bearer $VTCLONE_API_KEY"
```

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

## Requirements (summary)

See **[RunPod requirements](#runpod-requirements-what-to-rent)** for full detail. Short version:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (e.g. RTX 4090) |
| Network volume | 50 GB | 80–100 GB if keeping many outputs |
| Container disk | 10 GB | 20 GB |
| System RAM | 16 GB | 32 GB+ |
| Max upload | — | **300 MB** default (`VTCLONE_MAX_UPLOAD_MB`) |
| Parallel jobs | 1 | 1 per GPU |

Step-Audio-EditX also needs its own Python deps (notably **vLLM**); `start.sh` attempts `pip install -e` on the SAEX checkout.

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

## Changelog

### v1.2.0 — WebSocket + 300 MB upload + RunPod sizing docs

- **`WS /v1/ws`**: auth, subscribe, live `progress` / `completed` / `failed` push
- In-socket **submit + binary upload** (`submit` → chunks → `upload_complete`), auto-subscribe
- Thread-safe fan-out from pipeline worker threads to connected clients
- HTTP kept for multipart/binary upload and **result download**
- Default **`VTCLONE_MAX_UPLOAD_MB=300`**; `/health` reports `max_upload_mb`
- README **RunPod requirements** section (GPU VRAM, volume size, RAM, ports, peak disk per job)

### v1.1.x — Public job API + upload-in-request

**API / public hosting**

- Job API on **`0.0.0.0:8000`** with RunPod proxy headers (`--proxy-headers`)
- **Upload video in the request**: multipart `POST /v1/jobs` (`video` / `file`) and `POST /v1/jobs/binary`
- Streamed uploads, empty-file rejection, size limit, clear **400** if file missing
- **API-key auth** on all `/v1/*` (auto-generate + persist; `X-API-Key` / Bearer / ApiKey)
- Public `/health` only; CORS; OpenAPI at `/docs`
- Job list/status/result/segments/delete

**Pipeline**

- Modular package `vtclone/`
- **Load-once** Step-Audio-EditX TTS; **atempo** duration matching
- Boolean CLI flags fixed; project dirs next to video; `run_manifest.json`
- Whisper CUDA with CPU fallback; VAD retry; no-audio overlay handling

**Ops**

- Safer Jupyter tokens; HF download markers; `requirements.txt`; `calebtt` docs
- Unit tests: utils, auth, multipart/binary upload, WebSocket auth/submit
