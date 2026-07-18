"""
FastAPI job API for offline video translation + voice cloning.

Designed for public exposure (e.g. RunPod HTTP/WS proxy) with API-key auth.

Auth (all /v1/* routes + WebSocket):
  X-API-Key: <key>
  Authorization: Bearer <key>
  Authorization: ApiKey <key>
  WebSocket: ?api_key= or first JSON message {"type":"auth","api_key":"..."}

HTTP:
  GET  /health                  public (no secrets)
  GET  /v1/models/defaults
  POST /v1/jobs                 multipart: video file + form fields
  POST /v1/jobs/binary          raw video body
  GET  /v1/jobs                 list jobs
  GET  /v1/jobs/{job_id}        status + metadata
  GET  /v1/jobs/{job_id}/result download translated video
  GET  /v1/jobs/{job_id}/segments download segments JSON
  DELETE /v1/jobs/{job_id}      remove job artifacts

WebSocket:
  WS   /v1/ws                   preferred client path — live progress + optional upload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .utils import setup_logging

setup_logging()
logger = logging.getLogger("vtclone.api")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _default_jobs_dir() -> Path:
    if os.environ.get("VTCLONE_JOBS_DIR"):
        return Path(os.environ["VTCLONE_JOBS_DIR"]).expanduser()
    # Prefer RunPod workspace when present; otherwise local ./jobs
    if Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        return Path("/workspace/jobs")
    return Path.cwd() / "jobs"


def _default_key_file() -> Path:
    if os.environ.get("VTCLONE_API_KEY_FILE"):
        return Path(os.environ["VTCLONE_API_KEY_FILE"]).expanduser()
    if Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        return Path("/workspace/.vtclone_api_key")
    return Path.cwd() / ".vtclone_api_key"


def _resolve_api_key() -> str:
    """
    Resolve API key for public access:
      1. VTCLONE_API_KEY env
      2. VTCLONE_API_KEY_FILE / default key file on volume
      3. Generate + persist if VTCLONE_REQUIRE_API_KEY (default true)
    """
    env_key = os.environ.get("VTCLONE_API_KEY", "").strip()
    if env_key:
        return env_key

    key_file = _default_key_file()
    try:
        if key_file.is_file():
            stored = key_file.read_text(encoding="utf-8").strip()
            if stored:
                logger.info("Loaded API key from %s", key_file)
                return stored
    except OSError as e:
        logger.warning("Could not read API key file %s: %s", key_file, e)

    require = _env_bool("VTCLONE_REQUIRE_API_KEY", True)
    if not require:
        logger.warning(
            "VTCLONE_REQUIRE_API_KEY=0 and no key set — /v1 routes are UNPROTECTED"
        )
        return ""

    generated = secrets.token_urlsafe(32)
    try:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(generated + "\n", encoding="utf-8")
        try:
            key_file.chmod(0o600)
        except OSError:
            pass
        logger.info("Generated API key and saved to %s", key_file)
    except OSError as e:
        logger.warning("Could not persist API key to %s: %s (using in-memory key)", key_file, e)
    return generated


JOBS_ROOT = _default_jobs_dir().resolve()
API_KEY = _resolve_api_key()
REQUIRE_API_KEY = _env_bool("VTCLONE_REQUIRE_API_KEY", True)
# If a key exists, always enforce it (even when REQUIRE is false but key was set)
AUTH_ENABLED = bool(API_KEY) or REQUIRE_API_KEY

DEFAULT_REPO = Path(
    os.environ.get("VTCLONE_REPO_PATH", "/workspace/Step-Audio-EditX")
)
DEFAULT_MODEL = Path(
    os.environ.get("VTCLONE_MODEL_PATH", "/workspace/models/Step-Audio-EditX")
)
DEFAULT_TOKENIZER = Path(
    os.environ.get("VTCLONE_TOKENIZER_PATH", "/workspace/models/Step-Audio-Tokenizer")
)
DEFAULT_MT = os.environ.get("VTCLONE_MT_MODEL", "Helsinki-NLP/opus-mt-de-en")
DEFAULT_WHISPER = os.environ.get("VTCLONE_WHISPER_MODEL", "large-v3")
MAX_WORKERS = int(os.environ.get("VTCLONE_MAX_WORKERS", "1"))
MAX_UPLOAD_MB = int(os.environ.get("VTCLONE_MAX_UPLOAD_MB", "300"))
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("VTCLONE_CORS_ORIGINS", "*").split(",")
    if o.strip()
]

try:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    # Defer hard failure until a job is created
    pass

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_lock = threading.Lock()
_jobs: Dict[str, "JobRecord"] = {}

# WebSocket fan-out: job_id -> set of asyncio.Queue (one per subscriber)
_ws_sub_lock = threading.Lock()
_job_queues: DefaultDict[str, Set[asyncio.Queue]] = defaultdict(set)
_global_queues: Set[asyncio.Queue] = set()


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    src_lang: str
    mt_model: str
    stage: str = "all"
    progress_stage: Optional[str] = None
    progress_detail: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    input_video: Optional[str] = None
    output_video: Optional[str] = None
    output_json: Optional[str] = None
    project_dir: Optional[str] = None
    elapsed_sec: Optional[float] = None
    tts_summary: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_event(rec: JobRecord, event_type: str = "job_update") -> Dict[str, Any]:
    return {
        "type": event_type,
        "job_id": rec.id,
        "job": rec.model_dump(),
        "ts": _now(),
    }


def _subscribe_queue(job_id: Optional[str], q: asyncio.Queue) -> None:
    with _ws_sub_lock:
        if job_id:
            _job_queues[job_id].add(q)
        else:
            _global_queues.add(q)


def _unsubscribe_queue(job_id: Optional[str], q: asyncio.Queue) -> None:
    with _ws_sub_lock:
        if job_id:
            _job_queues.get(job_id, set()).discard(q)
            if job_id in _job_queues and not _job_queues[job_id]:
                del _job_queues[job_id]
        else:
            _global_queues.discard(q)


def _broadcast_job(job_id: str, event: Dict[str, Any]) -> None:
    """Thread-safe: push event to all WS queues watching this job (or all jobs)."""
    with _ws_sub_lock:
        targets = list(_job_queues.get(job_id, set())) + list(_global_queues)
    for q in targets:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def _check_api_key_value(provided: Optional[str]) -> bool:
    if not AUTH_ENABLED or not API_KEY:
        if REQUIRE_API_KEY and not API_KEY:
            return False
        return True
    if not provided:
        return False
    try:
        return secrets.compare_digest(provided.strip(), API_KEY)
    except Exception:
        return False


def _extract_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> Optional[str]:
    """Pull key from X-API-Key or Authorization: Bearer|ApiKey."""
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("bearer", "apikey"):
            return parts[1].strip()
        # bare token
        if len(parts) == 1:
            return parts[0].strip()
    # Optional query param for simple clients (prefer headers in production)
    q = request.query_params.get("api_key")
    if q and q.strip():
        return q.strip()
    return None


def _require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    """
    Enforce API key when public auth is enabled.
    /health stays open (registered without this dependency).
    """
    if not AUTH_ENABLED or not API_KEY:
        if REQUIRE_API_KEY and not API_KEY:
            raise HTTPException(
                status_code=503,
                detail="API key not configured; set VTCLONE_API_KEY",
            )
        return

    provided = _extract_api_key(request, x_api_key, authorization)
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Send X-API-Key or Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _save_job(rec: JobRecord) -> None:
    d = _job_dir(rec.id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")


def _load_jobs_from_disk() -> None:
    if not JOBS_ROOT.is_dir():
        return
    for p in JOBS_ROOT.iterdir():
        meta = p / "job.json"
        if meta.is_file():
            try:
                rec = JobRecord.model_validate_json(meta.read_text(encoding="utf-8"))
                # Mark interrupted runs as failed on restart
                if rec.status in (JobStatus.queued, JobStatus.running):
                    rec.status = JobStatus.failed
                    rec.error = "Interrupted by server restart"
                    rec.updated_at = _now()
                    _save_job(rec)
                _jobs[rec.id] = rec
            except Exception:
                continue


_load_jobs_from_disk()

# Common video extensions we accept on upload
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".ts"}


def _suffix_from_name(name: str) -> str:
    suf = Path(name).suffix.lower()
    return suf if suf in _VIDEO_SUFFIXES else ".mp4"


def _suffix_from_content_type(content_type: Optional[str]) -> str:
    if not content_type:
        return ".mp4"
    ct = content_type.split(";")[0].strip().lower()
    return {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/x-msvideo": ".avi",
        "application/octet-stream": ".mp4",
    }.get(ct, ".mp4")


async def _stream_upload_to_path(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """Write an UploadFile to disk; return size. Raises HTTPException on limits/empty."""
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded video is empty")
    return size


async def _stream_body_to_path(request: Request, dest: Path, max_bytes: int) -> int:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            400,
            "Empty body. Send the video as multipart field 'video' or as raw binary body.",
        )
    return size


def _enqueue_job(
    *,
    dest: Path,
    size: int,
    orig_name: str,
    src_lang: str,
    mt_model: str,
    stage: str,
    whisper_model: str,
    device: str,
    compute_type: str,
    duck_gain: float,
    ref_seconds: float,
    prompt_text: Optional[str],
    skip_existing: bool,
    fix_overlaps: bool,
    extend_video: bool,
    match_duration: bool,
    vad_filter: bool,
    repo_path: Optional[str],
    model_path: Optional[str],
    tokenizer_path: Optional[str],
) -> JobRecord:
    if stage not in ("stt", "tts", "overlay", "all"):
        raise HTTPException(400, f"Invalid stage: {stage}")
    if not src_lang or not src_lang.strip():
        raise HTTPException(400, "src_lang is required")

    job_id = uuid.uuid4().hex[:12]
    # Move into job-scoped directory
    jdir = _job_dir(job_id)
    input_dir = jdir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    final_path = input_dir / f"input{_suffix_from_name(orig_name)}"
    if dest.resolve() != final_path.resolve():
        shutil.move(str(dest), str(final_path))
        dest = final_path

    now = _now()
    rec = JobRecord(
        id=job_id,
        status=JobStatus.queued,
        created_at=now,
        updated_at=now,
        src_lang=src_lang.strip(),
        mt_model=mt_model,
        stage=stage,
        input_video=str(dest),
        params={
            "whisper_model": whisper_model,
            "device": device,
            "compute_type": compute_type,
            "duck_gain": duck_gain,
            "ref_seconds": ref_seconds,
            "prompt_text": prompt_text,
            "skip_existing": skip_existing,
            "fix_overlaps": fix_overlaps,
            "extend_video": extend_video,
            "match_duration": match_duration,
            "vad_filter": vad_filter,
            "repo_path": repo_path or str(DEFAULT_REPO),
            "model_path": model_path or str(DEFAULT_MODEL),
            "tokenizer_path": tokenizer_path or str(DEFAULT_TOKENIZER),
            "original_filename": orig_name,
            "upload_bytes": size,
        },
    )
    with _lock:
        _jobs[job_id] = rec
    _save_job(rec)
    _broadcast_job(job_id, _job_event(rec, "job_queued"))
    _executor.submit(_run_job, job_id)
    return rec


def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Translate Clone API",
        description=(
            "Offline video translation with voice cloning (job-based).\n\n"
            "**Preferred client path — WebSocket** `WS /v1/ws`:\n"
            "live progress, optional in-socket video upload, push on complete.\n\n"
            "**HTTP upload** (also supported):\n"
            "- `POST /v1/jobs` multipart field **`video`** (alias `file`)\n"
            "- `POST /v1/jobs/binary` raw video body\n\n"
            "**Auth:** `X-API-Key` / `Authorization: Bearer` on HTTP; "
            "WebSocket `?api_key=` or `{\"type\":\"auth\",\"api_key\":\"...\"}`.\n\n"
            "RunPod: expose port **8000** → "
            "`https://<POD_ID>-8000.proxy.runpod.net` and "
            "`wss://<POD_ID>-8000.proxy.runpod.net/v1/ws`"
        ),
        version="1.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None) or {},
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Public liveness probe (no auth). Does not expose the API key."""
        gpu = False
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            pass
        return {
            "status": "ok",
            "gpu": gpu,
            "max_workers": MAX_WORKERS,
            "max_upload_mb": MAX_UPLOAD_MB,
            "api_key_required": bool(AUTH_ENABLED and API_KEY),
            "public_auth": "X-API-Key or Authorization: Bearer",
        }

    @app.get("/v1/models/defaults")
    def defaults(_: None = Depends(_require_api_key)) -> Dict[str, Any]:
        return {
            "repo_path": str(DEFAULT_REPO),
            "model_path": str(DEFAULT_MODEL),
            "tokenizer_path": str(DEFAULT_TOKENIZER),
            "mt_model": DEFAULT_MT,
            "whisper_model": DEFAULT_WHISPER,
            "src_lang_examples": ["de", "ru", "fr", "es", "zh"],
            "mt_model_examples": {
                "de": "Helsinki-NLP/opus-mt-de-en",
                "ru": "Helsinki-NLP/opus-mt-ru-en",
                "fr": "Helsinki-NLP/opus-mt-fr-en",
                "es": "Helsinki-NLP/opus-mt-es-en",
                "zh": "Helsinki-NLP/opus-mt-zh-en",
            },
        }

    @app.get("/v1/jobs", response_model=List[JobRecord])
    def list_jobs(_: None = Depends(_require_api_key)) -> List[JobRecord]:
        with _lock:
            return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)

    @app.post(
        "/v1/jobs",
        response_model=JobRecord,
        status_code=202,
        summary="Create job (multipart video upload)",
        response_description="Job accepted and queued",
    )
    async def create_job(
        video: Optional[UploadFile] = File(
            default=None,
            description="Video file to translate (multipart field name: video)",
            media_type="video/*",
        ),
        file: Optional[UploadFile] = File(
            default=None,
            description="Alias for video (multipart field name: file)",
            media_type="video/*",
        ),
        src_lang: str = Form(..., description="Source language code, e.g. de", examples=["de"]),
        mt_model: str = Form(DEFAULT_MT, description="HuggingFace translation model id"),
        stage: str = Form("all"),
        whisper_model: str = Form(DEFAULT_WHISPER),
        device: str = Form("cuda"),
        compute_type: str = Form("float16"),
        duck_gain: float = Form(0.15),
        ref_seconds: float = Form(10.0),
        prompt_text: Optional[str] = Form(None),
        skip_existing: bool = Form(False),
        fix_overlaps: bool = Form(True),
        extend_video: bool = Form(True),
        match_duration: bool = Form(True),
        vad_filter: bool = Form(True),
        repo_path: Optional[str] = Form(None),
        model_path: Optional[str] = Form(None),
        tokenizer_path: Optional[str] = Form(None),
        _: None = Depends(_require_api_key),
    ) -> JobRecord:
        """
        **Upload the video in this request** as `multipart/form-data`.

        Required parts:
        - **`video`** (file) — or alias **`file`**
        - **`src_lang`** (text) — e.g. `de`, `ru`

        Example:
        ```bash
        curl -X POST \"$API/v1/jobs\" \\\\
          -H \"X-API-Key: $KEY\" \\\\
          -F \"video=@./my_clip.mp4;type=video/mp4\" \\\\
          -F \"src_lang=de\" \\\\
          -F \"mt_model=Helsinki-NLP/opus-mt-de-en\"
        ```
        """
        upload = video or file
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing video upload. Send multipart/form-data with a file field "
                    "named 'video' (or 'file'), e.g. "
                    "curl -F 'video=@clip.mp4' -F 'src_lang=de' ..."
                ),
            )

        # Reject clearly non-file empty placeholders
        orig_name = upload.filename or "input.mp4"
        if not orig_name or orig_name == "string":
            orig_name = "input.mp4"

        staging = JOBS_ROOT / "_staging" / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / f"upload{_suffix_from_name(orig_name)}"
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        try:
            size = await _stream_upload_to_path(upload, dest, max_bytes)
        except HTTPException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise HTTPException(400, f"Failed to read uploaded video: {e}") from e
        finally:
            try:
                await upload.close()
            except Exception:
                pass

        try:
            return _enqueue_job(
                dest=dest,
                size=size,
                orig_name=orig_name,
                src_lang=src_lang,
                mt_model=mt_model,
                stage=stage,
                whisper_model=whisper_model,
                device=device,
                compute_type=compute_type,
                duck_gain=duck_gain,
                ref_seconds=ref_seconds,
                prompt_text=prompt_text,
                skip_existing=skip_existing,
                fix_overlaps=fix_overlaps,
                extend_video=extend_video,
                match_duration=match_duration,
                vad_filter=vad_filter,
                repo_path=repo_path,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )
        except HTTPException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @app.post(
        "/v1/jobs/binary",
        response_model=JobRecord,
        status_code=202,
        summary="Create job (raw video body upload)",
        response_description="Job accepted and queued",
    )
    async def create_job_binary(
        request: Request,
        src_lang: str = Query(..., description="Source language code, e.g. de"),
        mt_model: str = Query(DEFAULT_MT),
        stage: str = Query("all"),
        whisper_model: str = Query(DEFAULT_WHISPER),
        device: str = Query("cuda"),
        compute_type: str = Query("float16"),
        duck_gain: float = Query(0.15),
        ref_seconds: float = Query(10.0),
        prompt_text: Optional[str] = Query(None),
        filename: Optional[str] = Query(
            None, description="Original filename for extension (default input.mp4)"
        ),
        _: None = Depends(_require_api_key),
    ) -> JobRecord:
        """
        Upload the video as the **raw HTTP body** (not multipart).

        ```bash
        curl -X POST \"$API/v1/jobs/binary?src_lang=de&mt_model=Helsinki-NLP/opus-mt-de-en\" \\\\
          -H \"X-API-Key: $KEY\" \\\\
          -H \"Content-Type: video/mp4\" \\\\
          --data-binary @./my_clip.mp4
        ```
        """
        content_type = request.headers.get("content-type", "")
        if content_type.lower().startswith("multipart/"):
            raise HTTPException(
                415,
                "This endpoint expects a raw video body. "
                "For multipart uploads use POST /v1/jobs with field 'video'.",
            )

        orig_name = filename or f"input{_suffix_from_content_type(content_type)}"
        staging = JOBS_ROOT / "_staging" / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / f"upload{_suffix_from_name(orig_name)}"
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        try:
            size = await _stream_body_to_path(request, dest, max_bytes)
        except HTTPException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        try:
            return _enqueue_job(
                dest=dest,
                size=size,
                orig_name=orig_name,
                src_lang=src_lang,
                mt_model=mt_model,
                stage=stage,
                whisper_model=whisper_model,
                device=device,
                compute_type=compute_type,
                duck_gain=duck_gain,
                ref_seconds=ref_seconds,
                prompt_text=prompt_text,
                skip_existing=False,
                fix_overlaps=True,
                extend_video=True,
                match_duration=True,
                vad_filter=True,
                repo_path=None,
                model_path=None,
                tokenizer_path=None,
            )
        except HTTPException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str, _: None = Depends(_require_api_key)) -> JobRecord:
        with _lock:
            rec = _jobs.get(job_id)
        if not rec:
            raise HTTPException(404, "Job not found")
        return rec

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str, _: None = Depends(_require_api_key)) -> FileResponse:
        with _lock:
            rec = _jobs.get(job_id)
        if not rec:
            raise HTTPException(404, "Job not found")
        if rec.status != JobStatus.completed:
            raise HTTPException(409, f"Job status is {rec.status}, not completed")
        if not rec.output_video or not Path(rec.output_video).is_file():
            raise HTTPException(404, "Result video missing")
        return FileResponse(
            rec.output_video,
            media_type="video/mp4",
            filename=f"{job_id}_translated.mp4",
        )

    @app.get("/v1/jobs/{job_id}/segments")
    def get_segments(job_id: str, _: None = Depends(_require_api_key)) -> FileResponse:
        with _lock:
            rec = _jobs.get(job_id)
        if not rec:
            raise HTTPException(404, "Job not found")
        if not rec.output_json or not Path(rec.output_json).is_file():
            raise HTTPException(404, "Segments JSON not available yet")
        return FileResponse(
            rec.output_json,
            media_type="application/json",
            filename=f"{job_id}_segments.json",
        )

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str, _: None = Depends(_require_api_key)) -> Dict[str, str]:
        with _lock:
            rec = _jobs.pop(job_id, None)
        if not rec:
            raise HTTPException(404, "Job not found")
        shutil.rmtree(_job_dir(job_id), ignore_errors=True)
        return {"status": "deleted", "id": job_id}

    @app.websocket("/v1/ws")
    async def jobs_websocket(
        websocket: WebSocket,
        api_key: Optional[str] = Query(default=None),
    ) -> None:
        """
        Preferred interactive client.

        Auth: `?api_key=` or first message `{"type":"auth","api_key":"..."}`.

        Client → server (JSON text unless noted):
          - `{"type":"auth","api_key":"..."}`
          - `{"type":"ping"}`
          - `{"type":"subscribe","job_id":"..."}`  (omit job_id = all jobs)
          - `{"type":"unsubscribe","job_id":"..."}`
          - `{"type":"submit","src_lang":"de","mt_model":"...","filename":"x.mp4", ...}`
            then binary frames of the video, then
            `{"type":"upload_complete"}`
          - `{"type":"get_job","job_id":"..."}`
          - `{"type":"list_jobs"}`

        Server → client:
          - `hello`, `auth_ok`, `pong`, `error`
          - `job_queued` / `job_update` / `progress` / `completed` / `failed`
          - `ready_for_upload` after submit
          - `upload_progress` while receiving bytes
        """
        await websocket.accept()
        out_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        subscribed: Set[Optional[str]] = set()
        authenticated = _check_api_key_value(api_key)

        # Optional upload session state
        upload_fh = None
        upload_path: Optional[Path] = None
        upload_staging: Optional[Path] = None
        upload_size = 0
        upload_meta: Optional[Dict[str, Any]] = None
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024

        async def send_event(event: Dict[str, Any]) -> None:
            if websocket.client_state != WebSocketState.CONNECTED:
                return
            await websocket.send_json(event)

        async def pump_out() -> None:
            while True:
                event = await out_q.get()
                try:
                    await send_event(event)
                except Exception:
                    break

        pump_task = asyncio.create_task(pump_out())

        def sub(job_id: Optional[str]) -> None:
            if job_id in subscribed:
                return
            _subscribe_queue(job_id, out_q)
            subscribed.add(job_id)

        def unsub(job_id: Optional[str]) -> None:
            if job_id not in subscribed:
                return
            _unsubscribe_queue(job_id, out_q)
            subscribed.discard(job_id)

        def cleanup_upload() -> None:
            nonlocal upload_fh, upload_path, upload_staging, upload_size, upload_meta
            if upload_fh is not None:
                try:
                    upload_fh.close()
                except Exception:
                    pass
                upload_fh = None
            if upload_staging is not None:
                shutil.rmtree(upload_staging, ignore_errors=True)
            upload_path = None
            upload_staging = None
            upload_size = 0
            upload_meta = None

        try:
            await send_event(
                {
                    "type": "hello",
                    "version": "1.2.0",
                    "auth_required": bool(AUTH_ENABLED and API_KEY),
                    "authenticated": authenticated,
                    "protocol": {
                        "submit": "JSON submit → binary video chunks → upload_complete",
                        "subscribe": "JSON subscribe for live job progress",
                        "result": "HTTP GET /v1/jobs/{id}/result after completed",
                    },
                }
            )

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                # ---- binary: video upload chunks ----
                if "bytes" in message and message["bytes"] is not None:
                    if not authenticated:
                        await send_event({"type": "error", "detail": "Not authenticated"})
                        continue
                    if upload_fh is None or upload_path is None:
                        await send_event(
                            {
                                "type": "error",
                                "detail": "Send {\"type\":\"submit\",...} before binary chunks",
                            }
                        )
                        continue
                    chunk = message["bytes"]
                    upload_size += len(chunk)
                    if upload_size > max_bytes:
                        cleanup_upload()
                        await send_event(
                            {
                                "type": "error",
                                "detail": f"Upload exceeds {MAX_UPLOAD_MB} MB",
                            }
                        )
                        continue
                    upload_fh.write(chunk)
                    # Throttle progress spam: every ~1 MiB boundary
                    if upload_size == len(chunk) or upload_size % (1024 * 1024) < len(chunk):
                        await send_event(
                            {
                                "type": "upload_progress",
                                "bytes": upload_size,
                            }
                        )
                    continue

                # ---- text JSON ----
                text = message.get("text")
                if text is None:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    await send_event({"type": "error", "detail": "Invalid JSON"})
                    continue
                if not isinstance(data, dict) or "type" not in data:
                    await send_event(
                        {"type": "error", "detail": "Message must be an object with type"}
                    )
                    continue

                msg_type = data["type"]

                if msg_type == "auth":
                    if _check_api_key_value(data.get("api_key") or data.get("key")):
                        authenticated = True
                        await send_event({"type": "auth_ok"})
                    else:
                        authenticated = False
                        await send_event({"type": "error", "detail": "Invalid API key"})
                    continue

                if msg_type == "ping":
                    await send_event({"type": "pong", "ts": _now()})
                    continue

                if not authenticated:
                    await send_event(
                        {
                            "type": "error",
                            "detail": "Authenticate first: ?api_key= or {\"type\":\"auth\",\"api_key\":\"...\"}",
                        }
                    )
                    continue

                if msg_type == "subscribe":
                    job_id = data.get("job_id")
                    # None / missing / "" → all jobs
                    if not job_id:
                        job_id = None
                    sub(job_id)
                    await send_event({"type": "subscribed", "job_id": job_id})
                    if job_id:
                        with _lock:
                            rec = _jobs.get(job_id)
                        if rec:
                            await send_event(_job_event(rec, "job_snapshot"))
                        else:
                            await send_event(
                                {"type": "error", "detail": f"Job not found: {job_id}"}
                            )
                    continue

                if msg_type == "unsubscribe":
                    job_id = data.get("job_id")
                    if not job_id:
                        job_id = None
                    unsub(job_id)
                    await send_event({"type": "unsubscribed", "job_id": job_id})
                    continue

                if msg_type == "list_jobs":
                    with _lock:
                        jobs = sorted(
                            (_jobs.values()),
                            key=lambda j: j.created_at,
                            reverse=True,
                        )
                    await send_event(
                        {
                            "type": "jobs",
                            "jobs": [j.model_dump() for j in jobs],
                        }
                    )
                    continue

                if msg_type == "get_job":
                    job_id = data.get("job_id")
                    if not job_id:
                        await send_event({"type": "error", "detail": "job_id required"})
                        continue
                    with _lock:
                        rec = _jobs.get(job_id)
                    if not rec:
                        await send_event({"type": "error", "detail": "Job not found"})
                    else:
                        await send_event(_job_event(rec, "job_snapshot"))
                    continue

                if msg_type == "submit":
                    if upload_fh is not None:
                        await send_event(
                            {
                                "type": "error",
                                "detail": "Upload already in progress; finish or cancel first",
                            }
                        )
                        continue
                    src_lang = (data.get("src_lang") or "").strip()
                    if not src_lang:
                        await send_event({"type": "error", "detail": "src_lang required"})
                        continue
                    filename = data.get("filename") or "input.mp4"
                    upload_staging = JOBS_ROOT / "_staging" / uuid.uuid4().hex
                    upload_staging.mkdir(parents=True, exist_ok=True)
                    upload_path = upload_staging / f"upload{_suffix_from_name(filename)}"
                    upload_fh = upload_path.open("wb")
                    upload_size = 0
                    upload_meta = {
                        "src_lang": src_lang,
                        "mt_model": data.get("mt_model") or DEFAULT_MT,
                        "stage": data.get("stage") or "all",
                        "whisper_model": data.get("whisper_model") or DEFAULT_WHISPER,
                        "device": data.get("device") or "cuda",
                        "compute_type": data.get("compute_type") or "float16",
                        "duck_gain": float(data.get("duck_gain", 0.15)),
                        "ref_seconds": float(data.get("ref_seconds", 10.0)),
                        "prompt_text": data.get("prompt_text"),
                        "skip_existing": bool(data.get("skip_existing", False)),
                        "fix_overlaps": bool(data.get("fix_overlaps", True)),
                        "extend_video": bool(data.get("extend_video", True)),
                        "match_duration": bool(data.get("match_duration", True)),
                        "vad_filter": bool(data.get("vad_filter", True)),
                        "repo_path": data.get("repo_path"),
                        "model_path": data.get("model_path"),
                        "tokenizer_path": data.get("tokenizer_path"),
                        "filename": filename,
                    }
                    await send_event(
                        {
                            "type": "ready_for_upload",
                            "filename": filename,
                            "max_upload_mb": MAX_UPLOAD_MB,
                        }
                    )
                    continue

                if msg_type == "upload_cancel":
                    cleanup_upload()
                    await send_event({"type": "upload_cancelled"})
                    continue

                if msg_type == "upload_complete":
                    if upload_fh is None or upload_path is None or upload_meta is None:
                        await send_event(
                            {"type": "error", "detail": "No upload in progress"}
                        )
                        continue
                    try:
                        upload_fh.close()
                    except Exception:
                        pass
                    upload_fh = None
                    if upload_size == 0 or not upload_path.is_file():
                        cleanup_upload()
                        await send_event({"type": "error", "detail": "Empty upload"})
                        continue

                    meta = upload_meta
                    path = upload_path
                    size = upload_size
                    # Keep staging until enqueue moves file
                    upload_path = None
                    upload_staging = None
                    upload_meta = None
                    upload_size = 0

                    try:
                        rec = _enqueue_job(
                            dest=path,
                            size=size,
                            orig_name=meta["filename"],
                            src_lang=meta["src_lang"],
                            mt_model=meta["mt_model"],
                            stage=meta["stage"],
                            whisper_model=meta["whisper_model"],
                            device=meta["device"],
                            compute_type=meta["compute_type"],
                            duck_gain=meta["duck_gain"],
                            ref_seconds=meta["ref_seconds"],
                            prompt_text=meta["prompt_text"],
                            skip_existing=meta["skip_existing"],
                            fix_overlaps=meta["fix_overlaps"],
                            extend_video=meta["extend_video"],
                            match_duration=meta["match_duration"],
                            vad_filter=meta["vad_filter"],
                            repo_path=meta["repo_path"],
                            model_path=meta["model_path"],
                            tokenizer_path=meta["tokenizer_path"],
                        )
                    except HTTPException as e:
                        cleanup_upload()
                        await send_event({"type": "error", "detail": e.detail})
                        continue
                    except Exception as e:
                        cleanup_upload()
                        await send_event({"type": "error", "detail": str(e)})
                        continue

                    # Auto-subscribe to this job for live progress
                    sub(rec.id)
                    await send_event(_job_event(rec, "job_queued"))
                    await send_event(
                        {
                            "type": "subscribed",
                            "job_id": rec.id,
                            "note": "Auto-subscribed after submit",
                        }
                    )
                    continue

                await send_event(
                    {
                        "type": "error",
                        "detail": f"Unknown message type: {msg_type}",
                    }
                )

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.exception("WebSocket error: %s", e)
            try:
                await send_event({"type": "error", "detail": str(e)})
            except Exception:
                pass
        finally:
            cleanup_upload()
            for jid in list(subscribed):
                unsub(jid)
            pump_task.cancel()
            try:
                await pump_task
            except Exception:
                pass
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close()
            except Exception:
                pass

    return app


def _update(job_id: str, **kwargs: Any) -> None:
    with _lock:
        rec = _jobs.get(job_id)
        if not rec:
            return
        data = rec.model_dump()
        data.update(kwargs)
        data["updated_at"] = _now()
        rec = JobRecord(**data)
        _jobs[job_id] = rec
    _save_job(rec)

    event_type = "job_update"
    if rec.status == JobStatus.completed:
        event_type = "completed"
    elif rec.status == JobStatus.failed:
        event_type = "failed"
    elif kwargs.get("progress_stage"):
        event_type = "progress"

    event = _job_event(rec, event_type)
    if event_type == "progress":
        event["stage"] = rec.progress_stage
        event["detail"] = rec.progress_detail
    if event_type == "completed":
        event["result_path"] = f"/v1/jobs/{job_id}/result"
        event["segments_path"] = f"/v1/jobs/{job_id}/segments"
    _broadcast_job(job_id, event)


def _run_job(job_id: str) -> None:
    with _lock:
        rec = _jobs.get(job_id)
    if not rec:
        return

    _update(job_id, status=JobStatus.running, progress_stage="starting")
    params = rec.params
    project_dir = _job_dir(job_id) / "project"

    def on_progress(stage: str, detail: Dict[str, Any]) -> None:
        _update(job_id, progress_stage=stage, progress_detail=detail)

    cfg = PipelineConfig(
        video=Path(rec.input_video),
        src_lang=rec.src_lang,
        mt_model=rec.mt_model,
        repo_path=Path(params["repo_path"]),
        model_path=Path(params["model_path"]),
        tokenizer_path=Path(params["tokenizer_path"]) if params.get("tokenizer_path") else None,
        project_dir=project_dir,
        whisper_model=params.get("whisper_model", DEFAULT_WHISPER),
        device=params.get("device", "cuda"),
        compute_type=params.get("compute_type", "float16"),
        duck_gain=float(params.get("duck_gain", 0.15)),
        ref_seconds=float(params.get("ref_seconds", 10.0)),
        prompt_text=params.get("prompt_text"),
        skip_existing=bool(params.get("skip_existing", False)),
        fix_overlaps=bool(params.get("fix_overlaps", True)),
        extend_video=bool(params.get("extend_video", True)),
        match_duration=bool(params.get("match_duration", True)),
        vad_filter=bool(params.get("vad_filter", True)),
        stage=rec.stage,
    )

    t0 = time.time()
    result: PipelineResult = run_pipeline(cfg, progress=on_progress)
    elapsed = time.time() - t0

    if result.success:
        _update(
            job_id,
            status=JobStatus.completed,
            progress_stage="done",
            output_video=result.output_video,
            output_json=result.output_json,
            project_dir=result.project_dir,
            elapsed_sec=elapsed,
            tts_summary=result.tts_summary,
            error=None,
        )
    else:
        _update(
            job_id,
            status=JobStatus.failed,
            progress_stage="error",
            output_json=result.output_json,
            project_dir=result.project_dir,
            elapsed_sec=elapsed,
            tts_summary=result.tts_summary,
            error=result.error or "Unknown error",
        )


app = create_app()
