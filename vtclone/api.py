"""
FastAPI job API for offline video translation + voice cloning.

Endpoints:
  GET  /health
  GET  /v1/models/defaults
  POST /v1/jobs                 multipart: video file + form fields
  GET  /v1/jobs                 list jobs
  GET  /v1/jobs/{job_id}        status + metadata
  GET  /v1/jobs/{job_id}/result download translated video
  GET  /v1/jobs/{job_id}/segments download segments JSON
  DELETE /v1/jobs/{job_id}      remove job artifacts
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .utils import setup_logging

setup_logging()

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

def _default_jobs_dir() -> Path:
    if os.environ.get("VTCLONE_JOBS_DIR"):
        return Path(os.environ["VTCLONE_JOBS_DIR"]).expanduser()
    # Prefer RunPod workspace when present; otherwise local ./jobs
    if Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        return Path("/workspace/jobs")
    return Path.cwd() / "jobs"


API_KEY = os.environ.get("VTCLONE_API_KEY", "").strip()
JOBS_ROOT = _default_jobs_dir().resolve()
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
MAX_UPLOAD_MB = int(os.environ.get("VTCLONE_MAX_UPLOAD_MB", "2048"))

try:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    # Defer hard failure until a job is created
    pass

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_lock = threading.Lock()
_jobs: Dict[str, "JobRecord"] = {}


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


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


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


def create_app() -> FastAPI:
    app = FastAPI(
        title="Video Translate Clone API",
        description="Offline video translation with voice cloning (job-based).",
        version="1.1.0",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        gpu = False
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            pass
        return {
            "status": "ok",
            "gpu": gpu,
            "jobs_root": str(JOBS_ROOT),
            "max_workers": MAX_WORKERS,
            "api_key_required": bool(API_KEY),
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

    @app.post("/v1/jobs", response_model=JobRecord, status_code=202)
    async def create_job(
        video: UploadFile = File(..., description="Input video file"),
        src_lang: str = Form(..., description="Source language code, e.g. de"),
        mt_model: str = Form(DEFAULT_MT),
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
        if stage not in ("stt", "tts", "overlay", "all"):
            raise HTTPException(400, f"Invalid stage: {stage}")

        job_id = uuid.uuid4().hex[:12]
        jdir = _job_dir(job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        input_dir = jdir / "input"
        input_dir.mkdir(exist_ok=True)

        # Preserve extension when possible
        orig_name = video.filename or "input.mp4"
        suffix = Path(orig_name).suffix or ".mp4"
        safe_name = f"input{suffix}"
        dest = input_dir / safe_name

        size = 0
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024
        with dest.open("wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    shutil.rmtree(jdir, ignore_errors=True)
                    raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB")
                f.write(chunk)

        now = _now()
        rec = JobRecord(
            id=job_id,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
            src_lang=src_lang,
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

        _executor.submit(_run_job, job_id)
        return rec

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
