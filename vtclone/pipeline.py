"""Orchestrates STT → MT → TTS → overlay stages."""

from __future__ import annotations

import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import mt as mt_mod
from . import overlay as overlay_mod
from . import stt as stt_mod
from . import tts_saex as tts_mod
from .utils import (
    fix_overlapping_segments,
    load_segments,
    save_segments,
    setup_logging,
    write_manifest,
)

logger = logging.getLogger("vtclone")

ProgressCb = Optional[Callable[[str, Dict[str, Any]], None]]


@dataclass
class PipelineConfig:
    video: Path
    src_lang: str
    mt_model: str
    repo_path: Path
    model_path: Path
    tokenizer_path: Optional[Path] = None

    output_json: Optional[Path] = None
    output_video: Optional[Path] = None
    work_dir: Optional[Path] = None
    audio_dir: Optional[Path] = None
    project_dir: Optional[Path] = None

    # STT
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    vad_filter: bool = True

    # MT
    mt_batch: int = 8
    mt_use_pipeline: bool = False

    # TTS
    prompt_text: Optional[str] = None
    n_edit_iter: int = 1
    ref_seconds: float = 10.0
    skip_existing: bool = False
    tts_fail_on_error: bool = False
    gpu_memory_utilization: float = 0.5

    # Overlay
    duck_gain: float = 0.15
    fix_overlaps: bool = True
    min_gap: float = 0.1
    max_delay: float = 5.0
    extend_video: bool = True
    match_duration: bool = True
    min_speed: float = 0.80
    max_speed: float = 1.25

    stage: str = "all"  # stt | tts | overlay | all

    def resolve_paths(self) -> "PipelineConfig":
        video = Path(self.video).resolve()
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video}")
        self.video = video

        # Project dir defaults next to the video
        if self.project_dir is None:
            self.project_dir = video.parent / video.stem
        else:
            self.project_dir = Path(self.project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.work_dir = Path(self.work_dir) if self.work_dir else self.project_dir / "work"
        self.audio_dir = Path(self.audio_dir) if self.audio_dir else self.project_dir / "audio_out"
        self.output_json = (
            Path(self.output_json) if self.output_json else self.project_dir / "segments.json"
        )
        self.output_video = (
            Path(self.output_video) if self.output_video else self.project_dir / "translated.mp4"
        )
        self.repo_path = Path(self.repo_path)
        self.model_path = Path(self.model_path)
        if self.tokenizer_path:
            self.tokenizer_path = Path(self.tokenizer_path)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class PipelineResult:
    success: bool
    output_video: Optional[str] = None
    output_json: Optional[str] = None
    project_dir: Optional[str] = None
    stages_run: List[str] = field(default_factory=list)
    tts_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0
    manifest_path: Optional[str] = None


def run_pipeline(
    cfg: PipelineConfig,
    progress: ProgressCb = None,
) -> PipelineResult:
    setup_logging()
    t_start = time.time()
    cfg = cfg.resolve_paths()
    stages_run: List[str] = []
    tts_summary: Optional[Dict[str, Any]] = None

    def emit(stage: str, **kwargs: Any) -> None:
        if progress:
            progress(stage, kwargs)
        logger.info("[%s] %s", stage, kwargs if kwargs else "…")

    try:
        python_exe = sys.executable

        # ---- STT + MT ----
        if cfg.stage in ("stt", "all"):
            emit("stt_start")
            stages_run.append("stt")
            audio_16k = cfg.work_dir / "audio_16k_mono.wav"
            logger.info("Extracting 16k mono audio -> %s", audio_16k)
            stt_mod.extract_audio_16k_mono(cfg.video, audio_16k)

            segments = stt_mod.stt_with_faster_whisper(
                audio_wav_16k=audio_16k,
                src_lang=cfg.src_lang,
                whisper_model=cfg.whisper_model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                vad_filter=cfg.vad_filter,
            )
            if not segments:
                logger.warning("No segments detected in video")

            emit("translate_start", segments=len(segments))
            stages_run.append("translate")
            mt_mod.translate_segments(
                segments,
                mt_model=cfg.mt_model,
                device=cfg.device,
                batch_size=cfg.mt_batch,
                use_pipeline=cfg.mt_use_pipeline,
            )

            if cfg.fix_overlaps:
                segments = fix_overlapping_segments(
                    segments, min_gap=cfg.min_gap, max_delay=cfg.max_delay
                )

            save_segments(cfg.output_json, segments)
            emit("stt_done", segments=len(segments), json=str(cfg.output_json))

        # ---- TTS ----
        if cfg.stage in ("tts", "all"):
            emit("tts_start")
            stages_run.append("tts")
            segments = load_segments(cfg.output_json)

            ref_wav = cfg.audio_dir / "ref_16k_mono.wav"
            if not ref_wav.exists() or ref_wav.stat().st_size == 0:
                logger.info("Extracting ref audio (%.1fs) -> %s", cfg.ref_seconds, ref_wav)
                tts_mod.extract_ref_audio_16k_mono(cfg.video, ref_wav, cfg.ref_seconds)
            else:
                logger.info("Using existing ref audio: %s", ref_wav)

            engine = tts_mod.SaexTtsEngine(
                model_path=cfg.model_path,
                tokenizer_path=cfg.tokenizer_path,
                repo_path=cfg.repo_path,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
            )
            mode = engine.load()
            emit("tts_engine", mode=mode)

            tts_summary = tts_mod.tts_generate_segments_saex(
                segments,
                python_exe=python_exe,
                repo_path=cfg.repo_path,
                model_path=cfg.model_path,
                tokenizer_path=cfg.tokenizer_path,
                ref_wav=ref_wav,
                audio_dir=cfg.audio_dir,
                prompt_text=cfg.prompt_text,
                n_edit_iter=cfg.n_edit_iter,
                skip_existing=cfg.skip_existing,
                engine=engine,
                fail_on_error=cfg.tts_fail_on_error,
            )
            save_segments(cfg.output_json, segments)
            emit("tts_done", **tts_summary)

        # ---- Overlay ----
        if cfg.stage in ("overlay", "all"):
            emit("overlay_start")
            stages_run.append("overlay")
            segments = load_segments(cfg.output_json)
            overlay_mod.overlay_to_video(
                video_in=cfg.video,
                video_out=cfg.output_video,
                segments=segments,
                duck_gain=cfg.duck_gain,
                work_dir=cfg.work_dir,
                fix_overlaps=cfg.fix_overlaps,
                min_gap=cfg.min_gap,
                max_delay=cfg.max_delay,
                extend_video=cfg.extend_video,
                match_duration=cfg.match_duration,
                min_speed=cfg.min_speed,
                max_speed=cfg.max_speed,
            )
            emit("overlay_done", output=str(cfg.output_video))

        elapsed = time.time() - t_start
        result = PipelineResult(
            success=True,
            output_video=str(cfg.output_video) if cfg.output_video.exists() else None,
            output_json=str(cfg.output_json) if cfg.output_json.exists() else None,
            project_dir=str(cfg.project_dir),
            stages_run=stages_run,
            tts_summary=tts_summary,
            elapsed_sec=elapsed,
        )
        manifest = {
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(cfg).items()
            },
            "result": asdict(result),
        }
        mpath = cfg.project_dir / "run_manifest.json"
        write_manifest(mpath, manifest)
        result.manifest_path = str(mpath)
        logger.info("SUCCESS in %.1fs -> %s", elapsed, result.output_video)
        return result

    except Exception as e:
        elapsed = time.time() - t_start
        logger.error("%s: %s", type(e).__name__, e)
        traceback.print_exc()
        result = PipelineResult(
            success=False,
            output_json=str(cfg.output_json) if cfg.output_json and cfg.output_json.exists() else None,
            project_dir=str(cfg.project_dir) if cfg.project_dir else None,
            stages_run=stages_run,
            tts_summary=tts_summary,
            error=f"{type(e).__name__}: {e}",
            elapsed_sec=elapsed,
        )
        if cfg.project_dir:
            try:
                write_manifest(
                    cfg.project_dir / "run_manifest.json",
                    {"result": asdict(result)},
                )
            except Exception:
                pass
        return result
