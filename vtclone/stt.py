"""Speech-to-text via Faster-Whisper."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .utils import run_cmd

logger = logging.getLogger("vtclone")


def extract_audio_16k_mono(video: Path, out_wav: Path) -> None:
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract audio: {out_wav}")


def stt_with_faster_whisper(
    audio_wav_16k: Path,
    src_lang: str,
    whisper_model: str,
    device: str = "cuda",
    compute_type: str = "float16",
    vad_filter: bool = True,
) -> List[Dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        ) from e

    if not audio_wav_16k.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_wav_16k}")

    t0 = time.time()
    logger.info("Loading Faster-Whisper model: %s", whisper_model)

    want_cuda = device.startswith("cuda")
    device_str = "cuda" if want_cuda else "cpu"
    compute_type_str = compute_type if want_cuda else "int8"

    model = None
    try:
        model = WhisperModel(
            whisper_model,
            device=device_str,
            compute_type=compute_type_str,
        )
        logger.info("Using device=%s compute_type=%s", device_str, compute_type_str)
    except Exception as e:
        if device_str == "cuda":
            logger.warning(
                "CUDA Whisper load failed (%s); falling back to CPU int8",
                e,
            )
            device_str = "cpu"
            compute_type_str = "int8"
            model = WhisperModel(
                whisper_model,
                device=device_str,
                compute_type=compute_type_str,
            )
            logger.info("Using device=%s compute_type=%s", device_str, compute_type_str)
        else:
            logger.error("Failed to load Faster-Whisper model: %s", e)
            raise

    logger.info("Model loaded in %.1fs", time.time() - t0)

    t1 = time.time()
    try:
        segments_iter, info = model.transcribe(
            str(audio_wav_16k),
            language=src_lang,
            beam_size=5,
            vad_filter=vad_filter,
            word_timestamps=False,
        )
        segments_list = list(segments_iter)
    except Exception as e:
        if vad_filter:
            logger.warning("Transcription with VAD failed (%s); retrying without VAD", e)
            segments_iter, info = model.transcribe(
                str(audio_wav_16k),
                language=src_lang,
                beam_size=5,
                vad_filter=False,
                word_timestamps=False,
            )
            segments_list = list(segments_iter)
        else:
            raise

    logger.info("Transcription done in %.1fs", time.time() - t1)
    logger.info(
        "Detected language: %s (p=%.2f), segments=%s",
        info.language,
        info.language_probability,
        len(segments_list),
    )

    norm: List[Dict[str, Any]] = []
    for i, s in enumerate(segments_list):
        norm.append(
            {
                "id": i,
                "start": float(s.start),
                "end": float(s.end),
                "text": (s.text or "").strip(),
            }
        )

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return norm
