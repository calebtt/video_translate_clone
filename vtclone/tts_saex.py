"""Step-Audio-EditX TTS: load-once in-process, with subprocess fallback."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import newest_wav_in_dir, run_cmd

logger = logging.getLogger("vtclone")

# Prefer Triton attention backend when available (SAEX/vLLM)
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


def extract_ref_audio_16k_mono(video: Path, out_wav: Path, ref_seconds: float) -> None:
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
            "-t",
            f"{ref_seconds:.3f}",
            str(out_wav),
        ]
    )
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract reference audio: {out_wav}")


def resolve_model_paths(
    model_path: Path,
    tokenizer_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Accept either:
      - model_path = .../models/Step-Audio-EditX  (preferred)
      - model_path = .../models  (parent containing both subdirs)
    """
    model_path = model_path.resolve()
    if tokenizer_path is not None:
        return model_path, tokenizer_path.resolve()

    # Parent-dir layout: models/{Step-Audio-EditX, Step-Audio-Tokenizer}
    sibling_tok = model_path / "Step-Audio-Tokenizer"
    sibling_edit = model_path / "Step-Audio-EditX"
    if sibling_edit.is_dir() and sibling_tok.is_dir():
        return sibling_edit, sibling_tok

    # Sibling of EditX dir
    if model_path.name.lower().startswith("step-audio-edit"):
        cand = model_path.parent / "Step-Audio-Tokenizer"
        if cand.is_dir():
            return model_path, cand

    # Last resort: same path (SAEX may auto-detect)
    return model_path, model_path.parent / "Step-Audio-Tokenizer"


class SaexTtsEngine:
    """Loads Step-Audio-EditX once and clones many segments."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Optional[Path] = None,
        repo_path: Optional[Path] = None,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 3072,
        dtype: str = "bfloat16",
    ):
        edit_path, tok_path = resolve_model_paths(model_path, tokenizer_path)
        self.edit_path = edit_path
        self.tok_path = tok_path
        self.repo_path = repo_path.resolve() if repo_path else None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._model = None
        self._load_mode: Optional[str] = None  # "inprocess" | "subprocess"

    def _ensure_repo_on_path(self) -> None:
        if self.repo_path and self.repo_path.is_dir():
            p = str(self.repo_path)
            if p not in sys.path:
                sys.path.insert(0, p)

    def load(self) -> str:
        """Load model. Returns mode used: 'inprocess' or 'subprocess'."""
        if self._load_mode == "inprocess" and self._model is not None:
            return "inprocess"

        self._ensure_repo_on_path()
        try:
            from tokenizer import StepAudioTokenizer  # type: ignore
            from tts import StepAudioTTS  # type: ignore

            logger.info("Loading SAEX in-process from %s", self.edit_path)
            if not self.tok_path.is_dir():
                raise FileNotFoundError(f"Tokenizer not found: {self.tok_path}")
            if not self.edit_path.is_dir():
                raise FileNotFoundError(f"EditX model not found: {self.edit_path}")

            tokenizer = StepAudioTokenizer(str(self.tok_path), model_source="local")
            self._model = StepAudioTTS(
                str(self.edit_path),
                tokenizer,
                model_source="local",
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                max_num_seqs=1,
            )
            self._load_mode = "inprocess"
            logger.info("SAEX loaded in-process (models stay warm across segments)")
            return "inprocess"
        except Exception as e:
            logger.warning(
                "In-process SAEX load failed (%s: %s); will use subprocess fallback per segment",
                type(e).__name__,
                e,
            )
            self._model = None
            self._load_mode = "subprocess"
            return "subprocess"

    def clone_to_file(
        self,
        *,
        prompt_audio: Path,
        prompt_text: str,
        generated_text: str,
        out_wav: Path,
        python_exe: str,
        n_edit_iter: int = 1,
    ) -> bool:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        if self._load_mode is None:
            self.load()

        if self._load_mode == "inprocess" and self._model is not None:
            return self._clone_inprocess(
                prompt_audio=prompt_audio,
                prompt_text=prompt_text,
                generated_text=generated_text,
                out_wav=out_wav,
            )

        return self._clone_subprocess(
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            generated_text=generated_text,
            out_wav=out_wav,
            python_exe=python_exe,
            n_edit_iter=n_edit_iter,
        )

    def _clone_inprocess(
        self,
        *,
        prompt_audio: Path,
        prompt_text: str,
        generated_text: str,
        out_wav: Path,
    ) -> bool:
        try:
            import torchaudio

            t0 = time.time()
            audio, sr = self._model.clone(
                prompt_wav_path=str(prompt_audio),
                prompt_text=prompt_text,
                target_text=generated_text,
            )
            torchaudio.save(str(out_wav), audio.cpu(), sr)
            logger.info("In-process clone -> %s (%.1fs)", out_wav.name, time.time() - t0)
            return out_wav.exists() and out_wav.stat().st_size > 0
        except Exception as e:
            logger.error("In-process clone failed: %s", e)
            return False

    def _clone_subprocess(
        self,
        *,
        prompt_audio: Path,
        prompt_text: str,
        generated_text: str,
        out_wav: Path,
        python_exe: str,
        n_edit_iter: int = 1,
    ) -> bool:
        if not self.repo_path:
            logger.error("Subprocess TTS requires repo_path to Step-Audio-EditX")
            return False
        tts_script = self.repo_path / "tts_infer.py"
        if not tts_script.exists():
            logger.error("TTS script not found: %s", tts_script)
            return False

        audio_dir = out_wav.parent
        files_before = set(audio_dir.glob("*.wav"))
        t0 = time.time()
        cmd = [
            python_exe,
            str(tts_script),
            "--model-path",
            str(self.edit_path),
            "--tokenizer-path",
            str(self.tok_path),
            "--model-source",
            "local",
            "--prompt-text",
            prompt_text,
            "--prompt-audio",
            str(prompt_audio),
            "--generated-text",
            generated_text,
            "--edit-type",
            "clone",
            "--output-dir",
            str(audio_dir),
        ]
        # Older scripts accepted --n-edit-iter; ignore if unsupported
        if n_edit_iter and n_edit_iter != 1:
            cmd.extend(["--n-edit-iter", str(n_edit_iter)])

        try:
            run_cmd(cmd)
        except Exception as e:
            # Retry without n-edit-iter if unknown arg
            if n_edit_iter and n_edit_iter != 1:
                cmd = [c for c in cmd if c not in ("--n-edit-iter", str(n_edit_iter))]
                try:
                    run_cmd(cmd)
                except Exception as e2:
                    logger.error("Subprocess TTS failed: %s", e2)
                    return False
            else:
                logger.error("Subprocess TTS failed: %s", e)
                return False

        files_after = set(audio_dir.glob("*.wav"))
        new_files = files_after - files_before
        newest = max(new_files, key=lambda p: p.stat().st_mtime) if new_files else newest_wav_in_dir(audio_dir, t0)
        if not newest:
            logger.error("SAEX did not produce a wav for %s", out_wav.name)
            return False
        if newest != out_wav:
            shutil.move(str(newest), str(out_wav))
        return out_wav.exists() and out_wav.stat().st_size > 0


def tts_generate_segments_saex(
    segments: List[Dict[str, Any]],
    *,
    python_exe: str,
    repo_path: Path,
    model_path: Path,
    tokenizer_path: Optional[Path],
    ref_wav: Path,
    audio_dir: Path,
    prompt_text: Optional[str],
    n_edit_iter: int,
    skip_existing: bool,
    engine: Optional[SaexTtsEngine] = None,
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Generate TTS for all segments. Returns a summary dict with counts/failures.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)

    if engine is None:
        engine = SaexTtsEngine(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            repo_path=repo_path,
        )
        engine.load()

    if prompt_text is None:
        for s in segments:
            text = (s.get("text") or "").strip()
            if text:
                prompt_text = text.split(".")[0].strip() or text[:100].strip()
                logger.info("Auto-selected prompt text: %r", prompt_text)
                break
        if prompt_text is None:
            prompt_text = "hey"
            logger.warning("No text found in segments, using default prompt")
    else:
        logger.info("Using provided prompt text: %r", prompt_text)

    todo = [s for s in segments if (s.get("translated_text") or "").strip()]
    total = len(todo)
    processed = 0
    skipped = 0
    failed: List[int] = []

    logger.info("TTS output dir: %s | segments: %s", audio_dir.resolve(), total)

    for s in segments:
        i = int(s["id"])
        out_wav = audio_dir / f"chunk_{i}.wav"

        if skip_existing and out_wav.exists() and out_wav.stat().st_size > 0:
            logger.info("[SKIP] existing TTS chunk %s: %s", i, out_wav.name)
            s["tts_wav"] = str(out_wav)
            skipped += 1
            continue

        text = (s.get("translated_text") or "").strip()
        if not text:
            logger.info("[SKIP] empty text for segment %s", i)
            continue

        processed += 1
        logger.info("=== TTS %s/%s (segment %s) ===", processed, total, i)
        logger.info("Text: %s%s", text[:100], "..." if len(text) > 100 else "")

        ok = engine.clone_to_file(
            prompt_audio=ref_wav,
            prompt_text=prompt_text,
            generated_text=text,
            out_wav=out_wav,
            python_exe=python_exe,
            n_edit_iter=n_edit_iter,
        )
        if ok:
            s["tts_wav"] = str(out_wav)
            logger.info("[OK] TTS chunk %s: %s", i, out_wav.name)
        else:
            failed.append(i)
            logger.error("[FAIL] TTS chunk %s", i)
            if fail_on_error:
                raise RuntimeError(f"TTS failed for segment {i}")

    summary = {
        "total_with_text": total,
        "generated": processed - len(failed),
        "skipped_existing": skipped,
        "failed_ids": failed,
    }
    if failed:
        logger.warning("TTS failed for %s segment(s): %s", len(failed), failed)
    return summary
