"""Shared utilities for the video translation pipeline."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vtclone")


def setup_logging(level: int = logging.INFO) -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def eprint(*a: Any) -> None:
    """Backward-compatible stderr print used by older call sites."""
    print(*a, file=sys.stderr, flush=True)
    logger.info(" ".join(str(x) for x in a))


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    capture_output: bool = False,
) -> Optional[str]:
    logger.info(">>> %s", " ".join(cmd))
    try:
        if capture_output:
            result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
            return result.stdout
        subprocess.run(cmd, cwd=cwd, check=True)
        return None
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %s", e.returncode)
        if getattr(e, "stderr", None):
            logger.error("stderr: %s", e.stderr)
        raise


def ffmpeg_exists() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", s.strip())
    s = re.sub(r"\s+", " ", s)
    return s[:120].strip() or "segment"


def ms(t: float) -> int:
    return int(round(t * 1000.0))


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_audio_duration(wav_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def video_has_audio(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def atempo_filter_chain(speed: float) -> str:
    """
    Build an ffmpeg atempo chain. Each atempo must be in [0.5, 2.0].
    speed > 1.0 shortens audio (faster speech); speed < 1.0 lengthens.
    """
    if speed <= 0:
        raise ValueError(f"Invalid speed: {speed}")
    # Clamp extreme ratios; caller should already keep within a sane range
    speed = max(0.25, min(4.0, speed))
    factors: List[float] = []
    remaining = speed
    while remaining > 2.0 + 1e-9:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def fit_audio_duration(
    src_wav: Path,
    dst_wav: Path,
    target_dur: float,
    min_speed: float = 0.80,
    max_speed: float = 1.25,
) -> float:
    """
    Time-stretch (atempo) src_wav toward target_dur.
    Returns the final duration of dst_wav.
    If ratio is outside [min_speed, max_speed], clamp to that range
    (residual mismatch is handled by pad/trim/delay later).
    """
    src_dur = get_audio_duration(src_wav)
    if src_dur <= 0.01 or target_dur <= 0.01:
        run_cmd(["ffmpeg", "-y", "-i", str(src_wav), "-c", "copy", str(dst_wav)])
        return get_audio_duration(dst_wav)

    # speed = src/target: >1 means speed up (shorten)
    raw_speed = src_dur / target_dur
    speed = max(min_speed, min(max_speed, raw_speed))

    if abs(speed - 1.0) < 0.02:
        # Close enough — pad or trim only
        if src_dur > target_dur:
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-af",
                    f"atrim=0:{target_dur:.6f},asetpts=PTS-STARTPTS",
                    "-c:a",
                    "pcm_s16le",
                    str(dst_wav),
                ]
            )
        elif src_dur < target_dur - 0.02:
            pad = target_dur - src_dur
            run_cmd(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_wav),
                    "-af",
                    f"apad=pad_dur={pad:.6f}",
                    "-c:a",
                    "pcm_s16le",
                    str(dst_wav),
                ]
            )
        else:
            run_cmd(["ffmpeg", "-y", "-i", str(src_wav), "-c", "copy", str(dst_wav)])
        return get_audio_duration(dst_wav)

    chain = atempo_filter_chain(speed)
    # After stretch, pad/trim to exact target when we clamped speed
    stretched_expected = src_dur / speed
    if abs(stretched_expected - target_dur) < 0.05:
        af = chain
    elif stretched_expected > target_dur:
        af = f"{chain},atrim=0:{target_dur:.6f},asetpts=PTS-STARTPTS"
    else:
        pad = target_dur - stretched_expected
        af = f"{chain},apad=pad_dur={pad:.6f}"

    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_wav),
            "-af",
            af,
            "-c:a",
            "pcm_s16le",
            str(dst_wav),
        ]
    )
    return get_audio_duration(dst_wav)


def fix_overlapping_segments(
    segments: List[Dict[str, Any]],
    min_gap: float = 0.1,
    max_delay: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Fix overlapping segments by adjusting timing.
    Prefer delaying the next segment; trim previous only if delay would be excessive.
    """
    if len(segments) <= 1:
        return segments

    fixed: List[Dict[str, Any]] = []
    overlaps_fixed = 0

    for i, seg in enumerate(segments):
        current = dict(seg)

        if i > 0:
            prev = fixed[-1]
            prev_end = float(prev["end"])
            current_start = float(current["start"])
            gap = current_start - prev_end

            if gap < min_gap:
                overlap = min_gap - gap
                overlaps_fixed += 1
                prev_duration = float(prev["end"]) - float(prev["start"])

                if overlap < max_delay:
                    delay = prev_end + min_gap - current_start
                    new_start = prev_end + min_gap
                    new_end = float(current["end"]) + delay
                    logger.info(
                        "Segment %s → %s: gap %.2fs < %.2fs, delaying curr to %.2fs (+%.2fs)",
                        i - 1,
                        i,
                        gap,
                        min_gap,
                        new_start,
                        delay,
                    )
                    current["start"] = new_start
                    current["end"] = new_end
                elif prev_duration > overlap + 0.5:
                    new_prev_end = current_start - min_gap
                    logger.warning(
                        "Segment %s → %s: massive overlap %.2fs, trimming prev to %.2fs",
                        i - 1,
                        i,
                        overlap,
                        new_prev_end,
                    )
                    prev["end"] = new_prev_end
                else:
                    delay = prev_end + min_gap - current_start
                    new_start = prev_end + min_gap
                    new_end = float(current["end"]) + delay
                    logger.warning(
                        "Segment %s → %s: large overlap %.2fs, delaying (+%.2fs) — may drift",
                        i - 1,
                        i,
                        overlap,
                        delay,
                    )
                    current["start"] = new_start
                    current["end"] = new_end

        fixed.append(current)

    if overlaps_fixed:
        logger.info("Fixed %s overlapping segment(s)", overlaps_fixed)
    else:
        logger.info("No overlapping segments detected")
    return fixed


def load_segments(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Segments JSON not found: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}") from e

    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected segments JSON shape in {json_path}")


def save_segments(json_path: Path, segments: List[Dict[str, Any]]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"segments": segments}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def newest_wav_in_dir(d: Path, after: float) -> Optional[Path]:
    candidates = []
    for p in d.glob("*.wav"):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime >= after:
            candidates.append((st.st_mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def write_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
