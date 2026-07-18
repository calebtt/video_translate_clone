"""FFmpeg overlay / mix of cloned speech onto original video."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import (
    ffmpeg_exists,
    fit_audio_duration,
    fix_overlapping_segments,
    get_video_duration,
    ms,
    run_cmd,
    video_has_audio,
)

logger = logging.getLogger("vtclone")


def prepare_segment_wavs(
    segments: List[Dict[str, Any]],
    work_dir: Path,
    match_duration: bool = True,
    min_speed: float = 0.80,
    max_speed: float = 1.25,
) -> List[Tuple[Dict[str, Any], Path]]:
    """
    Optionally time-stretch each TTS wav toward its segment slot duration.
    Returns list of (segment, wav_path) ready for mixing.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared: List[Tuple[Dict[str, Any], Path]] = []
    fit_dir = work_dir / "tts_fitted"
    fit_dir.mkdir(parents=True, exist_ok=True)

    for s in segments:
        wav = s.get("tts_wav")
        if not wav:
            continue
        wavp = Path(wav)
        if not wavp.exists():
            logger.warning("TTS wav not found: %s", wavp)
            continue

        start = float(s["start"])
        end = float(s["end"])
        target_dur = max(0.05, end - start)

        if match_duration:
            fitted = fit_dir / f"fit_{s['id']}.wav"
            final_dur = fit_audio_duration(
                wavp,
                fitted,
                target_dur,
                min_speed=min_speed,
                max_speed=max_speed,
            )
            # If residual longer than slot after max speedup, expand segment end
            if final_dur > target_dur + 0.05:
                s = dict(s)
                s["end"] = start + final_dur
                logger.info(
                    "Segment %s residual overflow +%.2fs after atempo (end=%.2f)",
                    s["id"],
                    final_dur - target_dur,
                    s["end"],
                )
            prepared.append((s, fitted))
        else:
            prepared.append((s, wavp))

    # Re-fix overlaps after possible end expansions
    if match_duration and prepared:
        segs_only = [p[0] for p in prepared]
        fixed = fix_overlapping_segments(segs_only, min_gap=0.05, max_delay=10.0)
        # Map back by id
        by_id = {int(x["id"]): x for x in fixed}
        prepared = [(by_id.get(int(s["id"]), s), w) for s, w in prepared]

    return prepared


def build_mix_filter_script(
    prepared: List[Tuple[Dict[str, Any], Path]],
    duck_gain: float,
    script_path: Path,
    has_original_audio: bool,
) -> List[Path]:
    lines: List[str] = []
    mix_inputs: List[str] = []
    seg_wavs: List[Path] = []

    if has_original_audio:
        lines.append(
            f"[0:a]aformat=channel_layouts=stereo,aresample=44100,volume={duck_gain}[a0];"
        )
        mix_inputs.append("[a0]")
        input_index = 1
    else:
        input_index = 1
        logger.warning("Input video has no audio stream; mixing TTS only")

    for s, wavp in prepared:
        start = float(s["start"])
        end = float(s["end"])
        dur = max(0.01, end - start)
        delay = ms(start)
        tag = f"a{input_index}"
        lines.append(
            f"[{input_index}:a]"
            f"aformat=channel_layouts=stereo,aresample=44100,"
            f"atrim=0:{dur:.6f},asetpts=PTS-STARTPTS,"
            f"apad=pad_dur={dur:.6f},"
            f"adelay={delay}|{delay}"
            f"[{tag}];"
        )
        mix_inputs.append(f"[{tag}]")
        seg_wavs.append(wavp)
        input_index += 1

    if not mix_inputs:
        raise RuntimeError("No audio inputs to mix")

    if len(mix_inputs) == 1:
        lines.append(f"{mix_inputs[0]}anull[aout]")
    else:
        lines.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0[aout]"
        )

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Filter script written with %s TTS segments", len(seg_wavs))
    return seg_wavs


def overlay_to_video(
    video_in: Path,
    video_out: Path,
    segments: List[Dict[str, Any]],
    duck_gain: float,
    work_dir: Path,
    fix_overlaps: bool = True,
    min_gap: float = 0.1,
    max_delay: float = 5.0,
    extend_video: bool = True,
    match_duration: bool = True,
    min_speed: float = 0.80,
    max_speed: float = 1.25,
) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found on PATH")
    if not video_in.exists():
        raise FileNotFoundError(f"Input video not found: {video_in}")

    video_duration = get_video_duration(video_in)
    logger.info("Video duration: %.3fs", video_duration)

    segs = [dict(s) for s in segments]
    if fix_overlaps:
        segs = fix_overlapping_segments(segs, min_gap=min_gap, max_delay=max_delay)

    prepared = prepare_segment_wavs(
        segs,
        work_dir=work_dir,
        match_duration=match_duration,
        min_speed=min_speed,
        max_speed=max_speed,
    )
    if not prepared:
        logger.warning("No TTS segments to overlay; copying original video")
        video_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_in, video_out)
        return

    # After fitting, recompute max end
    max_audio_end = max(float(s["end"]) for s, _ in prepared)
    logger.info("Final audio ends at: %.3fs", max_audio_end)

    target_duration = video_duration
    if max_audio_end > video_duration:
        extension_needed = max_audio_end - video_duration
        logger.info("Audio extends %.2fs beyond video end", extension_needed)
        if extend_video:
            target_duration = max_audio_end
            logger.info("Will extend video with freeze frame for %.2fs", extension_needed)
        else:
            logger.warning("Last %.2fs of audio may be cut off (extend_video=False)", extension_needed)

    has_audio = video_has_audio(video_in)
    filter_script = work_dir / "mix_filter.txt"
    seg_wavs = build_mix_filter_script(prepared, duck_gain, filter_script, has_audio)

    video_to_use = video_in
    if extend_video and max_audio_end > video_duration:
        extension = max_audio_end - video_duration
        extended_video = work_dir / "video_extended.mp4"
        extend_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_in),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={extension}[v]",
            "-map",
            "[v]",
        ]
        if has_audio:
            extend_cmd += ["-map", "0:a", "-c:a", "copy"]
        extend_cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(extended_video),
        ]
        logger.info("Step 1/2: Extending video with freeze frame...")
        run_cmd(extend_cmd)
        video_to_use = extended_video
        logger.info("Step 2/2: Overlaying translated audio...")

    cmd: List[str] = ["ffmpeg", "-y", "-i", str(video_to_use)]
    for w in seg_wavs:
        cmd += ["-i", str(w)]

    cmd += [
        "-filter_complex_script",
        str(filter_script),
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if video_to_use == video_in and not extend_video:
        cmd.append("-shortest")

    video_out.parent.mkdir(parents=True, exist_ok=True)
    cmd.append(str(video_out))
    run_cmd(cmd)

    if not video_out.exists() or video_out.stat().st_size == 0:
        raise RuntimeError(f"Failed to create output video: {video_out}")

    output_duration = get_video_duration(video_out)
    duration_diff = abs(output_duration - target_duration)
    logger.info(
        "Video created: %s | in=%.3fs out=%.3fs target=%.3fs diff=%.3fs",
        video_out,
        video_duration,
        output_duration,
        target_duration,
        duration_diff,
    )
    if duration_diff > 0.75:
        logger.warning("Duration mismatch > 0.75s vs target — check timing")

    verify_cmd = ["ffmpeg", "-v", "error", "-i", str(video_out), "-f", "null", "-"]
    result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if result.returncode == 0 and not (result.stderr or "").strip():
        logger.info("Output verification passed")
    elif result.stderr:
        logger.warning("Verification warnings: %s", result.stderr[:500])
