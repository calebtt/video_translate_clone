#!/usr/bin/env python3
"""
CLI for video translation + voice cloning pipeline.

Example:
  python video_translate_clone_perf.py \\
    --video /workspace/videos/input.mp4 \\
    --src_lang de \\
    --mt_model Helsinki-NLP/opus-mt-de-en \\
    --repo_path /workspace/Step-Audio-EditX \\
    --model_path /workspace/models/Step-Audio-EditX \\
    --tokenizer_path /workspace/models/Step-Audio-Tokenizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vtclone.pipeline import PipelineConfig, run_pipeline
from vtclone.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Video translation pipeline with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", required=True, help="Input video file")
    ap.add_argument("--src_lang", required=True, help="Source language code (e.g. de, ru)")
    ap.add_argument("--output_json", default=None, help="Output segments JSON")
    ap.add_argument("--output_video", default=None, help="Output video file")
    ap.add_argument(
        "--project_dir",
        default=None,
        help="Working project directory (default: <video_parent>/<video_stem>)",
    )

    # STT
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="vad_filter",
        help="Silero VAD in Faster-Whisper (default: on)",
    )

    # MT
    ap.add_argument("--mt_model", required=True, help="Translation model name")
    ap.add_argument("--mt_batch", type=int, default=8)
    ap.add_argument("--mt_use_pipeline", action="store_true")

    # TTS
    ap.add_argument("--repo_path", required=True, help="Path to Step-Audio-EditX repo")
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to Step-Audio-EditX weights (or parent models/ dir)",
    )
    ap.add_argument(
        "--tokenizer_path",
        default=None,
        help="Path to Step-Audio-Tokenizer (auto-detected if omitted)",
    )
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--prompt_text", default=None)
    ap.add_argument("--n_edit_iter", type=int, default=1)
    ap.add_argument("--ref_seconds", type=float, default=10.0)
    ap.add_argument(
        "--tts-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="tts_fail_on_error",
        help="Abort pipeline if any TTS segment fails",
    )
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    # Overlay
    ap.add_argument("--duck_gain", type=float, default=0.15)
    ap.add_argument(
        "--fix-overlaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="fix_overlaps",
    )
    ap.add_argument("--min_gap", type=float, default=0.1)
    ap.add_argument("--max_delay", type=float, default=5.0)
    ap.add_argument(
        "--extend-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="extend_video",
    )
    ap.add_argument(
        "--match-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="match_duration",
        help="Time-stretch TTS to fit segment slots via atempo (default: on)",
    )
    ap.add_argument("--min_speed", type=float, default=0.80)
    ap.add_argument("--max_speed", type=float, default=1.25)

    # Workflow
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument(
        "--stage",
        default="all",
        choices=["stt", "tts", "overlay", "all"],
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    cfg = PipelineConfig(
        video=Path(args.video),
        src_lang=args.src_lang,
        mt_model=args.mt_model,
        repo_path=Path(args.repo_path),
        model_path=Path(args.model_path),
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        output_json=Path(args.output_json) if args.output_json else None,
        output_video=Path(args.output_video) if args.output_video else None,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        audio_dir=Path(args.audio_dir) if args.audio_dir else None,
        project_dir=Path(args.project_dir) if args.project_dir else None,
        whisper_model=args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
        vad_filter=args.vad_filter,
        mt_batch=args.mt_batch,
        mt_use_pipeline=args.mt_use_pipeline,
        prompt_text=args.prompt_text,
        n_edit_iter=args.n_edit_iter,
        ref_seconds=args.ref_seconds,
        skip_existing=args.skip_existing,
        tts_fail_on_error=args.tts_fail_on_error,
        gpu_memory_utilization=args.gpu_memory_utilization,
        duck_gain=args.duck_gain,
        fix_overlaps=args.fix_overlaps,
        min_gap=args.min_gap,
        max_delay=args.max_delay,
        extend_video=args.extend_video,
        match_duration=args.match_duration,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        stage=args.stage,
    )

    result = run_pipeline(cfg)
    if not result.success:
        print(f"[ERROR] {result.error}", file=sys.stderr)
        return 1
    print(f"[SUCCESS] {result.output_video} ({result.elapsed_sec:.1f}s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user", file=sys.stderr)
        raise SystemExit(1)
