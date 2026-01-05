import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import collections

# -----------------------------
# Utilities
# -----------------------------

def eprint(*a):
    print(*a, file=sys.stderr, flush=True)

def run_cmd(cmd: List[str], cwd: Optional[str] = None, capture_output: bool = False):
    eprint("\n>>>", " ".join(cmd))
    try:
        if capture_output:
            result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        eprint(f"[ERROR] Command failed with exit code {e.returncode}")
        if hasattr(e, 'stderr') and e.stderr:
            eprint(f"[ERROR] stderr: {e.stderr}")
        raise

def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
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
    """Get video duration in seconds using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def fix_overlapping_segments(segments: List[Dict[str, Any]], min_gap: float = 0.1, max_delay: float = 5.0) -> List[Dict[str, Any]]:
    """
    Fix overlapping segments by adjusting timing.
    
    Strategy:
    - If segment N+1 starts before segment N ends, adjust timing
    - PREFER delaying the next segment to preserve full audio
    - Only trim previous segment as last resort if delay would be too large
    - Maintain a minimum gap between segments for natural speech
    
    Args:
        segments: List of segments with 'start' and 'end' times
        min_gap: Minimum gap in seconds between segments (default 0.1s)
        max_delay: Maximum acceptable delay before falling back to trim (default 5.0s)
    
    Returns:
        Modified segments with no overlaps
    """
    if len(segments) <= 1:
        return segments
    
    fixed = []
    overlaps_fixed = 0
    
    for i, seg in enumerate(segments):
        current = seg.copy()
        
        if i > 0:
            prev = fixed[-1]
            prev_end = prev["end"]
            current_start = current["start"]
            
            # Check for overlap or insufficient gap
            gap = current_start - prev_end
            if gap < min_gap:
                overlap = min_gap - gap
                overlaps_fixed += 1
                
                prev_duration = prev["end"] - prev["start"]
                current_duration = current["end"] - current["start"]
                
                # Strategy 1 (PREFERRED): Delay current segment to preserve full audio
                # Only skip this if the delay would be massive
                if overlap < max_delay:
                    delay = prev_end + min_gap - current_start
                    new_start = prev_end + min_gap
                    new_end = current["end"] + delay
                    eprint(f"[FIX] Segment {i-1} → {i}: gap {gap:.2f}s < {min_gap:.2f}s, delaying curr from {current_start:.2f}s to {new_start:.2f}s (+{delay:.2f}s)")
                    current["start"] = new_start
                    current["end"] = new_end
                else:
                    # Strategy 2 (FALLBACK): Only trim if delay would be excessive
                    # This should rarely happen unless there's a data quality issue
                    if prev_duration > overlap + 0.5:
                        new_prev_end = current_start - min_gap
                        eprint(f"[FIX] Segment {i-1} → {i}: massive overlap {overlap:.2f}s, trimming prev from {prev['end']:.2f}s to {new_prev_end:.2f}s")
                        eprint(f"[WARNING] Previous segment trimmed - may sound cut off!")
                        prev["end"] = new_prev_end
                    else:
                        # Both strategies would fail - just delay anyway and warn
                        delay = prev_end + min_gap - current_start
                        new_start = prev_end + min_gap
                        new_end = current["end"] + delay
                        eprint(f"[FIX] Segment {i-1} → {i}: large overlap {overlap:.2f}s, delaying curr from {current_start:.2f}s to {new_start:.2f}s (+{delay:.2f}s)")
                        eprint(f"[WARNING] Large timing shift - audio may drift from video")
                        current["start"] = new_start
                        current["end"] = new_end
        
        fixed.append(current)
    
    if overlaps_fixed > 0:
        eprint(f"[INFO] Fixed {overlaps_fixed} overlapping segment(s)")
    else:
        eprint(f"[INFO] No overlapping segments detected")
    
    return fixed

def load_segments(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Segments JSON not found: {json_path}")
    
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {json_path}: {e}")
    
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected segments JSON shape in {json_path}")

def save_segments(json_path: Path, segments: List[Dict[str, Any]]):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        json_path.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        eprint(f"[ERROR] Failed to save segments JSON: {e}")
        raise

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

# -----------------------------
# Stage 1: STT + Translation
# -----------------------------

def extract_audio_16k_mono(video: Path, out_wav: Path):
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")
    
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract audio: {out_wav}")

def stt_with_faster_whisper(
    audio_wav_16k: Path,
    src_lang: str,
    whisper_model: str,
    device: str,
    compute_type: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")

    if not audio_wav_16k.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_wav_16k}")

    t0 = time.time()
    eprint(f"[INFO] Loading Faster-Whisper model: {whisper_model}")
    
    # Force CPU due to CuDNN version conflicts with ctranslate2
    # This only affects transcription (~30s video = ~10s on CPU)
    # TTS will still use GPU which is what takes the most time
    device_str = "cpu"
    compute_type_str = "int8"
    
    eprint(f"[INFO] Using device: {device_str} (transcription only - TTS will use GPU)")
    
    try:
        model = WhisperModel(
            whisper_model,
            device=device_str,
            compute_type=compute_type_str,
        )
    except Exception as e:
        eprint(f"[ERROR] Failed to load Faster-Whisper model: {e}")
        raise
    
    eprint(f"[INFO] Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    segments_iter, info = model.transcribe(
        str(audio_wav_16k),
        language=src_lang,
        beam_size=5,
        vad_filter=False,  # Disable VAD to avoid onnxruntime dependency issues
        word_timestamps=False,
    )
    
    # Convert generator to list
    segments_list = list(segments_iter)
    eprint(f"[INFO] Transcription done in {time.time() - t1:.1f}s")
    eprint(f"[INFO] Detected language: {info.language} (probability: {info.language_probability:.2f})")
    eprint(f"[INFO] Found {len(segments_list)} segments")

    # Normalize to our format
    norm = []
    for i, s in enumerate(segments_list):
        norm.append({
            "id": i,
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
        })

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return norm

def translate_segments(
    segments: List[Dict[str, Any]],
    mt_model: str,
    device: str,
    batch_size: int = 8,
    use_pipeline: bool = False,
) -> None:
    texts = [s["text"] for s in segments]
    if not any(texts):
        eprint("[WARN] No text to translate")
        for s in segments:
            s["translated_text"] = ""
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        raise ImportError("transformers/torch not installed. Install with: pip install torch transformers")

    eprint(f"[INFO] Loading translation model: {mt_model}")
    
    torch_device = 0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1
    
    if torch_device == -1 and device.startswith("cuda"):
        eprint("[WARN] CUDA requested but not available, using CPU")

    if use_pipeline:
        from transformers import pipeline
        translator = pipeline("translation", model=mt_model, device=torch_device)
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            chunk_result = translator(chunk)
            outs.extend(chunk_result)
            eprint(f"[INFO] Translated batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        for s, o in zip(segments, outs):
            s["translated_text"] = (o.get("translation_text") or "").strip()
        return

    tok = AutoTokenizer.from_pretrained(mt_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mt_model)

    if torch_device >= 0:
        mdl = mdl.to("cuda")

    mdl.eval()

    outs: List[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            if torch_device >= 0:
                enc = {k: v.to("cuda") for k, v in enc.items()}
            gen = mdl.generate(**enc, max_new_tokens=256)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            outs.extend([d.strip() for d in dec])
            eprint(f"[INFO] Translated batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    for s, t in zip(segments, outs):
        s["translated_text"] = t

    del mdl
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# Stage 2: TTS per segment
# -----------------------------

def extract_ref_audio_16k_mono(video: Path, out_wav: Path, ref_seconds: float):
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")
    
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-t", f"{ref_seconds:.3f}",
        str(out_wav)
    ])
    
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract reference audio: {out_wav}")

def tts_generate_segments_saex(
    segments: List[Dict[str, Any]],
    python_exe: str,
    repo_path: Path,
    model_path: Path,
    ref_wav: Path,
    audio_dir: Path,
    prompt_text: Optional[str],
    n_edit_iter: int,
    skip_existing: bool,
):
    audio_dir.mkdir(parents=True, exist_ok=True)
    tts_script = repo_path / "tts_infer.py"
    
    if not tts_script.exists():
        raise FileNotFoundError(f"TTS script not found: {tts_script}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if not ref_wav.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_wav}")

    # Determine prompt text: use provided or extract from first segment
    if prompt_text is None:
        # Find first segment with text
        for s in segments:
            text = s.get("text", "").strip()
            if text:
                # Use first sentence or first 100 chars
                prompt_text = text.split('.')[0].strip()
                if not prompt_text:
                    prompt_text = text[:100].strip()
                eprint(f"[INFO] Auto-selected prompt text from first segment: '{prompt_text}'")
                break
        
        if prompt_text is None:
            prompt_text = "hey"
            eprint(f"[WARN] No text found in segments, using default prompt: '{prompt_text}'")
    else:
        eprint(f"[INFO] Using provided prompt text: '{prompt_text}'")

    total_segments = len([s for s in segments if (s.get("translated_text") or "").strip()])
    processed = 0
    
    eprint(f"[INFO] TTS output directory: {audio_dir.absolute()}")
    eprint(f"[INFO] Total segments to process: {total_segments}")
    
    for s in segments:
        i = s["id"]
        out_wav = audio_dir / f"chunk_{i}.wav"
        
        if skip_existing and out_wav.exists() and out_wav.stat().st_size > 0:
            eprint(f"[SKIP] existing TTS chunk {i+1}: {out_wav}")
            s["tts_wav"] = str(out_wav)
            continue

        text = s.get("translated_text") or ""
        if not text.strip():
            eprint(f"[SKIP] empty text for segment {i}")
            continue

        processed += 1
        eprint(f"\n[INFO] === Generating TTS {processed}/{total_segments} (segment {i}) ===")
        eprint(f"[INFO] Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # List files before TTS to track new files
        files_before = set(audio_dir.glob("*.wav"))
        
        t0 = time.time()
        cmd = [
            python_exe,
            str(tts_script),
            "--model-path", str(model_path),
            "--prompt-text", prompt_text,
            "--prompt-audio", str(ref_wav),
            "--generated-text", text,
            "--edit-type", "clone",
            "--n-edit-iter", str(n_edit_iter),
            "--output-dir", str(audio_dir)
        ]
        
        eprint(f"[DEBUG] Running: {' '.join(cmd)}")
        
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            eprint(f"[ERROR] TTS failed for segment {i}: {e}")
            continue
        
        elapsed = time.time() - t0
        eprint(f"[INFO] TTS process completed in {elapsed:.1f}s")

        # Find new files created
        files_after = set(audio_dir.glob("*.wav"))
        new_files = files_after - files_before
        
        if new_files:
            eprint(f"[INFO] New files created: {[f.name for f in new_files]}")
            # Use the most recent new file
            newest = max(new_files, key=lambda p: p.stat().st_mtime)
        else:
            # Fallback to newest_wav_in_dir
            newest = newest_wav_in_dir(audio_dir, t0)
        
        if not newest:
            eprint(f"[ERROR] SAEX did not produce a wav for segment {i}")
            eprint(f"[DEBUG] Files in {audio_dir}: {list(audio_dir.glob('*'))}")
            continue

        # Move/rename to expected location
        if newest != out_wav:
            eprint(f"[INFO] Moving {newest.name} -> {out_wav.name}")
            shutil.move(str(newest), str(out_wav))
        
        s["tts_wav"] = str(out_wav)
        eprint(f"[OK] TTS chunk {i}: {out_wav.name} ({elapsed:.1f}s)\n")

# -----------------------------
# Stage 3: Audio overlay/mix
# -----------------------------

def build_mix_filter_script(
    segments: List[Dict[str, Any]],
    duck_gain: float,
    script_path: Path
) -> Tuple[int, List[Path]]:
    lines = []
    lines.append(f"[0:a]aformat=channel_layouts=stereo,aresample=44100,volume={duck_gain}[a0];")

    seg_wavs: List[Path] = []
    mix_inputs = ["[a0]"]
    input_index = 1

    for s in segments:
        wav = s.get("tts_wav")
        if not wav:
            continue
        wavp = Path(wav)
        if not wavp.exists():
            eprint(f"[WARN] TTS wav not found: {wavp}")
            continue

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

    if len(mix_inputs) == 1:
        lines.append("[a0]anull[aout]")
    else:
        lines.append("".join(mix_inputs) + f"amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0[aout]")

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines), encoding="utf-8")
    eprint(f"[INFO] Filter script written with {len(seg_wavs)} segments")
    return input_index, seg_wavs

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
):
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found on PATH")
    
    if not video_in.exists():
        raise FileNotFoundError(f"Input video not found: {video_in}")

    # Get video duration for logging
    video_duration = get_video_duration(video_in)
    eprint(f"[INFO] Video duration: {video_duration:.3f}s")
    
    # Fix any overlapping segments (safety check even if done earlier)
    if fix_overlaps:
        segments = fix_overlapping_segments(segments, min_gap=min_gap, max_delay=max_delay)
    
    # Calculate final audio duration after all delays
    max_audio_end = video_duration  # Default to video duration
    if segments:
        max_audio_end = max(seg['end'] for seg in segments)
        eprint(f"[INFO] Final audio ends at: {max_audio_end:.3f}s")
        
        if max_audio_end > video_duration:
            extension_needed = max_audio_end - video_duration
            eprint(f"[INFO] Audio extends {extension_needed:.2f}s beyond video end")
            
            if extend_video:
                eprint(f"[INFO] Will extend video with freeze frame for {extension_needed:.2f}s")
            else:
                eprint(f"[WARNING] Last {extension_needed:.2f}s of audio will be cut off!")
                eprint(f"[INFO] Use --extend_video to preserve full audio")
                max_audio_end = video_duration  # Will use -shortest

    filter_script = work_dir / "mix_filter.txt"
    _, seg_wavs = build_mix_filter_script(segments, duck_gain, filter_script)

    if not seg_wavs:
        eprint("[WARN] No TTS segments to overlay, copying original video")
        shutil.copy2(video_in, video_out)
        return

    # Determine if we need to extend the video
    video_to_use = video_in
    if extend_video and max_audio_end > video_duration:
        extension = max_audio_end - video_duration
        eprint(f"[INFO] Extending video by {extension:.2f}s with freeze frame")
        
        # Create extended video first
        extended_video = work_dir / "video_extended.mp4"
        extend_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_in),
            "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={extension}[v]",
            "-map", "[v]",
            "-map", "0:a",  # Keep original audio for now
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            str(extended_video)
        ]
        eprint("[INFO] Step 1/2: Extending video with freeze frame...")
        run_cmd(extend_cmd)
        video_to_use = extended_video
        eprint("[INFO] Step 2/2: Overlaying translated audio...")
    
    # Now overlay audio on the (possibly extended) video
    cmd = ["ffmpeg", "-y", "-i", str(video_to_use)]
    for w in seg_wavs:
        cmd += ["-i", str(w)]

    cmd += [
        "-filter_complex_script", str(filter_script),
        "-map", "0:v",
        "-map", "[aout]",
    ]
    
    # Video codec: copy if using original, already encoded if extended
    if video_to_use == video_in:
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "copy"]  # Extended video is already encoded
    
    cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
    ]
    
    # Only use -shortest if we didn't extend video
    if video_to_use == video_in:
        cmd.append("-shortest")
    
    cmd.append(str(video_out))
    run_cmd(cmd)
    
    if not video_out.exists() or video_out.stat().st_size == 0:
        raise RuntimeError(f"Failed to create output video: {video_out}")
    
    # Verify output
    output_duration = get_video_duration(video_out)
    duration_diff = abs(output_duration - video_duration)
    
    eprint(f"[OK] Video created: {video_out}")
    eprint(f"[INFO] Duration - Input: {video_duration:.3f}s, Output: {output_duration:.3f}s, Diff: {duration_diff:.3f}s")
    
    if duration_diff > 0.5:
        eprint(f"[WARNING] Duration mismatch > 0.5s! This may indicate a problem.")
    
    # Check for corruption
    verify_cmd = ["ffmpeg", "-v", "error", "-i", str(video_out), "-f", "null", "-"]
    result = subprocess.run(verify_cmd, capture_output=True, text=True)
    if result.returncode == 0 and not result.stderr.strip():
        eprint(f"[OK] Output verification passed - no corruption detected")
    elif result.stderr.strip():
        eprint(f"[WARNING] Verification warnings:")
        eprint(result.stderr[:500])

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Video translation pipeline with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", required=True, help="Input video file")
    ap.add_argument("--src_lang", required=True, help="Source language code (e.g., de, ru, en)")
    ap.add_argument("--output_json", default=None, help="Output segments JSON file")
    ap.add_argument("--output_video", default=None, help="Output video file")

    # STT
    ap.add_argument("--whisper_model", default="large-v3", help="Faster-Whisper model name")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--compute_type", default="float16", help="Compute type for Faster-Whisper")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for transcription (unused with faster-whisper)")

    # MT
    ap.add_argument("--mt_model", required=True, help="Translation model name")
    ap.add_argument("--mt_batch", type=int, default=8, help="Translation batch size")
    ap.add_argument("--mt_use_pipeline", action="store_true", help="Use transformers pipeline for translation")

    # TTS (SAEX)
    ap.add_argument("--repo_path", required=True, help="Path to SAEX repo")
    ap.add_argument("--model_path", required=True, help="Path to TTS model")
    ap.add_argument("--audio_dir", default=None, help="Directory for TTS output")
    ap.add_argument("--prompt_text", default=None, help="Prompt text for TTS (if None, uses first segment text)")
    ap.add_argument("--n_edit_iter", type=int, default=1, help="Number of edit iterations")
    ap.add_argument("--ref_seconds", type=float, default=10.0, help="Reference audio duration")

    # Overlay
    ap.add_argument("--duck_gain", type=float, default=0.15, help="Original audio volume (0-1)")
    ap.add_argument("--fix_overlaps", action="store_true", default=True, help="Fix overlapping segments (default: True)")
    ap.add_argument("--min_gap", type=float, default=0.1, help="Minimum gap between segments in seconds")
    ap.add_argument("--max_delay", type=float, default=5.0, help="Maximum acceptable delay before trimming (seconds)")
    ap.add_argument("--extend_video", action="store_true", default=True, help="Extend video with freeze frame if audio is longer (default: True)")

    # Workflow
    ap.add_argument("--work_dir", default=None, help="Working directory")
    ap.add_argument("--skip_existing", action="store_true", help="Skip existing TTS files")
    ap.add_argument("--stage", default="all", choices=["stt", "tts", "overlay", "all"],
                    help="Which stage to run")

    args = ap.parse_args()

    # Setup paths
    video = Path(args.video).resolve()
    if not video.exists():
        eprint(f"[ERROR] Video file not found: {video}")
        sys.exit(1)
    
    video_stem = video.stem
    project_dir = Path(video_stem)
    project_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir) if args.work_dir else project_dir / "work"
    audio_dir = Path(args.audio_dir) if args.audio_dir else project_dir / "audio_out"
    out_json = Path(args.output_json) if args.output_json else project_dir / "segments.json"
    out_video = Path(args.output_video) if args.output_video else project_dir / "translated.mp4"

    python_exe = sys.executable
    repo_path = Path(args.repo_path)
    model_path = Path(args.model_path)

    # Stage 1: STT + Translation
    if args.stage in ("stt", "all"):
        eprint("\n" + "="*60)
        eprint("STAGE 1: Speech-to-Text + Translation")
        eprint("="*60)
        
        audio_16k = work_dir / "audio_16k_mono.wav"
        eprint(f"[INFO] Extracting 16k mono audio -> {audio_16k}")
        extract_audio_16k_mono(video, audio_16k)

        segments = stt_with_faster_whisper(
            audio_wav_16k=audio_16k,
            src_lang=args.src_lang,
            whisper_model=args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
        )

        if not segments:
            eprint("[WARN] No segments detected in video")
        
        eprint(f"[INFO] Translating {len(segments)} segments -> {args.mt_model}")
        translate_segments(
            segments,
            mt_model=args.mt_model,
            device=args.device,
            batch_size=args.mt_batch,
            use_pipeline=args.mt_use_pipeline,
        )
        
        # Fix overlapping segments before TTS
        if args.fix_overlaps:
            eprint(f"[INFO] Checking for overlapping segments (min_gap={args.min_gap}s, max_delay={args.max_delay}s)...")
            segments = fix_overlapping_segments(segments, min_gap=args.min_gap, max_delay=args.max_delay)

        save_segments(out_json, segments)
        eprint(f"[OK] Segments JSON saved: {out_json}")

    # Stage 2: TTS
    if args.stage in ("tts", "all"):
        eprint("\n" + "="*60)
        eprint("STAGE 2: Text-to-Speech Generation")
        eprint("="*60)
        
        segments = load_segments(out_json)

        ref_wav = audio_dir / "ref_16k_mono.wav"
        if not ref_wav.exists() or ref_wav.stat().st_size == 0:
            eprint(f"[INFO] Extracting ref audio ({args.ref_seconds}s) -> {ref_wav}")
            extract_ref_audio_16k_mono(video, ref_wav, args.ref_seconds)
        else:
            eprint(f"[INFO] Using existing ref audio: {ref_wav}")

        tts_generate_segments_saex(
            segments=segments,
            python_exe=python_exe,
            repo_path=repo_path,
            model_path=model_path,
            ref_wav=ref_wav,
            audio_dir=audio_dir,
            prompt_text=args.prompt_text,
            n_edit_iter=args.n_edit_iter,
            skip_existing=args.skip_existing,
        )

        save_segments(out_json, segments)
        eprint(f"[OK] Updated segments JSON with tts_wav paths: {out_json}")

    # Stage 3: Overlay
    if args.stage in ("overlay", "all"):
        eprint("\n" + "="*60)
        eprint("STAGE 3: Audio Overlay & Video Generation")
        eprint("="*60)
        
        segments = load_segments(out_json)
        eprint(f"[INFO] Creating output video -> {out_video}")
        overlay_to_video(
            video_in=video,
            video_out=out_video,
            segments=segments,
            duck_gain=args.duck_gain,
            work_dir=work_dir,
            fix_overlaps=args.fix_overlaps,
            min_gap=args.min_gap,
            max_delay=args.max_delay,
            extend_video=args.extend_video,
        )
        eprint(f"\n[SUCCESS] Completed! Output: {out_video}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint("\n[INFO] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        eprint(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)