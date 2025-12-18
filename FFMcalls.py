"""FFMcalls.py

Grandma Filter ffmpeg helpers ONLY.

Contract:
- No CLI / argparse / pipeline logic in this module.
- No imports of processText / whisperCalls / SWhelper.
- Pure functions used by main.py/processText.py.

Exports:
- extract_audio(...)
- make_mini_audio(...)
- censor_audio(...)
- replace_audio_track(...)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command and raise a helpful error if it fails."""
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install with `brew install ffmpeg` and ensure it is on PATH."
        ) from e


def extract_audio(input_media: str, output_mp3: str = "output.mp3") -> str:
    """Extract the main audio track from a media file into an MP3."""
    out_path = Path(output_mp3)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_media,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return str(out_path)


def make_mini_audio(
    input_media: str,
    start: float,
    end: Optional[float] = None,
    duration: Optional[float] = None,
    output_path: str = "mini.mp3",
) -> str:
    """Create a small audio clip from `input_media`.

    This exists because `processText.prepare_mini_audio_and_word_jsons()` expects
    `FFMcalls.make_mini_audio` to exist.

    You can specify either:
      - `end` (absolute seconds), or
      - `duration` (seconds)

    Returns the output path.
    """

    if end is None:
        if duration is None:
            raise ValueError("make_mini_audio requires either `end` or `duration`.")
        end = float(start) + float(duration)

    start_s = float(start)
    end_s = float(end)
    if end_s <= start_s:
        raise ValueError(f"Invalid mini range: start={start_s} end={end_s}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s}",
        "-to",
        f"{end_s}",
        "-i",
        input_media,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return str(out_path)


# Backwards-compatible aliases (in case other code uses different names)
makeMiniAudio = make_mini_audio


def _ffprobe_duration_seconds(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def _build_mute_filter(
    ranges: list[tuple[float, float]],
    *,
    pad: float = 0.0,
) -> str:
    """Build an ffmpeg -af filter string that mutes audio during `ranges`."""
    filters: list[str] = []
    for s, e in ranges:
        ss = max(0.0, float(s) - float(pad))
        ee = max(ss, float(e) + float(pad))
        filters.append(f"volume=enable='between(t,{ss:.6f},{ee:.6f})':volume=0")
    return ",".join(filters) if filters else "anull"


def censor_audio(
    *,
    ranges: list[tuple[float, float]],
    input_file: str,
    output_file: str,
    mode: str = "mute",
    beep_freq: int = 1000,
    beep_volume: float = 0.25,
    pad: float = 0.0,
) -> str:
    """Create a censored audio file.

    Parameters
    - ranges: list of (start,end) seconds to censor
    - input_file: source audio (e.g., output.mp3)
    - output_file: output audio path (e.g., output_censored.mp3)
    - mode: 'mute' or 'beep'
    - beep_freq: sine beep frequency (Hz) when mode='beep'
    - beep_volume: beep loudness multiplier (roughly 0.0–1.0)
    - pad: seconds to expand each range on both sides

    Returns output_file.
    """

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode_l = (mode or "mute").strip().lower()
    if mode_l not in {"mute", "beep"}:
        raise ValueError("censor_audio mode must be 'mute' or 'beep'")

    if not ranges:
        shutil.copyfile(input_file, str(out_path))
        return str(out_path)

    if mode_l == "mute":
        af = _build_mute_filter(ranges, pad=pad)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-vn",
            "-af",
            af,
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(out_path),
        ]
        _run_ffmpeg(cmd)
        return str(out_path)

    # mode == 'beep'
    duration = _ffprobe_duration_seconds(input_file)
    if duration <= 0:
        duration = 3600.0  # 1 hour fallback

    mute_af = _build_mute_filter(ranges, pad=pad)

    enable_filters: list[str] = ["volume=0"]
    for s, e in ranges:
        ss = max(0.0, float(s) - float(pad))
        ee = max(ss, float(e) + float(pad))
        enable_filters.append(
            f"volume=enable='between(t,{ss:.6f},{ee:.6f})':volume={float(beep_volume)}"
        )
    beep_af = ",".join(enable_filters)

    filter_complex = (
        f"[0:a]{mute_af}[a0];"
        f"sine=frequency={int(beep_freq)}:duration={duration:.6f}[s];"
        f"[s]{beep_af}[a1];"
        f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(out_path),
    ]
    _run_ffmpeg(cmd)
    return str(out_path)


def replace_audio_track(
    *,
    video_file: str,
    censored_audio: str,
    original_audio: str | None = None,
    output_video_file: str,
    original_language_name: str | None = None,
    original_language_code: str | None = None,
    subtitle_file: str | None = None,
) -> str:
    """Mux censored audio back into the video.

    - Track 0: censored audio (default)
    - Track 1: original audio (optional, secondary)
    - Optional subtitle track

    Returns output_video_file.
    """

    out_path = Path(output_video_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        video_file,
        "-i",
        censored_audio,
    ]

    if original_audio:
        cmd += ["-i", original_audio]

    if subtitle_file:
        cmd += ["-i", subtitle_file]

    # Map: video stream
    cmd += ["-map", "0:v:0"]

    # Map censored audio
    cmd += ["-map", "1:a:0"]

    # Map original audio if present
    if original_audio:
        cmd += ["-map", "2:a:0"]

    # Map subtitles if present
    if subtitle_file:
        sub_index = 3 if original_audio else 2
        cmd += ["-map", f"{sub_index}:s:0"]

    # Copy video, re-encode audio to AAC for compatibility
    cmd += [
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]

    # Metadata / disposition
    cmd += ["-disposition:a:0", "default"]

    if original_audio:
        cmd += ["-disposition:a:1", "0"]

    if original_language_name:
        cmd += ["-metadata:s:a:0", f"title={original_language_name} (Censored)"]

    if original_language_code:
        cmd += ["-metadata:s:a:0", f"language={original_language_code}"]
        if original_audio:
            cmd += ["-metadata:s:a:1", f"language={original_language_code}"]

    if subtitle_file:
        cmd += ["-c:s", "copy"]

    cmd += [str(out_path)]

    _run_ffmpeg(cmd)
    return str(out_path)
