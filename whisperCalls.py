"""whisperCalls.py

Grandma Filter transcription helpers.

Locked decisions (current):
- Primary/default transcription backend: faster-whisper
- Prefer word-level timestamps from faster-whisper (no WhisperX)

This module is designed to be library-friendly:
- No CLI parsing here
- Model is cached in-process so batch runs are fast

Output JSON schema (Whisper-like, extended):
{
  "engine": "faster-whisper",
  "model": "base",
  "language": "en" | null,
  "text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.2,
      "text": "...",
      "words": [
        {"start": 0.1, "end": 0.4, "word": "hello", "probability": 0.98},
        ...
      ]
    },
    ...
  ]
}

If word timestamps are not available, `words` will be an empty list.
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TranscribeConfig:
    model_name: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"  # or "translate"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    # Note: on macOS/Apple Silicon, faster-whisper via CTranslate2 typically runs on CPU.
    device: str = "cpu"
    compute_type: str = "int8"  # good default on CPU


# -------- Model cache --------

_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _get_model(cfg: TranscribeConfig):
    """Get (and cache) a WhisperModel instance for this config."""
    key = (cfg.model_name, cfg.device, cfg.compute_type)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    from faster_whisper import WhisperModel  # local import to keep module import light

    model = WhisperModel(cfg.model_name, device=cfg.device, compute_type=cfg.compute_type)
    _MODEL_CACHE[key] = model
    return model


# -------- Public API --------


def transcribe_to_dict(
    audio_file: str,
    *,
    cfg: Optional[TranscribeConfig] = None,
) -> Dict[str, Any]:
    """Transcribe `audio_file` and return a structured dict (see module docstring)."""
    cfg = cfg or TranscribeConfig()
    model = _get_model(cfg)

    segments_iter, info = model.transcribe(
        audio_file,
        language=cfg.language,
        task=cfg.task,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        word_timestamps=cfg.word_timestamps,
        condition_on_previous_text=False,
    )

    out: Dict[str, Any] = {
        "engine": "faster-whisper",
        "model": cfg.model_name,
        "language": getattr(info, "language", None) if info is not None else cfg.language,
        "text": "",
        "segments": [],
    }

    full_text_parts: List[str] = []

    # Strict gatekeeper: drop obvious hallucinations / silence segments
    # (keeps downstream transcript/censoring sane)
    last_text: str = ""
    repeat_count: int = 0

    for idx, seg in enumerate(segments_iter):
        seg_text = str(getattr(seg, "text", "")).strip()
        s = {
            "id": idx,
            "start": float(getattr(seg, "start", 0.0)),
            "end": float(getattr(seg, "end", 0.0)),
            "text": seg_text,
            "words": [],
        }

        # Optional diagnostic fields (present on many faster-whisper Segment objects)
        for k in ("no_speech_prob", "avg_logprob", "compression_ratio", "temperature"):
            v = getattr(seg, k, None)
            if v is not None:
                try:
                    s[k] = float(v)
                except Exception:
                    pass

        # Gatekeeper filter (conservative): skip segments likely to be silence/hallucination
        no_speech = float(s.get("no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(s.get("avg_logprob", 0.0) or 0.0)
        compression = float(s.get("compression_ratio", 0.0) or 0.0)

        txt_l = seg_text.lower().strip()
        short_text = (0 < len(txt_l) <= 6)

        # 1) High no-speech probability + very short or very low confidence text
        if no_speech >= 0.60 and (short_text or avg_logprob < -1.0):
            continue

        # 2) High compression ratio often indicates repetitive / degenerate decoding
        if compression >= 3.0 and short_text:
            continue

        # 3) Repetition guard: drop repeated identical short segments after a couple repeats
        if seg_text == last_text and short_text:
            repeat_count += 1
            if repeat_count >= 3:
                continue
        else:
            last_text = seg_text
            repeat_count = 0

        # faster-whisper returns seg.words when word_timestamps=True
        words = getattr(seg, "words", None)
        if words:
            for w in words:
                try:
                    s["words"].append(
                        {
                            "start": float(getattr(w, "start", 0.0)),
                            "end": float(getattr(w, "end", 0.0)),
                            "word": str(getattr(w, "word", "")).strip(),
                            "probability": float(getattr(w, "probability", 0.0)),
                        }
                    )
                except Exception:
                    # Keep going; word timing is a best-effort enhancement
                    continue

        out["segments"].append(s)
        if seg_text:
            full_text_parts.append(seg_text)

    out["text"] = " ".join(full_text_parts).strip()

    # Normalize language to a simple string when present
    lang = out.get("language")
    if isinstance(lang, str):
        out["language"] = lang.strip() or None

    return out


def transcribe_to_json_file(
    audio_file: str,
    *,
    output_json_path: str = "output.json",
    cfg: Optional[TranscribeConfig] = None,
) -> str:
    """Transcribe `audio_file` and write the JSON to `output_json_path`. Returns the path."""
    data = transcribe_to_dict(audio_file, cfg=cfg)
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(out_path)


# Backwards-compatible wrapper used by your existing main.py

def run_whisper_live(
    audio_file: str = "output.mp3",
    output_json: str = "output.json",
    *,
    model_name: str = "base",
    language: Optional[str] = None,
    timeout_s: int = 0,
) -> None:
    """Legacy entry point.

    - `timeout_s` is ignored in the direct-import implementation.
    - Writes `output_json` in the expected format.
    """
    _ = timeout_s
    cfg = TranscribeConfig(model_name=model_name, language=language)
    transcribe_to_json_file(audio_file, output_json_path=output_json, cfg=cfg)


def load_detected_language(transcript_json_path: str) -> Optional[str]:
    try:
        with open(transcript_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lang = data.get("language")
        return lang.strip() if isinstance(lang, str) and lang.strip() else None
    except Exception:
        return None


# Convenience function retained for callers that used `transcribe_audio()`

def transcribe_audio(
    audio_file: str,
    *,
    model_name: str = "base",
    language: Optional[str] = None,
    output_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.time()
    cfg = TranscribeConfig(model_name=model_name, language=language)
    if output_json_path:
        transcribe_to_json_file(audio_file, output_json_path=output_json_path, cfg=cfg)
        res = {"ok": True, "engine": "faster-whisper", "output_json": output_json_path, "error": None}
    else:
        _ = transcribe_to_dict(audio_file, cfg=cfg)
        res = {"ok": True, "engine": "faster-whisper", "output_json": None, "error": None}
    res["elapsed_s"] = round(time.time() - start, 3)
    return res


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python whisperCalls.py <audio_file> [output.json]")
        raise SystemExit(2)

    audio = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "output.json"
    transcribe_to_json_file(audio, output_json_path=out)
    print(out)