"""whisperCalls.py

Grandma Filter transcription helpers.

Design goals (aligned with your roadmap):
- Primary/default: faster-whisper (but guarded so it can’t wedge the whole process)
- Backup: whisperx (CLI) for word-level timestamps/alignment
- No heavy imports at module import time (torch/whisper/transformers can hang)
- App-ready: return structured dicts, write JSON deterministically, machine-readable logs
- Strict: caller can decide whether a failure is fatal

IMPORTANT NOTE (current reality):
On your Python 3.13 environment, importing faster-whisper may hang because it pulls in
ctranslate2 -> transformers (heavy import-structure scan). We therefore:
- try faster-whisper in a subprocess with a timeout
- if it times out, we fall back to whisperx CLI

Later, once you move to a Python 3.12 venv or pin versions, you can switch the
faster-whisper path to an in-process import for speed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -------------------------
# Utilities
# -------------------------

def _run(cmd: list[str], *, timeout_s: Optional[int] = None) -> Tuple[int, str]:
    """Run a command and return (returncode, combined_output)."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return p.returncode, (p.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout_s}s: {' '.join(cmd)}"
    except Exception as e:
        return 3, f"{type(e).__name__}: {e}"


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_output_json(audio_file: str | Path, output_dir: str | Path) -> str:
    audio_file = Path(audio_file)
    output_dir = Path(output_dir)
    return str(output_dir / f"{audio_file.stem}.json")


# -------------------------
# Primary: faster-whisper (guarded)
# -------------------------

def transcribe_faster_whisper_guarded(
    audio_file: str,
    *,
    model_name: str = "base",
    language: Optional[str] = None,
    output_json_path: Optional[str] = None,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """Attempt transcription via faster-whisper in a subprocess.

    Returns a dict:
      {
        "ok": bool,
        "engine": "faster-whisper",
        "timed_out": bool,
        "output_json": str|None,
        "error": str|None
      }

    We run it in a subprocess to avoid wedging the main process if imports hang.
    """

    out_json = output_json_path
    if out_json is None:
        out_json = _default_output_json(audio_file, ".")

    # Script executed in the child process. It writes a Whisper-like JSON with segments.
    # We keep it simple: segments with start/end/text plus a top-level language (if provided).
    lang_literal = "None" if language is None else repr(language)

    child_code = f"""
import json
from pathlib import Path

AUDIO = {audio_file!r}
MODEL = {model_name!r}
LANG = {lang_literal}
OUT = {out_json!r}

# Import inside child (can hang — parent has timeout)
from faster_whisper import WhisperModel

# Choose device. On Apple Silicon with torch+mps, faster-whisper still uses ctranslate2.
# 'cpu' is safest. You can switch to 'metal' once you verify your stack supports it.
model = WhisperModel(MODEL, device='cpu', compute_type='int8')

segments, info = model.transcribe(AUDIO, language=LANG)

out = {{
  'text': '',
  'language': getattr(info, 'language', LANG) if info is not None else LANG,
  'segments': []
}}

texts = []
for seg in segments:
    s = {{
      'start': float(getattr(seg, 'start', 0.0)),
      'end': float(getattr(seg, 'end', 0.0)),
      'text': str(getattr(seg, 'text', '')).strip(),
    }}
    out['segments'].append(s)
    if s['text']:
        texts.append(s['text'])

out['text'] = ' '.join(texts).strip()

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
"""

    rc, out = _run(["python", "-c", child_code], timeout_s=timeout_s)

    if rc == 0 and Path(out_json).exists():
        return {
            "ok": True,
            "engine": "faster-whisper",
            "timed_out": False,
            "output_json": out_json,
            "error": None,
        }

    if rc == 124:
        return {
            "ok": False,
            "engine": "faster-whisper",
            "timed_out": True,
            "output_json": None,
            "error": out,
        }

    return {
        "ok": False,
        "engine": "faster-whisper",
        "timed_out": False,
        "output_json": None,
        "error": out,
    }


# -------------------------
# Backup: whisperx (CLI)
# -------------------------

def transcribe_whisperx_cli(
    audio_file: str,
    *,
    model_name: str = "base",
    language: Optional[str] = None,
    output_dir: str = "./mini_audio",
    compute_type: str = "int8",
) -> Dict[str, Any]:
    """Run whisperx via CLI and return a structured result.

    whisperx will generate a JSON file in output_dir.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisperx",
        audio_file,
        "--model",
        model_name,
        "--device",
        "cpu",
        "--compute_type",
        compute_type,
        "--output_format",
        "json",
        "--output_dir",
        output_dir,
    ]
    if language:
        cmd.extend(["--language", language])

    rc, out = _run(cmd)

    out_json = _default_output_json(audio_file, output_dir)

    if rc == 0 and Path(out_json).exists():
        return {
            "ok": True,
            "engine": "whisperx",
            "output_json": out_json,
            "error": None,
        }

    return {
        "ok": False,
        "engine": "whisperx",
        "output_json": None,
        "error": out,
    }


# -------------------------
# Public entry points
# -------------------------

def transcribe_audio(
    audio_file: str,
    *,
    model_name: str = "base",
    language: Optional[str] = None,
    primary_timeout_s: int = 60,
    primary_output_json: Optional[str] = None,
    whisperx_output_dir: str = "./mini_audio",
) -> Dict[str, Any]:
    """Transcribe audio.

    Strategy:
    1) Try faster-whisper (guarded subprocess)
    2) If timeout/failure, fall back to whisperx CLI

    Returns:
      {
        "ok": bool,
        "engine": "faster-whisper"|"whisperx",
        "output_json": str|None,
        "timed_out": bool,
        "error": str|None,
        "elapsed_s": float
      }
    """

    start = time.time()

    # 1) Primary
    primary = transcribe_faster_whisper_guarded(
        audio_file,
        model_name=model_name,
        language=language,
        output_json_path=primary_output_json,
        timeout_s=primary_timeout_s,
    )
    if primary["ok"]:
        primary["elapsed_s"] = round(time.time() - start, 3)
        primary["output_json"] = primary.get("output_json")
        return {
            "ok": True,
            "engine": "faster-whisper",
            "output_json": primary.get("output_json"),
            "timed_out": False,
            "error": None,
            "elapsed_s": primary["elapsed_s"],
        }

    # 2) Backup
    backup = transcribe_whisperx_cli(
        audio_file,
        model_name=model_name,
        language=language,
        output_dir=whisperx_output_dir,
    )

    elapsed = round(time.time() - start, 3)

    if backup["ok"]:
        return {
            "ok": True,
            "engine": "whisperx",
            "output_json": backup.get("output_json"),
            "timed_out": bool(primary.get("timed_out")),
            "error": None,
            "elapsed_s": elapsed,
        }

    return {
        "ok": False,
        "engine": "whisperx" if backup else "faster-whisper",
        "output_json": None,
        "timed_out": bool(primary.get("timed_out")),
        "error": f"Primary failed: {primary.get('error')} | Backup failed: {backup.get('error')}",
        "elapsed_s": elapsed,
    }


def load_detected_language(transcript_json_path: str) -> Optional[str]:
    """Best-effort: extract detected language from a transcript JSON."""
    try:
        data = _read_json(transcript_json_path)
        lang = data.get("language")
        if isinstance(lang, str) and lang.strip():
            return lang.strip()
        # whisperx sometimes nests language differently; keep it simple for now.
        return None
    except Exception:
        return None


if __name__ == "__main__":
    # Minimal smoke test (won't run unless you call it directly)
    # Example:
    #   python whisperCalls.py ./mini_audio/0.mp3
    import sys

    if len(sys.argv) >= 2:
        audio = sys.argv[1]
        res = transcribe_audio(audio, model_name="base", language=None)
        print(json.dumps(res, indent=2))
    else:
        print("Usage: python whisperCalls.py <audio_file>")