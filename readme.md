# Grandma Filter 🎧🚫

Grandma Filter is a command-line tool that automatically **detects profanity in spoken audio**, creates a **censored audio track** (mute or beep), and **muxes it back into the original video** alongside the untouched original audio.

The output video:
- Keeps the **original audio track**
- Adds a **new default audio track** named `<Language>_censored`
- Optionally includes a **subtitle track** that displays `[CENSORED]` during censored moments

---

## Requirements

- macOS or Linux (tested on macOS Apple Silicon)
- Python **3.11+** (recommended: run inside a virtual environment)
- `ffmpeg` available on your PATH
- Internet connection (for Whisper models on first run)

### Python dependencies (recommended setup)

```bash
python3 -m venv venv
source venv/bin/activate
pip install openai-whisper whisperx torch
```

> ⚠️ On macOS with Homebrew Python, **do not install packages system-wide**. Use a virtual environment.

---

## Usage

> **Tip:** Use your project virtual environment (recommended):
>
> ```bash
> source venv/bin/activate
> ```

### Single file mode

```bash
python3 main.py <input_file>
```

- `<input_file>`: path to a single media file (e.g., `.mkv`, `.mp4`, `.mov`, `.avi`)
- Output is written next to the input as `<input>_censored<ext>`
- You can override the output path using `--output <file>` (single-file mode only).

### Directory mode (non-recursive)

```bash
python3 main.py --dir <folder>
```

- Processes all supported media files in the folder (non-recursive)
- Skips files that already end in `_censored` by default

### All flags

| Flag | Meaning | Notes |
|---|---|---|
| `-b`, `--beep` | Use a beep instead of mute for censored words | Default is **mute** |
| `--dir <folder>` | Process all supported files in a directory (non-recursive) | Mutually exclusive with `<input_file>` |
| `--output <file>` | Output file path (single-file mode only). Overrides default `_censored` naming | Mutually exclusive with `--dir` and cannot be combined with `--in-place` |
| `--out-dir <folder>` | Output directory (directory mode only). Defaults to input directory | Only valid with `--dir` and cannot be combined with `--in-place` |
| `--ext <ext>` | Restrict which extensions are processed | Repeatable. Example: `--ext mkv --ext mp4`. If omitted, defaults to `mp4,mov,mkv,avi` |
| `--dry-run` | Print what would be processed and output names, but do not run Whisper or FFmpeg | Works in both single-file and directory mode |
| `--no-skip-censored` | Also process files that already end in `_censored` | Only matters in directory mode |
| `--in-place` | Replace the original file with the censored output | Uses an atomic replace after a successful run |

### Examples

Mute (default):

```bash
python3 main.py "input.mkv"
```

Beep:

```bash
python3 main.py "input.mkv" --beep
```

```bash
# Single file with custom output path
python3 main.py "input.mkv" --output "./out/clean.mkv"
```

Batch process a folder, only MKV and MP4:

```bash
python3 main.py --dir ./videos --ext mkv --ext mp4
```

```bash
# Directory mode with separate output folder
python3 main.py --dir ./videos --out-dir ./out
```

Dry-run:

```bash
python3 main.py --dir ./videos --dry-run
```

Force re-processing of already-censored files:

```bash
python3 main.py --dir ./videos --no-skip-censored
```

In-place replacement (single file):

```bash
python3 main.py "input.mkv" --in-place
```

In-place replacement (directory):

```bash
python3 main.py --dir ./videos --in-place
```

---

## Output Details

For an English-language input video:

- **Audio Track 1 (default):** `English_censored`
- **Audio Track 2:** `English`
- **Subtitles:** `[CENSORED]` appears during muted/beeped sections

Most players (VLC, Plex, mpv) allow switching audio tracks at playback time.

---

## How It Works (High-Level)

1. Extracts audio from the input video
2. Transcribes speech using Whisper
3. Detects target words and precise timestamps
4. Creates a censored audio track (mute or beep)
5. Auto-detects spoken language
6. Muxes video + original audio + censored audio + subtitles into a single file

---

## Notes & Tips

- Batch mode is **non-recursive by design** (safe default)
- Temporary files are cleaned between each run
- Errors in one file will **not stop batch processing**
- Subtitle behavior varies slightly between players

---

## Example

```bash
python3 main.py --dir ./clips --beep
```

Produces:

```text
clip1_censored.mp4
clip2_censored.mp4
clip3_censored.mp4
```

---

## License / Disclaimer

This project is for educational and personal use.
Accuracy of speech recognition and profanity detection depends on the quality of the input audio and the Whisper model used.