

# Grandma Filter 🤬🚫🎧

Grandma Filter is a **command-line tool** that automatically **detects profanity in spoken audio**, creates a **censored audio track** (mute or beep), and **muxes it back into the original video** alongside the untouched original audio.

It is designed to be:
- **Accurate when possible** (word-level timestamps)
- **Safe when not** (fallback to estimated timing so swears are not missed)
- **Fast enough for full TV episodes**
- **Structured for future GUI / app integration**

---

## What the Output Looks Like

For a typical video input:

- **Video stream:** unchanged
- **Audio track 1 (default):** `<Language>_censored`
- **Audio track 2:** original audio
- **Report files:** JSON + CSV describing exactly what was censored

Most players (VLC, Plex, mpv) let you switch audio tracks at playback time.

---

## Platform Support

- **macOS:** fully supported and tested (Apple Silicon)
- **Windows:** untested
- **Linux:** untested

---

## Requirements

### System Requirements

- Python **3.11 or 3.12** (recommended: 3.12)
- `ffmpeg` available on your system `PATH`
- Internet connection on first run (Whisper models download automatically)

---

## Beginner Setup (Step-by-Step)

These steps assume **no prior Python project setup experience**.

### 1️⃣ Install Python

Make sure Python is installed:

```bash
python3 --version
```

You should see `Python 3.11.x` or `Python 3.12.x`.

---

### 2️⃣ Create and activate a virtual environment

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` in your terminal prompt.

> ⚠️ **Do not install packages globally.** Always use the virtual environment.

---

### 3️⃣ Install Python dependencies

Install required packages:

```bash
pip install faster-whisper torch
```

> On Apple Silicon, this will run efficiently on CPU. GPU acceleration is not required.

---

### 4️⃣ Install ffmpeg

#### macOS (Homebrew)
```bash
brew install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

#### Windows (experimental)
Install ffmpeg and ensure it is on your PATH (via winget, chocolatey, or manual install).

---

### ✅ Optional: Let SWhelper install missing dependencies

If you prefer, you can let the built‑in helper install missing **required** dependencies
(Python modules + ffmpeg) and re-check your setup:

```bash
python SWhelper.py check --install
```

Note: SWhelper will prompt before installing **ffmpeg**, since it is installed
system-wide (outside the venv). Python packages are installed inside the venv.
Use `--yes` to skip that prompt:

```bash
python SWhelper.py check --install --yes
```

### What installs outside the venv?

- `ffmpeg` (system package via Homebrew/apt/yum/winget)

### Uninstall everything (full cleanup)

This removes temp files, uninstalls Python deps, deletes `.venv`, and uninstalls ffmpeg:

```bash
python SWhelper.py uninstall
```

---

## Usage

Activate your virtual environment first:

```bash
source .venv/bin/activate
```

---

### Single File Mode

```bash
python main.py <input_file>
```

Example:
```bash
python main.py "video.mkv"
```

Result:
- Creates `video_censored.mkv` next to the original file
- Original file is untouched


### Automatic dependency check on every run

By default, `main.py` **auto-creates the venv (if missing), relaunches itself
inside that venv**, and then runs the SWhelper installer/check every time you run it.
It will attempt to install missing dependencies (Python packages + ffmpeg).
If Python 3.12 (or 3.11) is available, it will use that for the venv to keep
Whisper dependencies compatible.
On macOS, if Python 3.12 is missing, it will offer to install `python@3.12`
via Homebrew automatically.

Flags:
- `--no-check` to skip the check/installer
- `--yes-install` to auto-approve the system-wide ffmpeg install prompt

### Low-confidence verification (bigger models)

If a target word is detected with low confidence, Grandma Filter re-checks a short
audio clip using larger Whisper models (small → medium → large) and only mutes
the word if a larger model confirms it.

You can control this behavior:
- `--low-conf 0.6` (confidence cutoff; below this triggers verification)
- `--verify-models small,medium,large`
- `--verify-pad 0.30` (seconds of padding around the word)

### Subtitle track (censored words)

By default, output videos include a **soft subtitle track** with censored words.

Flags:
- `--no-subs` to disable subtitle muxing
- `--subs-raw` to include the full word in subtitles (default masks like `f---`)


You can still activate the venv manually if you prefer:

```bash
source .venv/bin/activate
python main.py "video.mkv"
```

---

### Beep Instead of Mute

```bash
python main.py "video.mkv" --beep
```

---

### Directory (Batch) Mode

Process all supported media files in a folder (non-recursive):

```bash
python main.py --dir ./videos
```

By default:
- Files ending in `_censored` are skipped
- Output is written next to the originals

---

### Directory Mode with Separate Output Folder

```bash
python main.py --dir ./videos --out-dir ./videos_censored
```

- Originals stay untouched
- All censored outputs go to `./videos_censored/`

---

### Restrict File Extensions

```bash
python main.py --dir ./videos --ext mkv --ext mp4
```

---

### Dry Run (No Processing)

```bash
python main.py --dir ./videos --dry-run
```

Shows what *would* be processed without running Whisper or ffmpeg.

---

### In-Place Replacement (Use Carefully)

Single file:
```bash
python main.py "video.mkv" --in-place
```

Directory:
```bash
python main.py --dir ./videos --in-place
```

The original file is replaced **only after** a successful run.

---

## Reports & Logs

Each run produces:

- `reports/<filename>.<timestamp>.report.json`
- `muted_words.csv`

The report includes:
- Input and output file paths
- Detected language
- Censor mode (mute / beep)
- Number of segments
- Word-level hits
- Estimated hits (fallback timing)
- Final censor ranges
- Total processing time

---

## How Profanity Detection Works

1. Audio is extracted from the video
2. Speech is transcribed using **faster-whisper**
3. Profanity words are matched against a target list
4. **If word-level timestamps exist**, they are used directly
5. **If word timestamps are missing**, timing is **estimated from transcript text** so swears are not missed
6. Censor ranges are normalized and merged
7. A censored audio track is generated and muxed back into the video

> This hybrid approach prioritizes **coverage over elegance** to avoid missed profanity.

---

## Notes & Tips

- Batch mode is **non-recursive by design** (safer default)
- Temporary files are cleaned between runs
- Errors in one file **do not stop** batch processing
- Timing is conservative to ensure audible profanity is fully muted

---

## License / Disclaimer

This project is for educational and personal use.
Speech recognition accuracy depends on audio quality and model behavior.
Censor timing may be approximate when word-level timestamps are unavailable.

---
