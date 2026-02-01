import argparse
import os
import shutil
import subprocess
import time
import sys
import threading
from typing import Iterable

import FFMcalls
import processText
import SWhelper
import whisperCalls

LOG = print
LOG_FILE = None


def set_logger(fn) -> None:
    global LOG
    LOG = fn


def set_log_file(path: str) -> None:
    global LOG_FILE
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        LOG_FILE = open(path, "a", encoding="utf-8")
    except Exception:
        LOG_FILE = None


def log_line(msg: str) -> None:
    LOG(msg)
    if LOG_FILE:
        try:
            LOG_FILE.write(str(msg) + "\n")
            LOG_FILE.flush()
        except Exception:
            pass




def checkFile(path: str) -> None:
    if os.path.exists(path):
        LOG(f"The file '{path}' exists.")
    else:
        LOG(f"The file '{path}' does not exist.")


def _cleanup_temp_outputs() -> None:
    """Remove temp outputs from a previous run to avoid cross-file contamination."""
    for p in [
        "output.mp3",
        "output.json",
        "output_censored.mp3",
        "censored.srt",
        "video_new.mp4",
    ]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass

    # mini_audio directory (can contain per-segment mp3/json)
    if os.path.isdir("mini_audio"):
        try:
            shutil.rmtree("mini_audio")
        except OSError:
            pass


def _iter_media_files(directory: str, extensions: Iterable[str], skip_censored: bool = True) -> list[str]:
    """Return a sorted list of media files in `directory` (non-recursive)."""
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    files: list[str] = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue

        base, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            continue

        if skip_censored and base.endswith("_censored"):
            continue

        files.append(full)

    files.sort()
    return files


# Compatibility audio extraction wrapper
def _extract_audio(input_media: str, output_mp3: str = "output.mp3") -> None:
    """Extract audio from the input media into an MP3.

    Tries to call whatever extraction function exists in FFMcalls. If none exists, falls back to ffmpeg.
    """
    # Try several likely names in FFMcalls
    candidate_names = [
        "extract_audio",
        "extractAudio",
        "extract_audio_track",
        "extractAudioTrack",
        "extract",
        "extractAudioFromVideo",
        "extract_audio_from_video",
        "extract_audio_from_media",
    ]

    for name in candidate_names:
        if hasattr(FFMcalls, name):
            fn = getattr(FFMcalls, name)
            # Try common signatures
            for args, kwargs in (
                ((input_media, output_mp3), {}),
                ((input_media,), {"output_audio": output_mp3}),
                ((input_media,), {"output_mp3": output_mp3}),
                ((input_media,), {"output_file": output_mp3}),
                ((input_media,), {}),
            ):
                try:
                    fn(*args, **kwargs)
                    return
                except TypeError:
                    continue

    # Fallback: run ffmpeg directly
    cmd = ["ffmpeg", "-y"]
    if os.environ.get("GF_FFMPEG_QUIET", "1") != "0":
        cmd += ["-hide_banner", "-loglevel", "error", "-nostats"]
    cmd += [
        "-i",
        input_media,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        output_mp3,
    ]
    LOG(f"Extracting audio via ffmpeg -> {output_mp3}")
    subprocess.run(cmd, check=True)


def _normalize_censor_ranges(
    ranges: list[tuple[float, float]],
    *,
    merge_gap: float = 0.05,   # 50 ms
    min_duration: float = 0.08 # 80 ms
) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent ranges and drop tiny ones.

    This protects against Whisper duplicate segments and zero/negative durations.
    """
    if not ranges:
        return []

    # Clean + sort
    cleaned = [(float(s), float(e)) for s, e in ranges if e > s]
    cleaned.sort(key=lambda x: x[0])

    merged: list[tuple[float, float]] = []
    for s, e in cleaned:
        if not merged:
            merged.append((s, e))
            continue

        ps, pe = merged[-1]
        if s <= pe + merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # Drop ultra-short ranges
    final = [(s, e) for s, e in merged if (e - s) >= min_duration]
    return final


def main(
    filepath: str,
    censor_mode: str,
    progress_cb=None,
    *,
    low_conf_threshold: float = 0.6,
    verify_models: list[str] | None = None,
    verify_pad: float = 0.30,
    debug: bool = False,
    no_subs: bool = False,
    subs_raw: bool = False,
) -> dict | None:
    start_time = time.time()
    timings: dict[str, float] = {}
    t0 = time.time()
    last_progress_line = ""
    log = log_line

    def _load_last_timings() -> dict[str, float]:
        try:
            if not os.path.isdir("reports"):
                return {}
            reports = [
                os.path.join("reports", f)
                for f in os.listdir("reports")
                if f.endswith(".report.json")
            ]
            if not reports:
                return {}
            reports.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            with open(reports[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("timings", {}) or {}
        except Exception:
            return {}

    expected_timings = _load_last_timings()
    default_expected = {
        "extract": 10.0,
        "transcribe": 90.0,
        "ranges": 5.0,
        "censor": 10.0,
        "mux": 10.0,
    }

    def _render_bar(pct: float, width: int = 20) -> str:
        pct = max(0.0, min(1.0, pct))
        filled = int(round(pct * width))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {int(pct*100):3d}%"

    def _print_progress(line: str) -> None:
        nonlocal last_progress_line
        last_progress_line = line
        # overwrite the current line in-place; cap width to avoid terminal wrap
        max_width = 72
        print("\r" + line[:max_width].ljust(max_width), end="", flush=True)

    def _stage_progress(stage: str, pct: float) -> None:
        if progress_cb:
            progress_cb(stage, pct)
        else:
            _print_progress(f"{stage:<8} {_render_bar(pct)}")

    def _run_stage(stage: str, pct_start: float, pct_end: float, fn):
        expected = float(expected_timings.get(stage, 0.0) or 0.0)
        if expected <= 0.0:
            expected = float(default_expected.get(stage, 0.0) or 0.0)
        stop_evt = threading.Event()

        def _ticker():
            if expected <= 0.0:
                return
            while not stop_evt.is_set():
                elapsed = time.time() - t_start
                frac = min(0.99, elapsed / expected) if expected > 0 else 0.0
                pct = pct_start + (pct_end - pct_start) * frac
                _stage_progress(stage, pct)
                time.sleep(0.1)

        t_start = time.time()
        thr = threading.Thread(target=_ticker, daemon=True)
        thr.start()
        try:
            return fn()
        finally:
            stop_evt.set()
            thr.join(timeout=0.2)
            _stage_progress(stage, pct_end)

    _cleanup_temp_outputs()

    checkFile(filepath)
    # Strict gatekeeper: input must exist and be a real file (fail fast before ffmpeg)
    if not os.path.exists(filepath):
        log(f"❌ Input file not found: {filepath}")
        log("Tip: include the full filename with extension (e.g., .mkv/.mp4) and use quotes if needed.")
        raise SystemExit(2)
    if not os.path.isfile(filepath):
        log(f"❌ Input path is not a file: {filepath}")
        raise SystemExit(2)

    target_words = processText.load_target_words("target_words.txt")

    _run_stage("extract", 0.00, 0.33, lambda: _extract_audio(filepath, "output.mp3"))
    timings["extract_audio_s"] = round(time.time() - t0, 3)
    _stage_progress("extract", 0.33)
    checkFile("output.mp3")

    # Produces output.json (segments)
    t0 = time.time()
    t0 = time.time()
    _run_stage("transcribe", 0.33, 0.66, lambda: whisperCalls.run_whisper_live("output.mp3", "output.json"))
    timings["transcribe_s"] = round(time.time() - t0, 3)
    _stage_progress("transcribe", 0.66)
    checkFile("output.json")

    segments = processText.load_segments("output.json")

    try:
        detected_language = whisperCalls.load_detected_language("output.json")
    except Exception:
        detected_language = None

    # Low-confidence verification: escalate to bigger models for uncertain hits.
    if verify_models is None:
        verify_models = ["small", "medium", "large"]
    verify_cache: dict[tuple[float, float, str], list[tuple[str, float]]] = {}
    verify_stats = {
        "low_conf_hits": 0,
        "model_attempts": {m: 0 for m in verify_models},
        "model_confirms": {m: 0 for m in verify_models},
    }
    exact_targets, prefix_targets = processText._parse_target_patterns(target_words)

    def _verify_low_confidence(tok: str, abs_start: float, abs_end: float) -> bool:
        clip_start = max(0.0, abs_start - verify_pad)
        clip_end = max(clip_start + 0.3, abs_end + verify_pad)

        os.makedirs("mini_audio", exist_ok=True)
        verify_stats["low_conf_hits"] += 1
        for model_name in verify_models:
            verify_stats["model_attempts"][model_name] = verify_stats["model_attempts"].get(model_name, 0) + 1
            cache_key = (round(clip_start, 3), round(clip_end, 3), model_name)
            tokens = verify_cache.get(cache_key)
            if tokens is None:
                mini_path = os.path.join(
                    "mini_audio",
                    f"verify_{int(time.time()*1000)}_{os.getpid()}_{model_name}.mp3",
                )
                try:
                    FFMcalls.make_mini_audio(
                        input_media=filepath,
                        start=clip_start,
                        end=clip_end,
                        output_path=mini_path,
                        quiet=True,
                    )
                    cfg = whisperCalls.TranscribeConfig(model_name=model_name, language=detected_language)
                    data = whisperCalls.transcribe_to_dict(mini_path, cfg=cfg)
                    tokens = []
                    for seg in data.get("segments", []):
                        for w in seg.get("words", []) or []:
                            ww = processText.remove_non_alpha(str(w.get("word", "")).lower())
                            if not ww:
                                continue
                            prob = float(w.get("probability", 0.0) or 0.0)
                            for t in ww.split():
                                tokens.append((t, prob))
                    verify_cache[cache_key] = tokens
                finally:
                    try:
                        if os.path.exists(mini_path):
                            os.remove(mini_path)
                    except Exception:
                        pass

            for t, prob in (tokens or []):
                if processText._token_matches(t, exact_targets, prefix_targets) and prob >= low_conf_threshold:
                    verify_stats["model_confirms"][model_name] = verify_stats["model_confirms"].get(model_name, 0) + 1
                    return True

        return False

    # 1) Identify which segments contain target words
    processText.mark_segments_with_targets(segments, target_words)

    # 2) Compute absolute time ranges to censor (in seconds)
    t0 = time.time()
    def _build_ranges():
        return processText.build_censor_ranges(
            segments=segments,
            target_words=target_words,
            pad=0.02,
            low_confidence_threshold=low_conf_threshold,
            verifier=_verify_low_confidence,
        )
    raw_ranges = _run_stage("ranges", 0.66, 0.75, _build_ranges)
    timings["build_ranges_s"] = round(time.time() - t0, 3)

    ranges = _normalize_censor_ranges(raw_ranges)
    duration_s = FFMcalls.get_audio_duration_seconds("output.mp3")
    ranges = FFMcalls.clamp_ranges_to_duration(ranges, duration_s)

    # Helpful debug when nothing gets censored despite target segments being flagged
    if len(ranges) == 0 and len(raw_ranges) == 0:
        flagged = sum(1 for s in segments if getattr(s, "contains_target_word", False))
        if flagged > 0:
            log(f"⚠ Debug: {flagged} segment(s) flagged, but 0 censor ranges built. Check transcription output + target word matching.")

    if duration_s > 0.0:
        log(f"Censor ranges: {len(raw_ranges)} → {len(ranges)} after normalization + clamp (duration={duration_s:.2f}s)")
    else:
        log(f"Censor ranges: {len(raw_ranges)} → {len(ranges)} after normalization")

        
    import csv

    def _format_srt_time(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        millis = int(round(seconds * 1000.0))
        hrs = millis // 3_600_000
        mins = (millis % 3_600_000) // 60_000
        secs = (millis % 60_000) // 1000
        ms = millis % 1000
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    def _mask_word(word: str, *, raw: bool) -> str:
        w = (word or "").strip()
        if not w:
            return ""
        if raw:
            return w
        return w[0] + ("-" * (len(w) - 1))

    def _write_srt(hits_abs: list[tuple[float, float, str]], srt_path: str, *, raw_words: bool) -> None:
        hits_abs.sort(key=lambda x: x[0])
        with open(srt_path, "w", encoding="utf-8") as f:
            for idx, (s, e, w) in enumerate(hits_abs, start=1):
                f.write(f"{idx}\n")
                f.write(f"{_format_srt_time(s)} --> {_format_srt_time(e)}\n")
                f.write(f"{_mask_word(w, raw=raw_words)}\n\n")

    def _log_muted_words(file: str, segments: list) -> tuple[str, int, str, list[tuple[float, float, str]]]:
        """Append muted-word hits to muted_words.csv.

        Preferred: word-level hits via segments[].words (from faster-whisper output.json).
        Fallback: if no word hits, we log flagged segments as coarse hits.

        Returns (csv_path, hit_count, granularity, hits_abs).
        """
        csv_file = "muted_words.csv"
        file_exists = os.path.isfile(csv_file)
        hits_abs: list[tuple[float, float, str]] = []

        hits = processText.extract_target_word_times(
            segments=segments,
            target_words=target_words,
            mini_audio_dir="mini_audio",  # ignored; kept for compatibility
            low_confidence_threshold=low_conf_threshold,
            verifier=_verify_low_confidence,
        )

        # Determine granularity for reporting
        granularity = "word" if len(hits) > 0 else "segment"

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["hit_type", "text", "start", "end", "granularity", "file"])

            if hits:
                for hit in hits:
                    seg_start = float(segments[hit.segment_number].start) if hit.segment_number is not None else 0.0
                    abs_start = max(0.0, float(hit.start) + seg_start)
                    abs_end = max(abs_start, float(hit.end) + seg_start)
                    writer.writerow(["word", hit.word, f"{abs_start:.3f}", f"{abs_end:.3f}", granularity, file])
                    hits_abs.append((abs_start, abs_end, hit.word))
            else:
                # Fallback: log the full segments that were flagged
                for seg in segments:
                    if getattr(seg, "contains_target_word", False):
                        text = str(getattr(seg, "text", "")).strip()
                        s = float(getattr(seg, "start", 0.0))
                        e = float(getattr(seg, "end", 0.0))
                        writer.writerow(["segment", text, f"{s:.3f}", f"{e:.3f}", granularity, file])
                        hits_abs.append((s, e, text))

        return (
            csv_file,
            (len(hits) if hits else sum(1 for seg in segments if getattr(seg, "contains_target_word", False))),
            granularity,
            hits_abs,
        )

    # Keep report() consistent with what we actually censor after normalization
    # Prefer the count of word-level hits when available; otherwise count flagged segments.
    processText.words_removed = 0  # will be set after logging

    csv_path, hit_count, granularity, hits_abs = _log_muted_words(filepath, segments)
    log(f"Logged muted hits ({granularity}, {hit_count}) -> {csv_path}")

    processText.words_removed = hit_count

    # 4) Build censored audio (mute by default, beep with -b)
    t0 = time.time()
    def _censor_audio():
        if not ranges:
            log("⚠ No valid censor ranges after normalization; copying original audio")
            shutil.copyfile("output.mp3", "output_censored.mp3")
        else:
            FFMcalls.censor_audio(
                ranges=ranges,
                input_file="output.mp3",
                output_file="output_censored.mp3",
                mode=censor_mode,
                beep_freq=1000,
                beep_volume=0.25,
                pad=0.0
            )
    _run_stage("censor", 0.75, 0.90, _censor_audio)
    timings["censor_audio_s"] = round(time.time() - t0, 3)
    checkFile("output_censored.mp3")

    # 5) Build censored-words subtitle file (SRT) and mux back into video
    # Default output name (caller may move/rename this when using --output/--out-dir)
    base, ext = os.path.splitext(filepath)
    output_video = f"{base}_censored{ext}"

    srt_path = None
    if not no_subs:
        srt_path = f"{base}_censored.srt"
        _write_srt(hits_abs, srt_path, raw_words=bool(subs_raw))
        log(f"Wrote censored words SRT -> {srt_path}")

    log(f"MUX VIDEO: {filepath}")
    log(f"OUTPUT VIDEO: {output_video}")
    t0 = time.time()
    _run_stage(
        "mux",
        0.90,
        1.0,
        lambda: FFMcalls.replace_audio_track(
            video_file=filepath,
            censored_audio="output_censored.mp3",
            original_audio="output.mp3",
            output_video_file=output_video,
            subtitle_file=srt_path,
        ),
    )
    timings["mux_s"] = round(time.time() - t0, 3)
    _stage_progress("mux", 1.0)
    if last_progress_line:
        print("")  # move past the progress line
    if progress_cb:
        progress_cb("mux", 1.0)

    # 6) Write per-file JSON report (app-ready interface)
    report = {
        "input_file": filepath,
        "output_file": output_video,
        "engine": "faster-whisper",
        "detected_language": detected_language,
        "censor_mode": censor_mode,
        "censor_granularity": granularity,
        "muted_hit_count": hit_count,
        "censor_ranges": [[float(s), float(e)] for (s, e) in ranges],
        "media_duration_s": round(duration_s, 3),
        "processing_time_s": round(time.time() - start_time, 3),
    }
    report["verify_stats"] = verify_stats
    report["timings"] = timings

    # Write report into reports/ directory (app-friendly)
    os.makedirs("reports", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_base = os.path.basename(os.path.splitext(output_video)[0])
    report_name = f"{report_base}.{stamp}.report.json"
    report_path = os.path.join("reports", report_name)

    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(f"Wrote report -> {report_path}")


    elapsed_time = time.time() - start_time
    processText.report(segments, elapsed_time)
    return report


def cli_main():
        log = log_line
        set_log_file(os.path.join("reports", "run.log"))
        # Always ensure we are running inside the project venv (auto-create if missing)
        SWhelper.ensure_runtime_ready(sys.argv)

        parser = argparse.ArgumentParser(description="Grandma Filter – speech-based audio censoring tool")

        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("filepath", nargs="?", help="Input video or audio file")
        source.add_argument("--dir", dest="directory", help="Process all supported media files in a directory (non-recursive)")

        parser.add_argument(
            "--output",
            dest="output_file",
            default=None,
            help="Output file path (single-file mode only). Overrides the default _censored naming.",
        )
        parser.add_argument(
            "--out-dir",
            dest="out_dir",
            default=None,
            help="Output directory (directory mode only). Defaults to the input directory.",
        )

        parser.add_argument("-b", "--beep", action="store_true", help="Use beep instead of mute for censored words")
        parser.add_argument(
            "--ext",
            dest="extensions",
            action="append",
            default=None,
            help="File extension to include (repeatable). Example: --ext mp4 --ext mov. Default: mp4,mov,mkv,avi",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be processed and outputs, but do not run ffmpeg/whisper",
        )
        parser.add_argument(
            "--no-skip-censored",
            action="store_true",
            help="Also process files that already end in _censored",
        )
        parser.add_argument(
            "--in-place",
            action="store_true",
            help="Replace original files with censored output (safe atomic replace)",
        )
        parser.add_argument(
            "--no-check",
            action="store_true",
            help="Skip the automatic dependency check/installer (not recommended)",
        )
        parser.add_argument(
            "--yes-install",
            action="store_true",
            help="Assume yes for system-wide installs during the auto-check (ffmpeg)",
        )
        parser.add_argument(
            "--low-conf",
            type=float,
            default=0.6,
            help="Low-confidence cutoff for target words; below this triggers verification (default: 0.6)",
        )
        parser.add_argument(
            "--verify-models",
            type=str,
            default="small,medium,large",
            help="Comma-separated models for low-confidence verification (default: small,medium,large)",
        )
        parser.add_argument(
            "--verify-pad",
            type=float,
            default=0.30,
            help="Seconds to pad around low-confidence word for verification (default: 0.30)",
        )
        parser.add_argument(
            "--no-subs",
            action="store_true",
            help="Do not mux the censored-words subtitle track",
        )
        parser.add_argument(
            "--subs-raw",
            action="store_true",
            help="Subtitle track uses the full censored word instead of masking (e.g., 'fuck' vs 'f---')",
        )

        args = parser.parse_args()

        # Validate flag combinations
        if args.output_file and args.directory:
            log("❌ --output is only valid in single-file mode (do not use with --dir)")
            raise SystemExit(2)

        if args.out_dir and not args.directory:
            log("❌ --out-dir is only valid with --dir (directory mode)")
            raise SystemExit(2)

        if args.in_place and (args.output_file or args.out_dir):
            log("❌ --in-place cannot be combined with --output or --out-dir")
            raise SystemExit(2)

        # Strict gatekeeper + auto-installer (unless --no-check)
        if not args.no_check:
            report = SWhelper.run_checks(install=True, assume_yes=bool(args.yes_install))
            if not report.required_ok:
                log("❌ Gatekeeper failed. Run: python SWhelper.py check --json")
                for it in report.items:
                    # Only print failing required items (keeps output readable)
                    if not it.ok and it.name in {
                        "env:venv",
                        "repo:root_contract",
                        "ffmpeg",
                        "torch:accel",
                        "import:FFMcalls",
                        "import:FFMcalls.make_mini_audio",
                        *{f"python:{m}" for m in getattr(SWhelper, "REQUIRED_MODULES", [])},
                    }:
                        log(f"❌ {it.name}: {it.details}")
                        if it.recommendation:
                            log(f"   -> {it.recommendation}")
                raise SystemExit(2)

        censor_mode = "beep" if args.beep else "mute"
        log(f"Censor mode: {censor_mode}")

        # Default extensions
        extensions = args.extensions or ["mp4", "mov", "mkv", "avi"]

        if args.directory:
            directory = args.directory
            if not os.path.isdir(directory):
                log(f"❌ Not a directory: {directory}")
                raise SystemExit(2)

            files = _iter_media_files(directory, extensions, skip_censored=(not args.no_skip_censored))

            if not files:
                log(f"No matching media files found in: {directory}")
                raise SystemExit(0)

            log(f"Found {len(files)} file(s) in {directory}")
            total_files = len(files)

            def _batch_bar(i: int) -> str:
                pct = i / total_files if total_files else 1.0
                width = 20
                filled = int(round(pct * width))
                return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {int(pct*100):3d}%"

            for idx, path in enumerate(files, start=1):
                base, ext = os.path.splitext(path)

                # Output location
                out_dir = args.out_dir or os.path.dirname(path)
                os.makedirs(out_dir, exist_ok=True)
                out_name = os.path.basename(base) + f"_censored{ext}"
                out_video = os.path.join(out_dir, out_name)

                log(f"\n[{idx}/{len(files)}] {path}")
                line = f"Batch {_batch_bar(idx-1)}"
                print("\r" + line[:72].ljust(72), end="", flush=True)
                log(f"    -> output: {out_video}")

                if args.dry_run:
                    continue

                try:
                    main(
                        path,
                        censor_mode,
                        low_conf_threshold=float(args.low_conf),
                        verify_models=[m.strip() for m in args.verify_models.split(",") if m.strip()],
                        verify_pad=float(args.verify_pad),
                        no_subs=bool(args.no_subs),
                        subs_raw=bool(args.subs_raw),
                    )
                    line = f"Batch {_batch_bar(idx)}"
                    print("\r" + line[:72].ljust(72), end="", flush=True)
                    print("")  # move past the progress line

                    # If output directory differs from input folder, move the default output into `out_video`
                    default_out = os.path.splitext(path)[0] + "_censored" + os.path.splitext(path)[1]
                    if out_video != default_out:
                        if os.path.exists(default_out):
                            os.replace(default_out, out_video)
                        else:
                            log(f"    ⚠ Expected output missing: {default_out}")


                    if args.in_place:
                        # Atomically replace original with censored output
                        if os.path.exists(out_video):
                            os.replace(out_video, path)
                            log(f"    ✔ Replaced in place: {path}")
                        else:
                            log(f"    ⚠ Expected output missing: {out_video}")
                except Exception as e:
                    # Keep going on errors in batch mode
                    log(f"❌ Failed: {path}")
                    log(f"   {type(e).__name__}: {e}")
                    continue

            raise SystemExit(0)

        # Single-file mode
        if args.filepath:
            base, ext = os.path.splitext(args.filepath)
            out_video = args.output_file or f"{base}_censored{ext}"

            if args.out_dir:
                log("❌ --out-dir is only valid with --dir (directory mode)")
                raise SystemExit(2)

            log(f"INPUT: {args.filepath}")
            log(f"OUTPUT: {out_video}")

            # Strict gatekeeper: input must exist before we call ffmpeg
            if not os.path.exists(args.filepath):
                log(f"❌ Input file not found: {args.filepath}")
                log("Tip: include the full filename with extension (e.g., .mkv/.mp4) and use quotes if needed.")
                raise SystemExit(2)
            if not os.path.isfile(args.filepath):
                log(f"❌ Input path is not a file: {args.filepath}")
                raise SystemExit(2)

            if args.dry_run:
                raise SystemExit(0)

            main(
                args.filepath,
                censor_mode,
                low_conf_threshold=float(args.low_conf),
                verify_models=[m.strip() for m in args.verify_models.split(",") if m.strip()],
                verify_pad=float(args.verify_pad),
                no_subs=bool(args.no_subs),
                subs_raw=bool(args.subs_raw),
            )

            # If the user specified --output, rename/move the produced default output into place
            if args.output_file:
                default_out = f"{base}_censored{ext}"
                if os.path.exists(default_out):
                    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
                    os.replace(default_out, args.output_file)
                    log(f"✔ Wrote output: {args.output_file}")
                else:
                    log(f"⚠ Expected output missing: {default_out}")

            if args.in_place:
                if os.path.exists(out_video):
                    os.replace(out_video, args.filepath)
                    log(f"✔ Replaced in place: {args.filepath}")
                else:
                    log(f"⚠ Expected output missing: {out_video}")

            raise SystemExit(0)

        parser.print_usage()
        raise SystemExit(1)

def _run_main():
    return cli_main()

if __name__ == "__main__":
    raise SystemExit(_run_main())
