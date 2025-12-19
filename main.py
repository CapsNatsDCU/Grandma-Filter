import argparse
import os
import shutil
import subprocess
import time
from typing import Iterable

import FFMcalls
import processText
import SWhelper
import whisperCalls


def checkFile(path: str) -> None:
    if os.path.exists(path):
        print(f"The file '{path}' exists.")
    else:
        print(f"The file '{path}' does not exist.")


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
        output_mp3,
    ]
    print(f"Extracting audio via ffmpeg -> {output_mp3}")
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


def main(filepath: str, censor_mode: str) -> None:
    start_time = time.time()

    _cleanup_temp_outputs()

    checkFile(filepath)
    # Strict gatekeeper: input must exist and be a real file (fail fast before ffmpeg)
    if not os.path.exists(filepath):
        print(f"❌ Input file not found: {filepath}")
        print("Tip: include the full filename with extension (e.g., .mkv/.mp4) and use quotes if needed.")
        raise SystemExit(2)
    if not os.path.isfile(filepath):
        print(f"❌ Input path is not a file: {filepath}")
        raise SystemExit(2)

    target_words = processText.load_target_words("target_words.txt")

    _extract_audio(filepath, "output.mp3")
    checkFile("output.mp3")

    # Produces output.json (segments)
    whisperCalls.run_whisper_live("output.mp3", "output.json")
    checkFile("output.json")

    segments = processText.load_segments("output.json")

    # 1) Identify which segments contain target words
    processText.mark_segments_with_targets(segments, target_words)

    # 2) Compute absolute time ranges to censor (in seconds)
    raw_ranges = processText.build_censor_ranges(
        segments=segments,
        target_words=target_words,
        pad=0.02
    )

    ranges = _normalize_censor_ranges(raw_ranges)
    duration_s = FFMcalls.get_audio_duration_seconds("output.mp3")
    ranges = FFMcalls.clamp_ranges_to_duration(ranges, duration_s)

    # Helpful debug when nothing gets censored despite target segments being flagged
    if len(ranges) == 0 and len(raw_ranges) == 0:
        flagged = sum(1 for s in segments if getattr(s, "contains_target_word", False))
        if flagged > 0:
            print(f"⚠ Debug: {flagged} segment(s) flagged, but 0 censor ranges built. Check transcription output + target word matching.")

    if duration_s > 0.0:
        print(f"Censor ranges: {len(raw_ranges)} → {len(ranges)} after normalization + clamp (duration={duration_s:.2f}s)")
    else:
        print(f"Censor ranges: {len(raw_ranges)} → {len(ranges)} after normalization")

        
    import csv

    def _log_muted_words(file: str, segments: list) -> tuple[str, int, str]:
        """Append muted-word hits to muted_words.csv.

        Preferred: word-level hits via segments[].words (from faster-whisper output.json).
        Fallback: if no word hits, we log flagged segments as coarse hits.

        Returns (csv_path, hit_count, granularity).
        """
        csv_file = "muted_words.csv"
        file_exists = os.path.isfile(csv_file)

        hits = processText.extract_target_word_times(
            segments=segments,
            target_words=target_words,
            mini_audio_dir="mini_audio",  # ignored; kept for compatibility
        )

        # Determine granularity for reporting
        granularity = "word" if len(hits) > 0 else "segment"

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["file", "hit_type", "text", "start", "end", "granularity"])

            if hits:
                for hit in hits:
                    abs_start = max(0.0, float(hit.start))
                    abs_end = max(abs_start, float(hit.end))
                    writer.writerow([file, "word", hit.word, f"{abs_start:.3f}", f"{abs_end:.3f}", granularity])
            else:
                # Fallback: log the full segments that were flagged
                for seg in segments:
                    if getattr(seg, "contains_target_word", False):
                        text = str(getattr(seg, "text", "")).strip()
                        s = float(getattr(seg, "start", 0.0))
                        e = float(getattr(seg, "end", 0.0))
                        writer.writerow([file, "segment", text, f"{s:.3f}", f"{e:.3f}", granularity])

        return csv_file, (len(hits) if hits else sum(1 for seg in segments if getattr(seg, "contains_target_word", False))), granularity

    # Keep report() consistent with what we actually censor after normalization
    # Prefer the count of word-level hits when available; otherwise count flagged segments.
    processText.words_removed = 0  # will be set after logging

    csv_path, hit_count, granularity = _log_muted_words(filepath, segments)
    print(f"Logged muted hits ({granularity}, {hit_count}) -> {csv_path}")

    processText.words_removed = hit_count

    # 4) Build censored audio (mute by default, beep with -b)
    if not ranges:
        print("⚠ No valid censor ranges after normalization; copying original audio")
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
    checkFile("output_censored.mp3")

    # 5) Mux censored + original into output video
    # Default output name (caller may move/rename this when using --output/--out-dir)
    base, ext = os.path.splitext(filepath)
    output_video = f"{base}_censored{ext}"

    print("MUX VIDEO:", filepath)
    print("OUTPUT VIDEO:", output_video)
    FFMcalls.replace_audio_track(
        video_file=filepath,
        censored_audio="output_censored.mp3",
        original_audio="output.mp3",
        output_video_file=output_video,
    )

    # DEBUG: print full transcript again right before report
    print("\n===== TRANSCRIPT BEFORE REPORT (DEBUG) =====")
    for seg in segments:
        txt = str(getattr(seg, "text", "")).strip()
        if txt:
            print(txt)
    print("===== END TRANSCRIPT =====\n")

    # 6) Write per-file JSON report (app-ready interface)
    try:
        detected_language = whisperCalls.load_detected_language("output.json")
    except Exception:
        detected_language = None

    report = {
        "input_file": filepath,
        "output_file": output_video,
        "engine": "faster-whisper",
        "detected_language": detected_language,
        "censor_mode": censor_mode,
        "censor_granularity": granularity,
        "muted_hit_count": hit_count,
        "censor_ranges": [[float(s), float(e)] for (s, e) in ranges],
        "processing_time_s": round(time.time() - start_time, 3),
    }

    # Write report into reports/ directory (app-friendly)
    os.makedirs("reports", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_base = os.path.basename(os.path.splitext(output_video)[0])
    report_name = f"{report_base}.{stamp}.report.json"
    report_path = os.path.join("reports", report_name)

    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote report -> {report_path}")

    elapsed_time = time.time() - start_time
    processText.report(segments, elapsed_time)


if __name__ == "__main__":
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

    args = parser.parse_args()

    # Validate flag combinations
    if args.output_file and args.directory:
        print("❌ --output is only valid in single-file mode (do not use with --dir)")
        raise SystemExit(2)

    if args.out_dir and not args.directory:
        print("❌ --out-dir is only valid with --dir (directory mode)")
        raise SystemExit(2)

    if args.in_place and (args.output_file or args.out_dir):
        print("❌ --in-place cannot be combined with --output or --out-dir")
        raise SystemExit(2)

    # Strict gatekeeper (updated SWhelper)
    report = SWhelper.run_checks(install=False)
    if not report.required_ok:
        print("❌ Gatekeeper failed. Run: python SWhelper.py check --json")
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
                print(f"❌ {it.name}: {it.details}")
                if it.recommendation:
                    print(f"   -> {it.recommendation}")
        raise SystemExit(2)

    censor_mode = "beep" if args.beep else "mute"
    print(f"Censor mode: {censor_mode}")

    # Default extensions
    extensions = args.extensions or ["mp4", "mov", "mkv", "avi"]

    if args.directory:
        directory = args.directory
        if not os.path.isdir(directory):
            print(f"❌ Not a directory: {directory}")
            raise SystemExit(2)

        files = _iter_media_files(directory, extensions, skip_censored=(not args.no_skip_censored))

        if not files:
            print(f"No matching media files found in: {directory}")
            raise SystemExit(0)

        print(f"Found {len(files)} file(s) in {directory}")

        for idx, path in enumerate(files, start=1):
            base, ext = os.path.splitext(path)

            # Output location
            out_dir = args.out_dir or os.path.dirname(path)
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.basename(base) + f"_censored{ext}"
            out_video = os.path.join(out_dir, out_name)

            print(f"\n[{idx}/{len(files)}] {path}")
            print(f"    -> output: {out_video}")

            if args.dry_run:
                continue

            try:
                main(path, censor_mode)

                # If output directory differs from input folder, move the default output into `out_video`
                default_out = os.path.splitext(path)[0] + "_censored" + os.path.splitext(path)[1]
                if out_video != default_out:
                    if os.path.exists(default_out):
                        os.replace(default_out, out_video)
                    else:
                        print(f"    ⚠ Expected output missing: {default_out}")


                if args.in_place:
                    # Atomically replace original with censored output
                    if os.path.exists(out_video):
                        os.replace(out_video, path)
                        print(f"    ✔ Replaced in place: {path}")
                    else:
                        print(f"    ⚠ Expected output missing: {out_video}")
            except Exception as e:
                # Keep going on errors in batch mode
                print(f"❌ Failed: {path}")
                print(f"   {type(e).__name__}: {e}")
                continue

        raise SystemExit(0)

    # Single-file mode
    if args.filepath:
        base, ext = os.path.splitext(args.filepath)
        out_video = args.output_file or f"{base}_censored{ext}"

        if args.out_dir:
            print("❌ --out-dir is only valid with --dir (directory mode)")
            raise SystemExit(2)

        print("INPUT:", args.filepath)
        print("OUTPUT:", out_video) #AHHHH

        # Strict gatekeeper: input must exist before we call ffmpeg
        if not os.path.exists(args.filepath):
            print(f"❌ Input file not found: {args.filepath}")
            print("Tip: include the full filename with extension (e.g., .mkv/.mp4) and use quotes if needed.")
            raise SystemExit(2)
        if not os.path.isfile(args.filepath):
            print(f"❌ Input path is not a file: {args.filepath}")
            raise SystemExit(2)

        if args.dry_run:
            raise SystemExit(0)

        main(args.filepath, censor_mode)        

        # If the user specified --output, rename/move the produced default output into place
        if args.output_file:
            default_out = f"{base}_censored{ext}"
            if os.path.exists(default_out):
                os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
                os.replace(default_out, args.output_file)
                print(f"✔ Wrote output: {args.output_file}")
            else:
                print(f"⚠ Expected output missing: {default_out}")

        if args.in_place:
            if os.path.exists(out_video):
                os.replace(out_video, args.filepath)
                print(f"✔ Replaced in place: {args.filepath}")
            else:
                print(f"⚠ Expected output missing: {out_video}")

        raise SystemExit(0)

    parser.print_usage()
    raise SystemExit(1)
