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

    # Keep filenames consistent with your existing workflow
    processText.filepath = filepath  # still used for mini-audio extraction offsets
    target_words = processText.load_target_words("target_words.txt")

    _extract_audio(filepath, "output.mp3")
    checkFile("output.mp3")

    # Produces output.json (segments)
    whisperCalls.run_whisper_live()
    checkFile("output.json")

    segments = processText.load_segments("output.json")

    # 1) Identify which segments contain target words
    processText.mark_segments_with_targets(segments, target_words)

    # 2) For those segments, extract mini clips and run word-level transcription
    processText.prepare_mini_audio_and_word_jsons(
        filepath=filepath,
        segments=segments,
        target_words=target_words,
        mini_audio_dir="mini_audio"
    )

    # 3) Compute absolute time ranges to censor (in seconds)
    raw_ranges = processText.build_censor_ranges(
        segments=segments,
        target_words=target_words,
        mini_audio_dir="mini_audio",
        pad=0.02
    )

    ranges = _normalize_censor_ranges(raw_ranges)

    # Keep report() consistent with what we actually censor after normalization
    processText.words_removed = len(ranges)

    # Helpful debug when nothing gets censored despite target segments being flagged
    if len(ranges) == 0 and len(raw_ranges) == 0 and getattr(processText, "words_found", 0) > 0:
        flagged = sum(1 for s in segments if getattr(s, "contains_target_word", False))
        mini_mp3 = 0
        mini_json = 0
        if os.path.isdir("mini_audio"):
            mini_mp3 = sum(1 for n in os.listdir("mini_audio") if n.endswith(".mp3"))
            mini_json = sum(1 for n in os.listdir("mini_audio") if n.endswith(".json"))
        print(f"⚠ Debug: {flagged} segment(s) flagged, but 0 censor ranges built. mini_audio has {mini_mp3} mp3 and {mini_json} json.")
        if mini_json == 0:
            print("   -> This usually means WhisperX did not produce word-level JSONs. Check WhisperX install / CLI availability and errors above.")

    print(f"Censor ranges: {len(raw_ranges)} → {len(ranges)} after normalization")

    import csv

    def _log_muted_words(file: str, segments: list, censor_ranges: list[tuple[float, float]]) -> None:
        """Append muted-word hits to muted_words.csv.

        We *do not* rely on `Segment.words` (it doesn't exist in the current dataclass).
        Instead we re-read the per-segment WhisperX word JSONs via `processText.extract_target_word_times`.
        """
        csv_file = "muted_words.csv"
        file_exists = os.path.isfile(csv_file)

        # Pull word hits (relative to each mini clip) and convert to absolute (relative to full audio)
        hits = processText.extract_target_word_times(
            segments=segments,
            target_words=target_words,
            mini_audio_dir="mini_audio",
        )

        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["file", "word", "start", "end"])

            for hit in hits:
                try:
                    offset = float(segments[hit.segment_number].start)
                except Exception:
                    offset = 0.0

                abs_start = max(0.0, float(hit.start) + offset)
                abs_end = max(abs_start, float(hit.end) + offset)
                writer.writerow([file, hit.word, f"{abs_start:.3f}", f"{abs_end:.3f}"])

    _log_muted_words(filepath, segments, ranges)
    print("Logged muted words -> muted_words.csv")

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

    SWhelper.checkSoftware()

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
