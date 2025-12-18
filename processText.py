import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import FFMcalls
import whisperCalls

# NOTE: this module now focuses on *finding ranges* to censor, not performing censoring.


target_words: List[str] = []
filepath: str = ""  # used by prepare_mini_audio_and_word_jsons if caller doesn't pass a path
words_removed: int = 0
words_found: int = 0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    number: int = 0
    contains_target_word: bool = False


@dataclass
class TargetWordTime:
    word: str
    start: float  # relative to the mini clip
    end: float    # relative to the mini clip
    segment_number: int


def remove_non_alpha(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s']", "", text)


def load_target_words(path: str = "target_words.txt") -> List[str]:
    global target_words
    target_words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            w = line.strip().lower()
            if w:
                target_words.append(w)
    return target_words


def load_segments(json_path: str) -> List[Segment]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: List[Segment] = []
    for i, seg in enumerate(data.get("segments", [])):
        segments.append(
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")),
                number=i
            )
        )
    return segments


def mark_segments_with_targets(segments: List[Segment], targets: Sequence[str]) -> None:
    """Mark segments whose text contains any target word (case-insensitive substring match)."""
    global words_found
    words_found = 0
    targets_l = [t.lower() for t in targets]
    for seg in segments:
        seg.contains_target_word = False
        seg_text = (seg.text or "").lower()
        for t in targets_l:
            if t and t in seg_text:
                seg.contains_target_word = True
                words_found += 1
                break


def prepare_mini_audio_and_word_jsons(filepath: str,
                                     segments: List[Segment],
                                     target_words: Sequence[str],
                                     mini_audio_dir: str = "mini_audio") -> None:
    """For each segment flagged as containing a target word:

    - extract a mini audio clip into mini_audio/<segment_number>.mp3
    - run word-level transcription on that mini clip producing mini_audio/<segment_number>.json

    This keeps FFmpeg / Whisper calls *outside* of the "range building" functions.
    """
    os.makedirs(mini_audio_dir, exist_ok=True)

    for seg in segments:
        if not seg.contains_target_word:
            continue

        out_mp3 = os.path.join(mini_audio_dir, f"{seg.number}.mp3")
        # IMPORTANT: pass output_path as a keyword so we don't accidentally fill the `duration` parameter
        FFMcalls.make_mini_audio(
            input_media=filepath,
            start=seg.start,
            end=seg.end,
            output_path=out_mp3,
        )

        # Produces JSON with word-level timestamps
        whisperCalls.get_word_level_transcription(out_mp3, seg.number)


def extract_target_word_times(segments: List[Segment],
                              target_words: Sequence[str],
                              mini_audio_dir: str = "mini_audio") -> List[TargetWordTime]:
    """Read word-level json for each flagged segment and return relative word hit times."""
    targets = {remove_non_alpha(w.lower()) for w in target_words}

    hits: List[TargetWordTime] = []
    for seg in segments:
        if not seg.contains_target_word:
            continue

        json_path = os.path.join(mini_audio_dir, f"{seg.number}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for s in data.get("segments", []):
            for w in s.get("words", []):
                ww = remove_non_alpha(str(w.get("word", "")).lower())
                if ww in targets:
                    start = float(w.get("start", 0.0)) - 0.02  # tiny lead-in
                    end = float(w.get("end", start))
                    hits.append(TargetWordTime(word=ww, start=start, end=end, segment_number=seg.number))
    return hits


def merge_ranges(ranges: Iterable[Tuple[float, float]], gap: float = 0.0) -> List[Tuple[float, float]]:
    """Merge overlapping/nearby ranges."""
    ranges = sorted([(float(s), float(e)) for s, e in ranges], key=lambda x: x[0])
    if not ranges:
        return []

    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def build_censor_ranges(segments: List[Segment],
                        target_words: Sequence[str],
                        mini_audio_dir: str = "mini_audio",
                        pad: float = 0.0,
                        merge_gap: float = 0.02) -> List[Tuple[float, float]]:
    """Return absolute (start,end) ranges in seconds for the full audio.

    Preferred path: use per-segment WhisperX word JSONs to get precise word timings.
    Fallback path: if no word timings can be found (e.g., missing JSONs), censor the entire flagged segment(s).
    """
    global words_removed

    hits = extract_target_word_times(segments, target_words, mini_audio_dir=mini_audio_dir)

    ranges: List[Tuple[float, float]] = []

    if hits:
        # Precise ranges from word-level timings
        for hit in hits:
            offset = segments[hit.segment_number].start
            s = max(0.0, hit.start + offset - float(pad))
            e = max(s, hit.end + offset + float(pad))
            ranges.append((s, e))

        merged = merge_ranges(ranges, gap=merge_gap)
        words_removed = len(merged)
        return merged

    # Fallback: no word-level hits found.
    # This usually means WhisperX JSONs are missing or didn't include word timing info.
    missing_json = 0
    flagged = 0

    for seg in segments:
        if not seg.contains_target_word:
            continue

        flagged += 1
        json_path = os.path.join(mini_audio_dir, f"{seg.number}.json")
        if not os.path.exists(json_path):
            missing_json += 1

        s = max(0.0, float(seg.start) - float(pad))
        e = max(s, float(seg.end) + float(pad))
        ranges.append((s, e))

    if flagged > 0:
        print(f"⚠ Fallback censoring: 0 word-level hits found; censoring {flagged} full segment(s). Missing JSONs: {missing_json}.")

    merged = merge_ranges(ranges, gap=merge_gap)
    words_removed = len(merged)
    return merged


def report(segments: Sequence[Segment], elapsed_seconds: float) -> None:
    minutes, seconds = divmod(elapsed_seconds, 60)
    print("--------------Report--------------")
    print(f"Target words: {len(target_words)}")
    print(f"Segments: {len(segments)}")
    print(f"Words found: {words_found}")
    print(f"Ranges to censor: {words_removed}")
    print(f"Time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds")
    print("--------------End Report--------------")
