import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


target_words: List[str] = []
filepath: str = ""  # used by prepare_mini_audio_and_word_jsons if caller doesn't pass a path
words_removed: int = 0
words_found: int = 0
word_hits: int = 0  # number of word-level target matches found
segments_flagged: int = 0  # number of segments that contained >=1 target


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    number: int = 0
    contains_target_word: bool = False
    words: List[Word] = None  # populated when transcription provides word timestamps

    def __post_init__(self):
        if self.words is None:
            self.words = []


@dataclass
class TargetWordTime:
    word: str
    start: float  # relative to the mini clip
    end: float    # relative to the mini clip
    segment_number: int
    probability: float = 0.0


def remove_non_alpha(text: str) -> str:
    """Normalize text for matching.

    - Lowercasing happens at call sites.
    - Replace non-alpha chars with spaces (not empty) so hyphenated words split.
    - Collapse repeated whitespace.
    """
    text = re.sub(r"[^a-zA-Z\s']+", " ", text)
    return " ".join(text.split())


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
        seg_obj = Segment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")),
            number=i,
        )

        # Prefer word-level timestamps when present (faster-whisper output)
        words = seg.get("words") or []
        for w in words:
            try:
                seg_obj.words.append(
                    Word(
                        word=str(w.get("word", "")),
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        probability=float(w.get("probability", 0.0) or 0.0),
                    )
                )
            except Exception:
                continue

        segments.append(seg_obj)
    return segments


def _parse_target_patterns(targets: Sequence[str]) -> tuple[set[str], list[str]]:
    """Return (exact_tokens, prefixes) for targets. Supports suffix '*' for prefix match."""
    exact: set[str] = set()
    prefixes: list[str] = []
    for t in targets:
        if not t:
            continue
        tt = remove_non_alpha(t.lower())
        if not tt:
            continue
        if t.strip().endswith("*"):
            prefixes.append(tt)
        else:
            exact.add(tt)
    return exact, prefixes


def _token_matches(tok: str, exact: set[str], prefixes: list[str]) -> bool:
    if tok in exact:
        return True
    for p in prefixes:
        if tok.startswith(p):
            return True
    return False


def mark_segments_with_targets(segments: List[Segment], targets: Sequence[str]) -> None:
    """Mark segments whose text contains any target word (case-insensitive, punctuation-stripped match).
    Supports wildcard suffix '*', e.g. 'fuck*' matches 'fucking' and 'motherfucker' tokens.
    """
    global segments_flagged
    segments_flagged = 0

    # Normalize target words once
    exact, prefixes = _parse_target_patterns(targets)

    for seg in segments:
        seg.contains_target_word = False

        # Normalize segment text: lowercase + strip punctuation
        seg_text_norm = remove_non_alpha((seg.text or "").lower())

        # Token-based match to avoid substring weirdness (e.g., 'ass' in 'pass')
        tokens = seg_text_norm.split()

        for tok in tokens:
            if _token_matches(tok, exact, prefixes):
                seg.contains_target_word = True
                segments_flagged += 1
                break


def prepare_mini_audio_and_word_jsons(filepath: str,
                                     segments: List[Segment],
                                     target_words: Sequence[str],
                                     mini_audio_dir: str = "mini_audio") -> None:
    # Word-level timestamps now come from the main transcription JSON (segments[].words).
    # This function is retained only for backwards compatibility.
    return


def extract_target_word_times(
    segments: List[Segment],
    target_words: Sequence[str],
    mini_audio_dir: str = "mini_audio",
    *,
    low_confidence_threshold: float = 0.6,
    verifier=None,
) -> List[TargetWordTime]:
    """
    Return word-level hits.
    Uses precise word timestamps when available.
    Falls back to proportional timing from seg.text when not.
    """
    _ = mini_audio_dir

    exact, prefixes = _parse_target_patterns(target_words)
    hits: List[TargetWordTime] = []

    for seg in segments:
        seg_duration = max(0.001, seg.end - seg.start)

        # -------------------------------------------------
        # Preferred path: real word timestamps exist
        # -------------------------------------------------
        if seg.words:
            for w in seg.words:
                ww_norm = remove_non_alpha(w.word.lower())
                if not ww_norm:
                    continue

                for tok in ww_norm.split():
                    if _token_matches(tok, exact, prefixes):
                        # Optional verification for low-confidence hits
                        if verifier is not None and w.probability < low_confidence_threshold:
                            abs_start = float(w.start)
                            abs_end = float(w.end)
                            if not verifier(tok, abs_start, abs_end):
                                continue
                        # faster-whisper word timestamps are absolute (global) seconds.
                        # Convert to segment-relative so build_censor_ranges can add the segment offset once.
                        start_rel = max(0.0, float(w.start) - float(seg.start) - 0.02)
                        end_rel = max(start_rel, float(w.end) - float(seg.start))
                        hits.append(
                            TargetWordTime(
                                word=tok,
                                start=start_rel,
                                end=end_rel,
                                segment_number=seg.number,
                                probability=float(w.probability),
                            )
                        )
            continue

        # -------------------------------------------------
        # Fallback path: approximate timing from seg.text
        # -------------------------------------------------
        text_norm = remove_non_alpha((seg.text or "").lower())
        tokens = text_norm.split()
        if not tokens:
            continue

        seconds_per_word = seg_duration / len(tokens)

        for i, tok in enumerate(tokens):
            if _token_matches(tok, exact, prefixes):
                if verifier is not None and low_confidence_threshold > 0.0:
                    if not verifier(tok, float(start), float(end)):
                        continue
                start = seg.start + i * seconds_per_word
                end = start + seconds_per_word

                hits.append(
                    TargetWordTime(
                        word=tok,
                        start=start - seg.start,  # relative to segment
                        end=end - seg.start,
                        segment_number=seg.number,
                        probability=0.0,
                    )
                )

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
                        merge_gap: float = 0.02,
                        *,
                        low_confidence_threshold: float = 0.6,
                        verifier=None) -> List[Tuple[float, float]]:
    """Return absolute (start,end) ranges in seconds for the full audio.

    Preferred path: use word timestamps present in the transcription JSON (segments[].words).
    Fallback path: if no word timings can be found (e.g., missing JSONs), censor the entire flagged segment(s).
    """
    global words_removed, word_hits

    hits = extract_target_word_times(
        segments,
        target_words,
        mini_audio_dir=mini_audio_dir,
        low_confidence_threshold=low_confidence_threshold,
        verifier=verifier,
    )
    word_hits = len(hits)

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
    # This usually means word timestamps were not available in the transcription output.
    word_hits = 0
    flagged = 0

    for seg in segments:
        if not seg.contains_target_word:
            continue

        flagged += 1

        s = max(0.0, float(seg.start) - float(pad))
        e = max(s, float(seg.end) + float(pad))
        ranges.append((s, e))

    if flagged > 0:
        print(f"⚠ Fallback censoring: 0 word-level hits found; censoring {flagged} full segment(s).")

    merged = merge_ranges(ranges, gap=merge_gap)
    words_removed = len(merged)
    return merged


def report(segments: Sequence[Segment], elapsed_seconds: float) -> None:
    minutes, seconds = divmod(elapsed_seconds, 60)
    print("--------------Report--------------")
    print(f"Target words: {len(target_words)}")
    print(f"Segments: {len(segments)}")
    print(f"Segments flagged: {segments_flagged}")
    print(f"Word-level hits: {word_hits}")
    print(f"Ranges to censor: {words_removed}")
    print(f"Time elapsed: {int(minutes)} minutes and {seconds:.2f} seconds")
    print("--------------End Report--------------")
