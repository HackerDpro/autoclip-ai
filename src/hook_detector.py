"""
Hook detection module for AutoClip AI.

Analyzes a full transcript and identifies the most viral-worthy segments
using Gemini 2.0 Flash, enforcing a strict JSON schema via Pydantic so the
rest of the pipeline can trust the output structurally.
"""

import json
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator

from config import (
    logger,
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    DEFAULT_NUM_CLIPS,
    MIN_CLIP_DURATION_SEC,
    MAX_CLIP_DURATION_SEC,
)
from transcriber import TranscriptionResult


MAX_WORDS_PER_HOOK_CHUNK = 2500  # keeps us safely under the 8000 TPM limit
                                 # even with timestamp annotation overhead


class HookDetectionError(Exception):
    """Raised when hook detection fails or returns invalid data."""
    pass


class ViralSegment(BaseModel):
    """A single candidate viral clip identified from the transcript."""
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    headline: str = Field(..., min_length=3, max_length=120)
    hook_reason: str = Field(..., min_length=3)
    virality_score: int = Field(..., ge=1, le=10)

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_time")
        if start is not None and v <= start:
            raise ValueError("end_time must be greater than start_time")
        return v

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ViralSegmentList(BaseModel):
    """Wrapper for the full list of detected segments (used for schema validation)."""
    segments: List[ViralSegment]


SYSTEM_PROMPT_TEMPLATE = """You are a viral short-form video strategist. You analyze podcast/video transcripts and identify the exact moments that would perform best as standalone 9:16 vertical clips on TikTok, YouTube Shorts, and Instagram Reels.

RULES:
1. Select exactly {num_clips} segments from the transcript below.
2. Each segment MUST be between {min_duration} and {max_duration} seconds long (end_time - start_time).
3. Segments must NOT overlap with each other.
4. Prioritize moments with: strong emotional hooks, surprising claims, concise standalone stories, controversial or bold statements, clear payoffs/punchlines, or practical high-value advice that makes sense without the rest of the video.
5. AVOID segments that are mid-thought, require earlier context to understand, or are just setup without a payoff.
6. start_time and end_time MUST correspond to actual timestamps in the provided transcript — do not invent times outside the transcript's duration.
7. "headline" should be a short, punchy, clickable title for the clip (max 10 words).
8. "hook_reason" should briefly explain WHY this moment is compelling (1 sentence).
9. "virality_score" is your honest 1-10 rating of how likely this clip is to perform well standalone.

Respond ONLY with valid JSON matching this exact schema, no markdown formatting, no extra commentary:
{{
  "segments": [
    {{
      "start_time": float,
      "end_time": float,
      "headline": string,
      "hook_reason": string,
      "virality_score": integer
    }}
  ]
}}

TRANSCRIPT (word-level timestamps included as [word @ start-end]):
{transcript_text}
"""


def _build_timestamped_transcript(transcript: TranscriptionResult) -> str:
    """
    Build a compact, timestamp-annotated version of the transcript so the
    LLM can accurately pick start/end times instead of guessing from plain
    text. Groups words into readable chunks rather than one-per-line to
    keep token usage reasonable.
    """
    if not transcript.words:
        # Fallback if word-level data wasn't available for some reason.
        return transcript.text

    lines = []
    chunk_words = []
    chunk_start = transcript.words[0].start

    for i, w in enumerate(transcript.words):
        chunk_words.append(w.word)
        is_last = i == len(transcript.words) - 1
        if len(chunk_words) >= 15 or is_last:
            chunk_end = w.end
            text = " ".join(chunk_words).strip()
            lines.append(f"[{chunk_start:.1f}-{chunk_end:.1f}] {text}")
            chunk_words = []
            if not is_last:
                chunk_start = transcript.words[i + 1].start

    return "\n".join(lines)


def _call_groq_llm(prompt: str, max_retries: int = 3) -> str:
    """
    Send the prompt to Groq's LLM endpoint and return the raw text response.
    Uses JSON mode to force structurally valid output. Retries on transient
    server errors, but fails fast on payload-too-large errors since retrying
    an oversized request never helps.
    """
    if not GROQ_API_KEY:
        raise HookDetectionError("GROQ_API_KEY is not set. Check your .env file.")

    try:
        from groq import Groq
    except ImportError as e:
        raise HookDetectionError(
            "The 'groq' package is not installed. Run: pip install groq"
        ) from e

    client = Groq(api_key=GROQ_API_KEY)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You respond only with valid JSON, no markdown formatting, no commentary."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )

            content = response.choices[0].message.content
            if not content:
                raise HookDetectionError("Groq returned an empty response.")

            return content

        except Exception as e:
            error_str = str(e)
            if "413" in error_str or "too large" in error_str.lower():
                raise HookDetectionError(
                    f"Request too large for the model's token limit. "
                    f"This should not happen with chunking enabled — "
                    f"check MAX_WORDS_PER_HOOK_CHUNK. Error: {e}"
                ) from e

            last_error = e
            wait_time = min(attempt * 3, 15)
            logger.warning(
                f"Groq LLM error (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise HookDetectionError(
        f"Groq LLM failed after {max_retries} attempts. Last error: {last_error}"
    )


def _split_transcript_into_chunks(
    transcript: TranscriptionResult,
    max_words: int = MAX_WORDS_PER_HOOK_CHUNK,
) -> List[TranscriptionResult]:
    """
    Split a long transcript into smaller TranscriptionResult chunks by word
    count, preserving word-level timestamps. Used so hook detection can run
    on very long videos (podcasts, lectures) without exceeding LLM token
    limits in a single request.
    """
    if len(transcript.words) <= max_words:
        return [transcript]

    chunks: List[TranscriptionResult] = []
    words = transcript.words

    for i in range(0, len(words), max_words):
        chunk_words = words[i : i + max_words]
        if not chunk_words:
            continue
        chunk_text = " ".join(w.word for w in chunk_words)
        chunk_duration = chunk_words[-1].end - chunk_words[0].start
        chunks.append(
            TranscriptionResult(
                text=chunk_text,
                words=chunk_words,
                duration=chunk_duration,
            )
        )

    logger.info(
        f"Transcript has {len(words)} words, split into {len(chunks)} chunks "
        f"of ~{max_words} words each for hook detection"
    )
    return chunks


def _detect_hooks_single_chunk(
    transcript: TranscriptionResult,
    num_clips: int,
) -> List[ViralSegment]:
    """
    Run hook detection on a single transcript chunk that's guaranteed to fit
    within token limits. This is the original single-request logic, now
    used as a building block for the chunked version.
    """
    timestamped_text = _build_timestamped_transcript(transcript)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        num_clips=num_clips,
        min_duration=MIN_CLIP_DURATION_SEC,
        max_duration=MAX_CLIP_DURATION_SEC,
        transcript_text=timestamped_text,
    )

    raw_response = _call_groq_llm(prompt)
    transcript_end_time = (
        transcript.words[-1].end if transcript.words else transcript.duration
    )
    return _parse_and_validate(raw_response, transcript_end_time)


def _parse_and_validate(raw_json: str, transcript_duration: float) -> List[ViralSegment]:
    """
    Parse the raw JSON string from Gemini and validate it against the
    ViralSegment schema. Clamps any out-of-bounds timestamps and drops
    invalid segments rather than crashing the whole pipeline over one
    bad entry.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise HookDetectionError(f"Gemini response was not valid JSON: {e}\nRaw: {raw_json[:500]}")

    raw_segments = data.get("segments", data if isinstance(data, list) else [])

    valid_segments: List[ViralSegment] = []
    for i, seg in enumerate(raw_segments):
        try:
            segment = ViralSegment(**seg)
        except Exception as e:
            logger.warning(f"Dropping invalid segment #{i}: {e}")
            continue

        if segment.end_time > transcript_duration:
            logger.warning(
                f"Segment #{i} end_time ({segment.end_time}) exceeds transcript "
                f"duration ({transcript_duration}), clamping."
            )
            segment.end_time = transcript_duration

        if segment.start_time >= segment.end_time:
            logger.warning(f"Dropping segment #{i} after clamping — invalid duration.")
            continue

        valid_segments.append(segment)

    return valid_segments


def _remove_overlaps(segments: List[ViralSegment]) -> List[ViralSegment]:
    """
    Sort segments by start_time and discard any that overlap with a
    previously kept segment, keeping the higher virality_score one.
    """
    sorted_segments = sorted(segments, key=lambda s: s.start_time)
    kept: List[ViralSegment] = []

    for seg in sorted_segments:
        overlaps_with = None
        for i, k in enumerate(kept):
            if seg.start_time < k.end_time and seg.end_time > k.start_time:
                overlaps_with = i
                break

        if overlaps_with is None:
            kept.append(seg)
        else:
            if seg.virality_score > kept[overlaps_with].virality_score:
                kept[overlaps_with] = seg

    return sorted(kept, key=lambda s: s.virality_score, reverse=True)


def detect_hooks(
    transcript: TranscriptionResult,
    num_clips: int = DEFAULT_NUM_CLIPS,
) -> List[ViralSegment]:
    """
    Analyze a transcript (of any length) and return the top N viral-worthy
    segments. Automatically chunks very long transcripts across multiple
    LLM requests to stay within token limits, then merges and ranks results
    across all chunks.

    Args:
        transcript: A TranscriptionResult from transcriber.py.
        num_clips: How many segments to return overall.

    Returns:
        A list of validated ViralSegment objects, sorted by virality_score
        descending, with no overlapping timestamps.

    Raises:
        HookDetectionError: If every chunk fails (total failure).
    """
    chunks = _split_transcript_into_chunks(transcript)

    # Ask each chunk for a few candidates — more than num_clips overall,
    # since we'll filter down to the best across all chunks afterward.
    candidates_per_chunk = max(2, num_clips)

    all_segments: List[ViralSegment] = []
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        logger.info(
            f"Analyzing chunk {i + 1}/{len(chunks)} "
            f"({len(chunk.words)} words, ~{chunk.duration / 60:.1f} min)"
        )
        try:
            segments = _detect_hooks_single_chunk(chunk, candidates_per_chunk)
            all_segments.extend(segments)
        except HookDetectionError as e:
            logger.warning(f"Chunk {i + 1}/{len(chunks)} failed, skipping: {e}")
            failed_chunks += 1
            continue

    if not all_segments:
        raise HookDetectionError(
            f"Hook detection failed on all {len(chunks)} chunks. "
            "No valid segments could be identified."
        )

    if failed_chunks:
        logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed but pipeline continued")

    all_segments = _remove_overlaps(all_segments)

    if len(all_segments) > num_clips:
        all_segments = all_segments[:num_clips]

    logger.info(f"Hook detection complete: {len(all_segments)} final segments selected")
    for s in all_segments:
        logger.info(
            f"  [{s.start_time:.1f}s - {s.end_time:.1f}s] "
            f"({s.duration:.1f}s) score={s.virality_score} — \"{s.headline}\""
        )

    return all_segments


if __name__ == "__main__":
    import sys
    from config import TEMP_DIR

    transcript_path = TEMP_DIR / "sample_transcript.json"
    if not transcript_path.exists():
        logger.error(
            f"No transcript found at {transcript_path}. "
            "Run transcriber.py first to generate it."
        )
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    transcript = TranscriptionResult(**data)

    segments = detect_hooks(transcript, num_clips=3)

    output_path = TEMP_DIR / "sample_hooks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in segments], f, indent=2, ensure_ascii=False)

    logger.info(f"Hooks saved to {output_path}")