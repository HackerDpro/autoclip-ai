"""
Video clipping module for AutoClip AI.

Takes a source video and a ViralSegment (start/end time + metadata), and
produces a final vertical (9:16) short-form clip with burned-in, karaoke-style
word-highlighted captions, using ffmpeg for all heavy lifting.
"""

import subprocess
from pathlib import Path
from typing import List

from config import (
    logger,
    OUTPUT_DIR,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
    VIDEO_CODEC,
    VIDEO_CRF,
    VIDEO_PRESET,
    AUDIO_CODEC,
    AUDIO_OUTPUT_BITRATE,
    CAPTION_FONT_NAME,
    CAPTION_FONT_SIZE,
    CAPTION_PRIMARY_COLOR,
    CAPTION_HIGHLIGHT_COLOR,
    CAPTION_OUTLINE_COLOR,
    CAPTION_OUTLINE_WIDTH,
    CAPTION_MARGIN_V,
)
from transcriber import TranscriptWord
from hook_detector import ViralSegment


class VideoClippingError(Exception):
    """Raised when trimming, cropping, or caption burning fails."""
    pass


def _get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """
    Return (width, height) of a video file using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width_str, height_str = result.stdout.strip().split("x")
        return int(width_str), int(height_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        raise VideoClippingError(f"Could not determine video dimensions: {e}") from e


def _format_ass_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) into ASS subtitle timestamp format: H:MM:SS.CC
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"


def _build_ass_header() -> str:
    """
    Build the ASS subtitle file header, including the style definition for
    karaoke-style captions (white base text, gold highlight on active word).
    """
    return f"""[Script Info]
Title: AutoClip AI Captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CAPTION_FONT_NAME},{CAPTION_FONT_SIZE},{CAPTION_PRIMARY_COLOR},{CAPTION_HIGHLIGHT_COLOR},{CAPTION_OUTLINE_COLOR},&H00000000,-1,0,0,0,100,100,0,0,1,{CAPTION_OUTLINE_WIDTH},0,2,60,60,{CAPTION_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _group_words_into_captions(
    words: List[TranscriptWord],
    max_words_per_caption: int = 4,
) -> List[List[TranscriptWord]]:
    """
    Group consecutive words into small caption chunks (e.g. 3-4 words shown
    on screen at a time), which is the standard short-form caption style —
    much more readable and dynamic than full sentences.
    """
    groups: List[List[TranscriptWord]] = []
    current_group: List[TranscriptWord] = []

    for word in words:
        current_group.append(word)
        if len(current_group) >= max_words_per_caption:
            groups.append(current_group)
            current_group = []

    if current_group:
        groups.append(current_group)

    return groups


def _build_karaoke_line(group: List[TranscriptWord], active_index: int) -> str:
    """
    Build a single ASS dialogue text line where the word at `active_index`
    is highlighted in the secondary (gold) color, and all others use the
    primary (white) color. Uses inline ASS override tags.
    """
    parts = []
    for i, w in enumerate(group):
        clean_word = w.word.strip()
        if i == active_index:
            # \c&H...& overrides color inline; switch to SecondaryColour (gold)
            parts.append(f"{{\\c{CAPTION_HIGHLIGHT_COLOR}}}{clean_word}{{\\c{CAPTION_PRIMARY_COLOR}}}")
        else:
            parts.append(clean_word)
    return " ".join(parts)


def _generate_ass_subtitles(
    words: List[TranscriptWord],
    segment_start: float,
    output_path: Path,
) -> None:
    """
    Generate a full ASS subtitle file with karaoke-style word highlighting
    for the given list of words, with all timestamps re-based relative to
    the start of the clipped segment (since the clip itself starts at 0).

    Args:
        words: Word-level transcript entries that fall within the segment.
        segment_start: The original video's timestamp where this segment
                       begins (used to re-base word times to 0).
        output_path: Where to write the .ass file.
    """
    if not words:
        raise VideoClippingError("No words provided for subtitle generation.")

    groups = _group_words_into_captions(words)
    lines = [_build_ass_header()]

    for group in groups:
        for active_index, active_word in enumerate(group):
            rel_start = active_word.start - segment_start
            rel_end = active_word.end - segment_start

            if rel_start < 0:
                rel_start = 0
            if rel_end <= rel_start:
                rel_end = rel_start + 0.1

            text = _build_karaoke_line(group, active_index)
            start_ts = _format_ass_timestamp(rel_start)
            end_ts = _format_ass_timestamp(rel_end)

            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated ASS subtitle file: {output_path.name} ({len(groups)} caption groups)")


def _filter_words_in_range(
    all_words: List[TranscriptWord],
    start: float,
    end: float,
) -> List[TranscriptWord]:
    """
    Return only the words whose timing falls within [start, end].
    """
    return [w for w in all_words if w.start >= start and w.end <= end]


def _build_crop_filter(src_width: int, src_height: int) -> str:
    """
    Build an ffmpeg filter string that crops the source video to a centered
    9:16 region, then scales it to the exact target output resolution.

    Logic: scale the video up so its height matches OUTPUT_HEIGHT while
    preserving aspect ratio, then crop the width down to OUTPUT_WIDTH,
    centered. This guarantees a consistent, clean 1080x1920 result
    regardless of the source video's original resolution/aspect ratio.
    """
    return (
        f"scale=-2:{OUTPUT_HEIGHT},"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
    )


def create_clip(
    source_video: str | Path,
    segment: ViralSegment,
    transcript_words: List[TranscriptWord],
    output_filename: str | None = None,
) -> Path:
    """
    Produce a final vertical short-form clip from a source video and a
    detected viral segment: trims, crops to 9:16, burns in karaoke-style
    captions, and encodes the result.

    Args:
        source_video: Path to the original long-form video file.
        segment: The ViralSegment describing which part to clip.
        transcript_words: Full word-level transcript of the source video
                           (used to extract captions for this segment).
        output_filename: Optional explicit output filename. Defaults to a
                          name derived from the segment's headline.

    Returns:
        Path to the final rendered .mp4 clip.

    Raises:
        FileNotFoundError: If the source video doesn't exist.
        VideoClippingError: If any ffmpeg step fails.
    """
    source_video = Path(source_video)
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    if output_filename is None:
        safe_headline = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "" for c in segment.headline
        ).strip().replace(" ", "_")[:50]
        output_filename = f"clip_{safe_headline}.mp4"

    output_path = OUTPUT_DIR / output_filename
    ass_path = output_path.with_suffix(".ass")

    logger.info(
        f"Creating clip '{output_filename}' from "
        f"{segment.start_time:.1f}s to {segment.end_time:.1f}s "
        f"({segment.duration:.1f}s duration)"
    )

    segment_words = _filter_words_in_range(
        transcript_words, segment.start_time, segment.end_time
    )

    if not segment_words:
        logger.warning(
            "No transcript words found within this segment's time range. "
            "Clip will be produced WITHOUT captions."
        )

    src_width, src_height = _get_video_dimensions(source_video)
    crop_filter = _build_crop_filter(src_width, src_height)

    if segment_words:
        _generate_ass_subtitles(segment_words, segment.start_time, ass_path)
        ass_path_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
        full_filter = f"{crop_filter},ass='{ass_path_escaped}'"
    else:
        full_filter = crop_filter

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(segment.start_time),
        "-i", str(source_video),
        "-t", str(segment.duration),
        "-vf", full_filter,
        "-r", str(OUTPUT_FPS),
        "-c:v", VIDEO_CODEC,
        "-crf", VIDEO_CRF,
        "-preset", VIDEO_PRESET,
        "-c:a", AUDIO_CODEC,
        "-b:a", AUDIO_OUTPUT_BITRATE,
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info("Running ffmpeg to trim, crop, and burn captions (this may take a moment)...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"ffmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        raise VideoClippingError(
            f"ffmpeg failed while producing '{output_filename}'. "
            f"Error: {e.stderr[-800:] if e.stderr else 'unknown error'}"
        ) from e
    finally:
        if ass_path.exists():
            ass_path.unlink()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoClippingError(
            f"ffmpeg reported success but output is missing or empty: {output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Clip created: {output_path.name} ({size_mb:.2f} MB)")

    return output_path


def create_clips_from_segments(
    source_video: str | Path,
    segments: List[ViralSegment],
    transcript_words: List[TranscriptWord],
) -> List[Path]:
    """
    Convenience wrapper: produce one final clip per detected viral segment.

    Returns:
        List of paths to all successfully created clip files. Segments that
        fail are logged and skipped rather than halting the whole batch.
    """
    output_paths: List[Path] = []

    for i, segment in enumerate(segments):
        logger.info(f"Processing clip {i + 1}/{len(segments)}: \"{segment.headline}\"")
        try:
            path = create_clip(source_video, segment, transcript_words)
            output_paths.append(path)
        except Exception as e:
            logger.error(f"Failed to create clip for segment \"{segment.headline}\": {e}")
            continue

    logger.info(f"Batch complete: {len(output_paths)}/{len(segments)} clips created successfully")
    return output_paths


if __name__ == "__main__":
    import sys
    import json
    from config import INPUT_DIR, TEMP_DIR

    sample_video = INPUT_DIR / "sample.mp4"
    transcript_path = TEMP_DIR / "sample_transcript.json"
    hooks_path = TEMP_DIR / "sample_hooks.json"

    if not sample_video.exists():
        logger.error(f"No sample video found at {sample_video}")
        sys.exit(1)
    if not transcript_path.exists():
        logger.error(f"No transcript found at {transcript_path}. Run transcriber.py first.")
        sys.exit(1)
    if not hooks_path.exists():
        logger.error(f"No hooks found at {hooks_path}. Run hook_detector.py first.")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    all_words = [TranscriptWord(**w) for w in transcript_data["words"]]

    with open(hooks_path, "r", encoding="utf-8") as f:
        hooks_data = json.load(f)
    segments = [ViralSegment(**h) for h in hooks_data]

    logger.info(f"Test mode: creating {len(segments)} clips from sample.mp4")
    results = create_clips_from_segments(sample_video, segments, all_words)

    for path in results:
        logger.info(f"  -> {path}")