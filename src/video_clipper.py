"""
Video clipping module for AutoClip AI.

Takes a source video and a ViralSegment (start/end time + metadata), and
produces a final vertical (9:16) short-form clip with burned-in, karaoke-style
word-highlighted captions, using ffmpeg for all heavy lifting. Uses
face-aware cropping (face_cropper.py) to decide the best framing strategy
per clip instead of blindly center-cropping everything.
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
from face_cropper import analyze_segment_faces, build_video_filter


class VideoClippingError(Exception):
    """Raised when trimming, cropping, or caption burning fails."""
    pass


def _get_video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
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
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centiseconds = int((secs - int(secs)) * 100)
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centiseconds:02d}"


def _build_ass_header() -> str:
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


def _group_words_into_captions(words: List[TranscriptWord], max_words_per_caption: int = 4) -> List[List[TranscriptWord]]:
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
    parts = []
    for i, w in enumerate(group):
        clean_word = w.word.strip()
        if i == active_index:
            parts.append(f"{{\\c{CAPTION_HIGHLIGHT_COLOR}}}{clean_word}{{\\c{CAPTION_PRIMARY_COLOR}}}")
        else:
            parts.append(clean_word)
    return " ".join(parts)


def _generate_ass_subtitles(words: List[TranscriptWord], segment_start: float, output_path: Path) -> None:
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

            lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated ASS subtitle file: {output_path.name} ({len(groups)} caption groups)")


def _filter_words_in_range(all_words: List[TranscriptWord], start: float, end: float) -> List[TranscriptWord]:
    return [w for w in all_words if w.start >= start and w.end <= end]


def create_clip(
    source_video: str | Path,
    segment: ViralSegment,
    transcript_words: List[TranscriptWord],
    output_filename: str | None = None,
) -> Path:
    """
    Produce a final vertical short-form clip: trims, applies face-aware
    9:16 cropping (single-speaker dynamic crop, dual-speaker split-screen,
    or center-crop fallback), burns in karaoke-style captions, and encodes.
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
        f"{segment.start_time:.1f}s to {segment.end_time:.1f}s ({segment.duration:.1f}s)"
    )

    segment_words = _filter_words_in_range(transcript_words, segment.start_time, segment.end_time)
    if not segment_words:
        logger.warning("No transcript words in this range. Clip will have NO captions.")

    src_width, src_height = _get_video_dimensions(source_video)

    # --- Face-aware crop decision ---
    crop_decision = analyze_segment_faces(source_video, segment.start_time, segment.end_time)
    video_filter, is_complex = build_video_filter(crop_decision, src_width, src_height)

    if segment_words:
        _generate_ass_subtitles(segment_words, segment.start_time, ass_path)
        ass_path_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

        if is_complex:
            # Chain subtitles onto the split-screen output, re-label as [vout]
            video_filter = video_filter.replace(
                "[vcrop]", f"[vcrop];[vcrop]ass='{ass_path_escaped}'[vout]"
            )
            # The replace above duplicates [vcrop] intentionally: once as the
            # stack's real output, once as the input to the ass filter.
        else:
            video_filter = f"{video_filter},ass='{ass_path_escaped}'"
    else:
        if is_complex:
            video_filter = video_filter.replace("[vcrop]", "[vcrop];[vcrop]copy[vout]")

    cmd = ["ffmpeg", "-y", "-ss", str(segment.start_time), "-i", str(source_video), "-t", str(segment.duration)]

    if is_complex:
        cmd += ["-filter_complex", video_filter, "-map", "[vout]", "-map", "0:a"]
    else:
        cmd += ["-vf", video_filter]

    cmd += [
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
        logger.warning(
            f"Dynamic face-tracked crop failed for '{output_filename}', "
            f"retrying with safe static center-crop fallback. "
            f"Original error: {e.stderr[-400:] if e.stderr else 'unknown'}"
        )
        fallback_filter = f"scale=-2:{OUTPUT_HEIGHT},crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"
        if segment_words:
            ass_path_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
            fallback_filter = f"{fallback_filter},ass='{ass_path_escaped}'"

        fallback_cmd = [
            "ffmpeg", "-y", "-ss", str(segment.start_time), "-i", str(source_video),
            "-t", str(segment.duration), "-vf", fallback_filter,
            "-r", str(OUTPUT_FPS), "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF,
            "-preset", VIDEO_PRESET, "-c:a", AUDIO_CODEC, "-b:a", AUDIO_OUTPUT_BITRATE,
            "-movflags", "+faststart", str(output_path),
        ]

        try:
            subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as fallback_error:
            raise VideoClippingError(
                f"Both dynamic crop AND fallback center-crop failed for "
                f"'{output_filename}'. Fallback error: "
                f"{fallback_error.stderr[-800:] if fallback_error.stderr else 'unknown'}"
            ) from fallback_error
    finally:
        if ass_path.exists():
            ass_path.unlink()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise VideoClippingError(f"ffmpeg reported success but output is missing/empty: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Clip created: {output_path.name} ({size_mb:.2f} MB)")

    return output_path


def create_clips_from_segments(
    source_video: str | Path,
    segments: List[ViralSegment],
    transcript_words: List[TranscriptWord],
) -> List[Path]:
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

    if not sample_video.exists() or not transcript_path.exists() or not hooks_path.exists():
        logger.error("Missing test files. Run the earlier pipeline steps first.")
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        all_words = [TranscriptWord(**w) for w in json.load(f)["words"]]

    with open(hooks_path, "r", encoding="utf-8") as f:
        segments = [ViralSegment(**h) for h in json.load(f)]

    results = create_clips_from_segments(sample_video, segments, all_words)
    for path in results:
        logger.info(f"  -> {path}")