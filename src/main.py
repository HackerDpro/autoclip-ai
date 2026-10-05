"""
AutoClip AI — CLI entry point.

Orchestrates the full pipeline: audio extraction -> transcription ->
viral hook detection -> vertical clip generation with burned-in captions.

Usage:
    python3 src/main.py --input path/to/video.mp4 --clips 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import logger, OUTPUT_DIR, TEMP_DIR, DEFAULT_NUM_CLIPS
from audio_extractor import extract_audio, AudioExtractionError
from transcriber import transcribe_audio, TranscriptionError, TranscriptionResult
from hook_detector import detect_hooks, HookDetectionError
from video_clipper import create_clips_from_segments, VideoClippingError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autoclip-ai",
        description=(
            "AutoClip AI — turn a long-form video into viral 9:16 vertical "
            "short clips with AI-selected moments and burned-in captions."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Path to the source video file (.mp4, .mov, .mkv, etc.)",
    )
    parser.add_argument(
        "--clips", "-c",
        type=int,
        default=DEFAULT_NUM_CLIPS,
        help=f"Number of viral clips to generate (default: {DEFAULT_NUM_CLIPS})",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate audio/transcript files in data/temp (default: kept anyway for now)",
    )
    parser.add_argument(
        "--transcript-cache",
        type=str,
        default=None,
        help=(
            "Optional path to a pre-existing transcript JSON file to skip "
            "re-transcribing (useful for testing/iterating on hook detection "
            "or clipping without burning API calls)."
        ),
    )
    return parser.parse_args()


def print_banner() -> None:
    banner = r"""
    ___        __        ________       ___    ____
   /   | __  __/ /_____  / ____/ (_)___  /   |  /  _/
  / /| |/ / / / __/ __ \/ /   / / / __ \/ /| |  / /
 / ___ / /_/ / /_/ /_/ / /___/ / / /_/ / ___ |_/ /
/_/  |_\__,_/\__/\____/\____/_/_/ .___/_/  |_/___/
                                /_/
    """
    print(banner)
    print("  AI-powered long-form -> viral short-form video pipeline\n")


def run_pipeline(input_path: str, num_clips: int, transcript_cache: str | None) -> list[Path]:
    """
    Run the full AutoClip AI pipeline end-to-end.

    Returns:
        List of paths to successfully generated clip files.
    """
    source_video = Path(input_path)
    if not source_video.exists():
        logger.error(f"Input file not found: {source_video}")
        sys.exit(1)

    pipeline_start = time.time()

    # --- Step 1: Transcription (with optional cache to skip re-work) ---
    if transcript_cache:
        cache_path = Path(transcript_cache)
        if not cache_path.exists():
            logger.error(f"Transcript cache file not found: {cache_path}")
            sys.exit(1)
        logger.info(f"[1/3] Loading cached transcript from {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            transcript = TranscriptionResult(**json.load(f))
    else:
        logger.info("[1/3] Extracting audio...")
        try:
            audio_path = extract_audio(source_video)
        except AudioExtractionError as e:
            logger.error(f"Audio extraction failed: {e}")
            sys.exit(1)

        logger.info("[1/3] Transcribing audio...")
        try:
            transcript = transcribe_audio(audio_path)
        except TranscriptionError as e:
            logger.error(f"Transcription failed: {e}")
            sys.exit(1)

        transcript_cache_path = TEMP_DIR / f"{source_video.stem}_transcript.json"
        with open(transcript_cache_path, "w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript cached at {transcript_cache_path.name} for future reuse")

    # --- Step 2: Hook detection ---
    logger.info(f"[2/3] Detecting top {num_clips} viral moments...")
    try:
        segments = detect_hooks(transcript, num_clips=num_clips)
    except HookDetectionError as e:
        logger.error(f"Hook detection failed: {e}")
        sys.exit(1)

    # --- Step 3: Clip generation ---
    logger.info(f"[3/3] Generating {len(segments)} vertical clips with captions...")
    try:
        output_paths = create_clips_from_segments(source_video, segments, transcript.words)
    except VideoClippingError as e:
        logger.error(f"Clip generation failed: {e}")
        sys.exit(1)

    elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline complete in {elapsed:.1f} seconds")

    return output_paths


def print_summary(output_paths: list[Path]) -> None:
    print("\n" + "=" * 60)
    print(f"  DONE — {len(output_paths)} clip(s) generated")
    print("=" * 60)
    for path in output_paths:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  -> {path} ({size_mb:.2f} MB)")
    print("=" * 60 + "\n")


def main() -> None:
    print_banner()
    args = parse_args()

    logger.info(f"Input video: {args.input}")
    logger.info(f"Requested clips: {args.clips}")

    output_paths = run_pipeline(args.input, args.clips, args.transcript_cache)

    if not output_paths:
        logger.error("No clips were successfully generated. Check logs above for errors.")
        sys.exit(1)

    print_summary(output_paths)


if __name__ == "__main__":
    main()