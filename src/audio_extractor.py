"""
Audio extraction module for AutoClip AI.

Takes a source video file (.mp4) and extracts a lightweight audio track
(.mp3) using local ffmpeg, optimized for speech-to-text transcription.
"""

import subprocess
import shutil
from pathlib import Path

from config import (
    logger,
    TEMP_DIR,
    AUDIO_SAMPLE_RATE,
    AUDIO_BITRATE,
    AUDIO_FORMAT,
)


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video file."""
    pass


def _check_ffmpeg_installed() -> None:
    """
    Verify that ffmpeg is available on the system PATH.
    Raises AudioExtractionError with a clear message if not found.
    """
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError(
            "ffmpeg is not installed or not found on PATH. "
            "If running in a Codespace, rebuild the dev container "
            "(Ctrl+Shift+P -> 'Codespaces: Rebuild Container')."
        )


def extract_audio(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    Extract a mono audio track from a video file and save it as an mp3.

    Args:
        video_path: Path to the source video file (.mp4, .mov, .mkv, etc).
        output_path: Optional explicit output path for the .mp3 file.
                     If not given, saves into TEMP_DIR using the video's
                     stem name (e.g. "podcast.mp4" -> "podcast.mp3").

    Returns:
        Path to the extracted .mp3 audio file.

    Raises:
        FileNotFoundError: If the input video does not exist.
        AudioExtractionError: If ffmpeg is missing or extraction fails.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    if not video_path.is_file():
        raise ValueError(f"Expected a file, got a directory: {video_path}")

    _check_ffmpeg_installed()

    if output_path is None:
        output_path = TEMP_DIR / f"{video_path.stem}.{AUDIO_FORMAT}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting audio from '{video_path.name}' -> '{output_path.name}'")

    # ffmpeg command breakdown:
    # -y            overwrite output without prompting
    # -i            input file
    # -vn           strip video stream entirely (audio-only output)
    # -ac 1         mono channel (smaller file, Whisper doesn't need stereo)
    # -ar           sample rate (16kHz is the sweet spot for speech models)
    # -b:a          audio bitrate (keeps file small for upload/chunking)
    # -codec:a      libmp3lame for reliable mp3 encoding
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-b:a", AUDIO_BITRATE,
        "-codec:a", "libmp3lame",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"ffmpeg stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed with exit code {e.returncode}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        raise AudioExtractionError(
            f"Failed to extract audio from '{video_path.name}'. "
            f"ffmpeg error: {e.stderr[-500:] if e.stderr else 'unknown error'}"
        ) from e
    except FileNotFoundError as e:
        raise AudioExtractionError(
            "ffmpeg executable could not be run. Is it installed correctly?"
        ) from e

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioExtractionError(
            f"ffmpeg reported success but output file is missing or empty: {output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Audio extraction complete: {output_path.name} ({size_mb:.2f} MB)")

    return output_path


def get_audio_duration(audio_path: str | Path) -> float:
    """
    Return the duration of an audio/video file in seconds using ffprobe.

    Args:
        audio_path: Path to the media file.

    Returns:
        Duration in seconds as a float.

    Raises:
        AudioExtractionError: If ffprobe fails or output can't be parsed.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        raise AudioExtractionError(
            f"Could not determine duration of '{audio_path.name}': {e}"
        ) from e


if __name__ == "__main__":
    # Quick manual test: place a sample video at data/input/sample.mp4 and run
    # `python3 src/audio_extractor.py` to verify extraction works end-to-end.
    import sys
    from config import INPUT_DIR

    sample = INPUT_DIR / "sample.mp4"
    if not sample.exists():
        logger.error(
            f"No test file found at {sample}. "
            "Drop a small .mp4 there to test this module."
        )
        sys.exit(1)

    output = extract_audio(sample)
    duration = get_audio_duration(output)
    logger.info(f"Test passed. Extracted audio duration: {duration:.2f} seconds")