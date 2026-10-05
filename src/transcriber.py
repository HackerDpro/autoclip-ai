"""
Transcription module for AutoClip AI.

Converts an audio file into a word-level timestamped transcript using the
Groq API (whisper-large-v3-turbo, free tier). Automatically chunks audio
files that exceed the API size limit, transcribing each piece and merging
timestamps back into one continuous transcript. Falls back to a local
faster-whisper model if Groq is unavailable for any reason.
"""

import math
import subprocess
from pathlib import Path
from typing import List, Any

from pydantic import BaseModel, Field

from config import (
    logger,
    GROQ_API_KEY,
    GROQ_WHISPER_MODEL,
    MAX_AUDIO_SIZE_MB,
    TEMP_DIR,
)
from audio_extractor import get_audio_duration


class TranscriptionError(Exception):
    """Raised when transcription fails through all available methods."""
    pass


class TranscriptWord(BaseModel):
    """A single transcribed word with its timing in the source audio."""
    word: str
    start: float
    end: float


class TranscriptionResult(BaseModel):
    """Full transcription result: plain text plus word-level timestamps."""
    text: str
    words: List[TranscriptWord] = Field(default_factory=list)
    duration: float


def _get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _word_to_model(w: Any) -> TranscriptWord:
    """
    Normalize a word entry from the Groq/OpenAI-style response into a
    TranscriptWord, handling both dict-style and attribute-style objects
    depending on SDK version.
    """
    if isinstance(w, dict):
        return TranscriptWord(word=w["word"], start=w["start"], end=w["end"])
    return TranscriptWord(word=w.word, start=w.start, end=w.end)


def _split_audio_into_chunks(audio_path: Path, num_chunks: int) -> List[tuple[Path, float]]:
    """
    Split an audio file into `num_chunks` roughly equal segments using ffmpeg
    stream-copy (no re-encoding, fast and lossless for mp3).

    Returns a list of (chunk_path, chunk_start_time_seconds) tuples.
    """
    total_duration = get_audio_duration(audio_path)
    chunk_duration = total_duration / num_chunks
    chunks: List[tuple[Path, float]] = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = TEMP_DIR / f"{audio_path.stem}_chunk{i}.mp3"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c", "copy",
            str(chunk_path),
        ]

        logger.info(f"Splitting chunk {i + 1}/{num_chunks} (start={start_time:.1f}s)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise TranscriptionError(
                f"Failed to split audio chunk {i}: {result.stderr[-500:]}"
            )

        chunks.append((chunk_path, start_time))

    return chunks


def _transcribe_chunk_with_groq(chunk_path: Path, client) -> Any:
    """Send a single audio chunk to the Groq Whisper API."""
    with open(chunk_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(chunk_path.name, f.read()),
            model=GROQ_WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
    return response


def _transcribe_with_groq(audio_path: Path) -> TranscriptionResult:
    """
    Transcribe an audio file using the Groq API, automatically chunking
    if the file exceeds MAX_AUDIO_SIZE_MB.
    """
    if not GROQ_API_KEY:
        raise TranscriptionError("GROQ_API_KEY is not set. Check your .env file.")

    try:
        from groq import Groq
    except ImportError as e:
        raise TranscriptionError(
            "The 'groq' package is not installed. Run: pip install groq"
        ) from e

    client = Groq(api_key=GROQ_API_KEY)
    file_size_mb = _get_file_size_mb(audio_path)
    total_duration = get_audio_duration(audio_path)

    all_words: List[TranscriptWord] = []
    all_text_parts: List[str] = []

    if file_size_mb <= MAX_AUDIO_SIZE_MB:
        logger.info(f"Transcribing '{audio_path.name}' ({file_size_mb:.2f} MB) in a single request")
        response = _transcribe_chunk_with_groq(audio_path, client)

        all_text_parts.append(response.text)
        for w in (response.words or []):
            all_words.append(_word_to_model(w))

    else:
        num_chunks = math.ceil(file_size_mb / MAX_AUDIO_SIZE_MB)
        logger.info(
            f"'{audio_path.name}' is {file_size_mb:.2f} MB, exceeds {MAX_AUDIO_SIZE_MB} MB limit. "
            f"Splitting into {num_chunks} chunks."
        )
        chunks = _split_audio_into_chunks(audio_path, num_chunks)

        for i, (chunk_path, start_offset) in enumerate(chunks):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunks)}")
            try:
                response = _transcribe_chunk_with_groq(chunk_path, client)
            finally:
                chunk_path.unlink(missing_ok=True)

            all_text_parts.append(response.text)
            for w in (response.words or []):
                word_model = _word_to_model(w)
                word_model.start += start_offset
                word_model.end += start_offset
                all_words.append(word_model)

    return TranscriptionResult(
        text=" ".join(all_text_parts).strip(),
        words=all_words,
        duration=total_duration,
    )


def _transcribe_with_faster_whisper(audio_path: Path) -> TranscriptionResult:
    """
    Local fallback transcription using faster-whisper. Only triggered if
    the Groq API fails. Slower on CPU, but free and fully offline.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise TranscriptionError(
            "Groq transcription failed and 'faster-whisper' fallback is not "
            "installed. Uncomment it in requirements.txt and run: "
            "pip install -r requirements.txt"
        ) from e

    logger.warning("Falling back to local faster-whisper model (slower on CPU).")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    all_words: List[TranscriptWord] = []
    all_text_parts: List[str] = []

    for segment in segments:
        all_text_parts.append(segment.text)
        for word in (segment.words or []):
            all_words.append(TranscriptWord(word=word.word, start=word.start, end=word.end))

    return TranscriptionResult(
        text=" ".join(all_text_parts).strip(),
        words=all_words,
        duration=info.duration,
    )


def transcribe_audio(audio_path: str | Path) -> TranscriptionResult:
    """
    Transcribe an audio file into a word-level timestamped transcript.

    Tries the Groq API first (fast, free tier). Falls back to a local
    faster-whisper model if Groq fails for any reason.

    Args:
        audio_path: Path to the .mp3 (or other ffmpeg-readable) audio file.

    Returns:
        TranscriptionResult containing full text and word-level timestamps.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        TranscriptionError: If both Groq and the local fallback fail.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        result = _transcribe_with_groq(audio_path)
        logger.info(
            f"Transcription complete via Groq: {len(result.words)} words, "
            f"{result.duration:.1f}s duration"
        )
        return result
    except Exception as e:
        logger.warning(f"Groq transcription failed: {e}")
        logger.info("Attempting local faster-whisper fallback...")

        try:
            result = _transcribe_with_faster_whisper(audio_path)
            logger.info(
                f"Transcription complete via faster-whisper fallback: "
                f"{len(result.words)} words"
            )
            return result
        except Exception as fallback_error:
            raise TranscriptionError(
                f"All transcription methods failed. "
                f"Groq error: {e} | Fallback error: {fallback_error}"
            ) from fallback_error


if __name__ == "__main__":
    import sys
    import json
    from config import TEMP_DIR as _TEMP_DIR

    test_audio = _TEMP_DIR / "sample.mp3"
    if not test_audio.exists():
        logger.error(
            f"No test audio found at {test_audio}. "
            "Run audio_extractor.py first to generate it."
        )
        sys.exit(1)

    result = transcribe_audio(test_audio)

    output_json = _TEMP_DIR / "sample_transcript.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    logger.info(f"Transcript saved to {output_json}")
    logger.info(f"First 200 chars of transcript: {result.text[:200]}")