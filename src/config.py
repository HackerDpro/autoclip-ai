"""
Central configuration for AutoClip AI.
All paths, API keys, model names, and rendering specs live here.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (used across every module)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoclip")

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in .env — transcription will fail.")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in .env — hook detection will fail.")

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TEMP_DIR = DATA_DIR / "temp"
ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

for directory in (INPUT_DIR, OUTPUT_DIR, TEMP_DIR, ASSETS_DIR, FONTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio extraction settings
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000  # Hz — optimal for Whisper models
AUDIO_BITRATE = "64k"
AUDIO_FORMAT = "mp3"
MAX_AUDIO_SIZE_MB = 25  # Groq/Whisper API limit threshold for chunking

# ---------------------------------------------------------------------------
# Video output settings (9:16 vertical)
# ---------------------------------------------------------------------------
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_CRF = "20"          # lower = higher quality, 18-23 is a good range
VIDEO_PRESET = "medium"    # ffmpeg encoding speed/quality tradeoff
AUDIO_CODEC = "aac"
AUDIO_OUTPUT_BITRATE = "192k"

# ---------------------------------------------------------------------------
# Caption / subtitle styling (ASS format)
# ---------------------------------------------------------------------------
CAPTION_FONT_NAME = "Arial Black"
CAPTION_FONT_SIZE = 72
CAPTION_PRIMARY_COLOR = "&H00FFFFFF"   # white
CAPTION_HIGHLIGHT_COLOR = "&H0000D7FF"  # gold/yellow (BGR order in ASS)
CAPTION_OUTLINE_COLOR = "&H00000000"   # black outline
CAPTION_OUTLINE_WIDTH = 4
CAPTION_MARGIN_V = 220  # vertical offset from bottom, in pixels

# ---------------------------------------------------------------------------
# Hook detection settings
# ---------------------------------------------------------------------------
DEFAULT_NUM_CLIPS = 3
MIN_CLIP_DURATION_SEC = 20
MAX_CLIP_DURATION_SEC = 90