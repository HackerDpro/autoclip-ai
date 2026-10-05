"""
Utility script: lists all models currently available to your Groq API key.
Run this to find the correct current model name if hook_detector.py ever
throws a model-not-found error.
"""

from groq import Groq
from config import GROQ_API_KEY, logger

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not set in .env")

client = Groq(api_key=GROQ_API_KEY)

logger.info("Available Groq models for your API key:")
for model in client.models.list().data:
    print(f"  - {model.id}")