# AutoClip AI — Project State

> Single source of truth for project context. If chat history is lost, paste
> this file back to the AI assistant to restore full context instantly.

Last updated: Phase 0 — Module 4 (video_clipper.py) starting

---

## WHAT THIS PROJECT IS
AutoClip AI — ingests long-form video (MP4/YouTube), outputs viral 9:16 vertical
short clips with burned-in dynamic captions. Open-core strategy: core pipeline
is open-sourced on GitHub, monetization comes later via manual productized
service (€15-25/video) then a self-serve micro-SaaS.

## WHO IS BUILDING THIS
Diego, 15, Belgium. €0 capital. Building in a GitHub Codespace (cloud Linux VM,
not local machine). Uses Claude (via arena.ai, no Claude account/Claude Code)
for architecture + full code, GitHub Copilot inside the Codespace for inline
autocomplete/small fixes only.

## KEY STRATEGIC DECISIONS (don't re-litigate these without reason)
- Open-source first (MIT license), monetize later.
- Transcription: Groq API (whisper-large-v3-turbo). CONFIRMED WORKING.
- Hook detection LLM: SWITCHED FROM GEMINI TO GROQ. Using
  `openai/gpt-oss-120b` on Groq. Gemini free tier was unreliable (constant
  503 "high demand" errors across every model tested — gemini-2.0-flash,
  gemini-3.8-flash, gemini-flash-latest, gemini-2.5-flash-lite all failed or
  were deprecated within the same session). Groq is now the SOLE LLM
  provider for the whole pipeline — one API key, one SDK, proven reliable.
- Payments (later): direct IBAN bank transfer for manual service tier.
  Legal/Bryo/parent stuff deliberately DEFERRED until real paying volume.
- Production hosting (later): Oracle Cloud Always Free VM. Not relevant yet.
- Target niche: Benelux podcasters/creators underserved by English-first
  tools like Opus Clip/Submagic; CapCut is the real free competitor to watch.

## ENVIRONMENT
- GitHub repo: https://github.com/HackerDpro/autoclip-ai (public, MIT license)
- Dev: GitHub Codespaces, devcontainer.json installs ffmpeg + pip deps.
- `.env` holds GROQ_API_KEY, GROQ_LLM_MODEL (gpt-oss-120b). GEMINI_API_KEY
  no longer used — can leave in .env harmlessly or remove.
- Confirmed working: ffmpeg 7.1.5, python-dotenv, groq==1.7.0 (NOT 0.11.0).

## KNOWN GOTCHAS (don't repeat these mistakes)
- groq==0.11.0 is BROKEN (httpx incompatibility). Must use groq>=1.7.0.
- devcontainer.json changes require "Codespaces: Rebuild Container" command.
- Gemini free tier is unreliable (503 errors) — DO NOT go back to Gemini for
  the core pipeline without strong reason. Groq is the chosen provider.
- Groq free tier has an 8000 TPM (tokens-per-minute) limit on gpt-oss-120b.
  Large transcripts (3600+ words) can briefly exceed this on first attempt —
  our retry-with-backoff logic already handles this gracefully (succeeds on
  attempt 2 typically). If this becomes a recurring problem on longer videos,
  consider trimming the timestamped transcript sent to the LLM (e.g. strip
  to every other chunk, or summarize first).
- Model names on both Gemini AND Groq can be deprecated/renamed without much
  warning. If hook_detector.py ever 404s on model name, run
  src/list_groq_models.py to see current available models and update
  GROQ_LLM_MODEL in .env accordingly. No code changes needed, just .env.

## REPO STRUCTURE (current)
autoclip-ai/
├── .devcontainer/devcontainer.json
├── assets/fonts/
├── data/
│ ├── input/sample.mp4
│ ├── temp/sample.mp3
│ ├── temp/sample_transcript.json
│ └── temp/sample_hooks.json (NEW — 3 validated viral segments)
├── src/
│ ├── init.py
│ ├── config.py ✅ DONE
│ ├── audio_extractor.py ✅ DONE
│ ├── transcriber.py ✅ DONE
│ ├── hook_detector.py ✅ DONE — tested, real results, see below
│ ├── list_groq_models.py ✅ utility, keep for future diagnostics
│ ├── video_clipper.py 🔄 NEXT UP
│ └── main.py ⬜ NOT STARTED
├── tests/
├── .env (gitignored)
├── .env.example
├── .gitignore
├── LICENSE (MIT)
├── README.md
├── requirements.txt
└── PROJECT_STATE.md (this file)

text


## MODULE STATUS

### config.py — ✅ DONE
GROQ_LLM_MODEL added (env-overridable, default openai/gpt-oss-120b).
GEMINI_MODEL/GEMINI_API_KEY constants can remain unused or be removed later.

### audio_extractor.py — ✅ DONE
Tested on real ~300MB/30min podcast video.

### transcriber.py — ✅ DONE
Tested: 1414.6s audio -> 3646 words via Groq Whisper in ~3 seconds.

### hook_detector.py — ✅ DONE
detect_hooks() now calls Groq (openai/gpt-oss-120b) instead of Gemini.
Pydantic validation (ViralSegment, ViralSegmentList), overlap removal,
timestamp clamping, retry-with-backoff all working.
REAL TEST RESULT on sample podcast transcript (3 clips requested):
  [968.8s-1017.5s] (48.7s) score=9 "Who Really Defines Success? The
    Blueprint Myth"
  [275.8s-306.7s] (30.9s) score=8 "Golden Handcuffs: When a Raise Traps You"
  [517.6s-562.8s] (45.2s) score=8 "Why Leaving a Top Job Feels Terrifying"
These are genuinely good, punchy, standalone-viable headlines — validates
the core "intelligence" layer of the product works as intended.

### video_clipper.py — NEXT UP
Job: given the original video + a ViralSegment, trim to exact timestamps,
center-crop 16:9 to 9:16 (1080x1920), generate word-level karaoke-style ASS
captions, burn them in, encode final mp4 per clip.

### main.py — NOT STARTED
CLI orchestrator tying all 4 modules together end-to-end.

## NEXT IMMEDIATE STEP
Write and paste src/video_clipper.py. Test it using sample.mp4 +
sample_hooks.json (already have real timestamps to test against).

## OPEN QUESTIONS / THINGS TO DECIDE LATER
- Exact caption animation style (karaoke word-highlight vs simple bold).
- Whether to support YouTube URL ingestion via yt-dlp in v1 or defer to v2.
- Web frontend framework — not relevant until Phase 3.
- Long-term: should we add a secondary/backup LLM provider in case Groq ever
  has issues? Not urgent — Groq has been 100% reliable in testing so far.