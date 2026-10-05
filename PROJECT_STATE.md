# AutoClip AI — Project State

> Single source of truth for project context. If chat history is lost, paste
> this file back to the AI assistant to restore full context instantly.

Last updated: Phase 1 — README done, demo content validated, rate-limit fix applied

---

## WHAT THIS PROJECT IS
AutoClip AI — ingests long-form video (MP4/YouTube), outputs viral 9:16 vertical
short clips with burned-in dynamic captions. Open-core strategy: core pipeline
is open-sourced on GitHub, monetization comes later via manual productized
service (€15-25/video) then a self-serve micro-SaaS.

## WHO IS BUILDING THIS
Diego, 15, Belgium. €0 capital. Building in a GitHub Codespace. Uses Claude
(via arena.ai) for architecture + full code, GitHub Copilot for inline
autocomplete only.

## 🎉 MILESTONES SO FAR
1. Full pipeline validated end-to-end on 3 real test cases: ~24min podcast,
   ~2.5hr podcast (26,796 words — proved no length cap needed), and a public-
   domain NASA Houston We Have a Podcast episode (~58min, 10,925 words) used
   specifically as README/demo material (public domain = no copyright risk).
2. NASA demo run produced 5/5 clips, all scoring 8-9/10 virality, with
   genuinely strong, specific, clickable headlines (e.g. "The Call That
   Changed My Astronaut Career", "Rainstorm Rescue: The Water-Funnel Story").
   Confirmed demo-quality, good candidate for README showcase GIF.
3. README.md fully rewritten with architecture diagram, setup guide, honest
   limitations section, and roadmap — repo is now publicly presentable.

## KEY STRATEGIC DECISIONS (don't re-litigate these without reason)
- Open-source first (MIT license), monetize later.
- Transcription: Groq API (whisper-large-v3-turbo). Handles any length via
  automatic chunking + ffmpeg stream-copy splitting.
- Hook detection LLM: Groq (openai/gpt-oss-120b), transcript chunked into
  ~2500-word windows to stay under 8000 TPM limit, results merged + ranked.
- Payments (later): direct IBAN bank transfer for manual service tier.
  Legal/Bryo/parent stuff deliberately DEFERRED until real paying volume.
- Production hosting (later): Oracle Cloud Always Free VM. Not relevant yet.
- Target niche: Benelux podcasters/creators underserved by English-first
  tools like Opus Clip/Submagic; CapCut is the real free competitor to watch.
- For public demo content: use public-domain sources (e.g. NASA podcasts)
  to avoid any copyright complications in README/marketing material.

## KNOWN PRODUCT LIMITATION (still accurate, documented in README too)
Center-crop 9:16 fails on: (1) videos with baked-in subtitles near bottom
of frame, (2) multi-person side-by-side horizontal layouts. Shared weakness
across ALL competitors, not unique to us. v2 opportunity: smart content-
aware cropping (face/speaker detection). NOT required for v1 launch.

## ENVIRONMENT
- GitHub repo: https://github.com/HackerDpro/autoclip-ai (public, MIT license)
- Dev: GitHub Codespaces, devcontainer.json installs ffmpeg + pip deps.
- `.env` holds GROQ_API_KEY, GROQ_LLM_MODEL (gpt-oss-120b).
- Confirmed working: ffmpeg 7.1.5, python-dotenv, groq==1.7.0 (NOT 0.11.0).

## KNOWN GOTCHAS (don't repeat these mistakes)
- groq==0.11.0 is BROKEN (httpx incompatibility). Must use groq>=1.7.0.
- devcontainer.json changes require "Codespaces: Rebuild Container" command.
- Gemini free tier abandoned entirely — unreliable 503s across all models.
- Groq free tier has TWO separate limits to respect:
    1. TPM (tokens per minute) — ~8000 on gpt-oss-120b. Handled via
       transcript chunking (MAX_WORDS_PER_HOOK_CHUNK in hook_detector.py).
    2. RPM (requests per minute) — hit this on a 5-chunk run (4/5 chunks got
       429 errors). The Groq SDK's built-in retry handled it automatically
       (waited up to 48s per retry) so the run still succeeded, but this
       wastes significant time on longer videos with more chunks. FIX
       APPLIED: added a 2-second time.sleep() between sequential chunk
       requests in detect_hooks() to proactively avoid most 429s rather
       than relying on reactive retries.
- Model names on Groq can be deprecated/renamed. Run src/list_groq_models.py
  to check current availability if needed.
- Git LFS pre-push hook can block pushes if git-lfs isn't installed. Fix:
  `rm .git/hooks/pre-push` (we don't use LFS).

## REPO STRUCTURE (current)
autoclip-ai/
├── .devcontainer/devcontainer.json
├── assets/fonts/
├── data/
│ ├── input/demo.mp4 (NASA podcast, public domain, for demo)
│ ├── temp/demo_transcript.json (cached, reusable)
│ └── output/clip_*.mp4 (5 real demo clips, all high quality)
├── src/
│ ├── init.py
│ ├── config.py ✅ DONE
│ ├── audio_extractor.py ✅ DONE
│ ├── transcriber.py ✅ DONE
│ ├── hook_detector.py ✅ DONE — chunked, rate-limit-aware
│ ├── video_clipper.py ✅ DONE
│ ├── main.py ✅ DONE
│ └── list_groq_models.py ✅ utility
├── tests/ ⬜ EMPTY — no automated tests yet
├── .env (gitignored) / .env.example
├── .gitignore
├── LICENSE (MIT)
├── README.md ✅ DONE — full rewrite with demo plan
├── requirements.txt
└── PROJECT_STATE.md (this file)

text


## PHASE 0: ✅ COMPLETE | PHASE 1: 🔄 IN PROGRESS

## WHAT'S NEXT
1. Pick the BEST single clip from the NASA demo batch (visually check all 5
   downloaded clips) — likely "The Call That Changed My Astronaut Career"
   or "Rainstorm Rescue" based on headline strength — and turn it into a
   short GIF/screen-recording for the README's demo section.
2. Clean up requirements.txt (remove unused google-genai reference if still
   present).
3. Begin outreach prep: identify 5-10 Belgian/Dutch podcasters/creators to
   DM with free sample clips made from their own content.
4. Longer-term (v2, not urgent): smart content-aware cropping, YouTube URL
   ingestion, web UI wrapper.

## OPEN QUESTIONS / THINGS TO DECIDE LATER
- Smart content-aware cropping (face/speaker detection) — real v2 feature.
- Whether to support YouTube URL ingestion via yt-dlp in v1 or defer to v2.
- Web frontend framework — not relevant until Phase 3.
- Exact outreach message template for first podcaster DMs — draft needed.