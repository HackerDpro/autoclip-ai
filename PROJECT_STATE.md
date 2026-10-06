# AutoClip AI — Project State

> Single source of truth for project context. If chat history is lost, paste
> this file back to the AI assistant to restore full context instantly.

Last updated: Phase 1.5 COMPLETE — Face tracking fully stable across all clip lengths

---

## WHAT THIS PROJECT IS
AutoClip AI — ingests long-form video (MP4/YouTube), outputs viral 9:16 vertical
short clips with burned-in dynamic captions AND face-aware tracked cropping.
Open-core: pipeline open-sourced, monetization later via manual service
(€15-25/video) then self-serve micro-SaaS.

## WHO IS BUILDING THIS
Diego, 15, Belgium. €0 capital. GitHub Codespace. Claude (via arena.ai) for
architecture + code, GitHub Copilot for inline autocomplete only.

## 🎉 MILESTONES SO FAR
1. Full pipeline validated end-to-end on short, very long (2.5hr), and
   public-domain demo (NASA podcast) content.
2. README.md public-presentable with demo GIF.
3. Face-tracking cropping built and FULLY STABILIZED:
   - 3 modes: SINGLE (dynamic tracking), SPLIT (dual speaker split-screen,
     not yet tested on real 2-person footage), CENTER (safe fallback).
   - Samples faces across full clip duration (not a snapshot), builds
     smoothed time-varying crop path via ffmpeg expression interpolation.
   - THREE real bugs found via testing and fixed:
     a) 3+ faces caused wrong-person tracking -> now treated as a gap,
        holds last known good position instead of guessing.
     b) ffmpeg crash on clips 55s+ ("Invalid argument" filter config
        error) -> root cause: too many interpolation points produced an
        ffmpeg expression too complex to parse. FIXED two ways: lowered
        SAMPLE_FPS from 2.0 to 0.5 (still smooth after interpolation+
        smoothing), AND added a hard cap of 25 max expression points in
        _build_crop_expression (downsamples evenly if exceeded) — this
        failure mode is now structurally impossible regardless of clip
        length.
     c) (safety net, not a fix but good engineering) video_clipper.py has
        an automatic fallback: if dynamic crop ever fails for any reason,
        automatically retries with safe static center-crop rather than
        losing the clip. Belt-and-suspenders alongside fix (b).
   - CONFIRMED WORKING on real content: 55.2s and 54.1s clips (previously
     crashing) now succeed with full dynamic tracking, zero fallback
     triggers, in the most recent test run.

## KEY STRATEGIC DECISIONS (don't re-litigate these without reason)
- Open-source first (MIT license), monetize later.
- Groq API only for both transcription (whisper-large-v3-turbo) and hook
  detection (openai/gpt-oss-120b). Gemini abandoned (unreliable free tier).
- Hook detection chunks transcripts (~2500 words/chunk, TPM limit) with
  inter-chunk delay (RPM limit mitigation, not fully eliminated — see
  known issue below).
- Face-aware cropping is core, not optional — addresses real market-fit
  gap (most real content isn't perfectly centered single-speaker).
- Payments (later): direct IBAN bank transfer. Legal/Bryo/parent stuff
  deliberately DEFERRED until real paying volume.
- Target niche: Benelux podcasters/creators/vloggers/streamers, going
  SMALL/LOCAL in outreach (not big accounts who already have editors).

## KNOWN REMAINING ISSUES (not urgent, just tracked honestly)
- Groq 429 rate-limiting still occurs ~2-4 times per 5-chunk hook detection
  run, adding 1-2 min of wait time via SDK auto-retry. Self-heals every
  time, never fails the pipeline, but adds latency. Possible future fix:
  smarter adaptive delay between chunks based on actual TPM usage reported
  in error messages, or reduce chunk count. NOT URGENT — works reliably.
- Total pipeline time ~5-6 min for 3 clips (cached transcript) — mostly
  rate-limit waits + ffmpeg encoding. Fine for personal testing/demos, will
  need revisiting once real client turnaround expectations exist.
- SPLIT mode (dual speaker) still UNTESTED on real footage — NASA demo
  didn't trigger it. Diego to test with real 2-person side-by-side video.
- No active-speaker detection for 3+ person scenes (documented, accepted
  limitation, not urgent — true fix needs lip-sync/audio correlation).

## ENVIRONMENT
- GitHub repo: https://github.com/HackerDpro/autoclip-ai (public, MIT)
- `.env`: GROQ_API_KEY, GROQ_LLM_MODEL (gpt-oss-120b)
- Confirmed working: ffmpeg 7.1.5, groq==1.7.0 (NOT 0.11.0),
  opencv-python-headless, face_cropper.py SAMPLE_FPS=0.5

## KNOWN GOTCHAS (don't repeat these mistakes)
- groq==0.11.0 is BROKEN (httpx incompatibility). Must use groq>=1.7.0.
- devcontainer.json changes require "Codespaces: Rebuild Container".
- Gemini free tier abandoned — unreliable 503s across all models.
- Groq TPM (8000) handled via transcript chunking. RPM 429s still occur
  occasionally — SDK auto-retry handles them, costs time, not urgent fix.
- Haar Cascade face detection does NOT return faces in speaker-order.
  Never assume "biggest 2 faces" = "the 2 speakers" when 3+ present.
- ffmpeg dynamic crop expressions MUST be capped in complexity (max ~25
  interpolation points) or filter graph parsing fails on longer clips —
  this was a real production bug, not theoretical. Always keep the
  center-crop fallback path alongside any future crop logic changes.
- Git LFS pre-push hook can block pushes if git-lfs isn't installed. Fix:
  `rm .git/hooks/pre-push`.

## REPO STRUCTURE (current)
autoclip-ai/
├── .devcontainer/devcontainer.json
├── assets/ (fonts/, demo/demo.gif)
├── data/ (input/, temp/, output/ — gitignored)
├── src/
│ ├── config.py, audio_extractor.py, transcriber.py,
│ │ hook_detector.py, face_cropper.py, video_clipper.py,
│ │ main.py, list_groq_models.py — ALL ✅ DONE & STABLE
├── tests/ ⬜ EMPTY — no automated tests yet
├── .env / .env.example / .gitignore
├── LICENSE (MIT) / README.md / requirements.txt
└── PROJECT_STATE.md (this file)

text


## WHAT'S NEXT
1. Diego to test SPLIT mode on real 2-person side-by-side footage —
   last major untested code path.
2. Resume outreach: target SMALL/local Benelux creators, lead with free
   finished sample clips, not cold pitches.
3. Not urgent: smarter Groq rate-limit handling, active-speaker detection,
   YouTube URL ingestion, web UI wrapper.

## OPEN QUESTIONS
- Active-speaker detection approach — v2+, not urgent.
- YouTube URL ingestion — defer.
- Web frontend — defer to Phase 3.