Plan


  1) Text normalization and reading-friendly rewriting

  • Add text normalization rules for speech:
    • Numbers, ordinals, years, dates, times, currencies, measurements, roman numerals, math
      symbols, URLs, email/hashtags.
    • Acronyms and all-caps: detect and either spell out or reduce emphasis (not literally
      spelled unless appropriate).
    • Expand common abbreviations contextual to sentence position (“Dr.”, “e.g.” already
      guarded; add many more).
  • Dialogue and quotes awareness:
    • Detect quoted speech vs narration to inform prosody (slightly different pacing and
      emphasis).
    • Detect parentheticals and appositives; compress slightly and add micro-pauses
      before/after.
  • Optional de-emoji/emoticon mapping to sentiment cues rather than stripping or read as
    “smiley”.

  Where: new text/normalize_for_speech.py, called before segmentation.
  Defaults:
  • “2025” → “twenty twenty-five”
  • “$15.99” → “fifteen dollars and ninety-nine cents”
  • “3rd” → “third”; “III” (roman) → “the third”
  • “10 km” → “ten kilometers”


  2) Smarter segmentation (prosody-aware)

  • Sentence splitting:
    • Extend abbreviations list; handle ellipses, em-dashes, questions in quotes, “?!”.
    • Merge very short sentences into neighbors to avoid choppy delivery.
  • Phrase-level segmentation:
    • Within sentences, split on commas/semicolons/colon/parentheticals/em-dash into phrases.
    • Annotate phrase type to drive pause lengths and micro-speed.

  Where: enhance segmentation.py and produce Sentence with phrase boundaries, or new structure
  Utterance with phrases.

  3) Prosody planning layer

  • Introduce a ProsodyPlan over chunks:
    • For each phrase: target speech rate multiplier, optional emphasis markers, intended pause
      after.
    • Compute pause durations by punctuation and context:
      • comma: 120–180 ms
      • semicolon/colon: 180–260 ms
      • sentence end: 400–650 ms
      • paragraph end: 900–1400 ms
      • chapter break: 2–3.5 s
    • Scale by global speed and a “naturalness” factor (wider dynamic range when naturalness
      high).
    • Shorten pauses inside quotes (snappier dialogue), lengthen before/after scene or topic
      shifts.
    • Add brief pre-breath for long sentences (>12 seconds) or at paragraph starts.

  Where: new prosody/planner.py. Input: List[Sentence] or List[Chunk]. Output: list of ProsodyStep
   { text_span, speed_mult, emphasis, pause_ms, optional breath }.

  4) Synthesis control

  • If KittenTTS supports prosody/SSML-like controls:
    • Prefer native controls: per-phrase speed, emphasis, breaks.
  • If not (current API only supports voice/speed):
    • Emulate prosody:
      • Synthesize per-phrase with per-phrase speed multiplier (call engine with speed = base *
        speed_mult).
      • Concatenate with targeted silence buffers.
      • Optional tiny crossfades (10–20 ms) to eliminate clicks.
    • Fallback/backoff remains, but now uses phrase units before character-length splits.

  Where: extend synthesis.py with synthesize_with_prosody(engine, prosody_steps, voice,
  base_speed) reusing cache keyed by (text_span, voice, effective_speed).

  5) Dialogue handling and speaker styling

  • Dialogue detection (quotes or em-dash at line start):
    • Option A (single-voice): subtle variation only—slightly slower, softer, more pitch
      variance when “speaking”, normal for narration.
    • Option B (multi-voice optional): allow assigning a secondary voice ID for dialogue, with
      automatic blending back to narrator at quotation end.
  • Parentheticals (“he said”, stage directions): faster pace, reduced volume and slight
    band-limit to signal aside.

  Where: dialogue heuristics in prosody planner; pass a voice_override or style per step.

  6) Audio assembly and post-processing

  • Joins:
    • Always crossfade joins (10–20 ms) to avoid clicks; apply DC offset correction if needed.
  • Breaths:
    • Optional realistic breaths at paragraph starts or after long utterances:
      • Generate with shaped noise + band-pass (300–1500 Hz) + envelope 200–350 ms, or use a
        small library of recorded breaths.
      • Mix at -36 to -30 dBFS relative to voice; skip if preceding context is a hard stop with
        long pause.
  • De-essing and light compression:
    • Mild de-esser around 5–8 kHz; 1–2 dB GR on sibilant bursts.
    • Gentle compressor (1.6:1, slow attack 20–30 ms, medium release 200–300 ms) to smooth
      peaks.
  • Loudness:
    • Keep integrated target: -18 LUFS mono (already implemented).
    • Add true-peak limiting at -1 dBTP to avoid MP3 overs.
  • Room tone (optional):
    • Very low-level noise bed (-60 to -50 dBFS) to mask complete digital silence and smooth
      pauses.

  Where: expand audio/post.py with DSP helpers; add assemble_with_pauses crossfades and breath
  mix.

  7) Chaptering and metadata

  • Improve chapter detection beyond regex:
    • Use layout features from PDF (font size, y-position, isolated lines).
    • Build chapter index and write MP3/MP4 chapters and ID3 metadata (title, author).
  • Insert longer lead-in/out silences around chapters; optional chime.

  Where: enhance ingestion/pdf_reader.py to capture heading candidates; write tags with mutagen.

  8) Controls and UX

  • CLI and UI knobs:
    • Voice, base speed.
    • Naturalness (scales pause and speed variance).
    • Dialogue mode: single-voice vs dual-voice.
    • Pause scaling (global).
    • Breath enable and level.
    • Target LUFS.
    • Test run: page/paragraph ranges, chunk limit (already partly present).
  • UI preview of the ProsodyPlan and pause markers; quick A/B of two settings.

  Where: extend cli/main.py and ui/app.py.

  9) Performance and reliability

  • Caching: already per-chunk—extend to per-phrase with speed keying.
  • Parallelization: synthesize multiple phrases concurrently but preserve order in assembly;
    throttle to avoid GPU/CPU over-subscription.
  • Robust backoff: keep character-based fallback but prefer phrase boundaries; estimate silence
    when synthesis fails to preserve timing.


  10) Quality checks and tests

  • Automated checks:
    • Catch too-short pauses at sentence ends (<250 ms), too-long micro-pauses (>400 ms for
      commas).
    • Speech rate stats (words per minute) within 140–190 for narration baseline; configurable.
    • Detect clipping, DC offset, clicks at joins.
  • Snapshot tests on small passages: ensure determinism of ProsodyPlan and stable audio length
    ranges.
  • Listening tests:
    • Standardized passages (dialogue, technical text, list, dramatic scene).
    • AB compare with/without planner; track MOS over iterations.


  Recommended defaults

  • Base speed 0.95–1.05 per voice; naturalness 0.6.
  • Pause multipliers:
    • comma 150 ms, semicolon/colon 220 ms, sentence 520 ms, paragraph 1100 ms, chapter 2500 ms.
    • Scale by speed (inverse) and naturalness (range).
  • Crossfade 15 ms at all joins.
  • De-esser: threshold tuned per voice; compressor 1.6:1, -18 LUFS target, -1 dBTP limiter.
  • Dialogue: +5% slower, -2 dB, slightly increased pitch variance; asides -3 dB, +15% faster.


  Integration map to current code

  • text/normalize_for_speech.py: full TTS text normalization rules; call from cli/main.py
    before sentence split.
  • text/segmentation.py: expand abbreviations, richer punctuation, phrase detection API.
  • prosody/planner.py: produce ProsodyPlan/ProsodyStep.
  • tts/synthesis.py: add synthesize_with_prosody, caching per phrase+speed; update
    assemble_with_pauses to do micro-pauses and crossfades.
  • audio/post.py: add crossfade, breath generator/mixer, de-esser, compressor, true-peak
    limiter, optional room tone.
  • ingestion/pdf_reader.py: capture heading candidates; utils/io for chapters metadata with
    mutagen.
  • cli/main.py and ui/app.py: expose controls, preview plan.
  • Implement incrementally: start with planner + per-phrase speed/pause + crossfades; then
    dialogue heuristics; then breaths; then DSP polish.
  • Rollout test scripts for evaluation; keep caching and backoff as safety net.
  • Ensure everything stays model-agnostic: if KittenTTS later exposes SSML, map planner to
    native controls and remove emulation.
  • Add a “strict-lines” enhanced path for technical books with tables/indexes (you already have
    a mode—keep improving skipping logic and normalization to avoid reading headers/footers).
  • Consider an optional “emotion mode” flag that applies slightly longer pauses and wider speed
    variance for fiction; keep a tighter profile for non-fiction.
  • For long books, write chaptered M4B with embedded chapters and cover art; add cue sheet
    export for audiobook platforms.
  • CI step to run a 2-page sample through the pipeline, asserting no regressions in timing and
    loudness.
  • Document recommended voice presets per genre with target speed/pauses.
  • Eventually: add a small prosody ML model to predict pause/emphasis from text features; keep
    current heuristic planner as baseline.
  • Add a “breathing budget” per minute: avoid sounding breathless or too breathy—dynamic
    placement by sentence length.
  • Provide a “consistency lock” option that fixes base pitch/speed profile across chapters to
    avoid drift.
  • Keep the backoff’s silence estimation logic to preserve rhythm in worst-case model failures.
  • Use deterministic random seeds for noise/breaths so previews are reproducible.
  • Tune defaults with 30–60 minutes of listening tests across genres; update presets
    accordingly.