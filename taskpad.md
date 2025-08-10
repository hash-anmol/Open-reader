# KittenTTS Audiobook Project – Taskpad

This document captures goals, constraints, architecture, processing strategy, performance considerations, and implementation milestones for a local audiobook generator built on `KittenML/kitten-tts-nano-0.1`.

## Goals & Requirements

- Build a local, GPU-free audiobook generator similar to “elevenreader”.
- Input: PDF, later EPUB.
- Output: MP3 (default), optionally WAV/M4B later.
- Preferred interface: Gradio UI (primary) after a successful CLI run.
- Defaults:
  - Voice: female (easy switching between voices).
  - Speed: model default (expose configurable speed).
  - Output destination: `output/test_mp3/` for now.
- Robust text handling: remove/normalize non-TTS-safe characters, handle images/tables, fix reading order issues as feasible.
- Efficiency & low latency: streaming, caching, minimal memory footprint, compiled regex, avoid heavy NLP unless necessary.
- Respect short context of the nano model: chunk text ~300–350 characters, hard cap ~400.

## Constraints & Model Notes

- Model: `KittenML/kitten-tts-nano-0.1` (Apache-2.0)
  - Assets: `kitten_tts_nano_v0_1.onnx`, `voices.npz`, `config.json`
  - Sample rate: 24 kHz mono
  - Voices: typical set `expr-voice-{2,3,4,5}-{m,f}`; enumerate via `KittenTTS.available_voices`
  - API (from wheel): `KittenTTS(repo).generate(text, voice=..., speed=...) -> np.float32 array`
- Context limits: best results under ~400 characters per synthesis call.

## Folder Architecture

```
kitten_tts_audiobook/
├── taskpad.md                    # This document
├── pyproject.toml                # Build/config (later)
├── requirements.txt              # Runtime deps (wheel + libs)
├── README.md                     # Usage guide (later)
├── src/
│   └── kitten_audiobook/
│       ├── __init__.py
│       ├── cli/
│       │   └── main.py           # CLI entry: pdf -> mp3
│       ├── ui/
│       │   └── app.py            # Gradio UI (after CLI success)
│       ├── ingestion/
│       │   ├── pdf_reader.py     # PDF -> structured text (pages, paragraphs)
│       │   └── structure.py      # Data classes for document structure
│       ├── text/
│       │   ├── cleaning.py       # Unicode normalize, strip/replace, de-hyphen, etc.
│       │   ├── segmentation.py   # Sentence/paragraph segmentation
│       │   └── chunking.py       # Chunk builder honoring model limits
│       ├── tts/
│       │   ├── engine.py         # KittenTTS wrapper, voice mgmt, caching
│       │   └── synthesis.py      # Chunk -> audio, pause planning
│       ├── audio/
│       │   ├── assembly.py       # Concatenate audio, insert silences
│       │   ├── post.py           # Loudness norm, MP3 encode, metadata
│       │   └── utils.py          # Audio helpers (silence gen, formats)
│       └── utils/
│           ├── io.py             # File paths, temp dirs, hashing
│           ├── logging.py        # Structured logging / progress
│           └── timers.py         # Simple timing utilities
├── data/
│   ├── samples/                  # Sample PDFs (user-provided)
│   └── cache/                    # Per-chunk synthesis cache (text+voice+speed hash)
├── output/
│   └── test_mp3/                 # Default MP3 output location
└── tests/
    ├── test_text_processing.py
    ├── test_chunking.py
    ├── test_tts_smoke.py
    └── test_audio_assembly.py
```

## Dependencies

- kittentts wheel: `https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl`
- numpy, soundfile, tqdm, pymupdf, pydub, pyloudnorm, regex
- OS tool: `ffmpeg` (MP3/M4B encode)

`requirements.txt` (initial):

```
https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl
numpy
soundfile
pymupdf
regex
pydub
pyloudnorm
tqdm
```

## End-to-End Pipeline

1. Ingest PDF
   - Extract text with `pymupdf` preserving reading order per page.
   - Drop images/tables; attempt to skip figure/table captions when detected.
   - Remove headers/footers via repetition heuristics (e.g., same line appears on many pages at top/bottom).
   - Collect structure: `Document -> Chapters? -> Pages -> Paragraphs -> Lines`.

2. Text Normalization (cleaning)
   - Unicode normalize (NFKC). Strip/control-char removal (except necessary whitespace and standard punctuation).
   - Replace smart quotes/dashes with ASCII equivalents; turn bullets (•, ◦, ▪) to commas or pauses.
   - Remove ligatures (ﬁ → fi), zero-width chars, directional marks, BOMs.
   - Replace emojis and unsupported symbols with verbalizable tokens or drop (configurable; default drop).
   - Collapse whitespace; de-hyphenate line-break splits (e.g., `hy-
     phen` → `hyphen`) with conservative rules.
   - Normalize punctuation spacing; cap repeated punctuation.

3. Segmentation
   - Paragraph detection from PDF blocks; fallback to double newlines or large vertical gaps.
   - Sentence segmentation via lightweight regex rules with abbreviation guard list (e.g., `Mr., Dr., i.e., e.g., etc.`) to avoid over-splitting.
   - Output: `Segment` objects with type: sentence/paragraph boundary markers.

4. Chunking (context-aware)
   - Build chunks targeting 300–350 chars; hard cap 400.
   - Prefer ending at sentence boundaries; join short sentences if safe.
   - Annotate boundaries for pause planning: sentence, paragraph, chapter.
   - Hash each chunk with voice+speed to key cache filenames.

5. Synthesis
   - For each chunk, use `KittenTTS.generate(text, voice, speed)`.
   - Sequential by default (ONNX CPU); expose threads via env if available.
   - Write per-chunk WAV to `data/cache/` to support resume and reuse.
   - Track progress with `tqdm`.

6. Assembly & Pauses
   - Insert silences: 0.20 s between sentences/chunks, 0.8 s between paragraphs, 2.0 s at chapter boundaries.
   - Concatenate using `soundfile` for WAV or `ffmpeg concat` for efficient MP3 finalization.

7. Post-processing & Export
   - Optional EBU R128 loudness normalization (pyloudnorm) for consistent perceived loudness.
   - Encode to MP3 using `ffmpeg` (e.g., mono 64–96 kbps). Default: 64 kbps mono VBR or CBR 80 kbps.
   - Add basic ID3 metadata (title, author if parsed from PDF metadata or filename).

## Performance & Latency Strategy

- Streaming & memory: do not hold full audio in RAM; write chunk WAVs and perform a single-pass concat+encode.
- Caching: per-chunk cache keyed by `sha1(text + voice + speed + model_version)`; reuse across runs.
- Compiled regex for cleaning/segmentation; avoid heavy NLP stacks.
- I/O efficiency: batch writes, use temp concat list for `ffmpeg`.
- Parallelism: TTS is sequential for coherence; consider limited parallel text pre-processing and I/O.
- Failure isolation: retries on transient ONNX errors for a chunk; skip with log if irrecoverable.

## CLI (primary for test)

- Command: `kitten-audiobook INPUT_PDF [options]`
- Options:
  - `--voice`: default `expr-voice-5-f` (female), list via `--list-voices`.
  - `--speed`: default `1.0`.
  - `--out-dir`: default `output/test_mp3/`.
  - `--chapterize`: `auto|none` (auto uses PDF bookmarks/headings heuristics).
  - `--max-chars`: default `350` (hard cap handled internally at 400).
  - `--silence`: `chunk=0.20,paragraph=0.80,chapter=2.0`.
  - `--normalize-loudness`: toggle; default on.
  - `--resume`: reuse cache, skip existing chunks.
  - `--verbose`: detailed logs.

## Gradio UI (after CLI validates)

- File upload (PDF), voice dropdown (default female), speed slider.
- Real-time progress bar; tail logs.
- Start/pause/cancel; output download link to MP3 in `output/test_mp3/`.

## Edge Cases & Handling

- Headers/footers/page numbers: detect repetition and remove.
- Hyphenation across lines: conservative join only when line ends with hyphen and next begins with lowercase.
- Figures/tables: drop blocks with low text density or containing many non-letters; drop OCR-needed pages (optional OCR later).
- Non-TTSable chars: emojis, control chars, math, dingbats → remove or replace with pauses.
- Long quotes/code snippets: trim or compress spacing; treat as normal text unless excessive.
- Non-English: assume English for segmentation; future: language detection & voice switch.

## Installation (macOS, Python 3.10)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Also ensure ffmpeg is present
# macOS: brew install ffmpeg
```

## Milestones

1. Bootstrap repo structure and requirements.
2. Implement ingestion + cleaning + segmentation + chunking.
3. Implement TTS engine wrapper + caching + per-chunk WAV.
4. Implement assembly + MP3 export + loudness normalization.
5. Wire CLI (`kitten-audiobook`) and test on sample 50-page PDF.
6. Add Gradio UI leveraging the same pipeline.
7. Add chapter metadata and M4B (optional).

## Open Decisions & Defaults (current)

- Default voice: `expr-voice-5-f` (switchable).
- Default speed: `1.0`.
- Default output format: MP3; out dir: `output/test_mp3/`.
- Target chunk size: 350 chars; cap 400.
- Pauses: 0.20s (chunk), 0.80s (paragraph), 2.0s (chapter).

## Test Plan

- Smoke test with short paragraph.
- End-to-end with user-provided 50-page PDF.
- Validate: voice selection, speed, chunk boundaries audible, no clipping, reasonable loudness, MP3 plays in common players.

## Notes

- Keep code modular, readable, and well documented with clear function names and docstrings.
- Prefer early returns, explicit types where helpful, and unit tests for text functions.
