from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from ..utils.io import ensure_dirs, derive_output_target
from ..utils.logging import configure_logging
from ..ingestion.pdf_reader import read_pdf
from ..ingestion.txt_reader import read_txt
from ..text.cleaning import normalize_text
from ..text.segmentation import split_into_sentences, Sentence
from ..text.chunking import build_chunks
from ..tts.engine import TTSEngine, TTSConfig
from ..tts.synthesis import (
    synthesize_chunks,
    assemble_with_pauses,
    PausePlan,
    compute_chapter_ranges,
    assemble_chapter_audios,
)
from ..audio.assembly import save_wav
from ..audio.post import loudness_normalize, encode_mp3


def run(pdf_path: Path, voice: str, speed: float, out_dir: Path, max_chars: int, normalize_loudness: bool, verbose: bool, limit_chunks: int | None = None, limit_pages: int | None = None, txt_path: Path | None = None, strict_lines: bool = False, start_from: str | None = None) -> Path:
    ensure_dirs()
    configure_logging(verbose=verbose)

    if txt_path is not None:
        logging.info("Reading TXT: %s", txt_path)
        document = read_txt(txt_path)
    else:
        logging.info("Reading PDF: %s", pdf_path)
        document = read_pdf(pdf_path)

    # Clean and segment
    sentences: list[Sentence] = []
    if strict_lines:
        # Strict line-by-line: accumulate across lines until sentence end; optionally skip index until heading
        paragraph_index = 0
        buffer = ""
        started = start_from is None
        after_heading = False
        heading = (start_from or "").strip().lower() if start_from else None
        for page in document.pages:
            if limit_pages is not None and page.number > limit_pages:
                break
            for paragraph in page.paragraphs:
                line = normalize_text(paragraph.text)
                if not line:
                    continue
                low = line.strip().lower()
                if not started and heading and low == heading:
                    # found the heading; now wait for first body line
                    started = True
                    after_heading = True
                    continue
                if not started:
                    continue  # skip before heading entirely
                if after_heading:
                    # skip index-like lines until we hit a plausible body line
                    letters = sum(ch.isalpha() for ch in line)
                    ratio = letters / max(1, len(line))
                    is_index = (
                        low.startswith("page no") or
                        low.startswith("chapter ") or
                        "the aidea of india" in low or
                        low in {"contents", "executive summary"} or
                        len(line) <= 3 or ratio < 0.3
                    )
                    if is_index:
                        continue
                    else:
                        after_heading = False

                # Hyphen wrap join
                if buffer.endswith('-'):
                    buffer = buffer[:-1] + line.lstrip()
                else:
                    # Skip running title/footer lines like "The AIdea of India: 2025" with optional page number
                    low_no_space = low.replace(' ', '')
                    if low.startswith("the aidea of india:") or "the aidea of india" in low:
                        # e.g., "The AIdea of India: 2025" or the same followed by a number
                        import re as _re
                        if _re.match(r"the\s+aidea\s+of\s+india:\s*2025(\s*\d+)?$", low):
                            continue
                    buffer = (buffer + ' ' + line).strip() if buffer else line

                # Extract complete sentences from buffer
                parts = split_into_sentences(buffer + ' ', paragraph_index)
                if len(parts) >= 2:
                    *complete, tail = parts
                    for s in complete:
                        sentences.append(Sentence(text=s.text.strip(), paragraph_index=paragraph_index))
                        paragraph_index += 1
                    buffer = tail.text.strip()
        if buffer.strip():
            sentences.append(Sentence(text=buffer.strip(), paragraph_index=paragraph_index))
    else:
        paragraph_counter = 0
        for page in document.pages:
            if limit_pages is not None and page.number > limit_pages:
                break
            # Paragraph-like buffering within a page
            line_buffer: list[str] = []
            for paragraph in page.paragraphs:
                cleaned_line = normalize_text(paragraph.text)
                if not cleaned_line:
                    continue
                # Merge hyphenated wrap
                if line_buffer and line_buffer[-1].endswith('-') and cleaned_line[:1].islower():
                    line_buffer[-1] = line_buffer[-1][:-1] + cleaned_line.lstrip()
                    continue
                line_buffer.append(cleaned_line)
                if cleaned_line.endswith(('.', '!', '?')) and len(" ".join(line_buffer)) >= 80:
                    paragraph_text = " ".join(line_buffer)
                    sentences.extend(split_into_sentences(paragraph_text, paragraph_index=paragraph_counter))
                    paragraph_counter += 1
                    line_buffer = []
            if line_buffer:
                paragraph_text = " ".join(line_buffer)
                sentences.extend(split_into_sentences(paragraph_text, paragraph_index=paragraph_counter))
                paragraph_counter += 1

    logging.info("Building chunks with target/hard cap: %d/360", max_chars)
    chunks = build_chunks(sentences, target_chars=max_chars, hard_cap=360)
    if limit_chunks is not None and limit_chunks > 0:
        chunks = chunks[:limit_chunks]
    if not chunks:
        raise RuntimeError("No text chunks produced from input PDF.")

    # Prepare output target and write a human-reviewable transcript of chunks
    target = derive_output_target(pdf_path, out_dir)
    transcript_path = target.directory / f"{target.base_name}.transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(chunks):
            # Remove noisy markers; show page/paragraph ids clearly
            f.write(f"[chunk {idx:04d} | paragraph {ch.paragraph_index:04d}]\n{ch.text}\n\n")
    logging.info("Wrote transcript: %s", transcript_path)

    # TTS
    engine = TTSEngine(TTSConfig())
    logging.info("Using voice=%s speed=%.2f", voice, speed)
    audios = synthesize_chunks(engine, chunks, voice=voice, speed=speed, cache=True)

    # Detect chapter breaks robustly: pause BEFORE headings like "Chapter 5" (short, no sentence punctuation)
    chapter_breaks = [False] * len(chunks)
    import re as _re
    chapter_re = _re.compile(r"^chapter\s+\d+\b", _re.IGNORECASE)
    for i, ch in enumerate(chunks):
        t = ch.text.strip()
        if chapter_re.match(t) and len(t) <= 60 and not any(p in t for p in ['.', '!', '?']):
            if i - 1 >= 0:
                chapter_breaks[i - 1] = True

    # Assemble combined and per-chapter audios
    pause_plan = PausePlan()
    audio_full = assemble_with_pauses(audios, chunks, pause_plan, chapter_breaks=chapter_breaks)

    # Compute per-chapter ranges and assemble audios
    chapter_ranges = compute_chapter_ranges(chapter_breaks, total_chunks=len(chunks))
    chapter_audios = assemble_chapter_audios(audios, chapter_ranges)

    # Loudness normalization
    if normalize_loudness and len(audio_full) > 0:
        audio_full = loudness_normalize(audio_full, 24000, target_lufs=-18.0)

    # Save WAV/MP3 for combined
    save_wav(target.wav_path, audio_full, 24000)
    encode_mp3(target.wav_path, target.mp3_path, bitrate="80k")

    # Save each chapter as base_name.chapterNN.mp3
    if chapter_audios:
        for idx, ch_audio in enumerate(chapter_audios, start=1):
            ch_wav = target.directory / f"{target.base_name}.chapter{idx:02d}.wav"
            ch_mp3 = target.directory / f"{target.base_name}.chapter{idx:02d}.mp3"
            save_wav(ch_wav, ch_audio, 24000)
            encode_mp3(ch_wav, ch_mp3, bitrate="80k")

    logging.info("Wrote MP3: %s", target.mp3_path)
    return target.mp3_path


def main() -> None:
    parser = argparse.ArgumentParser(description="KittenTTS Audiobook Generator (CLI)")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--txt", type=Path, default=None, help="Optional pre-converted .txt to use instead of PDF")
    parser.add_argument("--voice", default="expr-voice-4-f", help="Voice ID (default female)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/test_mp3"))
    parser.add_argument("--max-chars", type=int, default=350)
    parser.add_argument("--strict-lines", action="store_true", help="Line-strict mode: only split on full-stops across lines in order")
    parser.add_argument("--start-from", type=str, default=None, help="Skip content until a line exactly matching this heading (e.g., 'Foreword')")
    parser.add_argument("--no-normalize-loudness", action="store_true")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Limit number of chunks for a quick sample run")
    parser.add_argument("--limit-pages", type=int, default=None, help="Process only the first N pages")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mp3_path = run(
        pdf_path=args.input_pdf,
        voice=args.voice,
        speed=args.speed,
        out_dir=args.out_dir,
        max_chars=args.max_chars,
        normalize_loudness=not args.no_normalize_loudness,
        verbose=args.verbose,
        limit_chunks=args.limit_chunks,
        limit_pages=args.limit_pages,
        txt_path=args.txt,
        strict_lines=args.strict_lines,
        start_from=args.start_from,
    )
    print(str(mp3_path))


if __name__ == "__main__":
    main()
