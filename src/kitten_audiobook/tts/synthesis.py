from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import json
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..text.chunking import Chunk
from ..utils.io import chunk_cache_path
from .engine import TTSEngine


@dataclass
class PausePlan:
    chunk_pause_s: float = 0.0
    paragraph_pause_s: float = 0.0
    chapter_pause_s: float = 1.2


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    idx: int


def generate_word_timings(text: str, audio: np.ndarray, sample_rate: int = 24000) -> List[WordTiming]:
    """Generate word-level timings using heuristic distribution based on character length.
    
    Args:
        text: The input text that was synthesized
        audio: The generated audio array
        sample_rate: Audio sample rate (default 24000 for KittenTTS)
    
    Returns:
        List of WordTiming objects with start/end times for each word
    """
    # Calculate total audio duration
    duration = len(audio) / sample_rate
    
    # Split text into words, preserving punctuation
    words = re.findall(r'\S+', text)
    if not words:
        return []
    
    # Calculate character-based timing distribution
    total_chars = sum(len(word) for word in words)
    if total_chars == 0:
        return []
    
    timings = []
    current_time = 0.0
    
    for idx, word in enumerate(words):
        # Distribute duration proportionally based on character length
        word_duration = (len(word) / total_chars) * duration
        
        # Ensure minimum duration for very short words
        word_duration = max(word_duration, 0.1)
        
        # Adjust for last word to ensure we don't exceed total duration
        if idx == len(words) - 1:
            word_duration = duration - current_time
        
        start_time = current_time
        end_time = current_time + word_duration
        
        timings.append(WordTiming(
            word=word,
            start=start_time,
            end=end_time,
            idx=idx
        ))
        
        current_time = end_time
    
    return timings


def generate_word_timings_for_chunks(text_chunks: List[str], audio_chunks: List[np.ndarray], 
                                   sample_rate: int = 24000) -> List[WordTiming]:
    """Generate word timings for multiple text/audio chunks with continuous timing.
    
    Args:
        text_chunks: List of text chunks that were synthesized
        audio_chunks: List of corresponding audio arrays
        sample_rate: Audio sample rate
    
    Returns:
        List of WordTiming objects with continuous timing across all chunks
    """
    all_timings = []
    current_offset = 0.0
    word_idx = 0
    
    for text_chunk, audio_chunk in zip(text_chunks, audio_chunks):
        # Generate timings for this chunk
        chunk_timings = generate_word_timings(text_chunk, audio_chunk, sample_rate)
        
        # Adjust timings to account for previous chunks
        for timing in chunk_timings:
            adjusted_timing = WordTiming(
                word=timing.word,
                start=timing.start + current_offset,
                end=timing.end + current_offset,
                idx=word_idx
            )
            all_timings.append(adjusted_timing)
            word_idx += 1
        
        # Update offset for next chunk
        current_offset += len(audio_chunk) / sample_rate
    
    return all_timings


def word_timings_to_json(timings: List[WordTiming]) -> str:
    """Convert word timings to JSON format for frontend consumption."""
    return json.dumps([{
        'word': timing.word,
        'start': round(timing.start, 3),
        'end': round(timing.end, 3),
        'idx': timing.idx
    } for timing in timings])


def create_transcript_html(text: str, timings: List[WordTiming]) -> str:
    """Create HTML transcript with word spans for highlighting.
    
    Args:
        text: Original text
        timings: Word timings
    
    Returns:
        HTML string with clickable word spans
    """
    if not timings:
        # Fallback: create spans without timing data
        words = re.findall(r'\S+', text)
        spans = []
        for idx, word in enumerate(words):
            spans.append(f'<span class="w" id="w-{idx}">{word}</span>')
        return ' '.join(spans)
    
    # Create spans with timing data
    spans = []
    for timing in timings:
        span = (f'<span class="w" data-start="{timing.start:.3f}" '
                f'data-end="{timing.end:.3f}" id="w-{timing.idx}">{timing.word}</span>')
        spans.append(span)
    
    return ' '.join(spans)


def synthesize_chunks(
    engine: TTSEngine,
    chunks: List[Chunk],
    voice: str,
    speed: float,
    cache: bool = True,
) -> List[np.ndarray]:
    audios: List[np.ndarray] = []
    for chunk in tqdm(chunks, desc="Synthesize", unit="chunk"):
        cache_path = chunk_cache_path(chunk.text, voice, speed) if cache else None
        audio: Optional[np.ndarray] = None
        if cache and cache_path is not None and cache_path.exists():
            try:
                audio, sr = sf.read(cache_path, dtype='float32')
            except Exception:
                audio = None
        if audio is None:
            try:
                audio = engine.synthesize(chunk.text, voice=voice, speed=speed)
            except Exception as e:
                logging.warning("Direct synth failed for chunk, attempting backoff: %s", e)
                audio = synthesize_with_backoff(engine, chunk.text, voice, speed)
            if cache and cache_path is not None:
                sf.write(cache_path, audio, 24000)
        audios.append(audio)
    return audios


def assemble_with_pauses(
    audios: List[np.ndarray],
    chunks: List[Chunk],
    pause_plan: PausePlan,
    crossfade_ms: int = 0,
    chapter_breaks: List[bool] | None = None,
) -> np.ndarray:
    if not audios:
        return np.zeros(0, dtype=np.float32)
    # Simple concatenation: no gaps, no crossfades
    if chapter_breaks is None:
        return np.concatenate([a.astype('float32') for a in audios])
    out_parts: List[np.ndarray] = []
    for i, a in enumerate(audios):
        out_parts.append(a.astype('float32'))
        if i < len(chapter_breaks) and chapter_breaks[i]:
            # insert a small chapter pause after this chunk
            n = int(24000 * max(0.0, pause_plan.chapter_pause_s))
            if n > 0:
                out_parts.append(np.zeros(n, dtype=np.float32))
    return np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.float32)


# --- Chapter helpers for per-chapter outputs ---
def compute_chapter_ranges(chapter_breaks: List[bool], total_chunks: int) -> List[Tuple[int, int]]:
    """Convert per-chunk break flags into inclusive chapter (start, end) ranges.

    Rules:
    - A True at index i indicates a break AFTER chunk i.
    - The final chapter always closes at total_chunks - 1.
    - Trailing True is ignored (does not create empty chapter).
    """
    if total_chunks <= 0:
        return []
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(total_chunks):
        # If there is a break after i, close current chapter at i
        if i < len(chapter_breaks) and chapter_breaks[i]:
            end = i
            if end >= start:
                ranges.append((start, end))
            start = i + 1
    # Close the final chapter (even if start == total_chunks this yields empty; guard it)
    if start <= total_chunks - 1:
        ranges.append((start, total_chunks - 1))
    return ranges


def assemble_chapter_audios(audios: List[np.ndarray], chapter_ranges: List[Tuple[int, int]]) -> List[np.ndarray]:
    """Assemble per-chapter audio by concatenating chunk audio for each (start, end) range.
    Assumes all arrays are mono float32 at 24kHz.
    """
    out: List[np.ndarray] = []
    for start, end in chapter_ranges:
        if start < 0 or end >= len(audios) or start > end:
            # Skip invalid range gracefully
            out.append(np.zeros(0, dtype=np.float32))
            continue
        parts = [audios[i].astype('float32') for i in range(start, end + 1)]
        out.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32))
    return out


def detect_chapter_breaks_from_chunks(chunks_or_texts: List[Chunk | str]) -> List[bool]:
    """Heuristically detect chapter breaks based on chunk texts.

    Returns a list of flags same length as chunks, where True at i means
    a chapter break should be inserted AFTER chunk i (i.e., before i+1).

    Heuristic: if a chunk text matches /^chapter\s+\d+\b/i and is short
    (<= 60 chars) and contains no sentence punctuation, we mark a break
    before it by setting break at previous index (i-1) to True.
    """
    n = len(chunks_or_texts)
    flags = [False] * n
    chapter_re = re.compile(r"^chapter\s+\d+\b", re.IGNORECASE)
    for i in range(n):
        text = chunks_or_texts[i].text if hasattr(chunks_or_texts[i], 'text') else str(chunks_or_texts[i])
        t = text.strip()
        if chapter_re.match(t) and len(t) <= 60 and not any(p in t for p in ['.', '!', '?']):
            if i - 1 >= 0:
                flags[i - 1] = True
    return flags


def synthesize_with_backoff(engine: TTSEngine, text: str, voice: str, speed: float) -> tuple[np.ndarray, List[WordTiming]]:
    """Attempt synthesis with progressively smaller segments to avoid ONNX errors.

    Strategy:
    - Try direct
    - Split by major punctuation (.!?), then by commas/semicolons, then by words to ~80-120 chars
    - Insert short 150ms pauses between sub-segments
    - Returns both audio and word timings
    """
    try:
        audio = engine.synthesize(text, voice=voice, speed=speed)
        timings = generate_word_timings(text, audio)
        return audio, timings
    except Exception:
        pass

    # Helper to concatenate with small pauses
    def join(parts: List[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([p.astype('float32') for p in parts])

    # Level 1: split by sentence end
    sent_parts = re.split(r"(?<=[.!?])\s+", text)
    if len(sent_parts) > 1:
        auds: List[np.ndarray] = []
        text_chunks: List[str] = []
        for s in sent_parts:
            s = s.strip()
            if not s:
                continue
            try:
                audio = engine.synthesize(s, voice=voice, speed=speed)
                auds.append(audio)
                text_chunks.append(s)
            except Exception:
                audio, _ = _synthesize_by_length(engine, s, voice, speed)
                auds.append(audio)
                text_chunks.append(s)
        
        final_audio = join(auds)
        timings = generate_word_timings_for_chunks(text_chunks, auds)
        return final_audio, timings

    # Fallback: length-based chunking
    return _synthesize_by_length(engine, text, voice, speed)


def _synthesize_by_length(engine: TTSEngine, text: str, voice: str, speed: float, max_len: int = 120) -> tuple[np.ndarray, List[WordTiming]]:
    words = text.split()
    parts: List[str] = []
    buf: List[str] = []
    for w in words:
        candidate = (" ".join(buf + [w])).strip()
        if len(candidate) > max_len and buf:
            parts.append(" ".join(buf))
            buf = [w]
        else:
            buf.append(w)
    if buf:
        parts.append(" ".join(buf))
    auds: List[np.ndarray] = []
    text_chunks: List[str] = []
    for p in parts:
        if not p.strip():
            continue
        try:
            audio = engine.synthesize(p, voice=voice, speed=speed)
            auds.append(audio)
            text_chunks.append(p)
        except Exception:
            # As a last resort, skip or synthesize very small fragments
            tiny = p[:60]
            try:
                audio = engine.synthesize(tiny, voice=voice, speed=speed)
                auds.append(audio)
                text_chunks.append(tiny)
            except Exception as e:
                logging.error("Failed to synthesize fragment even after backoff: %s", e)
                # Do not drop content: insert estimated silence to preserve pacing
                # Estimate 12 chars/sec reading speed at speed=1.0; adjust by speed
                cps = 12.0 * max(speed, 0.5)
                seconds = max(0.3, min(3.0, len(p) / cps))
                silence = np.zeros(int(seconds * 24000), dtype=np.float32)
                auds.append(silence)
                text_chunks.append(p)  # Keep original text for timing
    if not auds:
        return np.zeros(0, dtype=np.float32), []
    
    # Concatenate audio and generate timings
    final_audio = np.concatenate([a.astype('float32') for a in auds])
    timings = generate_word_timings_for_chunks(text_chunks, auds)
    return final_audio, timings
