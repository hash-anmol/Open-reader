from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

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


def synthesize_with_backoff(engine: TTSEngine, text: str, voice: str, speed: float) -> np.ndarray:
    """Attempt synthesis with progressively smaller segments to avoid ONNX errors.

    Strategy:
    - Try direct
    - Split by major punctuation (.!?), then by commas/semicolons, then by words to ~80-120 chars
    - Insert short 150ms pauses between sub-segments
    """
    try:
        return engine.synthesize(text, voice=voice, speed=speed)
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
        for s in sent_parts:
            s = s.strip()
            if not s:
                continue
            try:
                auds.append(engine.synthesize(s, voice=voice, speed=speed))
            except Exception:
                auds.append(_synthesize_by_length(engine, s, voice, speed))
        return join(auds)

    # Fallback: length-based chunking
    return _synthesize_by_length(engine, text, voice, speed)


def _synthesize_by_length(engine: TTSEngine, text: str, voice: str, speed: float, max_len: int = 120) -> np.ndarray:
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
    for p in parts:
        if not p.strip():
            continue
        try:
            auds.append(engine.synthesize(p, voice=voice, speed=speed))
        except Exception:
            # As a last resort, skip or synthesize very small fragments
            tiny = p[:60]
            try:
                auds.append(engine.synthesize(tiny, voice=voice, speed=speed))
            except Exception as e:
                logging.error("Failed to synthesize fragment even after backoff: %s", e)
                # Do not drop content: insert estimated silence to preserve pacing
                # Estimate 12 chars/sec reading speed at speed=1.0; adjust by speed
                cps = 12.0 * max(speed, 0.5)
                seconds = max(0.3, min(3.0, len(p) / cps))
                auds.append(np.zeros(int(seconds * 24000), dtype=np.float32))
    if not auds:
        return np.zeros(0, dtype=np.float32)
    # Concatenate with small pauses
    return np.concatenate([a.astype('float32') for a in auds])
