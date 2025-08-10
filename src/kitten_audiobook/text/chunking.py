from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .segmentation import Sentence


@dataclass
class Chunk:
    text: str
    paragraph_index: int
    is_paragraph_end: bool


def build_chunks(sentences: List[Sentence], target_chars: int = 280, hard_cap: int = 360) -> List[Chunk]:
    chunks: List[Chunk] = []
    current: List[Sentence] = []
    current_len = 0

    def flush(is_par_end: bool) -> None:
        nonlocal current, current_len
        if current:
            text = " ".join(s.text for s in current).strip()
            chunks.append(Chunk(text=text, paragraph_index=current[-1].paragraph_index, is_paragraph_end=is_par_end))
            current = []
            current_len = 0

    last_par_index = -1
    for s in sentences:
        if s.paragraph_index != last_par_index and current:
            # Paragraph boundary: flush with paragraph_end
            flush(is_par_end=True)
        last_par_index = s.paragraph_index

        if current_len + len(s.text) + (1 if current else 0) > hard_cap:
            flush(is_par_end=False)
        current.append(s)
        current_len = len(" ".join(x.text for x in current))
        if current_len >= target_chars:
            flush(is_par_end=False)

    flush(is_par_end=True)
    return chunks
