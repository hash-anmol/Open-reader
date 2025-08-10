from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

# Common abbreviations to avoid sentence splits
ABBREVIATIONS = set(
    s.lower() for s in [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "e.g.", "i.e.", "Fig.", "Eq.", "No.", "pp.", "cf.", "al.",
    ]
)

SENTENCE_END_RE = re.compile(r"([.!?])\s+")


@dataclass
class Sentence:
    text: str
    paragraph_index: int


def split_into_sentences(paragraph_text: str, paragraph_index: int) -> List[Sentence]:
    text = paragraph_text.strip()
    if not text:
        return []

    sentences: List[Sentence] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text + " "):
        end = match.end()
        candidate = text[start:end].strip()
        token = candidate.split()[-1] if candidate.split() else ""
        if token in ABBREVIATIONS:
            continue
        if candidate:
            sentences.append(Sentence(text=candidate, paragraph_index=paragraph_index))
            start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(Sentence(text=tail, paragraph_index=paragraph_index))
    return sentences
