from __future__ import annotations

from pathlib import Path
from typing import List

from .structure import Document, Page, Paragraph


def read_txt(path: Path) -> Document:
    """Read a plain text file into a Document.

    Assumptions:
    - Pages are separated by form-feed (\f) if produced via pdftotext.
    - Otherwise, treat the whole file as a single page.
    - Each non-empty line becomes a Paragraph (line-preserving mode).
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    raw_pages = text.split("\f") if "\f" in text else [text]
    pages: List[Page] = []
    for idx, raw in enumerate(raw_pages, start=1):
        paragraphs: List[Paragraph] = []
        for ln in raw.splitlines():
            if not ln.strip():
                continue
            paragraphs.append(Paragraph(text=ln.rstrip("\r"), page_number=idx))
        pages.append(Page(number=idx, paragraphs=paragraphs))
    return Document(pages=pages)
