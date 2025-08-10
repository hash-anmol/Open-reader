from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # pymupdf

from .structure import Document, Page, Paragraph


def _extract_paragraphs_from_page(page: fitz.Page, page_number: int) -> List[Paragraph]:
    # Use blocks to get reading order; within each block, get lines to avoid merged headings
    blocks = page.get_text("blocks")
    # blocks: list of (x0, y0, x1, y1, text, block_no, block_type)
    # Sort by y (top to bottom), then x (left to right)
    blocks.sort(key=lambda b: (b[1], b[0]))

    paragraphs: List[Paragraph] = []
    width, height = page.rect.width, page.rect.height
    for b in blocks:
        x0, y0, x1, y1, text, _, btype = b
        if not text or not text.strip():
            continue
        raw = text.strip()
        # Skip obvious headers/footers and lone page numbers
        low = raw.lower()
        if low.startswith("page no:"):
            continue
        if raw.isdigit() and len(raw) <= 3:
            continue

        # Break block into lines to preserve original line granularity
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            # remove hyphen at end-of-line to rejoin words across lines later in cleaning
            normalized = ln.rstrip("-").strip()
            if not normalized:
                continue
            y_rel = float(y0) / float(height) if height else None
            paragraphs.append(Paragraph(text=normalized, page_number=page_number, y_pos=y0, y_rel=y_rel))
    return paragraphs


def read_pdf(path: Path) -> Document:
    doc = fitz.open(path)
    pages: List[Page] = []
    metadata = doc.metadata or {}
    for i, page in enumerate(doc, start=1):
        paragraphs = _extract_paragraphs_from_page(page, i)
        pages.append(Page(number=i, paragraphs=paragraphs))
    return Document(pages=pages, title=metadata.get("title"), author=metadata.get("author"))
