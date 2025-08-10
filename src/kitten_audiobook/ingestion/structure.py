from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Paragraph:
    text: str
    page_number: int
    y_pos: float | None = None   # absolute y coordinate
    y_rel: float | None = None   # relative y (0 top .. 1 bottom)


@dataclass
class Page:
    number: int
    paragraphs: List[Paragraph]


@dataclass
class Document:
    pages: List[Page]
    title: Optional[str] = None
    author: Optional[str] = None
