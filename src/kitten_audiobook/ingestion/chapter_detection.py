from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import fitz  # pymupdf

from .structure import Document, Page, Paragraph


@dataclass
class ChapterInfo:
    """Metadata for a detected chapter"""
    title: str
    start_page: int
    end_page: Optional[int] = None
    start_paragraph_idx: Optional[int] = None
    end_paragraph_idx: Optional[int] = None
    level: int = 1  # Heading level (1 = main chapter, 2 = subsection, etc.)
    source: str = "unknown"  # How this chapter was detected (bookmark, heading, pattern, manual)


@dataclass  
class ChapterBoundary:
    """Represents a chapter boundary within the document"""
    paragraph_idx: int
    page_number: int
    title: str
    level: int = 1
    source: str = "unknown"


@dataclass
class ChapterDetectorConfig:
    """Tunable thresholds for robust chapter detection."""
    # Heading analysis
    heading_min_relative_size: float = 1.25  # span size must be >= median*this
    heading_max_length: int = 120
    heading_top_of_page_rel: float = 0.25  # y_rel threshold to consider near top
    heading_require_no_sentence_punct: bool = True
    # Pattern matching
    fuzzy_threshold: float = 0.68
    # Deduplication/cleanups
    dedupe_same_page_window: int = 1  # collapse consecutive bookmarks on same/adjacent page
    max_title_len: int = 140


class ChapterDetector:
    """Enhanced chapter detection using multiple methods"""
    
    def __init__(self, document: Document, pdf_path: Optional[Path] = None, config: Optional[ChapterDetectorConfig] = None):
        self.document = document
        self.pdf_path = pdf_path
        self.config = config or ChapterDetectorConfig()
        self.chapters: List[ChapterInfo] = []
        self.boundaries: List[ChapterBoundary] = []
        
        # Chapter detection patterns
        # Cover numeric, roman numerals and word-based numbering
        word_nums = r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
        self.chapter_patterns = [
            re.compile(r"^chapter\s+(\d+|[ivxlcdm]+)\b", re.IGNORECASE),
            re.compile(rf"^chapter\s+({word_nums})\b", re.IGNORECASE),
            re.compile(r"^(\d+)\.\s+", re.IGNORECASE),  # "1. Introduction"
            re.compile(r"^part\s+(\d+|[ivxlcdm]+)\b", re.IGNORECASE),
            re.compile(rf"^part\s+({word_nums})\b", re.IGNORECASE),
            re.compile(r"^book\s+(\d+|[ivxlcdm]+)\b", re.IGNORECASE),
            re.compile(rf"^book\s+({word_nums})\b", re.IGNORECASE),
            re.compile(r"^section\s+(\d+|[ivxlcdm]+)\b", re.IGNORECASE),
            re.compile(r"^appendix\s*([a-z]|\d+)?\b", re.IGNORECASE),
            re.compile(r"^epilogue\b", re.IGNORECASE),
            re.compile(r"^prologue\b", re.IGNORECASE),
            re.compile(r"^preface\b", re.IGNORECASE),
            re.compile(r"^introduction\b", re.IGNORECASE),
            re.compile(r"^conclusion\b", re.IGNORECASE),
            re.compile(r"^bibliography\b", re.IGNORECASE),
            re.compile(r"^index\b", re.IGNORECASE),
            re.compile(r"^acknowledg?ements?\b", re.IGNORECASE),
            re.compile(r"^foreword\b", re.IGNORECASE),
            re.compile(r"^afterword\b", re.IGNORECASE),
            re.compile(r"^glossary\b", re.IGNORECASE),
            re.compile(r"^notes\b", re.IGNORECASE),
        ]

    def detect_chapters(self) -> List[ChapterInfo]:
        """Main method to detect chapters using multiple strategies"""
        logging.info("Starting enhanced chapter detection...")
        
        # Strategy 1: PDF bookmarks/outline (most reliable)
        bookmark_chapters = self._detect_from_bookmarks()
        if bookmark_chapters:
            logging.info(f"Found {len(bookmark_chapters)} chapters from PDF bookmarks")
            self.chapters = bookmark_chapters
            return self.chapters
        
        # Strategy 2: Heading analysis (font size, formatting)
        heading_chapters = self._detect_from_headings()
        if heading_chapters:
            logging.info(f"Found {len(heading_chapters)} chapters from heading analysis")
            self.chapters = heading_chapters
            return self.chapters
        
        # Strategy 3: Pattern matching (fallback)
        pattern_chapters = self._detect_from_patterns()
        logging.info(f"Found {len(pattern_chapters)} chapters from pattern matching")
        self.chapters = pattern_chapters
        
        return self.chapters

    def _detect_from_bookmarks(self) -> List[ChapterInfo]:
        """Detect chapters from PDF bookmarks/outline"""
        if not self.pdf_path or not self.pdf_path.exists():
            return []
        
        try:
            doc = fitz.open(self.pdf_path)
            toc = doc.get_toc()  # Table of contents (list of [level, title, page])
            
            if not toc:
                return []
            
            chapters = []
            for i, (level, title, page_num) in enumerate(toc):
                # Only consider level 1 and 2 headings as chapters
                if level <= 2:
                    title_norm = self._normalize_title(title)
                    if not title_norm:
                        continue
                    chapter = ChapterInfo(
                        title=title_norm,
                        start_page=page_num,
                        level=level,
                        source="bookmark"
                    )
                    chapters.append(chapter)
            
            # Dedupe consecutive duplicates and invalid title lines
            chapters = self._dedupe_and_filter_chapters(chapters)

            # Set end pages
            for i in range(len(chapters) - 1):
                chapters[i].end_page = chapters[i + 1].start_page - 1
            if chapters:
                chapters[-1].end_page = len(self.document.pages)
            
            return chapters
            
        except Exception as e:
            logging.warning(f"Failed to extract bookmarks: {e}")
            return []

    def _detect_from_headings(self) -> List[ChapterInfo]:
        """Detect chapters by analyzing font sizes and formatting"""
        if not self.pdf_path or not self.pdf_path.exists():
            return []
        
        try:
            doc = fitz.open(self.pdf_path)
            potential_headings = []
            
            for page_num, page in enumerate(doc, 1):
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue
                                
                                font_size = span.get("size", 0)
                                font_flags = span.get("flags", 0)
                                bbox = span.get("bbox")
                                y_rel = None
                                if bbox and hasattr(page, "rect") and page.rect.height:
                                    # bbox = [x0, y0, x1, y1]
                                    y_rel = float(bbox[1]) / float(page.rect.height)
                                
                                # Check if this looks like a heading
                                is_bold = bool(font_flags & 2**4)  # Bold flag
                                is_short = len(text) <= self.config.heading_max_length
                                no_punct = (not any(p in text for p in ['.', '!', '?', ';'])) if self.config.heading_require_no_sentence_punct else True

                                # Compute relative size vs page median size lazily per page
                                # Collect sizes per page to compute median
                                # For robustness without pre-pass, treat > 14 as a conservative large font
                                is_large = font_size >= 14

                                # consider near top of page if y_rel available
                                near_top = (y_rel is None) or (y_rel <= self.config.heading_top_of_page_rel)

                                if is_short and no_punct and near_top and (is_bold or is_large):
                                    # Check against chapter-like patterns OR strong uppercase ratio
                                    matches_pattern = any(p.match(text) for p in self.chapter_patterns)
                                    upper_ratio = _upper_ratio(text)
                                    if matches_pattern or upper_ratio >= 0.65:
                                        potential_headings.append({
                                            'text': self._normalize_title(text),
                                            'page': page_num,
                                            'font_size': font_size,
                                            'is_bold': is_bold,
                                        })
            
            # Convert potential headings to chapters
            chapters = []
            for i, heading in enumerate(potential_headings):
                chapter = ChapterInfo(
                    title=heading['text'],
                    start_page=heading['page'],
                    end_page=potential_headings[i + 1]['page'] - 1 if i + 1 < len(potential_headings) else len(self.document.pages),
                    level=1 if heading['font_size'] >= 16 else 2,
                    source="heading"
                )
                chapters.append(chapter)

            chapters = self._dedupe_and_filter_chapters(chapters)
            
            return chapters
            
        except Exception as e:
            logging.warning(f"Failed to analyze headings: {e}")
            return []

    def _detect_from_patterns(self) -> List[ChapterInfo]:
        """Detect chapters using pattern matching on paragraph text"""
        chapters = []
        current_chapter = None
        paragraph_idx = 0
        
        for page in self.document.pages:
            for paragraph in page.paragraphs:
                text = paragraph.text.strip()
                
                # Check if this paragraph matches any chapter pattern
                for pattern in self.chapter_patterns:
                    if pattern.match(text) and len(text) <= self.config.max_title_len:
                        # Prefer top-of-page short headings, mostly uppercase and no punctuation
                        y_ok = (paragraph.y_rel is None) or (paragraph.y_rel <= self.config.heading_top_of_page_rel)
                        no_punct = not any(p in text for p in ['.', '!', '?', ';', ':'])
                        if y_ok and (no_punct or pattern.pattern.startswith('^chapter')):
                            if current_chapter:
                                current_chapter.end_page = page.number - 1
                                current_chapter.end_paragraph_idx = paragraph_idx - 1
                            current_chapter = ChapterInfo(
                                title=self._normalize_title(text),
                                start_page=page.number,
                                start_paragraph_idx=paragraph_idx,
                                level=1,
                                source="pattern"
                            )
                            chapters.append(current_chapter)
                            break
                
                paragraph_idx += 1
        
        # Close the last chapter
        if current_chapter:
            current_chapter.end_page = len(self.document.pages)
            current_chapter.end_paragraph_idx = paragraph_idx - 1
        
        return chapters

    def get_chapter_boundaries_for_chunks(self, chunks: List[Any]) -> List[bool]:
        """
        Convert detected chapters to boundary flags for existing chunk-based processing
        
        Args:
            chunks: List of text chunks (with .text attribute or string)
            
        Returns:
            List of boolean flags indicating chapter breaks after each chunk
        """
        if not self.chapters:
            # Fallback to original pattern-based detection
            return self._detect_boundaries_from_chunk_patterns(chunks)
        
        # Create boundaries based on detected chapters
        boundaries = [False] * len(chunks)
        
        for chapter in self.chapters[1:]:  # Skip first chapter
            chapter_title = chapter.title.lower().strip()
            
            # Find the chunk that contains this chapter title
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                chunk_text = chunk_text.lower().strip()
                
                # Check if chunk contains or matches chapter title
                if (chapter_title in chunk_text or 
                    chunk_text in chapter_title or
                    self._fuzzy_match(chapter_title, chunk_text)):
                    
                    # Set boundary before this chunk (after previous chunk)
                    if i > 0:
                        boundaries[i - 1] = True
                    break
        
        return boundaries

    def _detect_boundaries_from_chunk_patterns(self, chunks: List[Any]) -> List[bool]:
        """Fallback method using original pattern detection on chunks"""
        boundaries = [False] * len(chunks)
        
        for i, chunk in enumerate(chunks):
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            text = text.strip()
            
            # Check against chapter patterns
            for pattern in self.chapter_patterns:
                if pattern.match(text) and len(text) <= 100:
                    # No sentence punctuation (likely a heading)
                    if not any(p in text for p in ['.', '!', '?']):
                        if i > 0:
                            boundaries[i - 1] = True
                        break
        
        return boundaries

    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Simple fuzzy matching for chapter titles"""
        # Simple word-based matching
        try:
            import difflib
            ratio = difflib.SequenceMatcher(a=text1.lower(), b=text2.lower()).ratio()
            return ratio >= (self.config.fuzzy_threshold if threshold is None else threshold)
        except Exception:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return False
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            similarity = len(intersection) / len(union) if union else 0
            return similarity >= (threshold or self.config.fuzzy_threshold)

    def get_chapter_metadata(self) -> Dict[str, Any]:
        """Get metadata about detected chapters"""
        if not self.chapters:
            return {"total_chapters": 0, "detection_method": "none"}
        
        return {
            "total_chapters": len(self.chapters),
            "detection_method": self.chapters[0].source if self.chapters else "none",
            "chapters": [
                {
                    "title": ch.title,
                    "start_page": ch.start_page,
                    "end_page": ch.end_page,
                    "level": ch.level
                }
                for ch in self.chapters
            ]
        }

    def create_chapter_manifest(self, audio_durations: Optional[List[float]] = None) -> Dict[str, Any]:
        """Create a chapter manifest/playlist"""
        manifest = {
            "title": self.document.title or "Audiobook",
            "author": self.document.author,
            "total_chapters": len(self.chapters),
            "chapters": []
        }
        
        for i, chapter in enumerate(self.chapters):
            chapter_data = {
                "index": i + 1,
                "title": chapter.title,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "duration": audio_durations[i] if audio_durations and i < len(audio_durations) else None
            }
            manifest["chapters"].append(chapter_data)
        
        return manifest


def detect_chapters_enhanced(document: Document, pdf_path: Optional[Path] = None) -> Tuple[List[ChapterInfo], ChapterDetector]:
    """
    Enhanced chapter detection function
    
    Args:
        document: Parsed document structure
        pdf_path: Optional path to original PDF for bookmark extraction
        
    Returns:
        Tuple of (detected chapters, detector instance)
    """
    detector = ChapterDetector(document, pdf_path)
    chapters = detector.detect_chapters()
    return chapters, detector


# ---- helpers ----
def _upper_ratio(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(1 for ch in letters if ch.isupper())
    return uppers / len(letters)


def _is_noise_title(t: str) -> bool:
    low = t.lower().strip()
    # Common noise or non-content markers that we avoid as chapters unless at level 1
    return low in {"table of contents", "copyright", "license"}


def _title_clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _valid_title(t: str) -> bool:
    if not t:
        return False
    if len(t) > 200:
        return False
    return True


# Bind helper methods to class via monkey pattern for cleanliness
def _normalize_title(self, t: str) -> str:
    t = _title_clean(t)
    # drop trailing dots and hyphens commonly seen in headers
    t = re.sub(r"[\s\-\.:]+$", "", t)
    return t


def _dedupe_and_filter_chapters(self, chapters: List[ChapterInfo]) -> List[ChapterInfo]:
    if not chapters:
        return []
    out: List[ChapterInfo] = []
    last_kept: Optional[ChapterInfo] = None
    for ch in chapters:
        if not _valid_title(ch.title) or _is_noise_title(ch.title):
            continue
        if last_kept and abs((ch.start_page or 0) - (last_kept.start_page or 0)) <= self.config.dedupe_same_page_window:
            # Prefer level 1 over level 2, and longer title if same level
            if ch.level < last_kept.level or (ch.level == last_kept.level and len(ch.title) > len(last_kept.title)):
                out[-1] = ch
                last_kept = ch
            continue
        out.append(ch)
        last_kept = ch
    # Ensure strictly increasing start pages
    out.sort(key=lambda c: (c.start_page, c.level))
    unique: List[ChapterInfo] = []
    seen_pages: set[int] = set()
    for ch in out:
        if ch.start_page in seen_pages:
            # keep only first occurrence per page at highest level
            if unique and unique[-1].start_page == ch.start_page and ch.level < unique[-1].level:
                unique[-1] = ch
            continue
        unique.append(ch)
        seen_pages.add(ch.start_page)
    return unique


# attach helpers to class
ChapterDetector._normalize_title = _normalize_title
ChapterDetector._dedupe_and_filter_chapters = _dedupe_and_filter_chapters
