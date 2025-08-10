from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Precompile regex patterns for performance
CONTROL_CHARS = ''.join(chr(i) for i in range(0, 32) if i not in (9, 10, 13)) + chr(127)
CONTROL_RE = re.compile(f"[{re.escape(CONTROL_CHARS)}]")
ZERO_WIDTH_RE = re.compile("[\u200B\u200C\u200D\u2060\uFEFF]")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAD6\U0001F1E6-\U0001F1FF\U00002700-\U000027BF]")
MULTI_SPACE_RE = re.compile(r"\s{2,}")
WHITESPACE_AROUND_PUNCT_RE = re.compile(r"\s*([,.;:!?])\s*")
REPEATED_PUNCT_RE = re.compile(r"([!?.,;:]){2,}")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\n(\w+)")
BULLETS_RE = re.compile(r"[•◦▪●○■□◆◇▶▷✔️✖️✳️✴️❖]")
LETTER_SPACED_WORD_RE = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}
SMARTS = {
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...",
}


def normalize_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    # Remove control and zero-width chars
    text = CONTROL_RE.sub(" ", text)
    text = ZERO_WIDTH_RE.sub("", text)
    # Replace ligatures and smarts
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    for k, v in SMARTS.items():
        text = text.replace(k, v)
    # Replace bullets with commas
    text = BULLETS_RE.sub(", ", text)
    # Remove emojis and symbols unlikely to be TTS-able
    text = EMOJI_RE.sub("", text)
    # De-hyphenate across line breaks
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    # Normalize whitespace and punctuation spacing
    text = MULTI_SPACE_RE.sub(" ", text)
    text = WHITESPACE_AROUND_PUNCT_RE.sub(r"\1 ", text)
    # Cap repeated punctuation to a single instance (or two for !!, ??)
    text = REPEATED_PUNCT_RE.sub(lambda m: m.group(1), text)
    # Collapse letter-spaced headings like 'C h a p t e r' -> 'Chapter'
    def _collapse_letter_spaced(m: re.Match) -> str:
        return m.group(0).replace(" ", "")
    text = LETTER_SPACED_WORD_RE.sub(_collapse_letter_spaced, text)
    # Whitelist characters likely safe for TTS: letters, digits, common punctuation, symbols
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;/'\"!?()%–—-\n\t[]()")
    text = ''.join(ch if ch in allowed else ' ' for ch in text)
    # Collapse whitespace again after filtering
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()
