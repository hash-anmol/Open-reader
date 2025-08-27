from __future__ import annotations

from src.kitten_audiobook.tts.synthesis import detect_chapter_breaks_from_chunks, compute_chapter_ranges


def test_detect_chapter_breaks_simple():
    chunks = [
        "Preface",
        "Chapter 1",
        "This is the first chapter text.",
        "Chapter 2",
        "More content.",
        "Appendix",
    ]

    flags = detect_chapter_breaks_from_chunks(chunks)
    # Expect a break before Chapter 1 and before Chapter 2 → break after indices 0 and 2
    assert flags == [True, False, True, False, False, False]

    ranges = compute_chapter_ranges(flags, total_chunks=len(chunks))
    assert ranges == [(0, 0), (1, 2), (3, 5)]


def test_detect_chapter_breaks_limits():
    # Headings with punctuation should not trigger a break
    chunks = [
        "Chapter 1: Introduction.",  # has colon + period -> not a standalone heading
        "Body text continues.",
        "CHAPTER 2",  # standalone heading (uppercase)
        "Next body.",
    ]
    flags = detect_chapter_breaks_from_chunks(chunks)
    # Only the standalone heading should induce a break before it → break after index 1
    assert flags == [False, True, False, False]





