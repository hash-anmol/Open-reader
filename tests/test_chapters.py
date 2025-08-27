from __future__ import annotations

import numpy as np

from src.kitten_audiobook.tts.synthesis import (
    PausePlan,
    assemble_with_pauses,
)


def test_compute_chapter_ranges_basic():
    # Delayed import to allow TDD (function may not exist initially)
    from src.kitten_audiobook.tts.synthesis import compute_chapter_ranges

    # 5 chunks, breaks after indices 1 and 3
    chapter_breaks = [False, True, False, True, False]
    ranges = compute_chapter_ranges(chapter_breaks, total_chunks=5)
    assert ranges == [(0, 1), (2, 3), (4, 4)]


def test_compute_chapter_ranges_no_breaks():
    from src.kitten_audiobook.tts.synthesis import compute_chapter_ranges

    chapter_breaks = [False, False, False]
    ranges = compute_chapter_ranges(chapter_breaks, total_chunks=4)
    assert ranges == [(0, 3)]


def test_compute_chapter_ranges_trailing_break_ignored():
    from src.kitten_audiobook.tts.synthesis import compute_chapter_ranges

    # Break after the last chunk should not create an empty chapter
    chapter_breaks = [False, False, True]
    ranges = compute_chapter_ranges(chapter_breaks, total_chunks=3)
    assert ranges == [(0, 2)]


def test_assemble_chapter_audios_lengths():
    from src.kitten_audiobook.tts.synthesis import compute_chapter_ranges, assemble_chapter_audios

    # Create 5 dummy chunk audios of distinct lengths (in samples)
    lengths = [100, 200, 300, 400, 500]
    audios = [np.ones(n, dtype=np.float32) for n in lengths]

    # breaks after indices 1 and 3 -> chapters: [0..1], [2..3], [4..4]
    chapter_breaks = [False, True, False, True, False]
    ranges = compute_chapter_ranges(chapter_breaks, total_chunks=len(audios))

    chapters = assemble_chapter_audios(audios, ranges)

    # Chapter lengths should be sums of their parts
    assert [len(ch) for ch in chapters] == [lengths[0] + lengths[1], lengths[2] + lengths[3], lengths[4]]


def test_combined_length_includes_chapter_pauses():
    # Build 4 audios of 1s, 2s, 3s, 4s at 24kHz
    sr = 24000
    secs = [1.0, 2.0, 3.0, 4.0]
    audios = [np.zeros(int(s * sr), dtype=np.float32) for s in secs]

    # One chapter break after first chunk (index 0)
    chapter_breaks = [True, False, False, False]

    # Combined should include exactly one chapter pause of 1.2s by default
    pause_plan = PausePlan()  # chapter_pause_s=1.2
    combined = assemble_with_pauses(audios, [None] * len(audios), pause_plan, chapter_breaks=chapter_breaks)

    expected_len = int(sr * sum(secs)) + int(sr * pause_plan.chapter_pause_s)
    assert len(combined) == expected_len





