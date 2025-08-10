from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def time_block(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label} took {(end - start)*1000:.1f} ms")
