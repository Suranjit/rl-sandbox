# pong/utils/timing.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Callable, Iterator

# --------------------------------------------------------------------------- #
# A tiny context-manager for wall-clock timing.  Usage:
#
#     with timing("my section"):
#         ...
# --------------------------------------------------------------------------- #
@contextmanager
def timing(section: str, log: Callable[[str], None] = print) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log(f"{section} took {elapsed:.3f}s")