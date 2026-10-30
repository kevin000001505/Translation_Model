"""Keep job logs alive during long silent steps (downloads, conversions, model loading)."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


def dir_size_gb(path: Path) -> float:
    """Total size of the files under `path` (follows a download as it grows)."""
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


@contextmanager
def heartbeat(log: Callable[[str], None], what: str, every: float = 60,
              detail: Callable[[], str] | None = None) -> Iterator[None]:
    """Log "still <what> (N min)" every `every` seconds until the block ends, then how long it took."""
    start = time.time()
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(every):
            extra = ""
            if detail:
                try:
                    extra = f", {detail()}"
                except OSError:
                    pass
            log(f"... still {what} ({(time.time() - start) / 60:.1f} min{extra})")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
    log(f"Finished {what} in {(time.time() - start) / 60:.1f} min.")
