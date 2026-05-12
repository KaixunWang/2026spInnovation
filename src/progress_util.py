"""Optional tqdm progress bars (stderr). Disable with env ``NO_TQDM=1``."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def tqdm_disabled() -> bool:
    return os.environ.get("NO_TQDM", "").strip().lower() in ("1", "true", "yes")


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    initial: int = 0,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
) -> Iterable[T]:
    """Wrap an iterable with a tqdm bar when progress is enabled."""
    if tqdm_disabled():
        return iterable  # type: ignore[return-value]
    from tqdm import tqdm

    kwargs: dict[str, Any] = {
        "iterable": iterable,
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "file": sys.stderr,
        "mininterval": 0.2,
        "dynamic_ncols": True,
    }
    if total is not None:
        kwargs["total"] = total
    if initial:
        kwargs["initial"] = initial
    if position is not None:
        kwargs["position"] = position
    return tqdm(**kwargs)


@contextmanager
def progress_counter(*, total: int | None, desc: str = "", unit: str = "it") -> Iterator[Any]:
    """Manual counter bar: ``yield`` object has ``update(n)`` and ``set_postfix_str``."""
    if tqdm_disabled():

        class _Null:
            def update(self, n: int = 1) -> None:
                pass

            def set_postfix_str(self, _s: str, *, refresh: bool = True) -> None:
                pass

        yield _Null()
        return

    from tqdm import tqdm

    bar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stderr,
        mininterval=0.2,
        dynamic_ncols=True,
    )
    try:
        yield bar
    finally:
        bar.close()
