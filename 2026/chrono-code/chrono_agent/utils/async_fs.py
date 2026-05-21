"""
Async-safe filesystem helpers.

LangGraph Studio runs graphs under an ASGI server. Synchronous filesystem calls (e.g. mkdir)
inside async code paths can block the event loop and trigger LangGraph blocking detectors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def ensure_dir(path: Path, *, parents: bool = True, exist_ok: bool = True) -> None:
    """Create a directory without blocking the event loop."""
    await asyncio.to_thread(path.mkdir, parents=parents, exist_ok=exist_ok)

