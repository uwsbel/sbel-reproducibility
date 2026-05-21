"""
Periodic screenshot capture from Xvfb virtual display during simulation execution.

Captured frames are saved to outputs/xvfb_captures/ and used by the VLM review agent
to analyze the actual simulation window rendering.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class XvfbFrameCapture:
    """Captures periodic screenshots of the Xvfb display using ImageMagick import."""

    def __init__(
        self,
        display: str = ":99",
        output_dir: str | Path = "./outputs/xvfb_captures",
        interval: float = 0.5,
    ) -> None:
        self.display = display
        self.output_dir = Path(output_dir)
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._frame_count = 0

    async def start(self) -> None:
        """Start periodic capture in the background."""
        if not shutil.which("import"):
            logger.warning(
                "ImageMagick 'import' not found. Install with: sudo apt install imagemagick"
            )
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clear old captures
        for f in self.output_dir.glob("*.png"):
            f.unlink(missing_ok=True)

        self._stop_event.clear()
        self._frame_count = 0
        self._task = asyncio.create_task(self._capture_loop())
        logger.info(f"Xvfb frame capture started (interval={self.interval}s, dir={self.output_dir})")

    async def stop(self) -> int:
        """Stop capture and return total frames captured."""
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info(f"Xvfb frame capture stopped ({self._frame_count} frames captured)")
        return self._frame_count

    async def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            frame_path = self.output_dir / f"frame_{self._frame_count:06d}.png"
            try:
                proc = await asyncio.create_subprocess_exec(
                    "import",
                    "-display", self.display,
                    "-window", "root",
                    str(frame_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=5)
                if proc.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
                    self._frame_count += 1
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Frame capture failed: {e}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                break  # stop_event was set
            except asyncio.TimeoutError:
                continue  # interval elapsed, capture next frame
