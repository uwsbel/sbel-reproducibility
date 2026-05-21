"""
Display server for streaming simulation window to browser via Xvfb + x11vnc + noVNC.

Lifecycle: start() → (simulation runs on virtual display) → stop()
The browser connects to the noVNC websockify endpoint to view the live VSG window.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Optional

from chrono_code.config import get_settings

logger = logging.getLogger(__name__)


class DisplayServer:
    """Manages Xvfb → x11vnc → websockify process chain for browser-based display."""

    def __init__(self) -> None:
        settings = get_settings()
        self.display = settings.xvfb_display
        self.resolution = settings.xvfb_resolution
        self.vnc_port = settings.x11vnc_port
        self.novnc_port = settings.novnc_port

        self._xvfb: Optional[asyncio.subprocess.Process] = None
        self._x11vnc: Optional[asyncio.subprocess.Process] = None
        self._websockify: Optional[asyncio.subprocess.Process] = None
        self._running = False

    @property
    def novnc_url(self) -> str:
        return f"http://localhost:{self.novnc_port}/vnc.html?autoconnect=true&resize=scale"

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> str:
        """Start the display server chain. Returns the noVNC URL."""
        if self._running:
            return self.novnc_url

        for cmd in ("Xvfb", "x11vnc", "websockify"):
            if not shutil.which(cmd):
                raise RuntimeError(
                    f"{cmd} not found. Install with: sudo apt install -y xvfb x11vnc novnc websockify"
                )

        # 1. Start Xvfb
        depth = 24
        screen_spec = f"{self.resolution}x{depth}"
        self._xvfb = await asyncio.create_subprocess_exec(
            "Xvfb", self.display, "-screen", "0", screen_spec, "-ac",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Give Xvfb time to initialize
        await asyncio.sleep(0.5)
        if self._xvfb.returncode is not None:
            raise RuntimeError(f"Xvfb failed to start (exit code {self._xvfb.returncode})")
        logger.info(f"Xvfb started on display {self.display} ({screen_spec})")

        # 2. Start x11vnc
        self._x11vnc = await asyncio.create_subprocess_exec(
            "x11vnc",
            "-display", self.display,
            "-rfbport", str(self.vnc_port),
            "-nopw",
            "-forever",
            "-shared",
            "-noxdamage",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(0.3)
        if self._x11vnc.returncode is not None:
            await self.stop()
            raise RuntimeError(f"x11vnc failed to start (exit code {self._x11vnc.returncode})")
        logger.info(f"x11vnc started on port {self.vnc_port}")

        # 3. Start websockify (noVNC bridge)
        # Try to find the noVNC web directory
        novnc_web = "/usr/share/novnc"
        websockify_args = [
            "websockify",
            "--web", novnc_web,
            str(self.novnc_port),
            f"localhost:{self.vnc_port}",
        ]
        self._websockify = await asyncio.create_subprocess_exec(
            *websockify_args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(0.3)
        if self._websockify.returncode is not None:
            await self.stop()
            raise RuntimeError(f"websockify failed to start (exit code {self._websockify.returncode})")
        logger.info(f"websockify (noVNC) started on port {self.novnc_port}")

        self._running = True
        logger.info(f"Display server ready: {self.novnc_url}")
        return self.novnc_url

    async def stop(self) -> None:
        """Stop all display server processes."""
        for name, proc in [
            ("websockify", self._websockify),
            ("x11vnc", self._x11vnc),
            ("Xvfb", self._xvfb),
        ]:
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                logger.info(f"{name} stopped")

        self._xvfb = None
        self._x11vnc = None
        self._websockify = None
        self._running = False

    async def __aenter__(self) -> "DisplayServer":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()
