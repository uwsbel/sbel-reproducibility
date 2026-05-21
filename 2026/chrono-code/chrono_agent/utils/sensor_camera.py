"""One-call sensor-camera helper: preview window + direct mp4 recording.

Returns a `CameraRecorder`; the simulation loop must call `recorder.tick()`
each step (after `manager.Update()`) and `recorder.close()` before exit. A
process-wide atexit/SIGINT/SIGTERM hook flushes any recorder whose `close()`
did not run (e.g., crash, or the user slamming the VSG window shut) so the
mp4 header is always written.

mp4 frames are pulled from `ChFilterRGBA8Access` and encoded directly via
`cv2.VideoWriter` — no PNG scratch dir, no external ffmpeg dependency.
"""

from __future__ import annotations

import atexit
import importlib
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from chrono_agent.utils.logger import get_logger

if TYPE_CHECKING:
    import pychrono as chrono  # noqa: F401
    import pychrono.sensor as sens  # noqa: F401

logger = get_logger(__name__)

_active_recorders: List["CameraRecorder"] = []
_handlers_installed = False


def get_registered_recorders() -> List["CameraRecorder"]:
    """Return a snapshot of every ``CameraRecorder`` that is currently
    live and registered for atexit finalization.

    ``setup_preview_camera`` appends every recorder it returns to the
    process-wide registry so the atexit hook can flush them on crash /
    SIGTERM. ``run_recording_loop`` calls this to default to "tick every
    recorder the user built", so callers don't have to remember to thread
    every `CameraRecorder` through the `recorders=` argument. The returned
    list is a defensive copy; mutating it does not affect the registry.
    """
    with _worker_lock:
        return list(_active_recorders)

# Optional background worker that polls every active recorder for new frames.
# It exists only for legacy call sites that do not explicitly call `tick()`.
# Default behavior keeps recording single-threaded (manager.Update() -> tick())
# because that is substantially more stable when VSG preview windows are also
# alive in the same process.
_worker_thread: Optional[threading.Thread] = None
_worker_stop = threading.Event()
_worker_lock = threading.Lock()
_WORKER_POLL_INTERVAL_S = 0.001  # 1 kHz polling; dedup via LaunchedCount
_cv2_module = None
_cv2_import_attempted = False
_cv2_warning_emitted = False


def _get_cv2():
    """Import cv2 once and cache the result.

    Returns the module when available, otherwise None.
    """
    global _cv2_module, _cv2_import_attempted
    if not _cv2_import_attempted:
        _cv2_import_attempted = True
        try:
            _cv2_module = importlib.import_module("cv2")
        except ImportError:
            _cv2_module = None
    return _cv2_module


def _warn_cv2_missing_once(mp4_path: Path) -> None:
    """Emit a single process-wide warning when cv2 is unavailable."""
    global _cv2_warning_emitted
    if _cv2_warning_emitted:
        return
    _cv2_warning_emitted = True
    logger.warning(
        "OpenCV (cv2) is not installed; disabling mp4 recording for sensor cameras. "
        "Preview windows still work. First skipped output: %s",
        mp4_path,
    )


def _default_output_base_dir() -> Path:
    """Resolve relative camera outputs against the running script directory.

    This matches user expectation for `python path/to/script.py`: default
    `output_root="cam"` should land in `path/to/cam/`, not the caller's cwd.
    Fall back to cwd when the main module has no `__file__` (REPL/notebook).
    """
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None) if main_mod else None
    if main_file:
        return Path(main_file).resolve().parent
    return Path.cwd()


def _inprogress_path(final_path: Path) -> Path:
    """Return the while-writing path for ``final_path``.

    Frames land in ``<stem>.inprogress.mp4`` while the recorder is live; only
    after ``cv2.VideoWriter.release()`` writes the moov atom does ``close()``
    rename it to the final ``<stem>.mp4``. The ``.mp4`` extension must be kept
    on the in-progress file because OpenCV's FFMPEG backend picks the container
    muxer from the filename suffix — an unknown suffix like ``.mp4.inprogress``
    causes every codec attempt (avc1/H264/mp4v) to fail to open. Using
    ``.inprogress.mp4`` still makes the "not yet done" status obvious to humans
    scanning the directory, and a surviving ``*.inprogress.mp4`` after a run is
    still a "process got killed before the final flush" signal.
    """
    return final_path.with_name(final_path.stem + ".inprogress" + final_path.suffix)


def _open_video_writer(cv2, mp4_path: Path, fps: int, width: int, height: int):
    """Open a VideoWriter at the .inprogress sibling path.

    Returns ``(writer, final_path, inprogress_path, codec_name)``. The caller
    keeps ``final_path`` as the user-visible target and is responsible for
    renaming ``inprogress_path`` → ``final_path`` after ``writer.release()``.
    """
    codec_candidates = [
        ("avc1", ".mp4"),  # H.264 in MP4; most compatible with VS Code preview
        ("H264", ".mp4"),
        ("mp4v", ".mp4"),  # fallback: MPEG-4 Part 2
    ]
    last_final = mp4_path
    for fourcc_name, suffix in codec_candidates:
        final_candidate = mp4_path.with_suffix(suffix)
        inprogress_candidate = _inprogress_path(final_candidate)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(
            str(inprogress_candidate), fourcc, float(fps), (width, height)
        )
        if writer.isOpened():
            return writer, final_candidate, inprogress_candidate, fourcc_name
        writer.release()
        last_final = final_candidate
    return None, last_final, None, None


def _make_camera_rotation(camera_pos, target_pos, up_hint):
    import pychrono as chrono

    forward = target_pos - camera_pos
    flen = forward.Length()
    if flen < 1e-9:
        raise ValueError("camera_pos and target_pos must differ")
    forward = forward / flen

    up = chrono.ChVector3d(up_hint.x, up_hint.y, up_hint.z)
    ulen = up.Length()
    if ulen < 1e-9:
        raise ValueError("up_direction must be non-zero")
    up = up / ulen

    if abs(forward.Dot(up)) > 0.98:
        up = chrono.ChVector3d(0, 1, 0)
        if abs(forward.Dot(up)) > 0.98:
            up = chrono.ChVector3d(1, 0, 0)

    lateral = up.Cross(forward)
    lateral = lateral / lateral.Length()
    true_up = forward.Cross(lateral)
    true_up = true_up / true_up.Length()

    rot = chrono.ChMatrix33d()
    rot.SetFromDirectionAxes(forward, lateral, true_up)
    return rot.GetQuaternion()


def _worker_loop() -> None:
    """Daemon thread: pull frames from every active recorder periodically."""
    while not _worker_stop.is_set():
        with _worker_lock:
            recs = list(_active_recorders)
        for rec in recs:
            try:
                rec._grab_frame()
            except Exception as e:  # never let one bad recorder kill the thread
                logger.warning("sensor_camera grab error for %s: %s", rec.mp4_path, e)
        time.sleep(_WORKER_POLL_INTERVAL_S)


def _ensure_worker_started() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_stop.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop, name="sensor-camera-recorder", daemon=True
    )
    _worker_thread.start()


def _stop_worker_if_idle() -> None:
    """Stop the recorder worker when no recorder still depends on it."""
    global _worker_thread
    with _worker_lock:
        if _active_recorders:
            return
        _worker_stop.set()
        thread = _worker_thread
        _worker_thread = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)


def _finalize_all() -> None:
    # Stop the worker first so it does not touch writers while we are releasing them.
    _worker_stop.set()
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=1.0)
    _worker_thread = None

    for rec in list(_active_recorders):
        try:
            rec.close()
        except Exception as e:  # never raise during shutdown
            logger.warning("sensor_camera finalize error for %s: %s", rec.mp4_path, e)


def _signal_handler(signum, frame):
    _finalize_all()
    sys.exit(0)


def _install_handlers_once() -> None:
    global _handlers_installed
    if _handlers_installed:
        return
    _handlers_installed = True
    atexit.register(_finalize_all)
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except (ValueError, OSError):
        pass  # not main thread; atexit alone has to do


class CameraRecorder:
    """Closure wrapping a `ChCameraSensor` + `cv2.VideoWriter`.

    Default usage is explicit and single-threaded: the simulation loop calls
    `manager.Update()` and then `recorder.tick()` once per step. An optional
    background daemon thread can still be enabled for legacy call sites, but it
    is not the default because concurrent frame pulling is a common source of
    shutdown instability when preview windows are active.
    """

    def __init__(self, cam, mp4_path: Path, fps: int, *, background_recording: bool = False):
        self.cam = cam
        self.mp4_path = mp4_path  # final, user-visible path (not where cv2 writes)
        self._inprogress_path: Optional[Path] = None  # actual cv2 target; set on first frame
        self.fps = max(1, int(fps))
        self._frame_interval = 1.0 / self.fps  # sim-seconds between recorded frames
        self._writer = None
        self._last_launched = -1
        self._last_written_ts = -1.0
        self._closed = False
        self._frames_written = 0
        self._lock = threading.Lock()  # serialize writer access between worker & explicit tick
        self._recording_enabled = _get_cv2() is not None
        self._background_recording = bool(background_recording)
        if not self._recording_enabled:
            _warn_cv2_missing_once(self.mp4_path)

    def _grab_frame(self) -> None:
        """Core frame-pull logic. Called by both the background worker and `tick()`."""
        if self._closed or not self._recording_enabled:
            return
        cv2 = _get_cv2()
        if cv2 is None:
            self._recording_enabled = False
            _warn_cv2_missing_once(self.mp4_path)
            return

        buf = self.cam.GetMostRecentRGBA8Buffer()
        if not buf.HasData():
            return
        launched = buf.LaunchedCount

        with self._lock:
            if self._closed:
                return
            if launched == self._last_launched:
                return  # sensor has not produced a new frame since last grab
            self._last_launched = launched

            # Decouple sensor update_rate from video fps: only record a frame when
            # enough simulation time has elapsed since the previous written frame.
            # Without this a 1 kHz sensor would produce a 1000 fps mp4 that most
            # players refuse to decode.
            ts = float(buf.TimeStamp)
            if self._last_written_ts >= 0.0 and (ts - self._last_written_ts) < self._frame_interval - 1e-9:
                return

            rgba = buf.GetRGBA8Data()
            if rgba is None or rgba.size == 0:
                return

            frame = cv2.cvtColor(rgba[::-1], cv2.COLOR_RGBA2BGR)

            if self._writer is None:
                h, w = frame.shape[:2]
                self.mp4_path.parent.mkdir(parents=True, exist_ok=True)
                (
                    self._writer,
                    final_path,
                    inprogress_path,
                    codec_name,
                ) = _open_video_writer(cv2, self.mp4_path, self.fps, w, h)
                if self._writer is None:
                    logger.warning(
                        "cv2.VideoWriter failed to open for %s", self.mp4_path
                    )
                    return
                self.mp4_path = final_path
                self._inprogress_path = inprogress_path
                logger.info(
                    "sensor_camera recording %s (writing to %s) using codec %s",
                    self.mp4_path,
                    self._inprogress_path.name,
                    codec_name,
                )

            # Fill gap frames: if simulation ran faster than the polling rate,
            # intermediate frames were lost. Duplicate the current frame to
            # cover the elapsed simulation time so the mp4 duration stays
            # faithful to the actual simulation duration.
            if self._last_written_ts >= 0.0:
                gap = ts - self._last_written_ts
                n_frames = max(1, int(round(gap / self._frame_interval)))
            else:
                n_frames = 1

            for _ in range(n_frames):
                self._writer.write(frame)
            self._frames_written += n_frames
            self._last_written_ts = ts

    def tick(self) -> None:
        """Optional explicit frame pull. Same as what the background worker does
        automatically each poll; idempotent thanks to `LaunchedCount` dedup."""
        self._grab_frame()

    def close(self) -> None:
        """Flush and release the mp4 writer, then atomically promote the
        ``<stem>.inprogress.mp4`` file to its final ``<stem>.mp4`` name.
        Idempotent and crash-safe; if release() / rename() fail the
        in-progress file stays put so operators can see the run did not
        finalize cleanly.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception as e:
                    logger.warning(
                        "VideoWriter release error for %s: %s", self.mp4_path, e
                    )
                self._writer = None
            # Atomically promote <stem>.inprogress.mp4 -> <stem>.mp4 so users
            # never see a half-written file under the final name. A zero-frame
            # writer (e.g. a recorder that was never ticked) leaves a tiny
            # unplayable stub; rename it anyway but log a warning — the presence
            # of a dangling 0-frame mp4 is a strong signal of a wiring bug like
            # "forgot to pass this recorder into run_recording_loop".
            inprogress = self._inprogress_path
            if inprogress is not None and inprogress.exists():
                try:
                    if self._frames_written == 0:
                        logger.warning(
                            "sensor_camera: 0 frames written to %s — likely the "
                            "recorder was created by setup_preview_camera() but "
                            "never ticked by run_recording_loop(). Promoting to "
                            "%s anyway so no stale .inprogress lingers.",
                            inprogress.name,
                            self.mp4_path.name,
                        )
                    inprogress.replace(self.mp4_path)  # POSIX-atomic rename
                except OSError as e:
                    logger.warning(
                        "sensor_camera: failed to rename %s -> %s (%s); leaving "
                        ".inprogress in place.",
                        inprogress,
                        self.mp4_path,
                        e,
                    )
            self._inprogress_path = None
        with _worker_lock:
            try:
                _active_recorders.remove(self)
            except ValueError:
                pass
        _stop_worker_if_idle()


def setup_preview_camera(
    manager,
    attach_body,
    target_pos,
    cam_pos,
    up_direction,
    *,
    update_rate: float,
    name: str = "camera",
    width: int = 1280,
    height: int = 720,
    fov: float = 1.408,
    fps: Optional[int] = None,
    output_root: str | Path = "cam",
    preview: bool = True,
    background_recording: bool = False,
) -> CameraRecorder:
    """Build a `ChCameraSensor` with live preview + direct mp4 recording.

    Inputs:
        manager: an existing `sens.ChSensorManager`.
        attach_body: the chrono body the camera rides on.
        target_pos, cam_pos, up_direction: `chrono.ChVector3d`.
        update_rate: sensor update rate in Hz (typically `1.0 / step_size`).
        name: basename for the output mp4 and preview window.
        fps: output mp4 playback rate. Defaults to `int(round(update_rate))`
            so the video plays at wall-clock speed (no temporal aliasing).
        output_root: directory for the mp4. Relative paths are resolved against
            the running script's directory; absolute paths are used as-is.

    Behavior:
        - Attaches `ChFilterVisualize` when `preview=True`, plus
          `ChFilterRGBA8Access` for direct encoding.
        - Returns a `CameraRecorder`. The simulation loop MUST call
          `recorder.tick()` each step after `manager.Update()`, and
          `recorder.close()` before exit. A process-wide atexit + SIGINT/
          SIGTERM hook flushes any recorder whose close() did not run.
        - `background_recording=False` is the stable default. Set it to True
          only for legacy code that cannot call `tick()` explicitly.
        - Multiple calls in one process each produce their own mp4.

    Returns: a `CameraRecorder` wrapping the underlying `ChCameraSensor`
    (available as `recorder.cam` if you need the raw sensor handle).
    """
    import pychrono as chrono
    import pychrono.sensor as sens

    base_dir = _default_output_base_dir()
    output_root_path = Path(output_root)
    if output_root_path.is_absolute():
        mp4_path_abs = (output_root_path / f"{name}.mp4").resolve()
    else:
        mp4_path_abs = (base_dir / output_root_path / f"{name}.mp4").resolve()

    rot = _make_camera_rotation(cam_pos, target_pos, up_direction)
    attach_body.SetPos(cam_pos)
    offset_pose = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), rot)

    cam = sens.ChCameraSensor(attach_body, update_rate, offset_pose, width, height, fov)
    cam.SetName(name)
    if preview:
        cam.PushFilter(sens.ChFilterVisualize(width, height, name))
    cam.PushFilter(sens.ChFilterRGBA8Access())
    manager.AddSensor(cam)

    # Default to update_rate / 10 (i.e. one stored frame per 10 sensor ticks).
    # Keeps mp4 fps at a player-friendly value while still following wall-clock
    # when update_rate == 1/timestep.
    effective_fps = int(round(fps)) if fps is not None else max(1, int(round(update_rate))/20)
    recorder = CameraRecorder(
        cam,
        mp4_path_abs,
        effective_fps,
        background_recording=background_recording,
    )
    if recorder._recording_enabled:
        # Always register for atexit / SIGTERM finalization so the mp4 moov
        # atom is written even when the caller forgets `finally: close()`.
        # The background worker itself is still gated on background_recording
        # because concurrent frame pulling is only needed for callers that
        # do not drive `tick()` themselves.
        with _worker_lock:
            _active_recorders.append(recorder)
        _install_handlers_once()
        if recorder._background_recording:
            _ensure_worker_started()

    return recorder
