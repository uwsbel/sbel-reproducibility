"""VSG → mp4 recording for FSI / vehicle scenes.

The companion to ``sensor_camera.py``: where ``setup_preview_camera``
records via ``ChCameraSensor`` (OptiX), this module records the
``ChVisualSystemVSG`` window directly. The two are NOT interchangeable —
SPH particles only render in the VSG pipeline (ChSphVisualizationVSG
plugin), so any FSI scene that needs water in the recording must use the
VSG path and not sensor cameras.

Three small helpers cover the canonical FSI-recording recipe:

* ``hide_vsg_gui(vis)`` — turn off the default ImGui panels and Chrono
  logo so the recorded frames are scene-only.

* ``lock_side_camera(vis, cam_pos, target_pos)`` — disable chase-camera
  follow logic and pin the view to a fixed world-frame side angle. MUST
  be called AFTER ``vis.Initialize()`` (Initialize resets the chase
  camera to its default ``Chase`` state).

* ``setup_vsg_recording(vis, mp4_path, fps, telemetry_log=None)`` — turn
  on VSG's per-render-frame BMP dump and return a ``finalize()`` closure
  that encodes those frames to mp4 (H.264 ``avc1`` first, falling back
  to ``mp4v`` for old OpenCV builds). Pass ``telemetry_log=[]`` and
  append a dict per render frame to overlay a HUD; pass ``None``
  (default) for a clean recording with no overlay.

Why we never call ``vis.WriteImageToFile()`` from a Python step callback:
that races VSG's swapchain and segfaults under load (verified). VSG's
own ``SetImageOutput(True)`` writes from inside the render context and
is the only safe way to snapshot every frame.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


def _resolve_relative_to_main_script(path: Any) -> Path:
    """Resolve a relative ``mp4_path`` against the running script's directory.

    Mirrors ``sensor_camera._default_output_base_dir`` so the two recording
    paths share one convention: ``setup_vsg_recording(vis, "cam/vsg.mp4")``
    inside ``history/iteration_NNN/simulation.py`` lands in
    ``history/iteration_NNN/cam/vsg.mp4`` regardless of the cwd from which
    the subprocess was launched. This matters because the ReviewAgent reads
    cam images from ``<visualization_output_path>/cam/`` where
    ``visualization_output_path`` is rebound to the iteration directory by
    ``ExecutionAgent`` — so cwd-relative or ``__file__/../results``-style
    paths silently miss the review pipeline (review then sees "no images"
    and falls back to the manifest-based pass, which masks real bugs).

    Absolute paths and falsey inputs are returned unchanged.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None) if main_mod else None
    if main_file:
        return (Path(main_file).resolve().parent / p).resolve()
    return p.resolve()


# ---------------------------------------------------------------------------
# GUI / camera helpers
# ---------------------------------------------------------------------------


def hide_vsg_gui(vis: Any) -> None:
    """Hide every default ImGui panel and the Chrono logo on a VSG window.

    VSG's default UI (Simulation / Vehicle / Camera State / Steering /
    Chrono Stats panels + the Chrono logo) overlaps the scene and clutters
    recordings. This collapses them all to "off". There is no "minimize" —
    VSG only exposes visible/hidden as a binary toggle per group.

    Idempotent and safe to call before or after ``vis.Initialize()``.
    """
    try:
        vis.SetGuiVisibility(False)
    except Exception:
        pass
    try:
        vis.SetBaseGuiVisibility(False)
    except Exception:
        pass
    try:
        vis.HideLogo()
    except Exception:
        pass


def lock_side_camera(vis: Any, cam_pos: Any, target_pos: Any) -> None:
    """Pin a VSG view to a fixed world-frame camera.

    Supports both vehicle VSG visual systems and the generic
    ``chronovsg.ChVisualSystemVSG``. Vehicle VSG exposes chase-camera methods;
    generic VSG exposes direct camera setters. Must be called AFTER
    ``vis.Initialize()`` because Initialize can reset camera state.

    Tutorial-canonical formula (vehicle's right side, 14 m off, 3 m up,
    looking at the pool surface center)::

        cam_pos    = chrono.ChVector3d(0, -7 * fyDim, 3 + bzDim / 2)
        target_pos = chrono.ChVector3d(0, 0, bzDim / 2)

    where ``fyDim`` is the tank Y dimension and ``bzDim`` is the wall
    top z. Negative Y = vehicle's right side in Chrono vehicle frame
    (+X forward, +Y left, +Z up).
    """
    # Vehicle visual systems need chase-follow disabled, otherwise they keep
    # recomputing the camera on render. Generic VSG does not have this API.
    if hasattr(vis, "SetChaseCameraState"):
        try:
            import pychrono.vehicle as veh  # local import keeps module importable without vehicle

            vis.SetChaseCameraState(veh.ChChaseCamera.Free)
        except Exception:
            # Some builds expose the method but not the enum binding. Continue
            # with explicit camera setters below; that still fixes generic VSG
            # and most first-frame camera placement issues.
            pass

    if hasattr(vis, "SetChaseCameraPosition"):
        vis.SetChaseCameraPosition(cam_pos, target_pos)
    else:
        if hasattr(vis, "SetCameraPosition"):
            vis.SetCameraPosition(cam_pos)
        if hasattr(vis, "SetCameraTarget"):
            vis.SetCameraTarget(target_pos)

    # UpdateCamera exists on vehicle VSG and some generic VSG builds. It makes
    # the first rendered frame use the requested pose immediately.
    if hasattr(vis, "UpdateCamera"):
        try:
            vis.UpdateCamera(cam_pos, target_pos)
        except TypeError:
            pass


# ---------------------------------------------------------------------------
# mp4 recording (with optional HUD overlay)
# ---------------------------------------------------------------------------


def _draw_telemetry_overlay(frame: Any, telemetry: dict) -> None:
    """Draw a multi-line telemetry HUD in the upper-left of the frame.

    Telemetry keys are rendered in insertion order (use a regular dict on
    Python 3.7+); values pass through ``str()``. Solid dark background
    keeps the text readable on bright water / sky.

    Silent no-op if ``cv2`` import fails or ``telemetry`` is empty.
    """
    if not telemetry:
        return
    try:
        import cv2
    except ImportError:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    line_h = 22
    pad_x, pad_y = 16, 28

    n_lines = len(telemetry)
    box_w = 270
    box_h = pad_y - 18 + n_lines * line_h + 8
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (24, 24, 24), -1)
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (180, 180, 180), 1)

    for i, (k, v) in enumerate(telemetry.items()):
        y = pad_y + i * line_h
        cv2.putText(
            frame, f"{k} = {v}", (pad_x, y), font, scale,
            (240, 240, 240), thick, cv2.LINE_AA,
        )


def _frame_dir_for(mp4_path: Path) -> Path:
    """Sibling directory next to ``mp4_path`` that holds raw VSG frame dumps.

    Deterministic so a parent process (e.g. ExecutionAgent) can find and
    encode any leftover frames after the child subprocess died — without
    having to inotify-track tempdirs or scan ``/tmp``.
    """
    return mp4_path.parent / f"_{mp4_path.stem}_frames"


def encode_vsg_frames(
    frame_dir: Any,
    mp4_path: Any,
    fps: float = 50.0,
    telemetry_log: Optional[list] = None,
) -> int:
    """Encode every BMP/PNG in ``frame_dir`` into ``mp4_path``. Idempotent.

    Two callers exercise this helper:

    * In-process: the closure returned by :func:`setup_vsg_recording`,
      called from ``finally:`` at the end of the simulation loop. Has
      access to the live ``telemetry_log`` mutated during the run, so
      the HUD overlay (if any) lands on the encoded frames.

    * Out-of-process recovery: the ExecutionAgent calls this on every
      iteration's ``cam/`` directory after the simulation subprocess
      exits — even when the subprocess was SIGKILL'd on timeout, which
      bypasses the in-process ``finally:``. The frames are on disk so
      encoding still works; the in-memory telemetry log is gone, so
      recovered mp4 has no HUD overlay (acceptable degradation — no
      overlay is better than no mp4).

    Behaviour:

    * If ``mp4_path`` already exists with size > 0 → already encoded by
      a previous caller; remove ``frame_dir`` and return its frame count
      (no re-encode). This is what makes the in-process and out-of-
      process callers safe to both run.
    * If ``frame_dir`` is missing or empty → return 0.
    * On successful encode → remove ``frame_dir``, return frame count.
    * If cv2 / a usable codec is unavailable → leave ``frame_dir`` in
      place so manual recovery is still possible; return 0.

    Codec preference: ``avc1`` (H.264) first for VS Code / browser /
    Slack compatibility; falls back to ``h264``, ``H264``, then ``mp4v``.
    """
    frame_dir = Path(frame_dir)
    mp4_path = Path(mp4_path)

    # Already done — clean up frame_dir and bail. Lets in-process and
    # post-mortem callers both run safely without races.
    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        if frame_dir.is_dir():
            shutil.rmtree(frame_dir, ignore_errors=True)
        return 0

    if not frame_dir.is_dir():
        return 0

    img_files = sorted(
        list(frame_dir.glob("*.bmp")) + list(frame_dir.glob("*.png"))
    )
    if not img_files:
        shutil.rmtree(frame_dir, ignore_errors=True)
        return 0

    try:
        import cv2
    except ImportError:
        print(f"VSG mp4: cv2 unavailable; raw frames left at {frame_dir}")
        return 0

    first = cv2.imread(str(img_files[0]))
    if first is None:
        print(f"VSG mp4: could not read {img_files[0]}; leaving raw frames at {frame_dir}")
        return 0
    h, w = first.shape[:2]

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for cc in ("avc1", "h264", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*cc)
        w_try = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
        if w_try.isOpened():
            writer = w_try
            print(f"VSG mp4: using fourcc={cc!r}")
            break
        w_try.release()
    if writer is None:
        print(f"VSG mp4: no usable codec available; raw frames left at {frame_dir}")
        return 0

    log = telemetry_log or []
    n = 0
    try:
        for f in img_files:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            if n < len(log):
                _draw_telemetry_overlay(frame, log[n])
            writer.write(frame)
            n += 1
    finally:
        writer.release()

    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"VSG mp4: {n} frames → {mp4_path}")
    return n


def setup_vsg_recording(
    vis: Any,
    mp4_path: Any,
    fps: float = 50.0,
    telemetry_log: Optional[list] = None,
):
    """Turn on VSG per-frame dump and return a ``finalize()`` callback.

    Frames are written to a deterministic sibling directory of
    ``mp4_path`` (``<mp4_dir>/_<stem>_frames/``), NOT a /tmp/ random dir.
    This lets the ExecutionAgent find and encode any frames left behind
    when the subprocess is SIGKILL'd on timeout — no signal-handling
    gymnastics inside the simulation needed.

    Args:
        vis: ``ChVisualSystemVSG`` instance (or vehicle subclass thereof).
        mp4_path: where to write the final mp4. Parent dir is created.
            **Relative paths are resolved against the running script's
            directory**, NOT the cwd — so ``"cam/vsg.mp4"`` always lands in
            ``<simulation.py-dir>/cam/vsg.mp4``, which is exactly where
            ReviewAgent looks (`<visualization_output_path>/cam/`). Pass an
            absolute path to override.
        fps: encoding fps. Should match the actual VSG render fps.
        telemetry_log: optional list of dicts. The i-th dict (if any) is
            burned in as a HUD overlay onto the i-th rendered frame.
            Default ``None`` = no overlay (clean scene-only recording).
            Pass an empty list and append per render frame to enable HUD.

    Returns:
        ``finalize()`` — call AFTER the simulation loop (typically inside
        a ``finally:`` block) to encode the dumped frames to mp4. Idempotent
        — safe to call multiple times; safe to skip entirely (parent agent
        will re-run encoding via :func:`encode_vsg_frames` on the same
        deterministic directory).

    Example::

        finalize = setup_vsg_recording(vis, "cam/vsg.mp4", fps=50)
        try:
            run_recording_loop(...)
        finally:
            finalize()
    """
    mp4_path = _resolve_relative_to_main_script(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = _frame_dir_for(mp4_path)
    frame_dir.mkdir(exist_ok=True)
    vis.SetImageOutputDirectory(str(frame_dir))
    vis.SetImageOutput(True)

    def finalize() -> int:
        return encode_vsg_frames(
            frame_dir, mp4_path, fps=fps, telemetry_log=telemetry_log,
        )

    return finalize


def encode_orphan_vsg_frames(
    cam_dir: Any,
    fps: float = 50.0,
) -> list:
    """Recover any orphan ``_<stem>_frames/`` dirs under ``cam_dir``.

    Called by ExecutionAgent after the simulation subprocess exits (any
    reason — clean exit, exception, SIGKILL on timeout). Walks
    ``cam_dir`` for sibling frame dirs created by :func:`setup_vsg_recording`
    and encodes each into ``<stem>.mp4`` next to it. Skips frame dirs
    whose mp4 already exists (in-process ``finalize()`` got there first).

    Returns the list of ``(mp4_path, frame_count)`` tuples actually
    encoded by this call (post-mortem only; the in-process path returns
    nothing here).
    """
    cam_dir = Path(cam_dir)
    if not cam_dir.is_dir():
        return []

    encoded: list = []
    for frame_dir in sorted(cam_dir.glob("_*_frames")):
        if not frame_dir.is_dir():
            continue
        # Strip leading "_" and trailing "_frames" → stem
        stem = frame_dir.name[1:-len("_frames")]
        mp4_path = cam_dir / f"{stem}.mp4"
        n = encode_vsg_frames(frame_dir, mp4_path, fps=fps)
        if n > 0:
            encoded.append((mp4_path, n))
    return encoded


__all__ = [
    "hide_vsg_gui",
    "lock_side_camera",
    "setup_vsg_recording",
    "encode_vsg_frames",
    "encode_orphan_vsg_frames",
]
