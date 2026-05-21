"""Main-loop driver for Chrono-Code generated simulations.

Centralizes the render-throttling + recorder-cleanup + exit-condition logic
that every generated `simulation.py` needs. Previously each codegen pass
copied an inline template from the skill docs and reassembled it from
scratch, which produced two classes of bugs:

1. Rendering VSG every physics step (instead of at a bounded `render_fps`)
   blew past the 300 s `ExecutionAgent` timeout on any non-trivial sim,
   triggering `SIGKILL`. `SIGKILL` skips `finally` blocks, so
   `recorder.close()` never ran, `cv2.VideoWriter.release()` never wrote
   the mp4 `moov` atom, and the downstream VLM got a 48-byte stub and
   returned HTTP 400.

2. Forgetting the `try/finally: recorder.close()` wrapper had the same
   outcome on any mid-run crash, even without the timeout.

`run_recording_loop` makes both problems structurally impossible: it owns
the loop, throttles VSG rendering to `render_fps`, pumps the sensor
manager + recorders every physics step, and runs `recorder.close()` in a
`finally` that fires on normal exit, exception, and `KeyboardInterrupt`
alike.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

from .logger import get_logger

logger = get_logger(__name__)


def run_recording_loop(
    sys,
    *,
    duration: float,
    time_step: float,
    vis=None,
    manager=None,
    recorders: Optional[Iterable] = None,
    render_fps: float = 100.0,
    step_fn: Optional[Callable[[int, float], None]] = None,
    on_step: Optional[Callable[[int, float], None]] = None,
    realtime: bool = False,
) -> dict:
    """Run a PyChrono simulation loop with correct render throttling and cleanup.

    Inputs:
        sys: a `ChSystem` (`ChSystemSMC`, `ChSystemNSC`, ...). The loop stops
            when `sys.GetChTime() >= duration`.
        duration: simulation-seconds to run.
        time_step: physics step size. `sys.DoStepDynamics(time_step)` is called
            each iteration unless `step_fn` overrides it.
        vis: optional visualization system (`ChVisualSystemVSG`-compatible).
            If `None`, the loop runs without any on-screen rendering. If
            provided, `BeginScene`/`Render`/`EndScene` run at roughly
            `render_fps`, NOT every physics step -- this is the critical
            throttling that keeps the wall-clock runtime bounded.
        manager: optional `ChSensorManager`. `manager.Update()` is called
            after each physics step.
        recorders: optional iterable of `CameraRecorder` (returned by
            `setup_preview_camera`). `recorder.tick()` is called each step,
            and `recorder.close()` is guaranteed to run via try/finally.
            WHEN LEFT AS None (default), the loop auto-collects every
            recorder that has been registered via `setup_preview_camera`
            so far in the process — you don't have to remember to thread
            each one through this argument. Pass an explicit list to
            override (e.g. in tests or for one-camera subsets); pass an
            empty list to skip recorder ticking entirely. When an explicit
            list is given, the loop logs a warning if any registered
            recorders are not included — that's the exact "forgot to pass
            my 5 rig cameras" bug.
        render_fps: target VSG frame rate. 50 Hz is smooth and cheap; pass
            0 to render every physics step (only safe for sims under ~1 s).
        step_fn: replaces the default `sys.DoStepDynamics(time_step)` call.
            Use for vehicle/terrain scenes that need per-step
            `Synchronize`/`Advance` orchestration. Signature
            `step_fn(step_index, sim_time)`; the callback owns the physics
            advance.
        on_step: optional `on_step(step_index, sim_time)` callback invoked
            AFTER the physics step, for CSV writes / contact reporting /
            logging. Exceptions in `on_step` are not caught -- they
            propagate and the finally-cleanup still flushes recorders.
        realtime: if True, sleep so wall-clock tracks sim time. Off by
            default because automated runs want to finish before the 300 s
            `ExecutionAgent` timeout; set True only for interactive demos.

    Returns: dict with keys `steps`, `sim_time`, `wall_time`, `render_count`.

    Why this helper exists (READ THIS BEFORE INLINING YOUR OWN LOOP):

        The generated `simulation.py` is run under `ExecutionAgent` with a
        hard 300 s timeout. A naive `while vis.Run(): ... vis.Render() ...
        DoStepDynamics(0.001)` loop with a 30 s sim + VSG preview will
        exceed that timeout, trigger `SIGKILL`, skip `finally`, and leave
        a corrupt mp4 that crashes the VLM review step. Use this helper
        instead of hand-writing the loop.
    """
    if recorders is None:
        # Default: tick every recorder that setup_preview_camera() registered
        # this process. Callers do not have to remember to thread each one in.
        # Lazy import: keeps simulation_loop importable without pychrono.sensor
        # loaded and avoids a hard circular dep on sensor_camera.
        try:
            from chrono_code.utils.sensor_camera import get_registered_recorders
            recorder_list = list(get_registered_recorders())
        except Exception as exc:  # noqa: BLE001 — defensive; must not block the loop
            logger.warning(
                "run_recording_loop: could not auto-collect registered "
                "recorders (%s); proceeding with an empty recorder list.",
                exc,
            )
            recorder_list = []
    else:
        recorder_list = list(recorders)
        # Warn loudly when the explicit list is missing recorders the user has
        # already built via setup_preview_camera(). This is the "iteration_006
        # built a 5-camera rig but only passed recorder=[1]" bug — silent today,
        # loud now.
        try:
            from chrono_code.utils.sensor_camera import get_registered_recorders
            registered = get_registered_recorders()
            passed_ids = {id(r) for r in recorder_list}
            missing = [r for r in registered if id(r) not in passed_ids]
            if missing:
                logger.warning(
                    "run_recording_loop: %d setup_preview_camera() recorder(s) "
                    "are registered but were NOT in the explicit recorders= "
                    "list. Their mp4 files will get zero frames. Pass "
                    "recorders=None to auto-collect all of them, or extend "
                    "the list. Missing: %s",
                    len(missing),
                    [getattr(r, "mp4_path", "?").name if hasattr(getattr(r, "mp4_path", None), "name") else "?" for r in missing],
                )
        except Exception:  # noqa: BLE001 — warning only, never raise
            pass

    if render_fps and render_fps > 0:
        render_every = max(1, int(round(1.0 / (time_step * render_fps))))
    else:
        render_every = 1

    if step_fn is None:
        def step_fn(step_index: int, sim_time: float) -> None:  # noqa: E306
            sys.DoStepDynamics(time_step)

    wall_start = time.time()
    step_number = 0
    render_count = 0
    exit_reason = "duration_reached"

    try:
        while True:
            sim_time = sys.GetChTime()
            if sim_time >= duration:
                exit_reason = "duration_reached"
                break
            if vis is not None and not vis.Run():
                exit_reason = "vis_closed"
                break

            if vis is not None and (step_number % render_every == 0):
                vis.BeginScene()
                vis.Render()
                vis.EndScene()
                render_count += 1

            step_fn(step_number, sim_time)

            if manager is not None:
                manager.Update()
            for rec in recorder_list:
                rec.tick()

            if on_step is not None:
                on_step(step_number, sim_time)

            step_number += 1

            if realtime:
                target = wall_start + step_number * time_step
                remaining = target - time.time()
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        # Guarantee mp4 finalization even if the loop raised or was SIGTERM-ed.
        # SIGKILL still bypasses this (nothing in userspace can catch it);
        # the separate ExecutionAgent fix is responsible for preferring
        # SIGTERM so this finally has a chance to run.
        for rec in recorder_list:
            try:
                rec.close()
            except Exception as e:
                logger.warning("recorder.close() failed for %s: %s", getattr(rec, "mp4_path", "?"), e)

    wall_time = time.time() - wall_start
    logger.info(
        "run_recording_loop done: steps=%d sim_time=%.3fs wall=%.2fs renders=%d reason=%s",
        step_number, sys.GetChTime(), wall_time, render_count, exit_reason,
    )
    return {
        "steps": step_number,
        "sim_time": sys.GetChTime(),
        "wall_time": wall_time,
        "render_count": render_count,
        "exit_reason": exit_reason,
    }
