"""
Execution Agent (Agent 4) for running PyChrono simulations.

The simulation stores sensor camera screenshots in ./outputs/cams/
and optional simulation-window frames in ./outputs/sim_window/ for
downstream visual review.
"""

import asyncio
import logging
import os
import re
import resource
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from chrono_agent.agents.base import BaseAgent
from chrono_agent.agents.exceptions import AgentLLMError
from chrono_agent.agents.prompts import execution_prompts
from chrono_agent.config import get_settings
from chrono_agent.models.code import GeneratedCode
from chrono_agent.models.execution import ExecutionResult
from chrono_agent.models.plan import SimulationPlan
from chrono_agent.utils.async_fs import ensure_dir

logger = logging.getLogger(__name__)

CAMERA_OUTPUT_DIR = "cam"
SIM_WINDOW_OUTPUT_DIRS = ("sim_window", "window_frames")
PRIMARY_SIM_WINDOW_DIR = "sim_window"
PRECREATE_OUTPUT_DIRS = (CAMERA_OUTPUT_DIR, PRIMARY_SIM_WINDOW_DIR)
IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "gif")
VIDEO_EXTENSIONS = ("mp4", "avi", "mov")
REQUIRED_TIME_STEP = 0.01
REQUIRED_VIDEO_FPS = int(round(1.0 / REQUIRED_TIME_STEP))


def prepare_next_iteration_dir(history_root: Path) -> Tuple[Path, int]:
    """Create and return the next ``iteration_NNN`` directory under ``history_root``.

    Shared helper used by both codegen (to pre-create the directory before the
    LLM tool loop writes ``simulation.py``) and execution (to run the script).
    """
    history_root = Path(history_root).resolve()
    history_root.mkdir(parents=True, exist_ok=True)

    existing: List[int] = []
    for candidate in history_root.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("iteration_"):
            continue
        suffix = candidate.name.removeprefix("iteration_")
        if suffix.isdigit():
            existing.append(int(suffix))

    iteration_index = (max(existing) + 1) if existing else 1
    iteration_dir = history_root / f"iteration_{iteration_index:03d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir, iteration_index


def _merge_unique_error_messages(*message_groups: Any) -> List[str]:
    seen = set()
    unique: List[str] = []

    for group in message_groups:
        if isinstance(group, str):
            candidates = group.splitlines()
        elif isinstance(group, (list, tuple)):
            candidates = [str(item) for item in group]
        else:
            candidates = [str(group)] if group else []

        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(text)

    return unique


class ExecutionAgent(BaseAgent):
    """
    Agent 4: Executes PyChrono simulations safely.

    Responsibilities:
    - Execute generated code in isolated environment
    - Capture sensor camera screenshots (cams/)
    - Handle runtime errors
    - Generate execution summary
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 300,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Execution Agent.

        Args:
            llm_provider: LLM provider to use
            model: Specific model to use
            temperature: Temperature for generation
            timeout: Execution timeout in seconds
            output_dir: Directory for output files
        """
        super().__init__(
            agent_name="ExecutionAgent",
            agent_number=4,
            llm_provider=llm_provider,
            model=model,
            temperature=temperature,
        )

        self.timeout = timeout
        self.settings = get_settings()
        # Always anchor on the canonical history root, NOT on visualization_output_path,
        # because visualization_output_path is mutated to point at the current iteration
        # during each execute() call. Reading it back on the next agent instance would
        # cause nested iteration_001/iteration_001/... folders.
        if output_dir is not None:
            history_root = Path(output_dir)
        else:
            history_root = self.settings.history_output_path()
        self._history_root = history_root
        self.output_dir = Path(history_root)

    def _next_iteration_dir(self) -> Tuple[Path, int]:
        """Create and return the next iteration directory under the history root."""
        return prepare_next_iteration_dir(self._history_root)

    async def _log_execution_round(
        self,
        *,
        iteration_index: int,
        iteration_dir: Optional[Path],
        script_name: str,
        script_contents: str,
        result: ExecutionResult,
    ) -> None:
        """Record this execution attempt to ``session/execution/NNN_*.txt``.

        ExecutionAgent is a subprocess runner, not an LLM, so the BaseAgent
        ``invoke_llm`` logging path never fires. Without this hook the per-round
        stderr / structured_error never lands in the session folder.
        """
        if self.dialog_manager is None:
            return

        try:
            import json as _json

            iter_dir_str = str(iteration_dir) if iteration_dir is not None else ""
            prompt_text = (
                f"Iteration: {iteration_index:03d}\n"
                f"Script: {script_name}\n"
                f"Iteration dir: {iter_dir_str}\n"
                f"Script length: {len(script_contents)} bytes\n"
                f"Timeout: {self.timeout}s\n\n"
                f"--- Script Source ---\n"
                f"{script_contents}"
            )

            structured = getattr(result, "structured_error", None) or {}
            err_msgs = getattr(result, "error_messages", []) or []
            try:
                structured_block = _json.dumps(structured, indent=2, ensure_ascii=False)
            except Exception:
                structured_block = repr(structured)

            response_parts = [
                f"Success: {bool(result.success)}",
                f"Return code: {result.return_code}",
                f"Runtime: {result.runtime_seconds:.2f}s",
                "",
                "--- Error Message ---",
                str(result.error_message or ""),
                "",
                "--- Error Messages (deduped) ---",
                "\n".join(err_msgs) if err_msgs else "",
                "",
                "--- Structured Error ---",
                structured_block,
                "",
                "--- Execution Log (STDOUT + STDERR) ---",
                str(result.execution_log or ""),
            ]
            response_text = "\n".join(response_parts)

            metadata = {
                "iteration_index": iteration_index,
                "script_name": script_name,
                "success": bool(result.success),
                "return_code": result.return_code,
                "runtime_seconds": result.runtime_seconds,
                "output_files": len(result.output_files or []),
            }

            await asyncio.to_thread(
                self.dialog_manager.log_prompt,
                agent_name=self.agent_name,
                prompt=prompt_text,
                metadata=metadata,
            )
            await asyncio.to_thread(
                self.dialog_manager.log_response,
                agent_name=self.agent_name,
                response=response_text,
                metadata=metadata,
            )
        except Exception as log_err:
            self.logger.warning(f"Failed to log execution round to session: {log_err}")

    async def execute(
        self,
        generated_code: GeneratedCode,
        iteration_dir: Optional[Path] = None,
        step_info: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute the generated simulation code.

        Args:
            generated_code: Code to execute
            iteration_dir: Pre-created iteration directory (typically prepared by
                codegen so that simulation.py is synced to the same place it will
                be executed from). If omitted, a fresh iteration directory is
                created here.
            step_info: Optional ``{"step_number": int, "total_steps": int}`` —
                when provided, the script's ``vis.SetWindowTitle(...)`` literal
                is force-synced to ``Step N of M ...`` matching the workflow's
                current step. Always (with or without ``step_info``) the
                title is ASCII-sanitized so VSG's default font does not render
                em-dashes / smart quotes / CJK as garbled glyphs.

        Returns:
            ExecutionResult object
        """
        self.logger.info("Executing simulation...")

        # Ensure output directory exists (async-safe under ASGI)
        await ensure_dir(self.output_dir)

        # Resolve to absolute path once (off event loop) to avoid os.getcwd() on event loop later
        self.output_dir = await asyncio.to_thread(self.output_dir.resolve)

        if iteration_dir is not None:
            iteration_dir = Path(iteration_dir).resolve()
            iteration_dir.mkdir(parents=True, exist_ok=True)
            suffix = iteration_dir.name.removeprefix("iteration_")
            iteration_index = int(suffix) if suffix.isdigit() else 0
        else:
            # Create a fresh iteration directory — all outputs for this run go here.
            iteration_dir, iteration_index = await asyncio.to_thread(self._next_iteration_dir)

        # Switch output_dir to the iteration directory so _find_output_files,
        # _cleanup_empty_output_dirs, etc. all operate inside it.
        self.output_dir = iteration_dir

        # Update visualization_output_path so downstream nodes (review, physics
        # analysis) look for cams/ and CSV files inside this iteration directory.
        object.__setattr__(self.settings, "visualization_output_path", str(iteration_dir))

        # Ensure only required output directories exist inside the iteration dir.
        for frame_dir_name in PRECREATE_OUTPUT_DIRS:
            await ensure_dir(iteration_dir / frame_dir_name)

        start_time = time.time()

        # Pre-bind for the outer except path so logging works even if the
        # script-write step raises before these are otherwise assigned.
        script_name = generated_code.file_name
        script_contents = generated_code.code or ""

        try:
            from chrono_agent.utils.vsg_title import rewrite_window_titles

            script_contents, n_titles = rewrite_window_titles(
                script_contents, step_info=step_info
            )
            if n_titles:
                self.logger.info(
                    "VSG window title sync: rewrote %d SetWindowTitle call(s)%s",
                    n_titles,
                    (
                        f" to Step {step_info['step_number']} of {step_info['total_steps']}"
                        if step_info
                        and "step_number" in step_info
                        and "total_steps" in step_info
                        else " (ASCII sanitization only)"
                    ),
                )

            script_path = (iteration_dir / script_name).resolve()
            script_path.write_text(script_contents, encoding="utf-8")

            self.logger.info(
                f"Running simulation: {script_path.name} "
                f"(iteration {iteration_index:03d})"
            )

            # Resolve the chrono-aware interpreter and env. This reuses the same
            # helper the bash/validator tools use, so LD_LIBRARY_PATH *and*
            # PYTHONPATH (which exposes `pychrono.core`) are set consistently.
            # Without PYTHONPATH the subprocess can't find `pychrono` even when
            # the interpreter is inside the chrono-agent env.
            from chrono_agent.tools.code_agent_tools import (
                _resolve_validator_python,
                _build_validator_env,
            )
            python_executable = _resolve_validator_python()
            self.logger.debug(f"Using Python interpreter: {python_executable}")

            env = _build_validator_env(python_executable)
            env['PYTHONFAULTHANDLER'] = '1'

            if "DISPLAY" not in env:
                self.logger.warning("DISPLAY is not set; GUI visualization windows may not appear")

            # Function to enable core dump generation in subprocess
            def enable_core_dump():
                """Enable core dump generation in subprocess."""
                try:
                    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
                except Exception:
                    pass  # Ignore if setting ulimit fails

            self.logger.info("Running simulation (capturing sensor + simulation-window frames)")

            result = await asyncio.create_subprocess_exec(
                python_executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(iteration_dir),
                env=env,
                preexec_fn=enable_core_dump,
            )

            try:
                async def _stream_pipe(pipe, pipe_name: str) -> str:
                    """Read lines from a subprocess pipe, emitting each as a stream event."""
                    from chrono_agent.workflow.events import emit_subprocess_event

                    collected: list[str] = []
                    async for raw_line in pipe:
                        line = raw_line.decode("utf-8", errors="replace")
                        collected.append(line)
                        emit_subprocess_event(pipe_name, line)
                    return "".join(collected)

                stdout_task = asyncio.create_task(_stream_pipe(result.stdout, "stdout"))
                stderr_task = asyncio.create_task(_stream_pipe(result.stderr, "stderr"))

                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, result.wait()),
                    timeout=self.timeout,
                )

                runtime = time.time() - start_time

                # Encode any VSG frame dumps the subprocess left behind.
                # Idempotent — no-op when the script's in-process finalize()
                # already produced the mp4 (clean exit / catchable exception).
                await asyncio.to_thread(self._recover_vsg_frames)

                # HR-16 audit: an FSI script that ran successfully but
                # produced no particles.csv is a codegen bug — the
                # downstream fluid_containment predicate cannot detect
                # leaks without it. Heuristic FSI detection: the script
                # imported / referenced an FSI-side symbol. Logging here
                # surfaces the issue early; the validator will also emit
                # FLUID_CONTAINMENT: SKIPPED in the structured findings.
                _is_fsi_script = (
                    "pychrono.fsi" in script_contents
                    or "sysFSI" in script_contents
                    or "ChFsiProblem" in script_contents
                    or "ChFluidSystemSPH" in script_contents
                )
                _particles_csv_path = iteration_dir / "particles.csv"
                if (
                    _is_fsi_script
                    and result.returncode == 0
                    and not _particles_csv_path.exists()
                ):
                    self.logger.warning(
                        "[HR-16] FSI script ran successfully but produced no "
                        "particles.csv at %s — fluid_containment review check "
                        "will be SKIPPED. Codegen must call "
                        "sysSPH.GetParticlePositions() at end of sim and "
                        "write particles.csv (HR-16 in fsi/sph/SKILL.md).",
                        _particles_csv_path,
                    )

                stdout_str = stdout_task.result()
                stderr_str = stderr_task.result()

                execution_log = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}"

                # Check if execution succeeded.
                # Treat SIGINT(-2) and SIGTERM(-15) as user-initiated window
                # closure / interruption — not a code error.
                _user_stop_codes = {0, -2, -15}  # OK, SIGINT, SIGTERM
                success = result.returncode in _user_stop_codes

                # Non-zero exit without a Python traceback in stderr: could be
                # the user closing the VSG window (benign) or a real crash.
                #
                # Check BOTH stdout and stderr for traceback — some environments
                # may redirect stderr to stdout.  Also look for common error
                # patterns (C++ exceptions, assertion failures, etc.).
                _combined_output = f"{stdout_str}\n{stderr_str}"
                _has_traceback = "Traceback" in _combined_output

                if not success and not _has_traceback:
                    _combined_lower = _combined_output.lower()
                    _error_indicators = (
                        "error", "fault", "abort", "exception",
                        "failed", "cannot", "not found", "no such",
                        "segmentation", "core dumped", "assertion",
                        "invalid", "undefined", "killed",
                    )
                    _has_error_signal = any(
                        indicator in _combined_lower
                        for indicator in _error_indicators
                    )
                    _min_runtime_for_user_close = 2.0

                    if not _has_error_signal and runtime >= _min_runtime_for_user_close:
                        # Ran long enough, no error keywords → user closed window
                        self.logger.info(
                            f"Process exited with code {result.returncode} but no "
                            "traceback and no error signals — treating as "
                            "user-interrupted success"
                        )
                        success = True
                    else:
                        # Real failure — extract error context from output.
                        _error_block = self._extract_error_block(_combined_output, result.returncode, runtime)
                        self.logger.warning(
                            f"Process exited with code {result.returncode}, no "
                            f"Python traceback but error detected:\n{_error_block}"
                        )
                        # Inject into stderr so the failure path picks it up.
                        stderr_str = (
                            f"{stderr_str}\n\n"
                            f"[No Python traceback — extracted error context]\n"
                            f"{_error_block}"
                        ).strip()
                        execution_log = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}"

                # Traceback found in stdout but not stderr — move it so the
                # failure path (which reads stderr_str) can see it.
                if not success and _has_traceback and "Traceback" not in stderr_str:
                    stderr_str = f"{stderr_str}\n\n{stdout_str}".strip()
                    execution_log = f"STDOUT:\n{stdout_str}\n\nSTDERR:\n{stderr_str}"

                if result.returncode in (-2, -15):
                    self.logger.info(
                        f"Process exited with signal {-result.returncode} "
                        "(user closed window) — treating as success"
                    )
                # Find raw output files (includes image frames before post-processing) off event loop.
                output_files_raw = await asyncio.to_thread(self._find_output_files, True)

                if success:
                    self.logger.info(
                        f"Execution SUCCESS in {runtime:.2f}s, "
                        f"{len(output_files_raw)} files generated"
                    )

                    # Collect CSV/PNG paths for physics analysis (no pre-analysis)
                    csv_files = [{"file": str(f)} for f in output_files_raw if f.suffix.lower() == ".csv"]
                    if csv_files:
                        self.logger.info(f"Found {len(csv_files)} CSV file(s) for physics analysis")

                    await asyncio.to_thread(self._cleanup_empty_output_dirs)
                    output_files = await asyncio.to_thread(self._find_output_files, False)

                    success_result = ExecutionResult(
                        success=True,
                        output_files=[str(f) for f in output_files],
                        execution_log=execution_log,
                        runtime_seconds=runtime,
                        return_code=result.returncode,
                        csv_data_summary={},
                        csv_files=csv_files,
                        simulation_data_metrics={},
                    )
                    await self._log_execution_round(
                        iteration_index=iteration_index,
                        iteration_dir=iteration_dir,
                        script_name=script_name,
                        script_contents=script_contents,
                        result=success_result,
                    )
                    return success_result

                self.logger.error(
                    f"Execution FAILED with return code {result.returncode}"
                )

                # Check if this is a signal-based crash (segfault, etc.)
                signal_crash_codes = {-11, -6, -8, -7}  # SIGSEGV, SIGABRT, SIGFPE, SIGBUS
                is_signal_crash = result.returncode in signal_crash_codes

                # If signal crash, get C/C++ backtrace: first try core file, then re-run under GDB
                gdb_backtrace = ""
                if is_signal_crash:
                    self.logger.info(f"Detected signal-based crash (signal {-result.returncode}), fetching C/C++ backtrace...")
                    gdb_backtrace = await self._analyze_core_dump(self.output_dir, python_executable)
                    if not gdb_backtrace:
                        gdb_backtrace = await self._get_backtrace_via_gdb_live(
                            python_executable=python_executable,
                            script_path=script_path,
                            cwd=str(iteration_dir),
                            env=env,
                            timeout=90,
                        )
                    if gdb_backtrace:
                        execution_log += gdb_backtrace
                        self.logger.info("Added C/C++ backtrace to execution log")
                    else:
                        execution_log += (
                            "\n\n[Note: C/C++ backtrace not available (no core file, gdb not installed, or GDB run failed). "
                            "Above is Python/faulthandler output: crash location only.]"
                        )

                # Still try to find and analyze output files even on failure
                # Partial data can help diagnose why the simulation crashed
                output_files_raw = await asyncio.to_thread(self._find_output_files, True)

                csv_files = [{"file": str(f)} for f in output_files_raw if f.suffix.lower() == ".csv"]
                if output_files_raw:
                    self.logger.info(f"Found {len(output_files_raw)} output files from failed execution")

                await asyncio.to_thread(self._cleanup_empty_output_dirs)
                output_files = await asyncio.to_thread(self._find_output_files, False)

                # Build detailed error message
                error_message = f"Process exited with code {result.returncode}\n{stderr_str}"
                if is_signal_crash:
                    signal_names = {-11: 'SIGSEGV', -6: 'SIGABRT', -8: 'SIGFPE', -7: 'SIGBUS'}
                    signal_name = signal_names.get(result.returncode, f'signal {-result.returncode}')
                    error_message = f"Process crashed with {signal_name}\n{stderr_str}"
                error_messages = _merge_unique_error_messages(error_message, stderr_str)

                # Structured error — feeds the codegen retry prompt with a
                # targeted introspection hint instead of a stderr blob.
                structured_error = self._structure_runtime_error(
                    stderr=stderr_str,
                    code=script_contents,
                    return_code=result.returncode,
                )
                if structured_error:
                    self.logger.info(
                        f"[structured_error] type={structured_error.get('error_type')} "
                        f"symbol={structured_error.get('failing_symbol')} "
                        f"line={structured_error.get('failing_line')}"
                    )

                failure_result = ExecutionResult(
                    success=False,
                    output_files=[str(f) for f in output_files],
                    execution_log=execution_log,
                    runtime_seconds=runtime,
                    return_code=result.returncode,
                    error_message=error_message,
                    error_messages=error_messages,
                    csv_data_summary={},
                    csv_files=csv_files,
                    simulation_data_metrics={},
                    structured_error=structured_error,
                )
                await self._log_execution_round(
                    iteration_index=iteration_index,
                    iteration_dir=iteration_dir,
                    script_name=script_name,
                    script_contents=script_contents,
                    result=failure_result,
                )
                return failure_result

            except asyncio.TimeoutError:
                # Timeout means the simulation ran without crashing - this is SUCCESS
                # The simulation may have visualization blocking or just long runtime
                self.logger.info(f"Execution ran for {self.timeout}s (timeout) - treating as SUCCESS")
                self.logger.info("  Simulation ran without errors. Physics validation handled by ReviewAgent.")

                # Cancel stream tasks and kill the process
                stdout_task.cancel()
                stderr_task.cancel()
                result.kill()
                await result.wait()

                runtime = time.time() - start_time

                # SIGKILL bypassed the script's in-process finalize() —
                # encode the orphan VSG frame dumps from the deterministic
                # cam/_*_frames/ directories.
                await asyncio.to_thread(self._recover_vsg_frames)

                # Find raw output files (includes image frames before post-processing) off event loop.
                output_files_raw = await asyncio.to_thread(self._find_output_files, True)

                csv_files = [{"file": str(f)} for f in output_files_raw if f.suffix.lower() == ".csv"]
                if output_files_raw:
                    self.logger.info(f"Found {len(output_files_raw)} output files")

                await asyncio.to_thread(self._cleanup_empty_output_dirs)
                output_files = await asyncio.to_thread(self._find_output_files, False)

                timeout_result = ExecutionResult(
                    success=True,  # Timeout = SUCCESS (code ran without errors)
                    output_files=[str(f) for f in output_files],
                    execution_log=f"Simulation ran for {runtime:.2f}s (reached timeout, no errors)",
                    runtime_seconds=runtime,
                    csv_data_summary={},
                    csv_files=csv_files,
                    simulation_data_metrics={},
                )
                await self._log_execution_round(
                    iteration_index=iteration_index,
                    iteration_dir=iteration_dir,
                    script_name=script_name,
                    script_contents=script_contents,
                    result=timeout_result,
                )
                return timeout_result

            finally:
                pass

        except Exception as e:
            runtime = time.time() - start_time

            self.logger.error(f"Execution ERROR: {e}")
            # Recover orphan VSG frames BEFORE the image cleanup wipes them.
            await asyncio.to_thread(self._recover_vsg_frames)
            await asyncio.to_thread(self._cleanup_image_outputs)
            await asyncio.to_thread(self._cleanup_empty_output_dirs)
            output_files = await asyncio.to_thread(self._find_output_files, False)

            exception_result = ExecutionResult(
                success=False,
                output_files=[str(f) for f in output_files],
                execution_log=str(e),
                runtime_seconds=runtime,
                error_message=f"Execution exception: {str(e)}",
                error_messages=_merge_unique_error_messages(f"Execution exception: {str(e)}"),
            )
            await self._log_execution_round(
                iteration_index=iteration_index,
                iteration_dir=iteration_dir,
                script_name=script_name,
                script_contents=script_contents,
                result=exception_result,
            )
            return exception_result

    # Map of pychrono module file → (skill name, focus phrase) used to
    # tailor the SIGSEGV introspection_hint based on the topmost native
    # frame in the traceback. The default hint (vsg/sens/camera) was
    # actively misleading for the iter_005/006 SIGSEGV @ vehicle.py:
    # InitializeTire chain in session_20260429_112754, which had nothing
    # to do with rendering.
    _SIGNAL_HINT_BY_PYCHRONO_MODULE: Dict[str, Tuple[str, str]] = {
        "vehicle.py": (
            "veh/wheeled_vehicle",
            "wheel/spindle FSI registration sequence — check that the "
            "vehicle was constructed with the existing sysMBS (overload "
            "WheeledVehicle(sysMBS, json), not WheeledVehicle(json, "
            "ChContactMethod_SMC) which silently spawns an orphan "
            "ChSystemSMC), and that no body migration / ChFsiSystemSPH "
            "re-creation happened between vehicle.Initialize() and "
            "InitializeTire()",
        ),
        "fsi.py": (
            "fsi/sph",
            "FSI body registration order — every AddFsiBody() must precede "
            "sysFSI.Initialize(); CreatePointsBoxInterior / "
            "CreatePointsBoxContainer must happen on bodies that already "
            "live in sysMBS",
        ),
        "vsg3d.py": (
            "vsg",
            "VSG visualizer construction order — AttachSystem before "
            "Initialize; SetCameraVertical before Initialize; do not call "
            "AddCamera() in vsg_only mode",
        ),
        "sensor.py": (
            "sens/camera",
            "sensor camera setup — OptiX cannot render SPH particles "
            "(use VSG instead in any FSI scene); ChSensorManager must be "
            "added after every body has its visual shapes",
        ),
        "robot.py": (
            "robot/go2_quadruped",
            "robot URDF/policy load order — verify the URDF path resolves "
            "and the policy file is reachable",
        ),
    }

    @staticmethod
    def _signal_introspection_hint(error_type: str, stderr: str, code: str) -> str:
        """Build a SIGSEGV/SIGABRT hint tuned to the topmost pychrono frame.

        The default vsg/sens/camera lead is wrong whenever the crash is
        inside vehicle/fsi/robot module C++ code — see the iter_005/006
        regression in session_20260429_112754 where the agent wasted two
        iterations chasing a VSG hint while the real bug was an iter_004
        body-migration that corrupted vehicle internal state. Routing on
        the top pychrono frame eliminates that mis-direction.
        """
        if error_type != "SIGSEGV":
            return f"Native fault ({error_type}). Check recent C-API calls for bad types or null handles."

        top_module = ExecutionAgent._top_pychrono_module(stderr)
        if top_module is not None:
            focus = ExecutionAgent._SIGNAL_HINT_BY_PYCHRONO_MODULE.get(top_module)
            if focus is not None:
                skill_name, phrase = focus
                return (
                    f"Native crash with topmost pychrono frame in `{top_module}`. "
                    f"Likely cause: {phrase}. Call `read_skill('{skill_name}')` "
                    f"or read_skill_section to verify recent edits to that area; "
                    f"consider reverting the most recent non-trivial mutation "
                    f"to that subsystem before re-running."
                )

        # Fallback: only mention vsg/sens when the simulation actually
        # uses sensor cameras — a VSG-only FSI scene shouldn't get this
        # hint.
        code_lower = (code or "").lower()
        if "chsensormanager" in code_lower or "chcamerasensor" in code_lower:
            return (
                "This is a native crash and the code uses ChSensorManager / "
                "ChCameraSensor — call `read_skill('sens/camera')` and "
                "`read_skill('vsg')` to verify setup order, and reduce scene "
                "complexity before re-running."
            )
        return (
            "This is a native crash. No pychrono frame was identifiable in "
            "stderr; inspect the most recent non-trivial mutation to body / "
            "system / visualizer state and revert it as a first probe."
        )

    @staticmethod
    def _top_pychrono_module(stderr: str) -> Optional[str]:
        """Return the basename (e.g. 'vehicle.py') of the topmost pychrono
        frame in *stderr*, or None.

        For ``Fatal Python error: Segmentation fault`` traces, the most
        recent native frame is listed FIRST (Python's faulthandler
        convention) — we honour that ordering and return the first match.
        """
        if not stderr:
            return None
        for match in re.finditer(
            r'File "(?P<file>[^"]*pychrono/[\w/]+\.py)", line \d+',
            stderr,
        ):
            file_path = match.group("file")
            return file_path.rsplit("/", 1)[-1]
        return None

    @staticmethod
    def _structure_runtime_error(
        stderr: str,
        code: str,
        return_code: int | None,
    ) -> Optional[Dict[str, Any]]:
        """Parse stderr into structured fields so the codegen retry prompt can
        surface a targeted introspection hint instead of a blocky stderr dump.

        Returns a dict with keys:
            error_type: str  (AttributeError, TypeError, SIGSEGV, ...)
            failing_symbol: str  (e.g. "ChAABB.max_pnt")
            failing_line: int | None
            file_line_excerpt: str  (3 lines of source around the failing line)
            introspection_hint: str  (shell command the code agent should run)

        Returns None when nothing informative can be parsed.
        """
        stderr = stderr or ""
        code = code or ""

        # Signal-based crash path — short-circuit.
        signal_map = {-11: "SIGSEGV", -6: "SIGABRT", -8: "SIGFPE", -7: "SIGBUS"}
        if return_code in signal_map:
            error_type = signal_map[return_code]
            hint = ExecutionAgent._signal_introspection_hint(
                error_type, stderr or "", code
            )
            return {
                "error_type": error_type,
                "failing_symbol": "",
                "failing_line": None,
                "file_line_excerpt": "",
                "introspection_hint": hint,
            }

        if not stderr.strip():
            return None

        # Parse the last traceback frame + exception line.
        tb_pattern = re.compile(
            r'File "(?P<file>[^"]+)", line (?P<line>\d+)',
        )
        frames = list(tb_pattern.finditer(stderr))
        # Prefer frames in the user's simulation.py, fall back to the last frame.
        user_frame = None
        for fr in frames:
            if fr.group("file").endswith("simulation.py"):
                user_frame = fr
        if user_frame is None and frames:
            user_frame = frames[-1]

        failing_line: Optional[int] = None
        if user_frame:
            try:
                failing_line = int(user_frame.group("line"))
            except ValueError:
                failing_line = None

        # Last non-empty line usually has the exception message.
        lines = [ln for ln in stderr.splitlines() if ln.strip()]
        exception_line = lines[-1].strip() if lines else ""
        m_ex = re.match(r"(?P<etype>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning))\s*:\s*(?P<msg>.*)", exception_line)
        if m_ex:
            error_type = m_ex.group("etype")
            ex_msg = m_ex.group("msg")
        else:
            # Couldn't identify a Python exception — not much to structure.
            if return_code is None:
                return None
            return {
                "error_type": "UnknownError",
                "failing_symbol": "",
                "failing_line": failing_line,
                "file_line_excerpt": "",
                "introspection_hint": (
                    "No recognisable Python exception in stderr. Inspect the last "
                    "~20 lines of the execution log and verify the most recent "
                    "PyChrono calls with bash('python -c ...')."
                ),
            }

        # Infer a failing symbol and a tailored introspection hint.
        failing_symbol = ""
        introspection_hint = ""
        # AttributeError patterns:
        # - "'pychrono.core.ChAABB' object has no attribute 'max_pnt'"
        # - "module 'pychrono' has no attribute 'ChFoo'"
        m_attr = re.search(
            r"'(?:pychrono\.[\w.]+\.)?(?P<cls>Ch[A-Za-z0-9_]+)' object has no attribute '(?P<attr>[A-Za-z_][A-Za-z0-9_]*)'",
            ex_msg,
        )
        m_mod_attr = re.search(
            r"module 'pychrono(?:\.[\w.]+)?' has no attribute '(?P<attr>[A-Za-z_][A-Za-z0-9_]*)'",
            ex_msg,
        )
        m_type = re.search(
            r"unsupported operand type\(s\) for [^:]+:\s*'(?:pychrono\.[\w.]+\.)?(?P<cls>Ch[A-Za-z0-9_]+)'",
            ex_msg,
        )
        m_sig = re.search(
            r"(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\(\) (?:missing \d+ required|takes \d+ positional)",
            ex_msg,
        )
        m_import = re.match(r"No module named '(?P<mod>[\w.]+)'", ex_msg) or re.match(
            r"cannot import name '(?P<name>[A-Za-z_][A-Za-z0-9_]*)' from '(?P<mod>[\w.]+)'",
            ex_msg,
        )

        if m_attr:
            cls = m_attr.group("cls")
            attr = m_attr.group("attr")
            failing_symbol = f"{cls}.{attr}"
            introspection_hint = (
                f'bash(\'python -c "import pychrono as c; print([a for a in dir(c.{cls}) '
                f'if not a.startswith(chr(95))])"\')'
            )
        elif m_mod_attr:
            attr = m_mod_attr.group("attr")
            failing_symbol = f"pychrono.{attr}"
            introspection_hint = (
                f'bash(\'python -c "import pychrono as c; print([a for a in dir(c) '
                f'if {attr.lower()!r} in a.lower()])"\')'
            )
        elif m_type:
            cls = m_type.group("cls")
            failing_symbol = cls
            introspection_hint = (
                f'bash(\'python -c "import pychrono as c; help(c.{cls})"\')'
            )
        elif m_sig:
            fn = m_sig.group("fn")
            failing_symbol = f"{fn}()"
            introspection_hint = (
                f"bash('grep -n \"def {fn}\" chrono_agent/utils/*.py') "
                f"or read_skill('...') to check the current signature of {fn}()"
            )
        elif m_import:
            mod = m_import.group("mod")
            failing_symbol = mod
            introspection_hint = (
                f"bash('pip show {mod.split(chr(46))[0]}') then "
                f"bash('python -c \"import {mod.split(chr(46))[0]}; print({mod.split(chr(46))[0]}.__file__)\"')"
            )

        # Pull 3-line source excerpt around failing_line from the user code.
        file_line_excerpt = ""
        if failing_line is not None and code:
            src_lines = code.splitlines()
            if 1 <= failing_line <= len(src_lines):
                lo = max(0, failing_line - 2)
                hi = min(len(src_lines), failing_line + 1)
                numbered = [f"{i + 1}: {src_lines[i]}" for i in range(lo, hi)]
                file_line_excerpt = "\n".join(numbered)

        return {
            "error_type": error_type,
            "failing_symbol": failing_symbol,
            "failing_line": failing_line,
            "file_line_excerpt": file_line_excerpt,
            "introspection_hint": introspection_hint or (
                "No specific hint — inspect the last frame and the pychrono "
                "bindings with bash('python -c ...') before editing."
            ),
        }

    @staticmethod
    def _extract_error_block(combined_output: str, return_code: int, runtime: float) -> str:
        """Extract a meaningful error block from combined stdout+stderr.

        Scans for recognisable error patterns (C++ exceptions, assertion
        failures, Python-like error lines, etc.) and returns the relevant
        context.  Falls back to the last non-empty lines if no pattern
        matches.
        """
        import re

        lines = combined_output.splitlines()

        # --- Strategy 1: find lines that look like errors and include context ---
        error_line_pattern = re.compile(
            r"(error|exception|fault|abort|assert|failed|cannot|not found|"
            r"no such|invalid|undefined|killed|terminated)",
            re.IGNORECASE,
        )
        error_indices = [
            i for i, line in enumerate(lines)
            if error_line_pattern.search(line)
        ]

        if error_indices:
            # Take from 3 lines before the first error to 3 lines after the last
            start = max(0, error_indices[0] - 3)
            end = min(len(lines), error_indices[-1] + 4)
            block = "\n".join(lines[start:end])
        else:
            # --- Strategy 2: no recognisable pattern — take last non-empty lines ---
            non_empty = [l for l in lines if l.strip()]
            block = "\n".join(non_empty[-20:]) if non_empty else "(no output)"

        header = f"Process exited with code {return_code} in {runtime:.2f}s."
        return f"{header}\n{block}"

    def _recover_vsg_frames(self) -> None:
        """Encode any VSG frame dumps the subprocess left behind.

        Idempotent: if the simulation's in-process ``finalize()`` already
        encoded the mp4 (clean exit / catchable exception), this is a
        no-op. If the subprocess was SIGKILL'd on timeout (in-process
        ``finally:`` skipped), this turns the orphan ``cam/_*_frames/``
        directories into ``cam/*.mp4``.

        Runs on EVERY exit path (success, failure, timeout) before the
        output-file enumeration so the recovered mp4 lands in the
        returned ``output_files`` list.
        """
        try:
            from chrono_agent.utils.vsg_recording import encode_orphan_vsg_frames
        except Exception as exc:
            self.logger.warning(f"[VSG recovery] import failed: {exc}")
            return
        cam_dir = self.output_dir / CAMERA_OUTPUT_DIR
        try:
            encoded = encode_orphan_vsg_frames(cam_dir, fps=50.0)
        except Exception as exc:
            self.logger.warning(f"[VSG recovery] encode failed: {exc}")
            return
        for mp4_path, n in encoded:
            self.logger.info(
                f"[VSG recovery] encoded {n} frames → {mp4_path.name} "
                "(in-process finalize skipped — subprocess SIGKILL'd or crashed)"
            )

    def _find_output_files(self, include_images: bool = True) -> List[Path]:
        """Find output files generated by the simulation."""
        output_files = []

        # Look for common output file types in the main output directory.
        patterns = ["*.mp4", "*.avi", "*.mov", "*.dat", "*.csv"]
        if include_images:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"] + patterns

        for pattern in patterns:
            output_files.extend(self.output_dir.glob(pattern))

        # Also look for generated files in known subdirectories.
        for frame_dir_name in (CAMERA_OUTPUT_DIR,) + SIM_WINDOW_OUTPUT_DIRS:
            frame_dir = self.output_dir / frame_dir_name
            if not frame_dir.exists():
                continue
            if include_images:
                for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]:
                    output_files.extend(frame_dir.glob(pattern))
            for pattern in ["*.mp4", "*.avi", "*.mov", "*.dat", "*.csv"]:
                output_files.extend(frame_dir.glob(pattern))

        # Sort by modification time (newest first)
        output_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return output_files

    def _cleanup_image_outputs(self) -> int:
        """Remove all image files from output directory tree."""
        removed_count = 0
        for ext in IMAGE_EXTENSIONS:
            for image_file in self.output_dir.rglob(f"*.{ext}"):
                try:
                    image_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove image file {image_file}: {e}")
        return removed_count

    def _cleanup_empty_output_dirs(self) -> int:
        """
        Remove empty camera/window output directories to avoid leaving unnecessary folders.
        """
        removed_count = 0
        for dir_name in (CAMERA_OUTPUT_DIR,) + SIM_WINDOW_OUTPUT_DIRS:
            directory = self.output_dir / dir_name
            if not directory.exists() or not directory.is_dir():
                continue
            try:
                # Only remove if directory is empty.
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed_count += 1
            except Exception as e:
                self.logger.debug(f"Skip removing non-empty or locked directory {directory}: {e}")
        if removed_count:
            self.logger.info(f"Removed {removed_count} empty output directories")
        return removed_count

    def _find_core_file_sync(self, output_dir: Path) -> Optional[Path]:
        """Find newest core dump file in output_dir (sync, for use in to_thread)."""
        core_patterns = ['core', 'core.*', 'core-*']
        core_file = None
        for pattern in core_patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                core_file = max(matches, key=lambda f: f.stat().st_mtime)
                break
        if not core_file or not core_file.exists():
            return None
        return core_file

    async def _analyze_core_dump(self, output_dir: Path, python_executable: str) -> Optional[str]:
        """
        Find and analyze core dump file to get C/C++ backtrace.

        Args:
            output_dir: Directory where core dump may be located
            python_executable: Path to Python executable for GDB

        Returns:
            C/C++ backtrace string, or None if no core file or analysis failed
        """
        core_file = await asyncio.to_thread(self._find_core_file_sync, output_dir)
        if not core_file:
            self.logger.info("No core dump file found")
            return None

        self.logger.info(f"Found core dump: {core_file}")

        try:
            # Use GDB to analyze core file
            result = await asyncio.create_subprocess_exec(
                'gdb', '-q', '-batch',
                '-ex', 'set pagination off',
                '-ex', 'bt full',                    # Full backtrace
                '-ex', 'thread apply all bt full',   # All threads backtrace
                python_executable, str(core_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)

            gdb_output = stdout.decode('utf-8')

            if gdb_output:
                self.logger.info(f"GDB analysis complete: {len(gdb_output)} chars")
                return f"\n\n=== C/C++ Backtrace (from core dump) ===\n{gdb_output}"
            else:
                return None

        except FileNotFoundError:
            self.logger.debug("GDB not installed or no core file, will try live GDB run")
            return None
        except asyncio.TimeoutError:
            self.logger.warning("GDB core analysis timed out")
            return None
        except Exception as e:
            self.logger.warning(f"GDB core analysis failed: {e}")
            return None

    async def _get_backtrace_via_gdb_live(
        self,
        python_executable: str,
        script_path: Path,
        cwd: str,
        env: dict,
        timeout: int = 90,
    ) -> Optional[str]:
        """
        Re-run the script under GDB to capture C/C++ backtrace on crash (no core file needed).

        When the script segfaults, GDB catches it and "bt full" gives the native backtrace.
        """
        try:
            self.logger.info("Re-running under GDB to capture C/C++ backtrace (no core file)...")
            proc = await asyncio.create_subprocess_exec(
                "gdb",
                "-q",
                "-batch",
                "-ex", "set pagination off",
                "-ex", "run",
                "-ex", "bt full",
                "-ex", "thread apply all bt full",
                "-ex", "quit",
                "--args",
                python_executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            if out:
                self.logger.info("GDB live run captured backtrace")
                return f"\n\n=== C/C++ Backtrace (GDB live run) ===\n{out}"
            if "SIGSEGV" in err or "Segmentation" in err or "signal" in err.lower():
                return f"\n\n=== GDB stderr (no bt captured) ===\n{err}"
            return None
        except FileNotFoundError:
            self.logger.warning("GDB not installed; install gdb to get C/C++ backtraces")
            return None
        except asyncio.TimeoutError:
            self.logger.warning("GDB live run timed out (simulation may not have crashed again)")
            return None
        except Exception as e:
            self.logger.warning(f"GDB live run failed: {e}")
            return None

    async def _save_csv_to_results(
        self,
        csv_files: List[Dict[str, Any]],
        results_dir: Path,
        timestamp: str,
    ) -> List[str]:
        """Copy CSV files and plot images to the results directory for persistent storage.

        Args:
            csv_files: List of CSV file dicts (each has a 'file' key with the path).
            results_dir: Destination directory (e.g. /results/agent).
            timestamp: Timestamp prefix string for the copied filenames.

        Returns:
            List of destination file paths that were successfully copied.
        """
        # Each run gets its own subdirectory so same-named CSVs from different runs
        # (or different source directories in the same run) never overwrite each other.
        run_dir = results_dir / timestamp
        await ensure_dir(run_dir)
        saved = []
        for entry in csv_files:
            src = Path(entry.get("file", ""))
            if not src.exists():
                self.logger.debug(f"CSV source not found, skipping: {src}")
                continue
            dst = run_dir / src.name
            try:
                await asyncio.to_thread(shutil.copy2, src, dst)
                saved.append(str(dst))
                self.logger.info(f"Saved CSV to results: {timestamp}/{src.name}")
            except Exception as e:
                self.logger.warning(f"Failed to copy CSV {src} to results: {e}")

        # Also copy any PNG plot files (e.g. simulation_timeseries.png) from the output dir
        for png_file in self.output_dir.glob("*.png"):
            dst = run_dir / png_file.name
            try:
                await asyncio.to_thread(shutil.copy2, png_file, dst)
                saved.append(str(dst))
                self.logger.info(f"Saved plot to results: {timestamp}/{png_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to copy plot {png_file} to results: {e}")

        return saved

    @staticmethod
    def _summarize_csv(
        path: Path,
        *,
        head_rows: int = 20,
        tail_rows: int = 20,
        sample_rows: int = 40,
        max_chars: int = 3500,
    ) -> str:
        """Produce a physics-diagnostic summary of one CSV.

        Dense head-only reads (the old 100-row head) bias the LLM's
        physics verdict toward early-time state — for a 30 s @ dt=0.001
        sim, 100 rows covers only the first 0.1 s. Driver ramps and
        settling transients dominate that window, which produces false
        "body stuck" verdicts and hides late-time drift / blowup /
        freeze failures entirely.

        The summary returned here has three parts:

        1. **Per-column statistics** over the FULL time range — min,
           max, mean, std, NaN count, and a "stuck" flag when
           max-min < 1e-9. This is where catastrophic physics failures
           show up most reliably: a stuck position column, NaN in any
           column, an energy column that grew 10x, a height column that
           went to -1e6.
        2. **Head** (first ``head_rows``) — captures spawn pose,
           initial conditions, and whether the sim started sane.
        3. **Evenly-spaced middle samples** (``sample_rows``) +
           **tail** (last ``tail_rows``) — captures steady state,
           late-time drift, and final pose.

        Budget: the whole block is clipped to ``max_chars`` so three
        CSVs at 3.5 KB each fit comfortably under the 12-15 KB prompt
        budget the upstream caller allows.
        """
        try:
            import pandas as pd
        except ImportError:
            pd = None

        # Fall back to a raw head read if pandas is unavailable for any
        # reason — better to send something than nothing. Matches the
        # old behavior but with a bigger window.
        if pd is None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.rstrip() for _, ln in zip(range(head_rows + sample_rows + tail_rows), f)]
            return "\n".join(lines)[:max_chars]

        try:
            df = pd.read_csv(path)
        except Exception:
            # Malformed CSV — raw-head fallback.
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.rstrip() for _, ln in zip(range(head_rows + sample_rows + tail_rows), f)]
            return "\n".join(lines)[:max_chars]

        n = len(df)
        if n == 0:
            return f"(CSV has {len(df.columns)} columns but 0 rows)"

        # --- per-column stats over the full range ---
        numeric = df.select_dtypes(include="number")
        stats_lines = [f"rows={n}, cols={len(df.columns)}, numeric={len(numeric.columns)}"]
        if not numeric.empty:
            desc = numeric.describe().T  # min, max, mean, std
            for col in numeric.columns:
                row = desc.loc[col]
                mn, mx = float(row["min"]), float(row["max"])
                mean, std = float(row["mean"]), float(row["std"]) if not pd.isna(row["std"]) else 0.0
                nan_ct = int(df[col].isna().sum())
                import math
                has_inf = bool(numeric[col].apply(lambda v: isinstance(v, float) and math.isinf(v)).any())
                flags = []
                if mx - mn < 1e-9:
                    flags.append("STUCK")
                if nan_ct > 0:
                    flags.append(f"NaN×{nan_ct}")
                if has_inf:
                    flags.append("INF")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                stats_lines.append(
                    f"  {col}: min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g}{flag_str}"
                )
        stats_block = "\n".join(stats_lines)

        # --- sampled rows: head + evenly-spaced middle + tail ---
        # Cap each slice against available rows so small CSVs don't crash.
        head_n = min(head_rows, n)
        tail_n = min(tail_rows, max(0, n - head_n))
        middle_start = head_n
        middle_end = n - tail_n
        middle_available = max(0, middle_end - middle_start)
        sample_n = min(sample_rows, middle_available)

        if sample_n > 0 and middle_available > 0:
            # Evenly-spaced indices across the middle segment.
            import numpy as np
            mid_idx = np.linspace(
                middle_start, middle_end - 1, num=sample_n, dtype=int
            )
        else:
            mid_idx = []

        parts = []
        if head_n > 0:
            parts.append(f"-- head (rows 0..{head_n - 1}) --")
            parts.append(df.head(head_n).to_csv(index=False))
        if len(mid_idx) > 0:
            parts.append(f"-- sampled middle ({len(mid_idx)} rows, evenly spaced {middle_start}..{middle_end - 1}) --")
            # Keep header for readability since the mid slice may not be contiguous.
            parts.append(df.iloc[mid_idx].to_csv(index=False))
        if tail_n > 0:
            parts.append(f"-- tail (rows {n - tail_n}..{n - 1}) --")
            parts.append(df.tail(tail_n).to_csv(index=False))

        body = "\n".join(parts)

        summary = (
            f"## stats (whole file)\n{stats_block}\n\n"
            f"## sampled rows\n{body}"
        )
        if len(summary) > max_chars:
            # Truncate from the body; keep stats intact since they
            # carry more diagnostic signal per character.
            keep = max_chars - len(stats_block) - 32
            summary = (
                f"## stats (whole file)\n{stats_block}\n\n"
                f"## sampled rows (truncated)\n{body[:max(0, keep)]}"
            )
        return summary

    async def analyze_physics_with_llm(
        self,
        plan: SimulationPlan,
        execution_result: ExecutionResult,
        results_dir: Path,
        video_description: str | None = None,
    ) -> Dict[str, Any]:
        """Call the LLM to validate simulation CSV output against the plan's physics laws."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = await self._save_csv_to_results(
            execution_result.csv_files, results_dir, timestamp
        )

        csv_paths = [p for p in saved_paths if Path(p).suffix.lower() == ".csv"]

        if not csv_paths:
            self.logger.info("No CSV output found; returning physics_uncertain")
            return {
                "verdict": "physics_uncertain",
                "reasoning": "No CSV output was produced. Cannot validate physics.",
                "violations": [],
                "suggested_fix": None,
                "saved_csv_paths": saved_paths,
            }

        # Stats-first summary per CSV. Old approach (first 100 rows) was
        # time-biased: for a 30 s @ dt=0.001 sim it covered the first
        # 0.1 s only, missing late-time drift/freeze and over-weighting
        # driver-input ramp transients. _summarize_csv returns whole-file
        # column stats + head + evenly-spaced middle samples + tail.
        csv_content_parts = []
        max_total_chars = 12000
        per_csv_budget = max(2000, max_total_chars // max(1, len(csv_paths)))
        total_chars = 0
        for p in csv_paths:
            try:
                text = await asyncio.to_thread(
                    self._summarize_csv, Path(p), max_chars=per_csv_budget
                )
                if text and total_chars < max_total_chars:
                    csv_content_parts.append(f"### {Path(p).name}\n{text}")
                    total_chars += len(text)
            except Exception as e:
                self.logger.warning(f"Failed to summarize CSV {p}: {e}")
        csv_content = "\n\n".join(csv_content_parts) if csv_content_parts else "(no CSV content)"

        objectives = "\n".join(getattr(plan, "objectives", []) or [])[:500]
        simulation_parameters = str(getattr(plan, "simulation_parameters", {}) or {})[:1000]

        prompt = execution_prompts.PHYSICS_ANALYSIS_PROMPT.substitute(
            objectives=objectives,
            simulation_parameters=simulation_parameters,
            csv_content=csv_content,
            video_description=(video_description or "No visual description available from ReviewAgent."),
        )

        valid_verdicts = {"physics_valid", "physics_invalid", "physics_uncertain"}
        result = await self.invoke_llm(
            prompt=prompt,
            parse_json=True,
            temperature=0.0,
        )
        if not isinstance(result, dict) or result.get("verdict") not in valid_verdicts:
            raise AgentLLMError(
                agent_name=self.agent_name,
                operation="physics_analysis",
                message=f"Unexpected physics analysis result: {result!r}",
            )

        result["saved_csv_paths"] = saved_paths
        self.logger.info(f"Physics analysis verdict: {result.get('verdict')}")
        return result
