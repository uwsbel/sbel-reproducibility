"""
Main CLI entry point for Chrono-Code.
"""

# Must be set before any imports that might load tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree

from chrono_code.config import get_settings, init_directories_sync
from chrono_code.agents.exceptions import (
    PlanModificationIncompleteError,
    PlanModificationValidationError,
)
from chrono_code.models.plan import SimulationPlan
from chrono_code.utils.logger import setup_logger
from chrono_code.utils.terminal_ui import AgentOutputManager, create_output_manager
from chrono_code.utils.textual_tui import is_textual_available, run_tui

# Create Typer app
app = typer.Typer(
    name="chrono-code",
    help="Multi-agent PyChrono simulation code generator",
    add_completion=False,
)

console = Console()
PATCH_PREVIEW_MAX_LINES = 260
PATCH_PREVIEW_MAX_CHARS = 18000

# Active live UI surface during run_workflow. ``_active_renderer`` is the
# ``LiveEventRenderer`` instance when the new event-driven UI is running;
# ``_active_progress`` is the legacy Rich Progress (kept for code paths that
# still use the old spinner). Interactive callbacks (plan approval, etc.)
# use ``_ProgressPauseContext`` to temporarily quiet whichever surface is
# active so their prompts aren't clobbered by the refresh loop.
_active_progress = None
_active_renderer = None


class _ProgressPauseContext:
    """Pause whichever live UI surface is active around blocking user input.

    Prefers the new ``LiveEventRenderer`` if set; otherwise falls back to
    the legacy Progress spinner.
    """

    def __enter__(self):
        renderer = _active_renderer
        if renderer is not None:
            try:
                renderer.pause()
            except Exception:
                pass
            return self
        prog = _active_progress
        if prog is not None:
            try:
                prog.stop()
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        renderer = _active_renderer
        if renderer is not None:
            try:
                renderer.resume()
            except Exception:
                pass
            return False
        prog = _active_progress
        if prog is not None:
            try:
                prog.start()
            except Exception:
                pass
        return False

_APPROVE_WORDS = {"ok", "approve", "approved", "yes", "y", "looks good", "lgtm",
                  "proceed", "go", "continue", "good", "\u786e\u8ba4", "\u901a\u8fc7", "\u597d"}
_REJECT_WORDS = {"cancel", "reject", "stop", "quit", "exit", "abort", "\u62d2\u7edd", "\u53d6\u6d88"}


def _extract_images_from_markdown(markdown_content: str, base_dir: Path) -> list[str]:
    """
    Extract image paths from markdown content.

    Parses markdown for ![...](path) patterns and returns absolute paths
    for any local image files (not URLs).

    Args:
        markdown_content: The markdown text to parse
        base_dir: Base directory for resolving relative image paths

    Returns:
        List of absolute image paths found in markdown
    """
    # Match ![alt text](image_path) or ![](image_path)
    pattern = r'!\[.*?\]\(([^)]+)\)'
    matches = re.findall(pattern, markdown_content)

    image_paths = []
    for match in matches:
        # Skip URLs
        if match.startswith(('http://', 'https://', 'data:')):
            continue

        # Resolve relative path against base_dir
        img_path = base_dir / match
        if img_path.exists():
            image_paths.append(str(img_path.resolve()))
        else:
            console.print(f"[yellow]Warning: Image not found: {img_path}[/yellow]")

    return image_paths


def _reset_cli_history(settings) -> None:
    """Clear all history artifacts at the start of a new CLI run."""
    history_dir = settings.history_output_path()
    if history_dir.exists():
        shutil.rmtree(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)


def _activate_cli_runtime_run(settings, run_id: str) -> Path:
    """Point the active visualization/runtime directory at the history root.

    Each execution iteration will create its own subdirectory
    (iteration_001/, iteration_002/, ...) under this path, keeping code and
    outputs self-contained per iteration.
    """
    history_dir = settings.history_output_path().resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings, "visualization_output_path", str(history_dir))
    return history_dir


def _should_display_results(result: dict) -> bool:
    if result.get("execution_complete"):
        return True
    if result.get("final_output"):
        return True
    if result.get("error"):
        return True
    return False


def check_environment():
    """Verify we're running in the correct conda environment with required packages."""
    import sys
    import subprocess

    # Check pychrono in a subprocess to avoid import-order / in-process conflicts.
    result = subprocess.run(
        [sys.executable, "-c", "import pychrono; print(getattr(pychrono, '__version__', 'ok'))"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        console.print("[red]ERROR: pychrono is not available in the current environment![/red]")
        console.print("[yellow]Please activate the chrono-code environment:[/yellow]")
        console.print("  conda activate chrono-code")
        if result.stderr:
            console.print("[dim]Details:[/dim]")
            for line in result.stderr.strip().split("\n")[:5]:
                console.print(f"  [dim]{line}[/dim]")
        console.print()
        return False

    # Optional: warn if conda env name is not chrono-code
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    if conda_env != "chrono-code":
        console.print(
            f"[yellow]WARNING: Running in '{conda_env}' environment, not 'chrono-code'[/yellow]"
        )
        console.print("[yellow]For best results, activate the correct environment:[/yellow]")
        console.print("  conda activate chrono-code")
        console.print()

    return True


def _setup_cli_logging_and_settings(verbose: bool):
    """Initialize logging and settings for CLI commands."""
    log_level = "DEBUG" if verbose else "INFO"

    try:
        settings = get_settings()
        log_file = settings.log_file
    except Exception:
        log_file = "./chrono_code.log"

    setup_logger(level=log_level, log_file=log_file, use_rich=True)

    try:
        settings = get_settings()
        init_directories_sync(settings)
        return settings
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        console.print("[yellow]Please check your .env file and API keys[/yellow]")
        raise typer.Exit(1)


def _show_cli_header() -> None:
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Chrono-Code[/bold cyan]\n"
        "Multi-Agent PyChrono Simulation Generator",
        border_style="cyan"
    ))
    console.print()


def _display_codegen_only_results(
    generated_code,
    updated_state: dict,
    plan_file: Path,
    detail_level: str = "normal",
) -> None:
    """Display codegen-only results without re-saving to history root."""
    console.print()
    console.print("[bold green]>>> Code Generation Complete[/bold green]")
    console.print()

    iteration_dir = Path(updated_state.get("iteration_dir", "")).resolve() if updated_state.get("iteration_dir") else None
    code_file = iteration_dir / "simulation.py" if iteration_dir else None

    summary_lines = [
        f"Plan file: [cyan]{plan_file}[/cyan]",
    ]
    if code_file:
        summary_lines.append(f"Iteration dir: [cyan]{iteration_dir}[/cyan]")
        summary_lines.append(f"Generated code: [cyan]{code_file}[/cyan]")
    summary_lines.append(
        f"Code length: {len(generated_code.code)} characters, {len(generated_code.code.splitlines())} lines"
    )

    console.print(Panel(
        "\n".join(summary_lines),
        title="Codegen Output",
        border_style="green",
    ))
    console.print()

    if detail_level == "verbose":
        syntax = Syntax(generated_code.code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Code Content", border_style="dim"))
        console.print()

    patch_status = getattr(generated_code, "patch_apply_status", None)
    if patch_status and patch_status != "not_attempted":
        console.print(f"[dim]Patch status:[/dim] {patch_status}")
    if updated_state.get("structured_error"):
        console.print(f"[yellow]Structured error:[/yellow] {updated_state['structured_error']}")
        console.print()


@app.command(name="codegen-from-plan")
def codegen_from_plan(
    plan_file: Path = typer.Option(
        ...,
        "--plan-file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a plan.json file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    detail_level: str = typer.Option(
        "normal",
        "--detail-level",
        "-d",
        help="Output detail level (minimal/normal/verbose)",
    ),
):
    """
    Skip planning and start directly from an existing plan.json with CodeGenerationAgent.
    """
    if not check_environment():
        raise typer.Exit(1)

    settings = _setup_cli_logging_and_settings(verbose)
    _show_cli_header()

    console.print(f"[dim]Mode:[/dim] codegen-from-plan")
    console.print(f"[dim]Plan file:[/dim] {plan_file}")
    console.print(f"[dim]Detail Level:[/dim] {detail_level}")
    console.print()

    try:
        raw_plan = json.loads(plan_file.read_text(encoding="utf-8"))
        plan = SimulationPlan.model_validate(raw_plan)
    except Exception as e:
        console.print(f"[red]Error parsing plan file: {e}[/red]")
        raise typer.Exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    import uuid
    thread_id = str(uuid.uuid4())
    _reset_cli_history(settings)
    runtime_output_dir = _activate_cli_runtime_run(settings, thread_id)

    try:
        from chrono_code.agents import CodeGenerationAgent

        agent = CodeGenerationAgent()

        # If the plan has implementation_steps (scene / mbs_in_scene),
        # synthesize a minimal step_loop so the code agent enters step
        # mode for the FIRST step. Pure-mbs plans (no steps) fall through
        # to the legacy one-shot codegen path.
        injected_state: Optional[dict] = None
        if plan.implementation_steps:
            first_step = plan.implementation_steps[0]
            first_ctx = plan.build_step_context(0, completed_step_descriptions=[])
            injected_state = {
                "plan": plan.dump_all(),
                "step_loop": {
                    "steps": [s.model_dump() for s in plan.implementation_steps],
                    "current_step_index": 0,
                    "current_step_description": first_step.description,
                    "step_retry_count": 0,
                    "max_step_retries": 6,
                    "step_no_progress_retry_count": 0,
                    "max_step_no_progress_retries": 2,
                    "step_last_failed_code_sha": None,
                    "completed_steps": [],
                    "step_feedback": None,
                    "all_steps_complete": False,
                    "relevant_bodies": [
                        a.get("name", "").lower()
                        for a in first_ctx.step_assets
                        if a.get("name")
                    ] + [
                        obj.get("name", "").lower()
                        for obj in first_ctx.step_scene_objects
                        if obj.get("name")
                    ],
                    "step_context": first_ctx.model_dump(),
                    "review_issues": [],
                },
                "build": {},
            }
            console.print(
                f"[dim]Step mode: running codegen for step 1/"
                f"{len(plan.implementation_steps)}: "
                f"{first_step.description[:80]}[/dim]"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating code from plan...", total=None)
            generated_code, updated_state = loop.run_until_complete(
                agent.execute(
                    plan=plan,
                    compilation_feedback=None,
                    previous_code=None,
                    state=injected_state,
                    fix_mode=False,
                )
            )

        iteration_dir = Path(updated_state["iteration_dir"]).resolve()
        iteration_dir.mkdir(parents=True, exist_ok=True)
        code_file = iteration_dir / "simulation.py"
        if not code_file.exists():
            code_file.write_text(generated_code.code, encoding="utf-8")

        _display_codegen_only_results(
            generated_code=generated_code,
            updated_state=updated_state,
            plan_file=plan_file,
            detail_level=detail_level,
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {rich_escape(str(e))}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


@app.command(name="run-from-plan")
def run_from_plan(
    plan_file: Path = typer.Option(
        ...,
        "--plan-file",
        "-f",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a plan.json file. scene / mbs_in_scene plans run the "
             "step loop (one codegen+review pass per implementation_step); "
             "pure mbs plans run monolithically.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    detail_level: str = typer.Option(
        "normal", "--detail-level", "-d",
        help="Output detail level (minimal/normal/verbose)",
    ),
    timeout: int = typer.Option(
        30, "--timeout", "-t",
        help="Execution timeout (seconds) per simulation run",
    ),
):
    """
    Run the full multi-agent workflow starting from an existing plan.json,
    skipping the planning agent entirely.

    This is the fixture-driven end-to-end entry point: useful for testing
    the step path (step_router → codegen → execution → review →
    next step / graceful_give_up) without spending tokens on planning,
    and for reproducibility when the plan is known good.
    """
    if not check_environment():
        raise typer.Exit(1)

    settings = _setup_cli_logging_and_settings(verbose)
    _show_cli_header()

    console.print(f"[dim]Mode:[/dim] run-from-plan (skips planning)")
    console.print(f"[dim]Plan file:[/dim] {plan_file}")
    console.print(f"[dim]Detail Level:[/dim] {detail_level}")
    console.print()

    global _PIPELINE_STATS_RENDERED
    _PIPELINE_STATS_RENDERED = False

    try:
        raw_plan = json.loads(plan_file.read_text(encoding="utf-8"))
        plan = SimulationPlan.model_validate(raw_plan)
    except Exception as e:
        console.print(f"[red]Error parsing plan file: {e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[dim]Plan type:[/dim] {plan.plan_type}  "
        f"[dim]Steps:[/dim] {len(plan.implementation_steps)}"
    )
    if plan.implementation_steps:
        console.print(
            "[dim]Step path:[/dim] "
            + " → ".join(
                f"step{i + 1}({', '.join(s.assets) or '—'})"
                for i, s in enumerate(plan.implementation_steps)
            )
        )
    console.print()

    # Show the plan once up front (same display used by approval flow)
    display_plan_for_approval(plan.dump_all())
    console.print()
    console.print("[dim]Plan loaded from file — skipping approval prompt.[/dim]")
    console.print()

    # Apply execution timeout override (matches `generate` CLI)
    if timeout and timeout > 0:
        settings.execution_timeout = timeout
        console.print(f"[dim]Execution timeout:[/dim] {timeout}s\n")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    import uuid
    thread_id = str(uuid.uuid4())
    _reset_cli_history(settings)
    runtime_output_dir = _activate_cli_runtime_run(settings, thread_id)
    output_dir = str(runtime_output_dir)

    try:
        from chrono_code.workflow.engine import run_workflow

        def run_with_spinner(description: str, coroutine):
            """Run the workflow coroutine with the event-driven live renderer.

            The renderer consumes workflow events (agent lifecycle, tool calls,
            streaming text, etc.) and renders Claude-Code-style per-agent
            blocks inline while keeping a sticky bottom status line.
            """
            global _active_renderer
            from chrono_code.ui import LiveEventRenderer
            with LiveEventRenderer(console) as renderer:
                _active_renderer = renderer
                try:
                    return loop.run_until_complete(coroutine)
                finally:
                    _active_renderer = None

        result = run_with_spinner(
            "Running workflow from preloaded plan...",
            run_workflow(
                user_prompt=f"(preloaded plan from {plan_file.name})",
                plan_mode="auto",
                images=None,
                on_plan_approval=None,  # bypassed because plan_approved=True
                preloaded_plan=plan.dump_all(),
            ),
        )

        def _save_transcript():
            from chrono_code.agents.base import BaseAgent
            dm = getattr(BaseAgent, "_shared_dialog_manager", None)
            if dm and dm.current_session and result.get("messages"):
                dm.save_session_transcript(result["messages"])

        _save_transcript()

        if _should_display_results(result):
            display_results(result, output_dir, detail_level=detail_level)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {rich_escape(str(e))}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    finally:
        # Force-show token / time totals regardless of success or failure.
        # Idempotent — if display_results already rendered the panel, this
        # is a no-op.
        try:
            _render_pipeline_stats_panel(locals().get("result") or {})
        except Exception:
            pass
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


def _ask_critical_clarifications(user_prompt: str) -> str:
    """Pre-planning Q&A gate.

    Before invoking the workflow, deterministically check whether the user
    named the critical simulation parameters (``time_step``, ``simulation_duration``,
    ``scene_size``). For each one that's missing, ask the user once and append
    the answer to ``user_prompt`` in a format that ``extract_user_spec_regex``
    parses cleanly.

    The point: by the time PlanningAgent runs, ``user_spec`` is already complete,
    so the planner emits a final plan with no ``clarifications_needed`` and no
    placeholder values. The post-plan approval loop only handles real user
    modifications, not unanswered questions.

    Press Enter on a question = defer to planner default for that one.
    Type 'skip' = dismiss the gate entirely (planner picks defaults).
    Ctrl-C = abort the run.
    """
    from chrono_code.models.user_spec import extract_user_spec_regex
    from chrono_code.agents.planning_agent import PlanningAgent

    user_spec = extract_user_spec_regex(user_prompt)
    # (key for regex-friendly augmentation, human question to ask)
    questions: list[tuple[str, str]] = []
    if user_spec.time_step_s is None:
        questions.append(("time_step", "What simulation time step do you want? (e.g., 0.01s)"))
    if user_spec.duration_s is None:
        questions.append(("simulation_duration", "What total simulation duration do you want? (e.g., 5s)"))
    if user_spec.scene_size_m is None and PlanningAgent._looks_like_scene(user_prompt):
        questions.append(("scene_size", "What scene size do you want? (e.g., 10x10m)"))

    if not questions:
        return user_prompt

    console.print()
    console.print(
        f"[yellow]The planner needs {len(questions)} simulation parameter(s) "
        f"before it can build the plan. Please answer each one.[/yellow]"
    )
    console.print(
        "[dim]Press Enter on any question to defer to the planner's default. "
        "Type 'skip' to dismiss the gate. Ctrl-C cancels the run.[/dim]"
    )
    console.print()

    answers: list[tuple[str, str]] = []
    for i, (key, q) in enumerate(questions, 1):
        console.print(f"[yellow]Q{i}/{len(questions)}:[/yellow] {q}")
        try:
            a = typer.prompt(f"  A{i}", default="").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]Cancelled[/red]")
            raise typer.Exit(1)
        if a.lower() == "skip":
            break
        if a:
            answers.append((key, a))

    if not answers:
        console.print("[dim]No answers provided — planner will pick defaults and may add clarifications_needed entries to the plan.[/dim]")
        return user_prompt

    # Verified-extractable formats: ``time_step: 0.01``,
    # ``simulation_duration: 5.0``, ``scene_size: 10x10m`` all parse via
    # ``extract_user_spec_regex``. Appending under an explicit header keeps the
    # original prompt intact for any downstream code that displays it back.
    augmentation = "\n\nUser-clarified parameters:\n" + "\n".join(
        f"- {key}: {a}" for key, a in answers
    )
    console.print(f"[dim]Augmenting prompt with {len(answers)} clarification answer(s).[/dim]")
    return user_prompt + augmentation


def _coerce_clarification_value(raw: str, value_type: str) -> Any:
    """Coerce the user's typed answer to the type expected by target_field.

    ``vec3`` extracts the first three numeric tokens (handles forms like
    ``5m x 2m x 0.3m``, ``[5, 2, 0.3]``, ``5 2 0.3``). ``float`` takes the
    first numeric token. ``str`` / ``relation_pattern`` / unknown types pass
    through unchanged.
    Returns ``None`` when the requested numeric form cannot be parsed —
    the caller treats that as ``skipped`` so the agent must surface the
    field as a plain-text clarification rather than a fabricated number.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if value_type in ("str", "relation_pattern"):
        return text
    nums = re.findall(r"-?\d+\.?\d*", text)
    if value_type == "vec3":
        if len(nums) < 3:
            return None
        return [float(nums[0]), float(nums[1]), float(nums[2])]
    if value_type == "float":
        if not nums:
            return None
        return float(nums[0])
    return text


async def _cli_batch_clarification(items) -> Dict[str, Any]:
    """Phase 4 batch UI. ``items`` is List[StructuredClarification].

    For each item, render a Rich panel and ask. Choice → letter picker,
    number → free-form numeric input. Returns {target_field: answer}.
    Skipped items are simply absent from the result map.
    """
    import asyncio

    answers: Dict[str, Any] = {}
    if not items:
        return answers

    with _ProgressPauseContext():
        console.print()
        console.print(
            f"[dim]Plan agent has {len(items)} clarification(s) for you. "
            "Press Enter to skip any one.[/dim]"
        )

    for idx, item in enumerate(items, 1):
        title = f"[{idx}/{len(items)}] {item.target_field or 'freeform'}"
        lines = [f"[bold]{rich_escape(item.question)}[/bold]", ""]

        if item.kind == "choice":
            letters: List[str] = []
            for i, label in enumerate(item.options or []):
                letter = chr(ord("A") + i)
                letters.append(letter)
                # Pull description / pattern from option_details when present.
                detail = (item.option_details or [None] * len(item.options))[i] if item.option_details else None
                desc = f" — {rich_escape(detail.description)}" if detail and detail.description else ""
                lines.append(f"  [cyan]{letter}[/cyan]  [bold]{rich_escape(label)}[/bold]{desc}")
            panel = Panel("\n".join(lines), title=f"[yellow]{rich_escape(title)}[/yellow]",
                          border_style="yellow", padding=(1, 2))
            with _ProgressPauseContext():
                console.print()
                console.print(panel)
                try:
                    user_input = await asyncio.to_thread(
                        lambda: typer.prompt(">", default="").strip()
                    )
                except (EOFError, KeyboardInterrupt):
                    continue
            head = (user_input or "")[:1].upper()
            if head in letters:
                answers[item.target_field] = item.options[letters.index(head)]

        else:  # number
            unit = item.unit or "1"
            lines.append(f"  [dim](enter a number; unit: {rich_escape(unit)})[/dim]")
            panel = Panel("\n".join(lines), title=f"[yellow]{rich_escape(title)}[/yellow]",
                          border_style="yellow", padding=(1, 2))
            with _ProgressPauseContext():
                console.print()
                console.print(panel)
                try:
                    user_input = await asyncio.to_thread(
                        lambda: typer.prompt(f"value [{unit}]", default="").strip()
                    )
                except (EOFError, KeyboardInterrupt):
                    continue
            try:
                answers[item.target_field] = float(user_input)
            except (TypeError, ValueError):
                pass  # skipped / invalid → leave out of map

    return answers


async def _cli_plan_approval(state: dict) -> dict:
    """Interactive plan approval for CLI mode.

    This callback is invoked by the workflow engine when the plan is ready
    for user review.  It displays the plan, prompts the user, and returns
    the updated state with the approval decision.

    The active workflow ``Progress`` spinner is paused around every blocking
    user-input call so it doesn't overwrite the prompt or swallow keystrokes.
    """
    plan = state.get("plan", {})

    # Pipeline ordering: when the planner emits open clarifications, do NOT
    # dump the full plan yet. The plan summary implies "this is final, please
    # approve"; showing it next to unresolved questions confuses the user
    # into approving early. Render only the clarifications panel first, and
    # ask the user to answer (or to type "show plan" to see the draft anyway).
    def _has_open_clarifications(p: dict) -> bool:
        """True iff at least one entry is an unanswered structured question.

        After ``modify_plan`` resolves a question the planner often keeps a
        plain-string note like "Camera placement resolved: y=-13.0" inside
        ``clarifications_needed`` — those are post-hoc breadcrumbs, not
        open questions. Counting them as "open" forces the UI into the
        questions-only panel and hides the full plan, so the user can
        never see the modified plan before pressing Enter to approve.
        Only count entries that still carry a multi-option structure.
        """
        entries = (p or {}).get("clarifications_needed") or []
        return any(_is_structured_clarification(e) for e in entries)

    def _count_inline_tokens(p: dict) -> int:
        """Count `<<ASK_*>>` tokens still embedded in ``plan_markdown``.

        New pipeline carries unanswered questions inline rather than in
        the ``clarifications_needed`` list; this helper exposes them to
        the approval UI so the user gets a clear warning when a plan
        with unresolved tokens reaches the approve step (which would
        otherwise let codegen see literal token strings).
        """
        from chrono_code.models.plan_markdown_parser import TOKEN_RE
        md = (p or {}).get("plan_markdown") or ""
        return sum(1 for _ in TOKEN_RE.finditer(md))

    def _render_unresolved_token_warning(p: dict) -> None:
        n = _count_inline_tokens(p)
        if n <= 0:
            return
        warning = Panel(
            f"[bold red]⚠️  {n} unresolved <<ASK_*>> token(s) remain in this plan[/bold red]\n"
            f"[red]Pressing Enter will approve as-is.[/red] codegen will see literal\n"
            f"[red]\"<<ASK_*>>\" strings where it expects values, which usually breaks the run.\n"
            f"Type 'modify' to revise / 'reject' to cancel / Enter to accept anyway.[/red]",
            title="[bold red]UNRESOLVED CLARIFICATIONS[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        console.print(warning)

    def _prompt_choice_hint(p: dict) -> str:
        """Build a Claude-Code-style choice hint for the bottom of the panel.

        For a single open structured clarification renders an explicit
        ``Choose [A/B/C/D]`` inline so the user sees what tokens are
        accepted without scrolling back to the panel — mirrors CC plan
        mode's "Type 1/2/3" affordance.
        """
        entries = [e for e in (p or {}).get("clarifications_needed") or []
                   if _is_structured_clarification(e)]
        if len(entries) != 1:
            return ""
        opts = entries[0].get("options") or []
        allow_other = entries[0].get("allow_other", True)
        letters = [_OPTION_LETTERS[i] for i in range(len(opts))
                   if i < len(_OPTION_LETTERS)]
        if allow_other and len(opts) < len(_OPTION_LETTERS):
            letters.append(_OPTION_LETTERS[len(opts)])
        if not letters:
            return ""
        return f"Choose [bold cyan]{'/'.join(letters)}[/bold cyan] · "

    with _ProgressPauseContext():
        console.print()
        if _has_open_clarifications(plan):
            display_plan_for_approval(plan, plan_panel=False)
            console.print()
            choice_hint = _prompt_choice_hint(plan)
            console.print(
                f"[dim]{choice_hint}or describe a modification · "
                "'show plan' to view the draft · "
                "'approve' to accept as-is · 'reject' to cancel · /help[/dim]"
            )
        else:
            display_plan_for_approval(plan)
            console.print()
            _render_unresolved_token_warning(plan)
            console.print("[dim]Enter to approve | type modifications | 'reject' to cancel | /help for commands[/dim]")

    # Conversational modification loop
    conversation_history: list[dict[str, str]] = []
    _plan_agent = None  # lazy-initialised, reused across rounds
    images = state.get("images")
    # ``pre_modify_plan`` is the snapshot taken right before each successful
    # modification; ``/diff`` uses it to show what last changed. Initialised
    # to the initial plan so an early ``/diff`` shows "no changes yet" rather
    # than crashing.
    pre_modify_plan: dict = dict(plan)

    while True:
        # When the plan has exactly one open structured clarification we
        # render an inline arrow-key picker (CC-style). The picker
        # returns shorthand like ``"1A"`` directly, which the existing
        # expander downstream consumes verbatim. Esc / non-TTY / picker
        # failure falls through to the legacy ``typer.prompt`` so the
        # user can type a free-form modification.
        picker_input: Optional[str] = None
        try:
            picker_input = _maybe_pick_clarification(plan)
        except KeyboardInterrupt:
            console.print("\n[red]Cancelled[/red]")
            planning = dict(state.get("planning", {}))
            planning["plan_rejected"] = True
            state["planning"] = planning
            return state

        if picker_input:
            user_input = picker_input.strip()
        else:
            with _ProgressPauseContext():
                try:
                    user_input = typer.prompt(">", default="").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[red]Cancelled[/red]")
                    planning = dict(state.get("planning", {}))
                    planning["plan_rejected"] = True
                    state["planning"] = planning
                    return state

        normalized = user_input.lower().strip()

        # "show plan" / "show" — explicit user override to view the draft
        # despite open clarifications. Render the full plan once, then
        # loop back to the prompt without changing approval state.
        if normalized in ("show plan", "show", "plan", "draft"):
            with _ProgressPauseContext():
                console.print()
                display_plan_for_approval(state.get("plan", plan))
                console.print()
            continue

        # --- Slash commands (handled before approve/reject/modify) ---
        if normalized.startswith("/"):
            from chrono_code.utils.slash_commands import (
                ACTION_EXIT,
                SlashContext,
                dispatch_slash_command,
            )
            slash_ctx = SlashContext(
                console=console,
                pre_modify_plan=pre_modify_plan,
                compute_plan_diff=_compute_plan_diff,
                render_plan_diff_panel=_render_plan_diff_panel,
            )
            with _ProgressPauseContext():
                console.print()
                action, state = dispatch_slash_command(user_input, state, slash_ctx)
            if action == ACTION_EXIT:
                return state
            # "stay" and "unknown" both re-prompt; re-display the plan so
            # the user sees the approval instructions again after any
            # extended output. Honor the clarification gate here too —
            # while questions are still open we keep the focus on them.
            cur_plan = state.get("plan", {})
            with _ProgressPauseContext():
                console.print()
                if _has_open_clarifications(cur_plan):
                    display_plan_for_approval(cur_plan, plan_panel=False)
                    console.print()
                    choice_hint = _prompt_choice_hint(cur_plan)
                    console.print(
                        f"[dim]{choice_hint}or describe a modification · "
                        "'show plan' to view the draft · "
                        "'approve' to accept as-is · 'reject' to cancel · /help[/dim]"
                    )
                else:
                    display_plan_for_approval(cur_plan)
                    console.print()
                    _render_unresolved_token_warning(cur_plan)
                    console.print("[dim]Enter to approve | type modifications | 'reject' to cancel | /help[/dim]")
            plan = cur_plan
            continue

        # Approve: empty or keyword match
        if normalized == "" or normalized in _APPROVE_WORDS:
            console.print()
            console.print("[bold green]>>> Plan approved[/bold green]")
            console.print()
            planning = dict(state.get("planning", {}))
            planning["plan_approved"] = True
            planning["plan_rejected"] = False
            planning["plan_needs_regeneration"] = False
            state["planning"] = planning
            return state

        # Reject: keyword match
        if normalized in _REJECT_WORDS:
            console.print()
            console.print("[red]Plan rejected[/red]")
            planning = dict(state.get("planning", {}))
            planning["plan_rejected"] = True
            state["planning"] = planning
            return state

        # Modification: everything else.
        # If the user typed compact answers like "1A 2B 3C" against a plan
        # whose clarifications include structured (multi-option) entries,
        # expand those into a clear sentence pointing at the chosen labels
        # / relation patterns before handing off to modify_plan. This
        # turns "1A 2B" — which modify_plan would otherwise read as
        # opaque shorthand — into the explicit answer the planner needs.
        expanded_request = _expand_clarification_shorthand(
            user_input, plan.get("clarifications_needed") or []
        )
        modification_request_to_send = expanded_request or user_input
        conversation_history.append(
            {"role": "user", "content": modification_request_to_send}
        )

        if _plan_agent is None:
            from chrono_code.agents import PlanningAgent
            _plan_agent = PlanningAgent()

        original_plan = SimulationPlan(**plan)
        pre_modify_plan = dict(plan)  # snapshot for diff
        modification_succeeded = False
        try:
            modified_plan = await _plan_agent.modify_plan(
                original_plan=original_plan,
                modification_request=modification_request_to_send,
                conversation_history=conversation_history,
                images=images,
            )
            # NOTE: dump_all() preserves implementation_steps / assets / topology
            state["plan"] = modified_plan.dump_all()
            plan = state["plan"]  # refresh local for the re-display below
            conversation_history.append({"role": "assistant", "content": "Plan updated."})
            modification_succeeded = True
            console.print()
            console.print("[bold green]>>> Plan modified[/bold green]")
        except PlanModificationIncompleteError as e:
            # Hard-fail path: the LLM's output was unparseable enough that
            # we can't trust the user's intent landed. Stay in the modify
            # loop so the user can retry with a clearer request.
            with _ProgressPauseContext():
                missing = ", ".join(e.missing_fields) if e.missing_fields else "critical fields"
                preview = (e.raw_preview or "")[:500]
                if getattr(e, "truncated", False):
                    headline = (
                        f"[red]Modification failed — response truncated mid-JSON.[/red] "
                        f"The LLM ran out of output tokens before finishing the plan, "
                        f"so [bold]{rich_escape(missing)}[/bold] didn't land and the "
                        f"original plan is unchanged."
                    )
                    hint = (
                        "Raise [bold]AGENT1_MAX_TOKENS[/bold] in .env (current default "
                        "is 16384; try 32768 for complex mbs_in_scene plans), or "
                        "narrow the modification to fewer fields."
                    )
                else:
                    headline = (
                        f"[red]Modification failed.[/red] The LLM did not emit "
                        f"[bold]{rich_escape(missing)}[/bold] in a parseable form, "
                        f"so the original plan is unchanged."
                    )
                    hint = "Try rephrasing your modification request (shorter / more specific)."
                console.print(Panel(
                    f"{headline}\n\n"
                    f"[dim]LLM output preview (first 500 chars):[/dim]\n"
                    f"[dim]{rich_escape(preview)}[/dim]\n\n"
                    f"{hint}",
                    title="[red]Plan modification error[/red]",
                    border_style="red",
                    padding=(1, 2),
                ))
        except PlanModificationValidationError as e:
            # Schema validation failed both on the first shot and the
            # auto-retry with error feedback. The weak provider (e.g. MiniMax,
            # small local models) can't reliably emit the nested shapes this
            # plan schema requires. Show a compact panel — truncate to the
            # top 5 field errors so the user isn't drowned in a 25-line
            # Pydantic dump.
            with _ProgressPauseContext():
                top = (e.field_errors or [])[:5]
                lines = []
                for fe in top:
                    loc = ".".join(str(p) for p in fe.get("loc", []))
                    msg = str(fe.get("msg", ""))[:80]
                    lines.append(f"[yellow]{rich_escape(loc)}[/yellow]: {rich_escape(msg)}")
                errors_block = "\n".join(lines) or "(no error details)"
                total = len(e.field_errors or [])
                more = f"\n[dim]...and {total - 5} more errors[/dim]" if total > 5 else ""
                preview = (e.raw_preview or "")[:400]
                console.print(Panel(
                    f"[red]Modification failed schema validation[/red] after "
                    f"[bold]{e.retries_attempted}[/bold] retry attempt(s). "
                    f"The LLM emitted the top-level fields but got nested "
                    f"shapes wrong.\n\n"
                    f"[dim]Top {len(top)} of {total} field errors:[/dim]\n"
                    f"{errors_block}{more}\n\n"
                    f"[dim]LLM output preview:[/dim]\n"
                    f"[dim]{rich_escape(preview)}[/dim]\n\n"
                    "This is often a weak provider (e.g. MiniMax, smaller "
                    "local models) misunderstanding nested schema shapes. "
                    "Consider switching this agent's provider to Anthropic "
                    "Claude or real OpenAI (which honor json_schema / "
                    "tool_use natively), or rephrase to touch fewer fields.",
                    title="[red]Plan validation error[/red]",
                    border_style="red",
                    padding=(1, 2),
                ))
        except Exception as e:
            console.print(f"\n[red]Modification failed: {e}[/red]")
            console.print("[dim]Try again or press Enter to approve current plan[/dim]")

        # Diff panel — only after successful modify; shows what actually changed
        if modification_succeeded:
            with _ProgressPauseContext():
                diff = _compute_plan_diff(pre_modify_plan, state["plan"])
                console.print()
                _render_plan_diff_panel(diff)

        # Re-display updated plan (spinner paused so the panel renders cleanly).
        # Honor the clarification gate: if the modified plan still has open
        # clarifications, render only the questions panel and prompt for
        # answers — don't dump the full plan summary again.
        cur_plan_after_mod = state.get("plan", {})
        with _ProgressPauseContext():
            console.print()
            if _has_open_clarifications(cur_plan_after_mod):
                display_plan_for_approval(cur_plan_after_mod, plan_panel=False)
                console.print()
                choice_hint = _prompt_choice_hint(cur_plan_after_mod)
                console.print(
                    f"[dim]{choice_hint}or describe a modification · "
                    "'show plan' to view the draft · "
                    "'approve' to accept as-is · 'reject' to cancel · /help[/dim]"
                )
            else:
                display_plan_for_approval(cur_plan_after_mod)
                console.print()
                console.print("[dim]Enter to approve | type modifications | 'reject' to cancel | /help for commands[/dim]")


@app.command(name="generate")
def generate(
    prompt: Optional[str] = typer.Argument(None, help="Simulation description"),
    prompt_file: Optional[Path] = typer.Option(
        None, "--prompt-file", "-f",
        help="Path to a .md or .txt file whose content is used as the prompt.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider (anthropic or openai)",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        "-m",
        help="Planning mode (simple, detailed, auto)",
    ),
    max_retries: Optional[int] = typer.Option(
        None,
        "--max-retries",
        "-r",
        help="Maximum retry attempts (defaults to MAX_RETRIES from .env)",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        "-t",
        help="Execution timeout (seconds)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for generated files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    detail_level: str = typer.Option(
        "normal",
        "--detail-level",
        "-d",
        help="Output detail level (minimal/normal/verbose)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Enable interactive Textual TUI mode",
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-img",
        help="Path to an image file to include in planning (can be specified multiple times; comma-separated for multiple paths)",
    ),
):
    """
    Generate a PyChrono simulation from natural language description.

    Example:
        chrono-code "Create a bouncing ball simulation"
    """
    # Check environment FIRST
    if not check_environment():
        raise typer.Exit(1)

    # Check if interactive TUI mode is requested
    if interactive:
        if not is_textual_available():
            console.print("[red]Error: Textual is not installed.[/red]")
            console.print("[yellow]Add it with: uv add textual[/yellow]")
            console.print("[dim]Falling back to normal mode...[/dim]")
            interactive = False
        else:
            console.print("[cyan]Starting interactive TUI mode...[/cyan]")
            console.print("[dim]Press 'q' to quit, 'e' to expand all, 'c' to collapse all[/dim]")
            detail_level = "verbose"
            console.print("[dim]Using verbose detail level for TUI-like experience[/dim]")

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"

    global _PIPELINE_STATS_RENDERED
    _PIPELINE_STATS_RENDERED = False

    try:
        settings = get_settings()
        log_file = settings.log_file
    except Exception:
        log_file = "./chrono_code.log"

    setup_logger(level=log_level, log_file=log_file, use_rich=True)

    # Load settings
    try:
        settings = get_settings()
        init_directories_sync(settings)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        console.print("[yellow]Please check your .env file and API keys[/yellow]")
        raise typer.Exit(1)

    # Determine max_retries (use CLI arg or fallback to settings)
    max_retries = max_retries or settings.max_retries

    # Show header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Chrono-Code[/bold cyan]\n"
        "Multi-Agent PyChrono Simulation Generator",
        border_style="cyan"
    ))
    console.print()

    # Initialize output manager for collapsible thinking display
    output_manager = create_output_manager(
        detail_level=detail_level,
        interactive=interactive
    )

    # Show configuration
    console.print(f"[dim]LLM Providers:[/dim] Individual agent configs (see .env)")
    console.print(f"[dim]Plan Mode:[/dim] {mode}")
    console.print(f"[dim]Max Retries:[/dim] {max_retries}")
    console.print(f"[dim]Detail Level:[/dim] {detail_level}")
    if interactive:
        console.print(f"[dim]Interactive Mode:[/dim] Enabled")
    console.print()

    # Resolve user_prompt from --prompt-file or positional argument
    extracted_images: list[str] = []
    if prompt_file:
        if not prompt_file.exists():
            console.print(f"[red]Error: prompt file not found: {prompt_file}[/red]", err=True)
            raise typer.Exit(1)
        user_prompt = prompt_file.read_text(encoding="utf-8").strip()
        console.print(f"[dim]Prompt loaded from:[/dim] {prompt_file}")
        # Extract images from markdown content
        extracted_images = _extract_images_from_markdown(user_prompt, prompt_file.parent)
    elif prompt:
        user_prompt = prompt
    else:
        console.print("[red]Error: provide either a prompt string or --prompt-file.[/red]", err=True)
        raise typer.Exit(1)

    # Merge CLI images with extracted images (dedupe by absolute path)
    cli_images: list[str] = []
    if image:
        for part in image.split(","):
            part = part.strip()
            if part:
                cli_images.append(part)

    all_images: list[str] = []
    seen_paths: set[str] = set()
    for img_list in [extracted_images, cli_images]:
        for img_path in img_list:
            abs_path = str(Path(img_path).resolve())
            if abs_path not in seen_paths:
                seen_paths.add(abs_path)
                all_images.append(abs_path)

    if all_images:
        console.print(f"[dim]Images for planning:[/dim] {len(all_images)} image(s)")
        for img in all_images:
            console.print(f"  [dim]- {img}[/dim]")
    console.print()

    # Pre-planning Q&A gate: ask the user for any critical simulation parameters
    # they didn't name in the prompt, then bake the answers in BEFORE the
    # planner runs. This way the planner gets a complete user_spec and emits a
    # final plan in one shot — no placeholder values, no clarifications_needed,
    # no post-plan modify_plan round trip.
    user_prompt = _ask_critical_clarifications(user_prompt)

    # Run workflow
    console.print("[bold]Starting multi-agent workflow...[/bold]")
    console.print()

    # Create a single event loop for the entire generate command
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    import uuid
    thread_id = str(uuid.uuid4())
    settings = get_settings()
    _reset_cli_history(settings)
    runtime_output_dir = _activate_cli_runtime_run(settings, thread_id)
    output_dir = str(runtime_output_dir)

    try:
        from chrono_code.workflow.engine import run_workflow

        def run_with_spinner(description: str, coroutine):
            """Run the workflow coroutine with the event-driven live renderer.

            The renderer consumes workflow events (agent lifecycle, tool calls,
            streaming text, etc.) and renders Claude-Code-style per-agent
            blocks inline while keeping a sticky bottom status line.
            """
            global _active_renderer
            from chrono_code.ui import LiveEventRenderer
            with LiveEventRenderer(console) as renderer:
                _active_renderer = renderer
                try:
                    return loop.run_until_complete(coroutine)
                finally:
                    _active_renderer = None

        # Run async workflow using the shared event loop
        result = run_with_spinner(
            "Processing...",
            run_workflow(
                user_prompt=user_prompt,
                plan_mode=mode,
                images=all_images if all_images else None,
                on_plan_approval=_cli_plan_approval,
                clarification_callback=_cli_batch_clarification,
            ),
        )

        def _save_transcript():
            from chrono_code.agents.base import BaseAgent
            dm = getattr(BaseAgent, "_shared_dialog_manager", None)
            if dm and dm.current_session and result.get("messages"):
                dm.save_session_transcript(result["messages"])

        _save_transcript()

        # Display thinking summary if in verbose mode
        if detail_level == "verbose" and output_manager.session.thinking_blocks:
            console.print()
            console.print("[bold cyan]Agent Thinking Summary[/bold cyan]")
            output_manager.display_tree(expanded_agents=list(output_manager.expanded_agents))
            console.print()

        # Display results only after the workflow has actually completed.
        if _should_display_results(result):
            display_results(result, output_dir, detail_level=detail_level)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(0)

    except Exception as e:
        console.print(f"\n[red]Error: {rich_escape(str(e))}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

    finally:
        # Force-show token / time totals regardless of success or failure.
        # Idempotent — if display_results already rendered the panel, this
        # is a no-op.
        try:
            _render_pipeline_stats_panel(locals().get("result") or {})
        except Exception:
            pass
        # Properly close the event loop to prevent "Event loop is closed" warnings
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop.close()


def _format_scalar(value) -> str:
    """Compact inline rendering for a scalar or small nested container."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            return "[" + ", ".join(_format_scalar(x) for x in value) + "]"
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_format_scalar(v)}" for k, v in value.items()) + "}"
    return str(value)


def _kv_table(data: dict, key_header: str = "key", val_header: str = "value"):
    """Two-column Rich Table for flat key/value rendering."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1), expand=False)
    table.add_column(key_header, style="cyan", no_wrap=True)
    table.add_column(val_header, style="white", overflow="fold")
    for k, v in data.items():
        table.add_row(str(k), _format_scalar(v))
    return table


def _render_plan_summary(plan: dict, outer_title: str, border_style: str) -> None:
    """Render a plan as structured sections (header, objectives, params, predicates, steps)."""
    from rich.console import Group
    from rich.table import Table
    from rich.rule import Rule

    plan_type = plan.get("plan_type", "unknown")
    sim_params = plan.get("simulation_parameters") or {}
    objectives = plan.get("objectives") or []
    steps = plan.get("implementation_steps") or []
    clarifications = plan.get("clarifications_needed") or []
    topology = plan.get("topology") or {}
    assets = plan.get("assets") or []
    visualization = plan.get("visualization") or {}

    scene_preds = (topology.get("scene_predicates") if isinstance(topology, dict) else None) or []
    phys_preds = (topology.get("physical_predicates") if isinstance(topology, dict) else None) or []

    sections: list = []

    # Header: quick stats
    stats = [f"[bold]type[/bold] {plan_type}"]
    if sim_params.get("simulation_duration") is not None:
        stats.append(f"[bold]duration[/bold] {_format_scalar(sim_params.get('simulation_duration'))}s")
    if sim_params.get("time_step") is not None:
        stats.append(f"[bold]dt[/bold] {_format_scalar(sim_params.get('time_step'))}")
    if assets:
        stats.append(f"[bold]assets[/bold] {len(assets)}")
    if scene_preds:
        stats.append(f"[bold]scene_predicates[/bold] {len(scene_preds)}")
    if steps:
        stats.append(f"[bold]steps[/bold] {len(steps)}")
    sections.append("  ".join(stats))

    # Objectives
    if objectives:
        sections.append(Rule("Objectives", style="dim"))
        sections.append("\n".join(f"  • {rich_escape(str(o))}" for o in objectives))

    # Simulation parameters (flat scalars first, nested dicts as sub-tables)
    if sim_params:
        flat = {k: v for k, v in sim_params.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in sim_params.items() if isinstance(v, dict)}
        sections.append(Rule("Simulation parameters", style="dim"))
        if flat:
            sections.append(_kv_table(flat, "parameter", "value"))
        for name, sub in nested.items():
            sections.append(f"[dim]{name}[/dim]")
            sections.append(_kv_table(sub, "parameter", "value"))

    # Topology (excluding predicates, which get their own table)
    if topology and isinstance(topology, dict):
        topo_flat = {k: v for k, v in topology.items()
                     if k not in ("scene_predicates", "physical_predicates") and v is not None}
        if topo_flat:
            sections.append(Rule("Topology", style="dim"))
            sections.append(_kv_table(topo_flat, "field", "value"))

    # Assets
    if assets:
        sections.append(Rule(f"Assets ({len(assets)})", style="dim"))
        a_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1))
        a_table.add_column("#", style="dim", no_wrap=True)
        a_table.add_column("type", style="cyan", no_wrap=True)
        a_table.add_column("filename", style="white", overflow="fold")
        a_table.add_column("apply_to", style="cyan", no_wrap=True)
        a_table.add_column("fixed", style="dim", no_wrap=True)
        for i, a in enumerate(assets, 1):
            if not isinstance(a, dict):
                continue
            a_table.add_row(
                str(i),
                str(a.get("type", "")),
                str(a.get("filename", "")),
                str(a.get("apply_to", "")),
                _format_scalar(a.get("fixed", "")),
            )
        sections.append(a_table)

    # Scene predicates
    if scene_preds:
        sections.append(Rule(f"Scene predicates ({len(scene_preds)})", style="dim"))
        sp_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1))
        sp_table.add_column("subject", style="cyan", no_wrap=True)
        sp_table.add_column("predicate", style="magenta", no_wrap=True)
        sp_table.add_column("object", style="cyan", no_wrap=True)
        sp_table.add_column("position (x,y,z)", style="white", no_wrap=True)
        sp_table.add_column("yaw°", style="white", no_wrap=True)
        for p in scene_preds:
            if not isinstance(p, dict):
                continue
            pos = p.get("position") or {}
            orient = p.get("orientation") or {}
            pos_str = (f"{_format_scalar(pos.get('x', ''))}, "
                       f"{_format_scalar(pos.get('y', ''))}, "
                       f"{_format_scalar(pos.get('z', ''))}") if pos else ""
            yaw_str = _format_scalar(orient.get("deg_z", "")) if orient else ""
            sp_table.add_row(
                str(p.get("subject", "")),
                str(p.get("predicate", "")),
                str(p.get("object", "")),
                pos_str,
                yaw_str,
            )
        sections.append(sp_table)

    # Physical predicates
    if phys_preds:
        sections.append(Rule(f"Physical predicates ({len(phys_preds)})", style="dim"))
        pp_table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1))
        pp_table.add_column("subject", style="cyan", no_wrap=True)
        pp_table.add_column("predicate", style="magenta", no_wrap=True)
        pp_table.add_column("object", style="cyan", no_wrap=True)
        for p in phys_preds:
            if not isinstance(p, dict):
                continue
            pp_table.add_row(
                str(p.get("subject", "")),
                str(p.get("predicate", "")),
                str(p.get("object", "")),
            )
        sections.append(pp_table)

    # Implementation steps (structured SimulationStep objects for scene /
    # mbs_in_scene; empty list for pure mbs).
    if steps:
        sections.append(Rule(f"Implementation steps ({len(steps)})", style="dim"))
        step_lines = []
        for i, s in enumerate(steps, 1):
            if isinstance(s, dict):
                desc = str(s.get("description") or "")
                assets_here = s.get("assets") or []
                cam = s.get("camera") or {}
                cam_pos = cam.get("position") or []
                cam_str = (
                    f"cam{_format_scalar(cam_pos[0]) if len(cam_pos) > 0 else '?'},"
                    f"{_format_scalar(cam_pos[1]) if len(cam_pos) > 1 else '?'},"
                    f"{_format_scalar(cam_pos[2]) if len(cam_pos) > 2 else '?'}"
                ) if cam_pos else "(no camera)"
                assets_str = ",".join(assets_here) if assets_here else "—"
                step_lines.append(
                    f"  [dim]{i:>2}.[/dim] [cyan]{rich_escape(assets_str)}[/cyan]  "
                    f"[dim]{rich_escape(cam_str)}[/dim]  {rich_escape(desc[:100])}"
                )
            else:
                # Legacy list-of-string fallback (shouldn't happen with new
                # schema but render defensively).
                step_lines.append(f"  [dim]{i:>2}.[/dim] {rich_escape(str(s))}")
        sections.append("\n".join(step_lines))

    # Visualization
    vis_out = {}
    if isinstance(visualization, dict):
        vis_out.update({f"viz.{k}": v for k, v in visualization.items() if not isinstance(v, dict)})
    if vis_out:
        sections.append(Rule("Visualization", style="dim"))
        sections.append(_kv_table(vis_out, "key", "value"))

    # Clarifications (warnings)
    if clarifications:
        sections.append(Rule("[yellow]Clarifications needed[/yellow]", style="yellow"))
        sections.append("\n".join(f"  [yellow]?[/yellow] {rich_escape(str(q))}" for q in clarifications))

    console.print(Panel(
        Group(*sections),
        title=outer_title,
        border_style=border_style,
        padding=(1, 2),
    ))


def _compute_plan_diff(old: Optional[dict], new: Optional[dict]) -> dict:
    """Shallow-plus-one-level diff between two plan dicts.

    Returns a dict of the shape::

        {
            "changed": {key: (old_val, new_val), ...},
            "added":   {key: new_val, ...},
            "removed": {key: old_val, ...},
            "sim_params": {key: (old_val, new_val), ...},   # recurse
            "counts": {"implementation_steps_old": int, "implementation_steps_new": int, ...},
        }

    Recurses ONLY into ``simulation_parameters`` (most frequently modified
    structured field). Other nested values are compared as opaque blobs and
    summarized as "changed" if they don't match — the full-plan panel below
    shows details if the user cares.
    """
    diff: Dict[str, Any] = {
        "changed": {},
        "added": {},
        "removed": {},
        "sim_params": {},
        "counts": {},
    }
    old = old or {}
    new = new or {}

    # Recurse into simulation_parameters
    old_sp = old.get("simulation_parameters") or {}
    new_sp = new.get("simulation_parameters") or {}
    if isinstance(old_sp, dict) and isinstance(new_sp, dict):
        for k in sorted(set(old_sp) | set(new_sp)):
            if old_sp.get(k) != new_sp.get(k):
                diff["sim_params"][k] = (old_sp.get(k), new_sp.get(k))

    # Top-level field diff (skip simulation_parameters + plan_markdown which are
    # either recursed-into or a noisy serialization artifact).
    skip = {"simulation_parameters", "plan_markdown"}
    for k in sorted(set(old) | set(new)):
        if k in skip:
            continue
        ov, nv = old.get(k), new.get(k)
        if k not in old:
            diff["added"][k] = nv
        elif k not in new:
            diff["removed"][k] = ov
        elif ov != nv:
            diff["changed"][k] = (ov, nv)

    # Useful count deltas for implementation_steps / assets
    for k in ("implementation_steps", "assets"):
        old_n = len(old.get(k) or []) if isinstance(old.get(k), list) else 0
        new_n = len(new.get(k) or []) if isinstance(new.get(k), list) else 0
        if old_n != new_n:
            diff["counts"][k] = (old_n, new_n)

    return diff


def _plan_diff_is_empty(diff: dict) -> bool:
    return not (
        diff.get("changed")
        or diff.get("added")
        or diff.get("removed")
        or diff.get("sim_params")
        or diff.get("counts")
    )


def _render_plan_diff_panel(diff: dict) -> None:
    """Render a compact diff of what the modification actually changed.

    Shown above the re-drawn plan panel so users can verify at a glance that
    their modification request landed (or didn't). Empty diffs get a yellow
    warning explicitly calling that out.
    """
    if _plan_diff_is_empty(diff):
        console.print(Panel(
            "[yellow]⚠ No changes detected.[/yellow] The LLM may have "
            "misunderstood your request or returned the original plan "
            "unchanged. Try rephrasing your modification, or type 'approve' "
            "to proceed with the unchanged plan.",
            title="[yellow]Plan modifications[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        return

    lines: list[str] = []
    if diff.get("sim_params"):
        lines.append("[bold cyan]simulation_parameters[/bold cyan]")
        for k, (ov, nv) in diff["sim_params"].items():
            lines.append(
                f"  [dim]{rich_escape(str(k))}:[/dim] "
                f"[red]{rich_escape(_format_scalar(ov))}[/red] "
                f"[green]→ {rich_escape(_format_scalar(nv))}[/green]"
            )
    if diff.get("counts"):
        lines.append("[bold cyan]count deltas[/bold cyan]")
        for k, (on, nn) in diff["counts"].items():
            arrow_color = "green" if nn >= on else "red"
            lines.append(
                f"  [dim]{rich_escape(str(k))}:[/dim] "
                f"{on} items [{arrow_color}]→ {nn} items[/{arrow_color}]"
            )
    if diff.get("changed"):
        lines.append("[bold cyan]changed fields[/bold cyan]")
        for k, (ov, nv) in diff["changed"].items():
            ov_s = rich_escape(_format_scalar(ov))[:60]
            nv_s = rich_escape(_format_scalar(nv))[:60]
            lines.append(
                f"  [dim]{rich_escape(str(k))}:[/dim] "
                f"[red]{ov_s}[/red] [green]→ {nv_s}[/green]"
            )
    if diff.get("added"):
        lines.append("[bold cyan]added[/bold cyan]")
        for k in diff["added"]:
            lines.append(f"  [green]+ {rich_escape(str(k))}[/green]")
    if diff.get("removed"):
        lines.append("[bold cyan]removed[/bold cyan]")
        for k in diff["removed"]:
            lines.append(f"  [red]- {rich_escape(str(k))}[/red]")

    total = (
        len(diff.get("sim_params", {}))
        + len(diff.get("changed", {}))
        + len(diff.get("added", {}))
        + len(diff.get("removed", {}))
        + len(diff.get("counts", {}))
    )
    console.print(Panel(
        "\n".join(lines),
        title=f"[green]Plan modifications ({total} change{'s' if total != 1 else ''})[/green]",
        border_style="green",
        padding=(1, 2),
    ))


_OPTION_LETTERS = "ABCDEFGH"


def _pt_clarification_picker(
    question: str,
    options: List[Any],
    allow_other: bool,
) -> Tuple[str, Optional[int]]:
    """Inline arrow-key picker for a single structured clarification.

    Renders below the existing question panel as a non-fullscreen
    ``prompt_toolkit.Application``. The user navigates options with ↑/↓
    or jumps directly with the corresponding letter (A/B/C/D…), Enter
    confirms, Esc bails to the legacy free-form text path.

    Returns ``(action, payload)``:
      * ``("CHOICE", idx)`` — user picked option ``idx`` (0-based; idx
        equal to ``len(options)`` is the implicit "Other" entry).
      * ``("MODIFY", None)`` — user pressed Esc; caller should fall
        back to ``typer.prompt`` for free-form text.
      * ``("INTERRUPTED", None)`` — Ctrl-C; caller treats as cancel.
      * ``("UNAVAILABLE", None)`` — non-TTY / prompt_toolkit init
        failed; caller should fall back to ``typer.prompt`` silently.
    """
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.dimension import D
        from prompt_toolkit.styles import Style
    except Exception:
        return ("UNAVAILABLE", None)

    n_choices = len(options) + (1 if allow_other else 0)
    if n_choices <= 0:
        return ("UNAVAILABLE", None)

    cursor = [0]

    def _opt_label_desc(i: int) -> Tuple[str, str]:
        if i >= len(options):
            return ("Other", "type your own custom answer")
        opt = options[i]
        if isinstance(opt, dict):
            return (
                str(opt.get("label", "")),
                str(opt.get("description", "") or ""),
            )
        return (
            str(getattr(opt, "label", "")),
            str(getattr(opt, "description", "") or ""),
        )

    def _render_options():
        rows = []
        for i in range(n_choices):
            label, desc = _opt_label_desc(i)
            letter = (
                _OPTION_LETTERS[i] if i < len(_OPTION_LETTERS) else f"#{i + 1}"
            )
            cursor_mark = "▶" if i == cursor[0] else " "
            head = f"  {cursor_mark} {letter}. {label}"
            if desc:
                head += f"  —  {desc[:80]}"
            cls = "class:selected" if i == cursor[0] else ""
            rows.append((cls, head + "\n"))
        return rows

    kb = KeyBindings()

    @kb.add("up")
    def _(event):  # noqa: ANN001
        cursor[0] = (cursor[0] - 1) % n_choices
        event.app.invalidate()

    @kb.add("down")
    def _(event):  # noqa: ANN001
        cursor[0] = (cursor[0] + 1) % n_choices
        event.app.invalidate()

    @kb.add("enter")
    def _(event):  # noqa: ANN001
        event.app.exit(result=("CHOICE", cursor[0]))

    @kb.add("escape")
    def _(event):  # noqa: ANN001
        event.app.exit(result=("MODIFY", None))

    @kb.add("c-c")
    def _(event):  # noqa: ANN001
        event.app.exit(exception=KeyboardInterrupt())

    # Direct letter selection (case-insensitive). Each letter both moves
    # the cursor and immediately confirms the choice — this is the
    # CC-style "press a number to pick" affordance.
    for idx in range(min(n_choices, len(_OPTION_LETTERS))):
        letter = _OPTION_LETTERS[idx]
        for ch in (letter.lower(), letter):

            @kb.add(ch)
            def _(event, _idx=idx):  # noqa: ANN001
                cursor[0] = _idx
                event.app.exit(result=("CHOICE", _idx))

    body = HSplit([
        Window(
            content=FormattedTextControl(text=_render_options),
            height=D.exact(n_choices),
        ),
        Window(
            content=FormattedTextControl(
                text=HTML(
                    "<i>↑/↓ navigate · Enter pick · A-{last} jump · "
                    "Esc to type free-form</i>"
                ).format_map(
                    {"last": _OPTION_LETTERS[min(n_choices, len(_OPTION_LETTERS)) - 1]}
                )
            ),
            height=D.exact(1),
        ),
    ])

    style = Style.from_dict({
        "selected": "bg:#005577 fg:#ffffff bold",
    })

    try:
        app = Application(
            layout=Layout(body),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            style=style,
            erase_when_done=True,  # leave the panel above intact
        )
        return app.run()
    except KeyboardInterrupt:
        return ("INTERRUPTED", None)
    except Exception:
        # Any other prompt_toolkit failure (non-TTY, terminal incompat) —
        # fall back to the legacy text prompt.
        return ("UNAVAILABLE", None)


def _maybe_pick_clarification(plan: dict) -> Optional[str]:
    """If the plan has exactly ONE open structured clarification AND we
    are on a TTY, run ``_pt_clarification_picker`` and translate the
    result into the same shorthand string ``_expand_clarification_shorthand``
    consumes (``"1A"`` etc.). Returns ``None`` when the picker is
    unavailable or the user chose to type a free-form modification —
    caller falls back to ``typer.prompt``.
    """
    import sys
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    entries = [
        e for e in (plan or {}).get("clarifications_needed") or []
        if _is_structured_clarification(e)
    ]
    if len(entries) != 1:
        return None
    entry = entries[0]
    options = entry.get("options") or []
    allow_other = bool(entry.get("allow_other", True))

    question = str(entry.get("question") or "(no question text)")
    # Hint line above picker — keeps panel above intact, picker is
    # purely the option list.
    with _ProgressPauseContext():
        console.print(f"[bold]→ {rich_escape(question)}[/bold]")

    action, payload = _pt_clarification_picker(question, options, allow_other)
    if action == "INTERRUPTED":
        raise KeyboardInterrupt()
    if action == "UNAVAILABLE" or action == "MODIFY":
        return None
    if action == "CHOICE":
        idx = int(payload or 0)
        letter = (
            _OPTION_LETTERS[idx] if idx < len(_OPTION_LETTERS) else f"#{idx + 1}"
        )
        # Single-question shorthand: "1A". The expander already accepts
        # bare letters too, but emit "1A" so the residual / consumed-span
        # bookkeeping is unambiguous.
        return f"1{letter}"
    return None


def _is_structured_clarification(entry: Any) -> bool:
    """A StructuredClarification round-trips through dump_all as a dict with
    an ``options`` list. Plain-string entries miss that key."""
    return isinstance(entry, dict) and isinstance(entry.get("options"), list) and entry.get("options")


def _render_one_clarification(idx: int, entry: Any) -> str:
    """Render one clarifications_needed entry as Rich-formatted text.

    Plain strings render as the bare question. Structured entries render
    with labelled options A/B/C plus an "Other (text input)" escape when
    ``allow_other`` is True. The output is a single multi-line block ready
    to be joined with blank lines into the panel body.
    """
    if not _is_structured_clarification(entry):
        return f"[yellow]{idx}.[/yellow] {rich_escape(str(entry))}"

    question = entry.get("question", "(no question)")
    options = entry.get("options") or []
    allow_other = entry.get("allow_other", True)

    lines = [f"[yellow]{idx}.[/yellow] [bold]{rich_escape(question)}[/bold]"]
    for i, opt in enumerate(options):
        letter = _OPTION_LETTERS[i] if i < len(_OPTION_LETTERS) else f"#{i + 1}"
        if isinstance(opt, dict):
            label = opt.get("label", "(no label)")
            description = opt.get("description", "")
            relation_pattern = opt.get("relation_pattern")
            preview = opt.get("preview")
        else:
            label = getattr(opt, "label", "(no label)")
            description = getattr(opt, "description", "")
            relation_pattern = getattr(opt, "relation_pattern", None)
            preview = getattr(opt, "preview", None)

        head = f"  [cyan]{letter})[/cyan] [bold]{rich_escape(str(label))}[/bold]"
        if relation_pattern:
            head += f" [dim](pattern: {rich_escape(str(relation_pattern))})[/dim]"
        lines.append(head)
        if description:
            lines.append(f"     {rich_escape(str(description))}")
        if preview:
            indented = "\n".join("       " + ln for ln in str(preview).splitlines())
            lines.append(f"[dim]{rich_escape(indented)}[/dim]")

    if allow_other:
        other_letter = _OPTION_LETTERS[len(options)] if len(options) < len(_OPTION_LETTERS) else "?"
        lines.append(
            f"  [cyan]{other_letter})[/cyan] [bold]Other[/bold] "
            f"[dim](type your own answer in the modification box below)[/dim]"
        )

    return "\n".join(lines)


_SHORTHAND_TOKEN_RE = re.compile(r"\b(\d+)\s*([A-Za-z])\b")


def _expand_clarification_shorthand(user_input: str, clarifications: List[Any]) -> str:
    """Translate '1A 2B' / '1=A, 2=B' / bare 'A' shorthand into an explicit sentence.

    Looks for ``<question_index><option_letter>`` tokens in ``user_input``
    and maps them against the ``clarifications`` list. Every recognised
    token is rendered as a sentence naming the chosen option's label,
    its ``relation_pattern``, and (when present) the option's concrete
    ``value`` so modify_plan can use the numeric content downstream.

    When exactly one structured clarification exists, a bare single
    letter (e.g. "A") is accepted as shorthand for "1A". Without this
    fall-through users hitting the side-camera question were typing "A"
    and seeing the planner ignore the answer because the shorthand regex
    required a numeric prefix.

    Returns an empty string when no recognisable token is found, so the
    caller can fall back to the raw input untouched.
    """
    matches = list(_SHORTHAND_TOKEN_RE.finditer(user_input))

    # Bare single-letter fallback when there's exactly one structured
    # clarification — typing "A" / "B" / "T" should mean "answer to
    # question 1", and "D camera at y=-7" should mean "Other choice
    # with custom answer 'camera at y=-7'". Common when the planner
    # emits a single camera-choice question and the user trusts the
    # implicit numbering.
    if not matches:
        stripped = user_input.strip()
        structured_indices = [
            i for i, c in enumerate(clarifications)
            if _is_structured_clarification(c)
        ]
        forged_match: Optional[Any] = None
        if len(structured_indices) == 1 and stripped:
            head = stripped[0].upper()
            # Single bare letter: "A"
            if len(stripped) == 1 and head in _OPTION_LETTERS:
                forged_span = (0, 1)
                forged_letter = head
            # Letter followed by space/punct + free text: "D camera at y=-7"
            # The leading letter is the choice; remaining text becomes the
            # residual carried as ``Additional notes from the user``.
            elif (
                len(stripped) > 1
                and head in _OPTION_LETTERS
                and stripped[1] in " ,;\t:"
            ):
                # Locate "head" in the original (possibly-padded) input so
                # the consumed span maps back correctly.
                idx = user_input.find(stripped[0])
                forged_span = (idx, idx + 1)
                forged_letter = head
            else:
                return ""

            class _ForgedMatch:
                def __init__(self, q_idx: int, letter: str,
                             span: Tuple[int, int]) -> None:
                    self._q = q_idx
                    self._l = letter
                    self._span = span
                def group(self, n: int) -> str:
                    return str(self._q) if n == 1 else self._l
                def span(self) -> Tuple[int, int]:
                    return self._span
            forged_match = _ForgedMatch(
                structured_indices[0] + 1, forged_letter, forged_span,
            )
            matches = [forged_match]
        else:
            return ""

    sentences: List[str] = []
    consumed_spans: List[Tuple[int, int]] = []
    for m in matches:
        try:
            q_idx = int(m.group(1))
        except ValueError:
            continue
        letter = m.group(2).upper()
        if q_idx < 1 or q_idx > len(clarifications):
            continue
        entry = clarifications[q_idx - 1]
        if not _is_structured_clarification(entry):
            continue
        options = entry.get("options") or []
        # The CLI offered letters A..Z for options[0..N-1]; an extra letter
        # corresponds to the implicit 'Other' (free-text) escape.
        letter_idx = _OPTION_LETTERS.find(letter)
        if letter_idx < 0:
            continue
        if letter_idx < len(options):
            opt = options[letter_idx] or {}
            label = opt.get("label", letter)
            pattern = opt.get("relation_pattern")
            description = opt.get("description", "")
            # Pull the option's concrete `value` if present (e.g. camera
            # y=[-13.0], plate size [3.5, 1.8, 0.2]) so modify_plan
            # actually receives the number the user picked. Without this
            # the planner sees only the label string and re-asks the
            # same question on the next round.
            value = opt.get("value")
            sentence_parts = [
                f'For clarification {q_idx} ({entry.get("question", "").strip()})',
                f'choose option "{label}"',
            ]
            if pattern:
                sentence_parts.append(f'(use relation pattern `{pattern}`)')
            if value is not None:
                sentence_parts.append(f'with value {value}')
            if description:
                sentence_parts.append(f'-- {description}')
            sentences.append(" ".join(sentence_parts))
            consumed_spans.append(m.span())
        elif entry.get("allow_other", True) and letter_idx == len(options):
            # 'Other' picked. Emit an explicit sentence so modify_plan
            # sees that the user rejected the listed labels for this
            # question — without this the bare token "D" / "1D" was
            # silently dropped (no sentence appended) and the raw input
            # leaked through with no question context. Any free-text the
            # user typed alongside the shorthand is captured below as
            # ``Additional notes from the user`` and serves as the
            # actual Other value.
            sentence_parts = [
                f'For clarification {q_idx} ({entry.get("question", "").strip()})',
                'choose option "Other"',
            ]
            sentences.append(" ".join(sentence_parts))
            consumed_spans.append(m.span())

    if not sentences:
        return ""

    # Strip the consumed shorthand tokens from the residual so the planner
    # doesn't see both "1A" and the expanded sentence describing it.
    residual = user_input
    for start, end in sorted(consumed_spans, reverse=True):
        residual = residual[:start] + residual[end:]
    residual = residual.strip(" ,;\t")

    expanded = ". ".join(sentences) + "."
    if residual:
        expanded += f" Additional notes from the user: {residual}"
    return expanded


def _render_clarifications_panel(plan: dict) -> None:
    """Render the planner's open questions as a separate panel above the plan.

    Clarifications also appear inside the structured plan summary, but burying
    them at the bottom of a long JSON/markdown blob makes them easy to miss.
    Showing them first invites the user to answer (via the modification loop)
    before approving.

    Structured entries (``StructuredClarification`` round-tripped via
    ``dump_all``) render with labelled options + 'Other'. Plain-string entries
    render as bare questions, preserving the legacy behaviour.
    """
    clarifications = plan.get("clarifications_needed") or []
    if not clarifications:
        return
    body = "\n\n".join(
        _render_one_clarification(i, q)
        for i, q in enumerate(clarifications, 1)
    )
    has_structured = any(_is_structured_clarification(q) for q in clarifications)
    if has_structured:
        subtitle = (
            "[dim]Reply with the option letters per question (e.g. '1A 2B') "
            "or type a free-form modification — both flow into modify_plan[/dim]"
        )
    else:
        subtitle = (
            "[dim]Type a reply below to modify the plan, or press Enter to "
            "accept the planner's default[/dim]"
        )
    console.print(Panel(
        body,
        title=f"[yellow]Clarifications needed ({len(clarifications)})[/yellow]",
        subtitle=subtitle,
        border_style="yellow",
        padding=(1, 2),
    ))


def display_plan_for_approval(plan: dict, *, plan_panel: bool = True):
    """Display plan details for user approval using markdown or structured summary.

    When ``plan_panel`` is False, only the clarifications panel is
    rendered — used to keep the user focused on answering open questions
    before the full plan is dumped. The plan summary appears once the
    clarification list is empty (or the user explicitly approves
    despite open clarifications).
    """
    from rich.markdown import Markdown

    plan_type = plan.get("plan_type", "unknown")
    plan_md = plan.get("plan_markdown", "") or ""

    # Surface open questions BEFORE the plan panel so they're unmissable.
    _render_clarifications_panel(plan)

    if not plan_panel:
        return

    # When the planner falls back to per-field JSON extraction, it stores the
    # raw LLM JSON response in ``plan_markdown`` (see planning_agent._parse_json_plan).
    # Rendering that as Markdown flattens it into an unreadable single paragraph,
    # so detect it and fall through to the structured summary renderer.
    looks_like_json = plan_md.lstrip().startswith(("{", "["))

    if plan_md and not looks_like_json:
        # Display the canonical markdown directly
        console.print(Panel(
            Markdown(plan_md),
            title="Proposed Simulation Plan",
            border_style="yellow",
            padding=(1, 2),
        ))
    else:
        _render_plan_summary(
            plan,
            outer_title=f"Proposed Simulation Plan (type: {plan_type})",
            border_style="yellow",
        )


def _extract_patch_from_result(result: dict) -> str:
    """Get unified diff patch from result payload if available."""
    code_payload = result.get("code")
    if isinstance(code_payload, dict):
        patch = code_payload.get("patch")
        if isinstance(patch, str) and patch.strip():
            return patch

    artifacts = result.get("code_artifacts") or []
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if isinstance(artifact, dict):
                patch = artifact.get("patch")
                if isinstance(patch, str) and patch.strip():
                    return patch
    return ""


def _truncate_patch_for_display(patch_text: str) -> tuple[str, bool]:
    if not patch_text:
        return "", False
    lines = patch_text.splitlines()
    truncated_by_lines = len(lines) > PATCH_PREVIEW_MAX_LINES
    clipped = "\n".join(lines[:PATCH_PREVIEW_MAX_LINES])
    truncated_by_chars = len(clipped) > PATCH_PREVIEW_MAX_CHARS
    if truncated_by_chars:
        clipped = clipped[:PATCH_PREVIEW_MAX_CHARS]
    truncated = truncated_by_lines or truncated_by_chars
    if truncated:
        clipped += "\n... [patch preview truncated]"
    return clipped, truncated


def _show_patch_preview(result: dict, detail_level: str) -> None:
    patch = _extract_patch_from_result(result)
    if not patch:
        return

    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1

    if detail_level == "minimal":
        console.print(f"[dim]Patch summary: +{added}/-{removed}[/dim]")
        console.print()
        return

    preview, truncated = _truncate_patch_for_display(patch)
    syntax = Syntax(preview, "diff", theme="monokai", line_numbers=False)
    title = f"Code Patch (+{added}/-{removed})"
    if truncated:
        title += " [truncated]"
    console.print(Panel(syntax, title=title, border_style="magenta"))
    console.print()


_PIPELINE_STATS_RENDERED: bool = False


def _render_pipeline_stats_panel(result: dict) -> None:
    """Print the final token / wall-clock summary as the LAST output block.

    The live event renderer already shows this during the run, but it can
    scroll off-screen by the time the workflow ends and ``display_results``
    finishes. Re-emit it here so the user always sees totals as the final
    thing printed.

    Idempotent: a module-level flag prevents double-render when both
    ``display_results`` (success path) and the outer ``finally`` block of
    a CLI command call this in the same invocation.
    """
    global _PIPELINE_STATS_RENDERED
    if _PIPELINE_STATS_RENDERED:
        return
    stats = (result or {}).get("pipeline_stats") or {}

    # When run_workflow raises before returning, the caller never sees
    # the populated state. The dialog manager's in-memory copy is still
    # valid, written from engine.py's finally block via dm.log_pipeline_stats.
    # Fall back to it so the user always sees costs even on partial runs.
    if not stats:
        try:
            from chrono_code.agents.base import BaseAgent
            dm = getattr(BaseAgent, "_shared_dialog_manager", None)
            if dm is not None:
                disk_record = (getattr(dm, "session_data", {}) or {}).get("pipeline_stats")
                if disk_record:
                    # On-disk shape uses elapsed_seconds; live event uses elapsed.
                    stats = {
                        "elapsed": float(disk_record.get("elapsed_seconds", 0.0)),
                        "usage": disk_record.get("usage", {}),
                        "per_agent": disk_record.get("per_agent", {}),
                        "sessions": int(disk_record.get("sessions", 0)),
                        "calls": int(disk_record.get("calls", 0)),
                    }
        except Exception:
            pass

    if not stats:
        # Truly nothing to show (workflow crashed before any LLM call).
        try:
            from chrono_code.agents.base import BaseAgent
            dm = getattr(BaseAgent, "_shared_dialog_manager", None)
            if dm is not None and getattr(dm, "current_session", None) is not None:
                console.print(
                    f"[dim]Pipeline stats unavailable — partial logs at "
                    f"{dm.current_session}[/dim]"
                )
                console.print()
        except Exception:
            pass
        return

    from chrono_code.ui.renderer import (
        _AGENT_COLORS,
        _DEFAULT_AGENT_COLOR,
        _fmt_elapsed,
        _fmt_usage_inline,
    )
    from rich.text import Text

    elapsed = float(stats.get("elapsed") or 0.0)
    usage = stats.get("usage") or {}
    per_agent = stats.get("per_agent") or {}
    sessions = int(stats.get("sessions") or 0)
    calls = int(stats.get("calls") or 0)

    lines: List[str] = []
    usage_str = _fmt_usage_inline(usage)
    header_bits = [f"[bold]total[/bold] · [dim]{_fmt_elapsed(elapsed)}[/dim]"]
    if usage_str:
        header_bits.append(f"[dim]{usage_str}[/dim]")
    meta = []
    if sessions:
        meta.append(f"{sessions} session{'s' if sessions != 1 else ''}")
    if calls:
        meta.append(f"{calls} call{'s' if calls != 1 else ''}")
    if meta:
        header_bits.append(f"[dim]({', '.join(meta)})[/dim]")
    lines.append(" · ".join(header_bits))

    def _agent_total(entry: dict) -> int:
        u = entry.get("usage") or {}
        return sum(
            int(u.get(k, 0) or 0)
            for k in ("input", "output", "cache_read", "cache_creation")
        )

    sorted_agents = sorted(
        per_agent.items(), key=lambda kv: _agent_total(kv[1]), reverse=True
    )
    for name, entry in sorted_agents:
        color = _AGENT_COLORS.get(name, _DEFAULT_AGENT_COLOR)
        agent_usage_str = _fmt_usage_inline(entry.get("usage") or {})
        ag_elapsed = float(entry.get("elapsed") or 0.0)
        ag_sessions = int(entry.get("sessions") or 0)
        ag_calls = int(entry.get("calls") or 0)
        row_bits = [
            f"[{color}]•[/{color}] [bold]{name}[/bold]",
            f"[dim]{_fmt_elapsed(ag_elapsed)}[/dim]",
        ]
        if agent_usage_str:
            row_bits.append(f"[dim]{agent_usage_str}[/dim]")
        sub = []
        if ag_sessions:
            sub.append(f"{ag_sessions} session{'s' if ag_sessions != 1 else ''}")
        if ag_calls:
            sub.append(f"{ag_calls} call{'s' if ag_calls != 1 else ''}")
        if sub:
            row_bits.append(f"[dim]({', '.join(sub)})[/dim]")
        lines.append(" · ".join(row_bits))

    # Footer: link to the session JSON for downstream tooling.
    try:
        from chrono_code.agents.base import BaseAgent
        dm = getattr(BaseAgent, "_shared_dialog_manager", None)
        if dm is not None and getattr(dm, "current_session", None) is not None:
            lines.append("")
            lines.append(
                f"[dim]Per-session JSONs: {dm.current_session}/[/dim]"
            )
            lines.append(
                f"[dim]Pipeline summary:  {dm.current_session}/pipeline_stats.json[/dim]"
            )
    except Exception:
        pass

    console.print(Panel(
        Text.from_markup("\n".join(lines)),
        title="Pipeline Cost & Time",
        border_style="white",
    ))
    console.print()
    _PIPELINE_STATS_RENDERED = True


def display_results(result: dict, output_dir: Optional[str] = None, detail_level: str = "normal"):
    """Display workflow results."""
    console.print()
    console.print("[bold green]>>> Workflow Complete[/bold green]")
    console.print()

    # Show plan
    if "plan" in result:
        from rich.markdown import Markdown

        plan = result["plan"]
        plan_md = plan.get("plan_markdown", "") or ""
        # Same JSON-stashed-as-markdown guard as display_plan_for_approval.
        looks_like_json = plan_md.lstrip().startswith(("{", "["))

        if plan_md and not looks_like_json:
            console.print(Panel(
                Markdown(plan_md),
                title="Simulation Plan",
                border_style="blue",
            ))
        else:
            _render_plan_summary(
                plan,
                outer_title=f"Simulation Plan (type: {plan.get('plan_type', 'unknown')})",
                border_style="blue",
            )
        console.print()

    # Show agent messages (conversation log) - as thinking process
    if "messages" in result and result["messages"]:
        # Agent color mapping
        agent_colors = {
            "PlanningAgent": "blue",
            "CodeGenerationAgent": "green",
            "ReviewAgent": "yellow",
            "ExecutionAgent": "magenta",
        }

        if detail_level == "verbose":
            # Verbose mode: Show as tree structure
            console.print("[bold cyan]Agent Thinking Process[/bold cyan]")
            console.print()

            tree = Tree("Workflow Execution", guide_style="dim")

            # Group messages by agent
            agents_msgs = {}
            for msg in result["messages"]:
                if hasattr(msg, 'agent'):
                    agent = msg.agent
                    content = msg.content
                    metadata = msg.metadata
                    timestamp = msg.timestamp
                else:
                    agent = msg.get("agent", "Unknown")
                    content = msg.get("content", "")
                    metadata = msg.get("metadata", {})
                    timestamp = msg.get("timestamp", "")

                if agent not in agents_msgs:
                    agents_msgs[agent] = []
                agents_msgs[agent].append({
                    "content": content,
                    "metadata": metadata,
                    "timestamp": timestamp
                })

            # Build tree
            for agent, msgs in agents_msgs.items():
                color = agent_colors.get(agent, "white")
                agent_branch = tree.add(f"[{color}]{agent}[/{color}] ({len(msgs)} actions)")

                for msg_data in msgs:
                    timestamp = msg_data["timestamp"]
                    if timestamp:
                        time_str = timestamp.split("T")[1].split(".")[0] if "T" in timestamp else timestamp
                    else:
                        time_str = ""

                    content = msg_data["content"]
                    metadata = msg_data["metadata"]

                    # Create message node
                    msg_text = f"[dim]{time_str}[/dim] {content}"
                    msg_node = agent_branch.add(msg_text)

                    # Add metadata as leaf
                    if metadata:
                        meta_items = [f"{k}={v}" for k, v in metadata.items()]
                        msg_node.add(f"[dim]{', '.join(meta_items)}[/dim]")

            console.print(tree)
            console.print()
        else:
            # Normal/minimal mode: Show as list
            console.print(Panel(
                "[bold cyan]Agent Communication History[/bold cyan]",
                border_style="cyan"
            ))
            console.print()

            for msg in result["messages"]:
                if hasattr(msg, 'timestamp'):
                    timestamp = msg.timestamp
                    agent = msg.agent
                    content = msg.content
                    metadata = msg.metadata
                else:
                    timestamp = msg.get("timestamp", "")
                    agent = msg.get("agent", "Unknown")
                    content = msg.get("content", "")
                    metadata = msg.get("metadata", {})

                if timestamp:
                    time_str = timestamp.split("T")[1].split(".")[0] if "T" in timestamp else timestamp
                else:
                    time_str = ""

                color = agent_colors.get(agent, "white")

                console.print(
                    f"[dim]{time_str}[/dim] [{color}]{agent}[/{color}]: {content}"
                )

                if metadata and detail_level != "minimal":
                    metadata_items = [f"{key}={value}" for key, value in metadata.items()]
                    if metadata_items:
                        console.print(f"  [dim]-- {', '.join(metadata_items)}[/dim]")

                console.print()

            console.print()

    # Show code
    final_output = result.get("final_output", {})
    if "code" in final_output:
        code = final_output["code"]

        # Save code to file
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(get_settings().visualization_output_path)

        output_path.mkdir(parents=True, exist_ok=True)
        code_file = output_path / "simulation.py"

        with open(code_file, "w") as f:
            f.write(code)

        # In verbose mode, show the code content with syntax highlighting
        if detail_level == "verbose":
            console.print(Panel(
                f"Code saved to: [cyan]{code_file}[/cyan]\n\n"
                f"[dim]Code length: {len(code)} characters, {len(code.splitlines())} lines[/dim]",
                title="Generated Code",
                border_style="green"
            ))
            console.print()

            # Show code with syntax highlighting
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Code Content", border_style="dim"))
            console.print()
        else:
            console.print(Panel(
                f"Code saved to: [cyan]{code_file}[/cyan]",
                title="Generated Code",
                border_style="green"
            ))
            console.print()

        _show_patch_preview(result, detail_level)

    # Show review
    if "review" in final_output:
        review = final_output["review"]
        status = "APPROVED" if review.get("approved") else "REJECTED"
        color = "green" if review.get("approved") else "red"

        console.print(Panel(
            f"[bold]{status}[/bold]\n"
            f"Physics Check: {review.get('physics_check', 'unknown')}\n"
            f"Common Sense: {review.get('common_sense_check', 'unknown')}\n\n"
            f"{review.get('feedback', 'No feedback')}",
            title="Code Review",
            border_style=color
        ))
        console.print()

    # Show execution
    if "execution" in final_output:
        execution = final_output["execution"]
        success = execution.get("success", False)
        status = "SUCCESS" if success else "FAILED"
        color = "green" if success else "red"

        info = (
            f"[bold]{status}[/bold]\n"
            f"Runtime: {execution.get('runtime_seconds', 0):.2f}s\n"
            f"Output Files: {len(execution.get('output_files', []))}"
        )

        if execution.get("error_message"):
            info += f"\n\nError: {execution['error_message']}"

        console.print(Panel(
            info,
            title="Execution Results",
            border_style=color
        ))
        console.print()

        # List output files
        output_files = execution.get("output_files", [])
        if output_files:
            console.print("[bold]Generated Files:[/bold]")
            for file in output_files:
                console.print(f"  - {file}")
            console.print()

    # Show simulation review (post-execution assessment)
    if "simulation_review" in result:
        sim_review = result["simulation_review"]
        approved = sim_review.get("approved", False)
        status = "SUCCESSFUL" if approved else "NEEDS IMPROVEMENT"
        color = "green" if approved else "yellow"

        review_content = [
            f"[bold]{status}[/bold]",
            f"Physics Assessment: {sim_review.get('physics_check', 'unknown')}",
            f"Common Sense Check: {sim_review.get('common_sense_check', 'unknown')}",
            "",
            sim_review.get('feedback', 'No feedback'),
        ]

        # Show issues if any
        issues = sim_review.get('issues', [])
        if issues:
            review_content.append("")
            review_content.append("[bold red]Issues Found:[/bold red]")
            for issue in issues:
                review_content.append(f"  - {issue}")

        # Show recommendations if any
        suggestions = sim_review.get('suggestions', [])
        if suggestions:
            review_content.append("")
            review_content.append("[bold cyan]Recommendations:[/bold cyan]")
            for suggestion in suggestions:
                review_content.append(f"  - {suggestion}")

        console.print(Panel(
            "\n".join(review_content),
            title="Simulation Assessment",
            border_style=color
        ))
        console.print()

    # Always end with the token / time summary so it doesn't scroll off
    # behind the per-section panels above.
    _render_pipeline_stats_panel(result)

    console.print()


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Chrono-Code[/bold] version 0.1.0")
    console.print("Multi-agent PyChrono simulation generator")


@app.command()
def setup_rag():
    """Setup RAG database by indexing PyChrono documentation."""
    console.print("[bold]Setting up RAG database...[/bold]")
    console.print()
    console.print("Please run: [cyan]python scripts/setup_rag.py[/cyan]")


if __name__ == "__main__":
    app()
