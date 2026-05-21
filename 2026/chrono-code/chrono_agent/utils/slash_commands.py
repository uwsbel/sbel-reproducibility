"""
Slash-command dispatcher for the CLI plan-approval loop.

During plan approval (see ``chrono_agent/main.py:_cli_plan_approval``) the
user can type free-form modification text, the word "reject", or empty-to-
approve. This module adds a fourth category: ``/command`` prefixed inputs
that trigger out-of-band operations on the workflow state without going
through the ``modify_plan`` LLM round-trip.

v1 command set (stateless / read-only except ``/retry``):

* ``/help``  — print available commands and the approval conventions.
* ``/dump``  — pretty-print the current workflow state (plan, planning
  flags, step_loop summary).
* ``/diff``  — show a structured diff between the last pre-modification
  snapshot and the current plan (reuses ``_compute_plan_diff`` from
  ``main.py``).
* ``/retry`` — set the regeneration flag and exit the approval loop so the
  engine re-runs the planning phase from scratch.
* ``/fast``  — toggle ``state['_fast_mode']`` so the next codegen iteration
  picks up the cheaper ``AGENT2_FAST_MODEL`` (codegen agent checks this).

Handlers return ``(action, new_state)`` where ``action`` is one of:

* ``"stay"``     — redraw prompt and continue the modify loop.
* ``"exit"``     — return the state to the engine (used by ``/retry``).
* ``"unknown"``  — no matching command; caller prints an error and stays.

Keeping the dispatcher free of ``main.py`` coupling makes it trivially
unit-testable — every handler accepts dict-only inputs and returns tuples.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple

# Slash-command exit codes for the CLI loop to act on.
ACTION_STAY = "stay"
ACTION_EXIT = "exit"
ACTION_UNKNOWN = "unknown"

# Type alias for handler signature.
SlashHandler = Callable[
    [str, Dict[str, Any], "SlashContext"], Tuple[str, Dict[str, Any]]
]


class SlashContext:
    """Bundle of optional helpers the dispatcher can pass to handlers.

    ``main.py`` builds one of these per approval turn so handlers can
    access the console, the pre-modification plan snapshot, and the diff
    renderer without reaching back into the CLI module.
    """

    def __init__(
        self,
        *,
        console: Any = None,
        pre_modify_plan: Optional[Dict[str, Any]] = None,
        compute_plan_diff: Optional[Callable[..., Any]] = None,
        render_plan_diff_panel: Optional[Callable[..., None]] = None,
    ) -> None:
        self.console = console
        self.pre_modify_plan = pre_modify_plan
        self.compute_plan_diff = compute_plan_diff
        self.render_plan_diff_panel = render_plan_diff_panel

    def print(self, msg: str) -> None:
        """Best-effort print that works with both Rich console and stdlib."""
        if self.console is not None and hasattr(self.console, "print"):
            self.console.print(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------


def _handle_help(arg: str, state: Dict[str, Any], ctx: SlashContext) -> Tuple[str, Dict[str, Any]]:
    lines = [
        "[bold]Plan approval commands[/bold]",
        "  [green]<enter>[/green]       approve the current plan",
        "  [yellow]<text>[/yellow]      type any non-command text → LLM modifies the plan",
        "  [red]reject[/red]        cancel this run",
        "",
        "[bold]Slash commands[/bold]",
        "  [cyan]/help[/cyan]         show this message",
        "  [cyan]/dump[/cyan]         pretty-print the current workflow state",
        "  [cyan]/diff[/cyan]         diff of last pre-modify snapshot vs. current plan",
        "  [cyan]/retry[/cyan]        force planning to regenerate from scratch",
        "  [cyan]/fast[/cyan]         toggle fast mode (use cheaper model in next codegen)",
    ]
    ctx.print("\n".join(lines))
    return ACTION_STAY, state


def _handle_dump(arg: str, state: Dict[str, Any], ctx: SlashContext) -> Tuple[str, Dict[str, Any]]:
    # Keep the dump compact: plan summary + planning flags + step_loop
    # headline. A full dump would flood the terminal; the user can grep
    # `dialog/sessions/...` for the raw state if needed.
    plan = state.get("plan") or {}
    summary: Dict[str, Any] = {
        "plan_type": plan.get("plan_type"),
        "sim_params": plan.get("simulation_parameters"),
        "implementation_steps": plan.get("implementation_steps", [])[:3],
        "milestone_count": len(plan.get("milestones") or []),
        "assets_count": len(plan.get("assets") or []),
        "clarifications_needed": plan.get("clarifications_needed") or [],
    }
    step_loop = state.get("step_loop") or {}
    loop_headline = {
        k: step_loop.get(k)
        for k in ("current_step_index", "current_milestone_id", "all_steps_complete")
        if k in step_loop
    }
    payload = {
        "plan_summary": summary,
        "planning_flags": state.get("planning") or {},
        "step_loop": loop_headline,
        "fast_mode": state.get("_fast_mode", False),
    }
    try:
        dumped = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 — fall back to repr so /dump never crashes
        dumped = f"(dump failed: {exc})\n{payload!r}"
    ctx.print("[bold cyan]Workflow state[/bold cyan]")
    ctx.print(dumped)
    return ACTION_STAY, state


def _handle_diff(arg: str, state: Dict[str, Any], ctx: SlashContext) -> Tuple[str, Dict[str, Any]]:
    if ctx.pre_modify_plan is None:
        ctx.print(
            "[yellow]No pre-modification snapshot yet.[/yellow] "
            "Type a modification first, then /diff will show what changed."
        )
        return ACTION_STAY, state
    if ctx.compute_plan_diff is None or ctx.render_plan_diff_panel is None:
        ctx.print("[red]Diff helper not wired up.[/red]")
        return ACTION_STAY, state
    try:
        diff = ctx.compute_plan_diff(ctx.pre_modify_plan, state.get("plan") or {})
        ctx.render_plan_diff_panel(diff)
    except Exception as exc:  # noqa: BLE001 — diff must never block the loop
        ctx.print(f"[red]Diff rendering failed: {exc}[/red]")
    return ACTION_STAY, state


def _handle_retry(arg: str, state: Dict[str, Any], ctx: SlashContext) -> Tuple[str, Dict[str, Any]]:
    planning = dict(state.get("planning") or {})
    planning["plan_needs_regeneration"] = True
    planning["plan_approved"] = False
    planning["plan_rejected"] = False
    new_state = dict(state)
    new_state["planning"] = planning
    ctx.print("[bold yellow]>>> Regenerating plan from scratch[/bold yellow]")
    return ACTION_EXIT, new_state


def _handle_fast(arg: str, state: Dict[str, Any], ctx: SlashContext) -> Tuple[str, Dict[str, Any]]:
    new_state = dict(state)
    current = bool(state.get("_fast_mode", False))
    new_state["_fast_mode"] = not current
    status = "ON" if new_state["_fast_mode"] else "OFF"
    ctx.print(
        f"[bold]Fast mode {status}.[/bold] Next codegen iteration will "
        f"{'use the AGENT2_FAST_MODEL when set' if new_state['_fast_mode'] else 'use the default model'}."
    )
    return ACTION_STAY, new_state


HANDLERS: Dict[str, SlashHandler] = {
    "help": _handle_help,
    "?": _handle_help,
    "dump": _handle_dump,
    "state": _handle_dump,
    "diff": _handle_diff,
    "retry": _handle_retry,
    "regen": _handle_retry,
    "fast": _handle_fast,
}


def dispatch_slash_command(
    raw_input: str,
    state: Dict[str, Any],
    ctx: Optional[SlashContext] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Parse and dispatch a slash command.

    Args:
        raw_input: The user's typed string. Accepts either with or without
            a leading ``/``; callers typically pre-check for ``startswith("/")``.
        state: The current workflow state dict (treated as immutable in
            read-only commands; cloned by handlers that mutate).
        ctx: Optional ``SlashContext`` with CLI rendering helpers. A
            ``SlashContext()`` instance with all fields None is used when
            omitted — useful for tests.

    Returns:
        ``(action, new_state)`` — see module docstring for actions.
    """
    ctx = ctx or SlashContext()
    text = raw_input.strip()
    if text.startswith("/"):
        text = text[1:]
    cmd, _, arg = text.partition(" ")
    cmd = cmd.strip().lower()
    if not cmd:
        ctx.print("[red]Empty slash command. Try /help.[/red]")
        return ACTION_UNKNOWN, state
    handler = HANDLERS.get(cmd)
    if handler is None:
        ctx.print(
            f"[red]Unknown command: /{cmd}.[/red] Type [cyan]/help[/cyan] "
            "for the list."
        )
        return ACTION_UNKNOWN, state
    return handler(arg.strip(), state, ctx)
