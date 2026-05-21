"""Event-driven Rich renderer that approximates the Claude Code terminal UX.

Listens to the workflow event stream (``chrono_code.workflow.events``) and
produces per-agent inline blocks plus a sticky bottom status line:

    ────── PlanningAgent · openai/gpt-5.2 ──────
      ▸ find_files('.obj', 'data/scene/')
        → 23 files
      ✎ The outdoor props live under sensor/offroad/...
      ...streamed draft text lands here, token by token...
      ✓ PlanningAgent done · 58.1s

    ⠙ planning · awaiting approval · 1m08s

Design notes:

* Completed blocks go via ``console.print`` so they enter scrollback.
  Only the status line uses Rich ``Live`` (non-transient so the history
  survives).
* Streaming text is held as a mutable ``Text`` and swapped from the
  ``Live`` region to ``console.print`` when the stream ends — this keeps
  the partial output visible without hijacking a terminal line.
* ``pause()`` / ``resume()`` bracket blocking user input (plan approval
  callback) so the Live refresh doesn't clobber the prompt.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.rule import Rule
from rich.text import Text

from chrono_code.workflow import events as ev


# --- style constants ------------------------------------------------------

_ICON_TOOL_START = "▸"
_ICON_TOOL_RESULT = "→"
_ICON_TOOL_ERROR = "✗"
_ICON_THINKING = "✎"
_ICON_AGENT_DONE = "✓"

# Per-agent accent colors — stable order so output looks consistent run to run.
_AGENT_COLORS = {
    "PlanningAgent": "cyan",
    "CodeGenerationAgent": "green",
    "ExecutionAgent": "magenta",
    "ReviewAgent": "yellow",
    "VLMReviewAgent": "yellow",
    "ReplanAgent": "bright_cyan",
}
_DEFAULT_AGENT_COLOR = "white"


def _fmt_elapsed(seconds: float) -> str:
    """``1m08s`` / ``42.3s`` style short elapsed string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m{secs:02d}s"


def _fmt_tokens(n: int) -> str:
    """Compact token count: ``742``, ``12.3k``, ``1.04M``."""
    n = int(n or 0)
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def _fmt_usage_inline(usage: Dict[str, Any]) -> str:
    """One-line summary of a usage dict for an end-of-session marker."""
    if not usage:
        return ""
    inp = int(usage.get("input", 0) or 0)
    out = int(usage.get("output", 0) or 0)
    cache_r = int(usage.get("cache_read", 0) or 0)
    cache_c = int(usage.get("cache_creation", 0) or 0)
    total = inp + out + cache_r + cache_c
    if total == 0:
        return ""
    parts = [f"in {_fmt_tokens(inp)}", f"out {_fmt_tokens(out)}"]
    if cache_r:
        parts.append(f"cache {_fmt_tokens(cache_r)}")
    if cache_c:
        parts.append(f"cache+ {_fmt_tokens(cache_c)}")
    parts.append(f"Σ {_fmt_tokens(total)}")
    return " · ".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class LiveEventRenderer:
    """Consume workflow events and render a Claude-Code-style terminal view."""

    def __init__(
        self,
        console: Console,
        *,
        refresh_per_second: int = 8,
        start_live: bool = True,
    ):
        self.console = console
        self._refresh = refresh_per_second
        self._start_live = start_live

        # Lifecycle / state
        self._live: Optional[Live] = None
        self._started = False
        self._paused = False
        self._prev_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._workflow_start: float = 0.0

        # Per-agent state (keyed by agent name)
        self._agent_starts: Dict[str, float] = {}
        self._agent_headered: Dict[str, bool] = {}

        # Current streaming text block (only one LLM stream active at a time)
        self._stream_agent: Optional[str] = None
        self._stream_text: Optional[Text] = None  # mutable, printed on end

        # Sticky status line state
        self._phase: str = "workflow"
        self._step: str = "starting"
        self._status_detail: str = ""

    # --- lifecycle --------------------------------------------------------

    def __enter__(self) -> "LiveEventRenderer":
        self._workflow_start = time.time()
        self._prev_callback = ev._event_callback  # snapshot in case nested
        ev.set_event_callback(self.handle)
        if self._start_live:
            self._live = Live(
                self._render_status(),
                console=self.console,
                refresh_per_second=self._refresh,
                transient=False,
                auto_refresh=True,
            )
            self._live.start()
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Final landing of any in-flight stream
        if self._stream_text is not None and self._stream_agent is not None:
            self._finalize_stream()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
        ev.set_event_callback(self._prev_callback)
        self._started = False

    def pause(self) -> None:
        """Stop the Live refresh so blocking user input isn't clobbered."""
        if self._paused:
            return
        self._paused = True
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass

    def resume(self) -> None:
        if not self._paused:
            return
        self._paused = False
        if self._live is not None:
            try:
                self._live.start()
            except Exception:
                pass

    # --- main dispatch ----------------------------------------------------

    def handle(self, event: Dict[str, Any]) -> None:
        """Route an event to its per-type handler. Never raises to the caller."""
        try:
            etype = event.get("type") or event.get("event_type")
            if etype == "agent_lifecycle":
                self._on_lifecycle(event)
            elif etype == "tool_call":
                self._on_tool_call(event)
            elif etype == "agent_thinking":
                self._on_thinking(event)
            elif etype == "llm_stream_start":
                self._on_stream_start(event)
            elif etype == "llm_text_delta":
                self._on_text_delta(event)
            elif etype == "llm_stream_end":
                self._on_stream_end(event)
            elif etype in ("progress_event", "progress"):
                self._on_progress(event)
            elif etype == "pipeline_stats":
                self._on_pipeline_stats(event)
            # Unknown event types are ignored — the event stream is best-effort.
            self._refresh_status()
        except Exception as exc:
            # Never let rendering bugs crash the workflow. Surface via stderr.
            try:
                self.console.print(f"[dim red][renderer error] {exc}[/dim red]")
            except Exception:
                pass

    # --- per-event handlers ----------------------------------------------

    def _on_lifecycle(self, e: Dict[str, Any]) -> None:
        agent = str(e.get("agent") or "")
        state = str(e.get("state") or "")
        if not agent:
            return
        if state == "started":
            if self._agent_headered.get(agent):
                return  # dedupe repeated starts
            self._agent_starts[agent] = time.time()
            self._agent_headered[agent] = True
            model = str(e.get("model") or "")
            provider = str(e.get("provider") or "")
            suffix = ""
            if provider and model:
                suffix = f" · {provider}/{model}"
            elif model:
                suffix = f" · {model}"
            color = _AGENT_COLORS.get(agent, _DEFAULT_AGENT_COLOR)
            rule = Rule(
                Text(f"{agent}{suffix}", style=f"bold {color}"),
                style=color,
            )
            self._emit(rule)
        elif state == "finished":
            started = self._agent_starts.pop(agent, None)
            elapsed = float(e.get("elapsed") or 0.0)
            if not elapsed and started is not None:
                elapsed = time.time() - started
            color = _AGENT_COLORS.get(agent, "green")
            usage = e.get("usage") or {}
            calls = int(e.get("calls") or 0)
            turns = int(e.get("turns") or 0)
            kind = str(e.get("session_kind") or "")
            usage_str = _fmt_usage_inline(usage)

            tail_bits = [f"[dim]{_fmt_elapsed(elapsed)}[/dim]"]
            if usage_str:
                tail_bits.append(f"[dim]{_escape_markup(usage_str)}[/dim]")
            meta_bits = []
            if calls:
                meta_bits.append(f"{calls} call{'s' if calls != 1 else ''}")
            if turns:
                meta_bits.append(f"{turns} turn{'s' if turns != 1 else ''}")
            if kind == "tool_loop" and not turns:
                meta_bits.append("tool_loop")
            elif kind == "invoke_llm" and not calls:
                meta_bits.append("invoke_llm")
            if meta_bits:
                tail_bits.append(f"[dim]({', '.join(meta_bits)})[/dim]")
            self._emit(
                Text.from_markup(
                    f"  [{color}]{_ICON_AGENT_DONE}[/{color}] "
                    f"[bold]{agent}[/bold] done · "
                    + " · ".join(tail_bits)
                )
            )
            # Blank line for visual separation between agent blocks
            self._agent_headered.pop(agent, None)
            self._emit(Text(""))

    def _on_tool_call(self, e: Dict[str, Any]) -> None:
        status = str(e.get("status") or "")
        tool_name = str(e.get("tool_name") or "?")
        if status == "start":
            args = e.get("tool_args") or {}
            args_preview = _truncate(_format_tool_args(args), 120)
            self._emit(
                Text.from_markup(
                    f"  [dim]{_ICON_TOOL_START}[/dim] "
                    f"[bold]{tool_name}[/bold][dim]({args_preview})[/dim]"
                )
            )
        elif status == "complete":
            preview = _truncate(str(e.get("result_preview") or ""), 200)
            if preview:
                self._emit(
                    Text.from_markup(
                        f"    [green]{_ICON_TOOL_RESULT}[/green] "
                        f"[dim]{_escape_markup(preview)}[/dim]"
                    )
                )
        elif status in ("error", "blocked"):
            preview = _truncate(str(e.get("result_preview") or ""), 200)
            self._emit(
                Text.from_markup(
                    f"    [red]{_ICON_TOOL_ERROR}[/red] "
                    f"[red]{_escape_markup(preview) or status}[/red]"
                )
            )

    def _on_thinking(self, e: Dict[str, Any]) -> None:
        content = _truncate(str(e.get("content") or "").strip(), 400)
        if not content:
            return
        self._emit(
            Text.from_markup(
                f"  [dim]{_ICON_THINKING}[/dim] [italic dim]{_escape_markup(content)}[/italic dim]"
            )
        )

    def _on_stream_start(self, e: Dict[str, Any]) -> None:
        agent = str(e.get("agent") or "")
        # Finalize any previous hanging stream first (safety)
        if self._stream_text is not None:
            self._finalize_stream()
        self._stream_agent = agent
        color = _AGENT_COLORS.get(agent, _DEFAULT_AGENT_COLOR)
        self._stream_text = Text("  ", style=color)

    def _on_text_delta(self, e: Dict[str, Any]) -> None:
        if self._stream_text is None:
            # Start implicitly if someone forgot a stream_start
            self._on_stream_start({"agent": e.get("agent") or ""})
        delta = str(e.get("delta") or "")
        if delta and self._stream_text is not None:
            self._stream_text.append(delta)

    def _on_stream_end(self, e: Dict[str, Any]) -> None:
        self._finalize_stream()

    def _finalize_stream(self) -> None:
        if self._stream_text is None:
            return
        # Clear state BEFORE emitting so _emit's "finalize in-flight stream"
        # guard doesn't recurse back into this method.
        text = self._stream_text
        self._stream_text = None
        self._stream_agent = None
        if len(text.plain.strip()) > 0:
            self._emit(text)

    def _on_pipeline_stats(self, e: Dict[str, Any]) -> None:
        """Render the workflow-end token / time summary block."""
        elapsed = float(e.get("elapsed") or 0.0)
        usage = e.get("usage") or {}
        per_agent = e.get("per_agent") or {}
        sessions = int(e.get("sessions") or 0)
        calls = int(e.get("calls") or 0)

        self._emit(Rule(Text("Pipeline summary", style="bold white"), style="white"))

        usage_str = _fmt_usage_inline(usage)
        header_bits = [f"[bold]total[/bold] · [dim]{_fmt_elapsed(elapsed)}[/dim]"]
        if usage_str:
            header_bits.append(f"[dim]{_escape_markup(usage_str)}[/dim]")
        meta = []
        if sessions:
            meta.append(f"{sessions} session{'s' if sessions != 1 else ''}")
        if calls:
            meta.append(f"{calls} call{'s' if calls != 1 else ''}")
        if meta:
            header_bits.append(f"[dim]({', '.join(meta)})[/dim]")
        self._emit(Text.from_markup("  " + " · ".join(header_bits)))

        # Per-agent breakdown, sorted by total tokens descending so the
        # heaviest agent surfaces first.
        def _agent_total(entry: Dict[str, Any]) -> int:
            u = entry.get("usage") or {}
            return sum(int(u.get(k, 0) or 0) for k in ("input", "output", "cache_read", "cache_creation"))

        for name, entry in sorted(per_agent.items(), key=lambda kv: _agent_total(kv[1]), reverse=True):
            color = _AGENT_COLORS.get(name, _DEFAULT_AGENT_COLOR)
            agent_usage_str = _fmt_usage_inline(entry.get("usage") or {})
            ag_elapsed = float(entry.get("elapsed") or 0.0)
            ag_sessions = int(entry.get("sessions") or 0)
            ag_calls = int(entry.get("calls") or 0)
            row_bits = [
                f"[{color}]•[/{color}] [bold]{_escape_markup(name)}[/bold]",
                f"[dim]{_fmt_elapsed(ag_elapsed)}[/dim]",
            ]
            if agent_usage_str:
                row_bits.append(f"[dim]{_escape_markup(agent_usage_str)}[/dim]")
            sub = []
            if ag_sessions:
                sub.append(f"{ag_sessions} session{'s' if ag_sessions != 1 else ''}")
            if ag_calls:
                sub.append(f"{ag_calls} call{'s' if ag_calls != 1 else ''}")
            if sub:
                row_bits.append(f"[dim]({', '.join(sub)})[/dim]")
            self._emit(Text.from_markup("  " + " · ".join(row_bits)))

        self._emit(Text(""))

    def _on_progress(self, e: Dict[str, Any]) -> None:
        # Track phase/step for the sticky status line only; the event log
        # already contains per-agent detail.
        phase = str(e.get("phase") or self._phase)
        step = str(e.get("step") or "")
        message = str(e.get("message") or "")
        self._phase = phase
        self._step = step
        self._status_detail = _truncate(message, 80)

    # --- emit + status line -----------------------------------------------

    def _emit(self, renderable) -> None:
        """Print a completed block above the Live status region.

        Rich auto-coordinates ``console.print`` with an active ``Live``
        (it temporarily hides the live region, prints to scrollback, then
        redraws), so we unconditionally route through the shared console.
        """
        # If any stream is in-flight, finalize it first so the new block
        # doesn't interleave with streaming text.
        if self._stream_text is not None:
            self._finalize_stream()
        self.console.print(renderable)

    def _render_status(self):
        elapsed = time.time() - self._workflow_start if self._workflow_start else 0.0
        parts = [f"[bold]{self._phase}[/bold]"]
        if self._step:
            parts.append(f"[dim]{self._step}[/dim]")
        if self._status_detail:
            parts.append(f"[dim]{_escape_markup(self._status_detail)}[/dim]")
        parts.append(f"[dim]· {_fmt_elapsed(elapsed)}[/dim]")
        status = Text.from_markup(" · ".join(parts))

        if self._stream_text is not None:
            # Show the in-flight stream above the status line so user sees
            # partial tokens land in real time.
            return Group(self._stream_text, status)
        return status

    def _refresh_status(self) -> None:
        if self._live is not None and not self._paused:
            try:
                self._live.update(self._render_status())
            except Exception:
                pass


# --- helpers ---------------------------------------------------------------


def _format_tool_args(args: Dict[str, Any]) -> str:
    """Compact one-line arg preview: ``key=value, key2='string'``."""
    if not isinstance(args, dict) or not args:
        return ""
    bits = []
    for k, v in args.items():
        if isinstance(v, str):
            bits.append(f"{k}={v!r}")
        elif isinstance(v, (list, tuple)):
            bits.append(f"{k}=[{len(v)} items]" if len(v) > 3 else f"{k}={list(v)!r}")
        elif isinstance(v, dict):
            bits.append(f"{k}={{{len(v)} keys}}")
        else:
            bits.append(f"{k}={v}")
    return ", ".join(bits)


def _escape_markup(s: str) -> str:
    """Escape Rich markup so user-provided strings render literally."""
    return s.replace("[", r"\[")
