"""Simple callback-based event system replacing LangGraph streaming."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

EventCallback = Callable[[Dict[str, Any]], None]
_event_callback: Optional[EventCallback] = None


def set_event_callback(cb: Optional[EventCallback]) -> None:
    """Register (or clear) the global event callback."""
    global _event_callback
    _event_callback = cb


def emit_custom_event(payload: Dict[str, Any]) -> bool:
    """Emit a custom event via the registered callback, if any."""
    if _event_callback:
        _event_callback(payload)
        return True
    return False


def emit_progress_event(
    phase: str,
    step: str,
    progress_pct: int,
    message: str | None = None,
    **payload: Any,
) -> bool:
    """Emit a normalized progress event for stream consumers."""
    event: Dict[str, Any] = {
        "type": "progress_event",
        "name": f"{phase}.{step}",
        "event_type": "progress",
        "phase": phase,
        "step": step,
        "progress_pct": int(progress_pct),
    }
    if message:
        event["message"] = message
    if payload:
        event.update(payload)
    event["payload"] = {
        key: value
        for key, value in event.items()
        if key not in {"type", "name", "payload"}
    }
    return emit_custom_event(event)


def emit_subprocess_event(pipe: str, line: str) -> bool:
    """Emit a subprocess output line (stdout/stderr) as a custom event."""
    return emit_custom_event({
        "type": "subprocess_output",
        "pipe": pipe,
        "line": line,
    })


def emit_frame_event(path: str, frame_idx: int) -> bool:
    """Emit a notification that a new frame is available."""
    return emit_custom_event({
        "type": "frame_available",
        "path": path,
        "frame_idx": frame_idx,
    })


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _sanitize_tool_args(args: Dict[str, Any], max_value_chars: int = 200) -> Dict[str, Any]:
    """Truncate large string values in tool args for streaming."""
    sanitized: Dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_value_chars:
            sanitized[key] = _truncate(value, max_value_chars)
        else:
            sanitized[key] = value
    return sanitized


def emit_tool_call_event(
    tool_name: str,
    tool_args: Dict[str, Any],
    status: str,
    result_preview: str | None = None,
    call_index: int = 0,
    loop_iter: int = 0,
) -> bool:
    """Emit a code agent tool call event for the activity feed.

    Args:
        tool_name: Name of the tool being called.
        tool_args: Tool arguments (large values auto-truncated).
        status: One of "start", "complete", "error", "blocked".
        result_preview: Truncated result text (for complete/error/blocked).
        call_index: Index of this call within the current LLM response.
        loop_iter: Tool loop iteration number.
    """
    event: Dict[str, Any] = {
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_args": _sanitize_tool_args(tool_args),
        "status": status,
        "call_index": call_index,
        "loop_iter": loop_iter,
    }
    if result_preview is not None:
        event["result_preview"] = _truncate(str(result_preview), 500)
    return emit_custom_event(event)


def emit_agent_thinking_event(
    agent: str,
    content: str,
    loop_iter: int = 0,
) -> bool:
    """Emit an agent thinking/reasoning event for the activity feed.

    Args:
        agent: Name of the agent (e.g. "CodeGenerationAgent").
        content: The LLM's reasoning text.
        loop_iter: Tool loop iteration number.
    """
    return emit_custom_event({
        "type": "agent_thinking",
        "agent": agent,
        "content": _truncate(content, 2000),
        "loop_iter": loop_iter,
    })


def emit_agent_lifecycle_event(
    agent: str,
    state: str,
    model: str = "",
    provider: str = "",
    elapsed: float = 0.0,
    usage: Optional[Dict[str, int]] = None,
    calls: int = 0,
    turns: int = 0,
    session_kind: str = "",
) -> bool:
    """Emit an agent start/finish marker for the activity feed.

    Args:
        agent: Name of the agent (e.g. "PlanningAgent").
        state: One of "started", "finished".
        model: Model identifier (e.g. "claude-opus-4-7").
        provider: Provider name (e.g. "anthropic", "openai").
        elapsed: For "finished" events, seconds since the agent started.
        usage: For "finished" events, per-session token-usage diff
            ``{"input", "output", "cache_read", "cache_creation"}``.
        calls: Number of LLM calls inside this session (raw API turns,
            including invoke_llm retries and tool-loop streamed turns).
        turns: Tool-loop turn count when applicable; 0 for plain invoke_llm.
        session_kind: ``"tool_loop"`` or ``"invoke_llm"``; helps the
            engine/renderer distinguish the two emission paths.
    """
    payload: Dict[str, Any] = {
        "type": "agent_lifecycle",
        "agent": agent,
        "state": state,
        "model": model,
        "provider": provider,
        "elapsed": float(elapsed),
    }
    if usage:
        payload["usage"] = {
            "input": int(usage.get("input", 0) or 0),
            "output": int(usage.get("output", 0) or 0),
            "cache_read": int(usage.get("cache_read", 0) or 0),
            "cache_creation": int(usage.get("cache_creation", 0) or 0),
        }
    if calls:
        payload["calls"] = int(calls)
    if turns:
        payload["turns"] = int(turns)
    if session_kind:
        payload["session_kind"] = session_kind
    return emit_custom_event(payload)


def emit_pipeline_stats_event(
    elapsed: float,
    usage: Dict[str, int],
    per_agent: Dict[str, Dict[str, Any]],
    sessions: int = 0,
    calls: int = 0,
) -> bool:
    """Emit a pipeline-level token / time summary at workflow end.

    Args:
        elapsed: Wall-clock seconds for the whole multi-agent pipeline.
        usage: Aggregated token usage across all agent sessions
            ``{"input", "output", "cache_read", "cache_creation"}``.
        per_agent: Breakdown keyed by agent name, each entry containing
            ``{"elapsed", "usage", "calls", "sessions"}``.
        sessions: Number of agent sessions observed in the pipeline.
        calls: Total LLM calls across all sessions.
    """
    return emit_custom_event({
        "type": "pipeline_stats",
        "elapsed": float(elapsed),
        "usage": {
            "input": int(usage.get("input", 0) or 0),
            "output": int(usage.get("output", 0) or 0),
            "cache_read": int(usage.get("cache_read", 0) or 0),
            "cache_creation": int(usage.get("cache_creation", 0) or 0),
        },
        "per_agent": per_agent,
        "sessions": int(sessions),
        "calls": int(calls),
    })


def emit_llm_stream_start_event(agent: str, loop_iter: int = 0) -> bool:
    """Emit when a streaming LLM call begins (one brackets each assistant turn)."""
    return emit_custom_event({
        "type": "llm_stream_start",
        "agent": agent,
        "loop_iter": loop_iter,
    })


def emit_llm_text_delta_event(agent: str, delta: str, loop_iter: int = 0) -> bool:
    """Emit a single token/chunk of streamed assistant text."""
    return emit_custom_event({
        "type": "llm_text_delta",
        "agent": agent,
        "delta": delta,
        "loop_iter": loop_iter,
    })


def emit_llm_stream_end_event(
    agent: str,
    final_text: str = "",
    loop_iter: int = 0,
) -> bool:
    """Emit when a streaming LLM call finishes.

    ``final_text`` is truncated (like thinking events) so downstream consumers
    can fall back to it without having to reconstruct from deltas.
    """
    return emit_custom_event({
        "type": "llm_stream_end",
        "agent": agent,
        "final_text": _truncate(final_text, 2000),
        "loop_iter": loop_iter,
    })
