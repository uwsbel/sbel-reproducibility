"""
UserSpec: deterministic pre-extraction of user-specified simulation constraints.

The PlanningAgent runs this over the raw user prompt BEFORE any LLM call so
that numeric / structural facts the user named explicitly (duration, time
step, scene size, required assets) survive every downstream handoff intact.
Without this the two-phase memo->JSON design would paraphrase numerics into
prose and silent defaults would fill in placeholder values.

Regex-first: covers the common phrasings ("30s", "dt=0.001", "50m x 50m")
without any LLM round-trip. The LLM fallback only fires when regex finds
nothing at all.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class UserSpec(BaseModel):
    """User-specified simulation constraints pulled deterministically from the prompt.

    Every field is optional; ``None`` means "not specified by the user". Downstream
    code MUST NOT silently substitute a default for a ``None`` field without
    surfacing the fact — that is the exact failure mode this model prevents.
    """

    duration_s: Optional[float] = Field(
        default=None,
        description="Simulation duration in seconds, when the user named one",
    )
    time_step_s: Optional[float] = Field(
        default=None,
        description="Integration time step in seconds, when the user named one",
    )
    scene_size_m: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Scene footprint in meters (x, y), when the user named one",
    )
    required_assets: List[str] = Field(
        default_factory=list,
        description="Asset tokens the user named explicitly (populated elsewhere)",
    )

    def any_numeric_set(self) -> bool:
        return (
            self.duration_s is not None
            or self.time_step_s is not None
            or self.scene_size_m is not None
        )

    def render_for_prompt(self) -> str:
        """Format as a short ``key: value`` block for inclusion in a system prompt.

        Keys the user did not specify are rendered as ``not specified`` so the
        LLM knows to pick a reasonable default — as opposed to guessing whether
        the field exists at all.
        """

        def _fmt(x: object) -> str:
            return "not specified" if x is None else str(x)

        scene = self.scene_size_m
        scene_str = "not specified" if scene is None else f"{scene[0]} x {scene[1]} m"

        lines = [
            f"- duration_s: {_fmt(self.duration_s)}",
            f"- time_step_s: {_fmt(self.time_step_s)}",
            f"- scene_size_m: {scene_str}",
        ]
        if self.required_assets:
            lines.append(f"- required_assets: {', '.join(self.required_assets)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex extractors (deterministic, zero-LLM path)
# ---------------------------------------------------------------------------

# Matches a plain number optionally followed by one of the scientific-notation
# forms we see in physics specs: "0.001", "1e-3", "1E-3", "5e-4".
_NUMBER = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def _extract_time_step(text: str) -> Optional[float]:
    """Find a "dt=..." / "time step ..." / "timestep ..." value.

    Requires a time-step context word before the number so a bare "0.001"
    buried in unrelated prose doesn't get misread as dt. We accept:
        dt = 1e-3
        dt: 0.001
        time step is 0.001
        time_step = 0.5e-3
        timestep 0.001
        Δt 1e-3
    """
    patterns = [
        r"(?:d\s*t|delta[_\s]*t|Δ\s*t|time[_\s]*step|timestep)"
        r"\s*(?:is|=|:|of|to|,)?\s*"
        + _NUMBER
        + r"\s*(?:s|sec|secs|second|seconds)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_duration(text: str) -> Optional[float]:
    """Find a simulation-duration value in seconds.

    Handles:
        duration is 30s
        simulate for 30 seconds
        run for 10 s
        duration=30
        30-second simulation
        over 10 s

    Deliberately REJECTS numbers that look like time steps (preceded by
    dt/timestep keywords) — those are handled by ``_extract_time_step``.
    Also ignores bare "30s" when it's immediately after a time-step context
    word (so "time step 0.001 s" doesn't double-count).
    """
    # Strategy 1: "duration", "simulation_duration", "run for", "simulate for"
    # followed by a number. Most explicit, tried first.
    explicit_patterns = [
        r"(?:simulation[_\s]*duration|duration)\s*(?:is|=|:|of|to|,)?\s*"
        + _NUMBER + r"\s*(?:s|sec|secs|second|seconds)?\b",
        r"(?:simulate|run|simulation)\s+(?:for|over)\s+" + _NUMBER
        + r"\s*(?:s|sec|secs|second|seconds)\b",
        r"\bover\s+" + _NUMBER + r"\s*(?:s|sec|secs|second|seconds)\b",
        # "30-second simulation", "30s simulation"
        _NUMBER + r"\s*[-\s]?(?:s|sec|secs|second|seconds)\s+(?:simulation|run|sim)\b",
    ]
    for pat in explicit_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if val > 0:
                return val

    # Strategy 2: "<n> seconds" not inside a time-step context. Scan each
    # match and check the 24 characters of left-context.
    ts_context = re.compile(
        r"(?:d\s*t|delta[_\s]*t|Δ\s*t|time[_\s]*step|timestep)\b",
        re.IGNORECASE,
    )
    for m in re.finditer(
        _NUMBER + r"\s*(?:s|sec|secs|second|seconds)\b", text, re.IGNORECASE
    ):
        left = text[max(0, m.start() - 24): m.start()]
        if ts_context.search(left):
            continue  # this number is a time step, not a duration
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        # Reject implausibly small (looks like a time step) when no context
        # word surrounded it — < 0.5 s is almost never a sim duration.
        if val >= 0.5:
            return val
    return None


def _extract_scene_size(text: str) -> Optional[Tuple[float, float]]:
    """Find a 2D scene footprint like "50m x 50m", "10 x 20 meters", "50 by 50 m"."""
    patterns = [
        # "50m x 50m", "50 m x 50 m", "50mx50m"
        _NUMBER + r"\s*m\s*[x×*]\s*" + _NUMBER + r"\s*m\b",
        # "50 x 50 m" / "50x50 m"
        _NUMBER + r"\s*[x×*]\s*" + _NUMBER + r"\s*m\b",
        # "50 by 50 meters"
        _NUMBER + r"\s*by\s*" + _NUMBER + r"\s*(?:m|meters?)\b",
        # "50m by 50m"
        _NUMBER + r"\s*m\s*by\s*" + _NUMBER + r"\s*m\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                x = float(m.group(1))
                y = float(m.group(2))
            except ValueError:
                continue
            if x > 0 and y > 0:
                return (x, y)
    return None


def extract_user_spec_regex(user_prompt: str) -> UserSpec:
    """Deterministic regex-only extraction. Always returns a UserSpec (fields None on miss)."""
    text = user_prompt or ""
    return UserSpec(
        duration_s=_extract_duration(text),
        time_step_s=_extract_time_step(text),
        scene_size_m=_extract_scene_size(text),
    )
