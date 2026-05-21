"""
Structured handoff, skill bundle, and error context models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StructuredError(BaseModel):
    """Normalized error payload that can move across workflow stages."""

    error_type: str
    phase: str
    summary: str
    raw_message: str = ""
    operation: Optional[str] = None
    source_location: Optional[str] = None
    retryable: bool = True
    recommended_action: Optional[str] = None
    context_snippet: Optional[str] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_message(
        cls,
        *,
        error_type: str,
        phase: str,
        summary: str,
        raw_message: Optional[str] = None,
        operation: Optional[str] = None,
        retryable: bool = True,
        recommended_action: Optional[str] = None,
        context_snippet: Optional[str] = None,
        signature: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StructuredError":
        return cls(
            error_type=error_type,
            phase=phase,
            summary=summary,
            raw_message=raw_message or summary,
            operation=operation,
            retryable=retryable,
            recommended_action=recommended_action,
            context_snippet=context_snippet,
            signature=signature,
            metadata=metadata or {},
        )


class FailureContext(BaseModel):
    """Compact failure handoff for the next LLM call.

    This block describes the current failure only.
    Prior-attempt memory must live outside this model.
    """

    source: str
    summary: str
    structured_error: Optional[StructuredError] = None
    recent_feedback: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PriorError(BaseModel):
    """A single prior error with fingerprint for semantic dedup."""

    fingerprint: str
    summary: str
    phase: str = "execution"
    timestamp: str = ""
    fingerprint_group: Optional[str] = None
    raw_message: str = ""


class AttemptedFix(BaseModel):
    """Record of a fix attempt and its outcome."""

    original_fingerprint: str
    fix_action: str
    outcome: str = "unknown"
    new_fingerprint: Optional[str] = None
    timestamp: str = ""


class HistoryContext(BaseModel):
    """Prior-attempt memory kept separate from the current failure."""

    summary: str = ""
    cycle_detected: bool = False
    recent_attempts_available: bool = False
    repeated_error_summaries: List[str] = Field(default_factory=list)
    do_not_repeat: List[str] = Field(default_factory=list)
    prior_errors: List[PriorError] = Field(default_factory=list)
    attempted_fixes: List[AttemptedFix] = Field(default_factory=list)


class SkillBundle(BaseModel):
    """Bundle of selected skills for one LLM call."""

    name: str = "selected"
    skills: List[str] = Field(default_factory=list)
    summary: str = ""
    full_text: str = ""
    section_index: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMHandoff(BaseModel):
    """Explicit context packet shared across LLM calls."""

    task_intent: str = ""
    input_artifacts: Dict[str, Any] = Field(default_factory=dict)
    plan_summary: Dict[str, Any] = Field(default_factory=dict)
    decisions: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    failure_context: Optional[FailureContext] = None
    history_context: Optional[HistoryContext] = None
    next_expected_action: str = ""
    skill_bundle: Optional[SkillBundle] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
