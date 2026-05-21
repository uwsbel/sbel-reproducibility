"""
Exceptions shared across agent implementations.
"""

from typing import List, Optional


class AgentLLMError(RuntimeError):
    """Raised when an agent LLM call fails and the workflow should stop."""

    def __init__(
        self,
        agent_name: str,
        operation: str,
        message: str,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        self.agent_name = agent_name
        self.operation = operation
        self.detail = str(message)
        self.original_exception = original_exception
        super().__init__(f"{agent_name} LLM failure during {operation}: {self.detail}")


class PlanModificationIncompleteError(AgentLLMError):
    """Raised when the planning LLM's ``modify_plan`` output was unparseable
    enough that the user's intent cannot be trusted to have landed.

    Triggered when critical plan fields (``simulation_parameters``,
    ``implementation_steps``) are missing from the parsed response — those
    fields carry the bulk of the semantic payload users typically modify, so
    silently restoring them from the original plan (the previous behavior)
    hid failed modifications behind a "success" message. Now we fail loudly
    and surface the raw LLM preview so the user can retry with a clearer
    request or recognize a formatting regression.
    """

    def __init__(
        self,
        agent_name: str,
        missing_fields: List[str],
        raw_preview: str,
        truncated: bool = False,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        self.missing_fields = list(missing_fields)
        self.raw_preview = raw_preview
        self.truncated = truncated
        reason = "truncated mid-JSON (likely hit max_tokens)" if truncated else "unparseable"
        detail = (
            f"Modified plan {reason}; missing critical fields: "
            f"{', '.join(missing_fields)}. LLM output preview: {raw_preview[:200]!r}"
        )
        super().__init__(
            agent_name=agent_name,
            operation="modify_plan",
            message=detail,
            original_exception=original_exception,
        )


class PlanModificationValidationError(AgentLLMError):
    """Raised when ``modify_plan``'s LLM output survives the top-level
    critical-fields check but fails deeper Pydantic schema validation on
    ``SimulationPlan(**plan_dict)`` — and an auto-retry with the validation
    errors fed back into the prompt also failed.

    Symptoms seen in the wild (2026-04-21): weaker providers (e.g. MiniMax,
    some local models) emit plans where nested milestone fields have wrong
    shapes — ``milestones[].constraints`` as string instead of list,
    ``verify.csv_cols`` as list of strings instead of list of objects, etc.

    Carries the top 5 validation errors (field path + short reason) plus the
    raw LLM preview so the CLI can render a compact, actionable red panel
    instead of dumping 25 lines of Pydantic stack trace at the user.
    """

    def __init__(
        self,
        agent_name: str,
        field_errors: List[dict],
        raw_preview: str,
        retries_attempted: int = 1,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        self.field_errors = list(field_errors)  # list of {"loc": [...], "msg": "..."}
        self.raw_preview = raw_preview
        self.retries_attempted = retries_attempted
        top = self.field_errors[:5]
        summary = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')[:60]}"
            for e in top
        )
        detail = (
            f"Modified plan failed Pydantic validation after {retries_attempted} "
            f"retry attempt(s). {len(self.field_errors)} total field error(s). "
            f"First {len(top)}: {summary}"
        )
        super().__init__(
            agent_name=agent_name,
            operation="modify_plan",
            message=detail,
            original_exception=original_exception,
        )
