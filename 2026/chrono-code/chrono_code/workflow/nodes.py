"""
Workflow nodes (merged from root graph + build subgraph + review subgraph).

Planning, plan approval, error handling, code generation, execution,
VLM visual review, physics analysis, and step-by-step review.

State is nested: state["planning"], state["build"], state["review"].
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from chrono_code.agents import PlanningAgent, CodeGenerationAgent, ExecutionAgent, ReviewAgent
from chrono_code.agents.exceptions import AgentLLMError
from chrono_code.workflow.state import WorkflowState, make_agent_message
from chrono_code.workflow.conditions import DEFAULT_MAX_EXECUTION_RETRIES
from chrono_code.workflow.events import emit_custom_event, emit_progress_event
from chrono_code.models.handoff import LLMHandoff, StructuredError
from chrono_code.models.plan import SimulationPlan, SimulationStep
from chrono_code.models.plan_markdown_parser import has_unresolved_tokens
from chrono_code.models.code import CodeArtifact, GeneratedCode
from chrono_code.models.execution import ExecutionResult
from chrono_code.config import get_settings
from chrono_code.utils.error_utils import fingerprint_error
from chrono_code.utils.logger import log_boxed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities (planning)
# ---------------------------------------------------------------------------

def _get_planning(state: WorkflowState) -> Dict[str, Any]:
    """Return planning dict, mutable copy for reads."""
    p = state.get("planning")
    return dict(p) if isinstance(p, dict) else {}


def _merged_planning_state(
    planning: Dict[str, Any],
    plan_mode: str,
    state: WorkflowState,
) -> Dict[str, Any]:
    """Merge planning updates; default auto-approve based on the field value.

    In the new project, CLI plan approval is handled by the workflow engine
    via callback, so we simply read ``auto_approve_on_continue`` directly.
    """
    if "auto_approve_on_continue" in planning:
        auto = bool(planning["auto_approve_on_continue"])
    else:
        auto = False
    return {
        **planning,
        "plan_mode": plan_mode,
        "auto_approve_on_continue": auto,
        "plan_approved": planning.get("plan_approved"),
        "plan_rejected": planning.get("plan_rejected", False),
        "plan_needs_regeneration": planning.get("plan_needs_regeneration", False),
    }


# ---------------------------------------------------------------------------
# Helper utilities (build / review)
# ---------------------------------------------------------------------------

def _get_build(state: WorkflowState) -> Dict[str, Any]:
    """Return build dict."""
    b = state.get("build")
    return dict(b) if isinstance(b, dict) else {}


def _get_review(state: WorkflowState) -> Dict[str, Any]:
    """Return review dict."""
    r = state.get("review")
    return dict(r) if isinstance(r, dict) else {}


def _backfill_previous_code(state: WorkflowState) -> None:
    """Copy current ``state['code']['code']`` into ``state['build']['previous_code']``.

    Used after any failure path (execution, CSV physics, LLM physics) so the
    next codegen attempt has a canonical baseline to patch against.
    """
    code_raw = state.get("code")
    code_text = code_raw.get("code") if isinstance(code_raw, dict) else ""
    if not code_text:
        return
    build = _get_build(state)
    state["build"] = {**build, "previous_code": code_text}


def _track_step_no_progress(step_loop: Dict[str, Any], state: WorkflowState) -> None:
    """Update the no-progress retry counter on the current step_loop.

    Compares the SHA-256 of the current ``state['code']['code']`` against
    ``step_loop['step_last_failed_code_sha']``. Equal SHA → this retry
    produced no change (codegen is stuck), bump
    ``step_no_progress_retry_count`` by 1. Differing SHA → forward
    progress, reset the counter to 0. The new SHA is always recorded so
    the next failed retry has something to compare against.

    Caller must invoke this AFTER ``state['code']['code']`` reflects the
    just-failed attempt (i.e. after ``_backfill_previous_code`` or
    equivalent). The retry-router (``route_after_step_review``) reads the
    counter to decide whether to abort early.
    """
    code_raw = state.get("code")
    code_text = code_raw.get("code") if isinstance(code_raw, dict) else ""
    if not code_text:
        # Without code we cannot compare; leave counters alone so the
        # plain step_retry_count remains the only retry budget for this
        # corner case.
        return
    new_sha = hashlib.sha256(code_text.encode("utf-8", errors="replace")).hexdigest()
    prior_sha = step_loop.get("step_last_failed_code_sha")
    if prior_sha is not None and prior_sha == new_sha:
        step_loop["step_no_progress_retry_count"] = (
            int(step_loop.get("step_no_progress_retry_count") or 0) + 1
        )
    else:
        step_loop["step_no_progress_retry_count"] = 0
    step_loop["step_last_failed_code_sha"] = new_sha


def _bump_execution_retry(
    state: WorkflowState,
    source: Literal["execution", "physics"],
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    """Increment the retry counter appropriate for ``source`` and tag it.

    Hard code-execution failures (``source="execution"``) bump
    ``review['execution_retry_count']``. Soft review-layer failures
    (``source="physics"``) bump ``review['vlm_review_retry_count']`` —
    these come from the CSV / LLM physics analysis paths and represent
    the reviewer rejecting otherwise-executing code. Keeping their
    budgets separate prevents a reviewer that is wrong about working
    code from burning the code-repair budget.

    Returns the new counter value for the bumped bucket.
    """
    review = _get_review(state)
    if source == "physics":
        counter_key = "vlm_review_retry_count"
    else:
        counter_key = "execution_retry_count"
    new_count = int(review.get(counter_key, 0)) + 1
    updated: Dict[str, Any] = {
        **review,
        counter_key: new_count,
        "last_failure_source": source,
    }
    if extra:
        updated.update(extra)
    state["review"] = updated
    return new_count


def _reset_per_step_retry_budget(state: WorkflowState) -> None:
    """Reset execution-retry + fingerprint state when advancing to a fresh step.

    Step-mode workflows should give each step an independent execution-retry
    budget. The old behavior let a step 1 with 2 exec failures plus a step 3
    with 2 exec failures (different root causes) exhaust the total 4-cap at
    step 3 — conflating unrelated code paths. Matching step_retry_count's
    per-step reset cadence keeps each step's budget self-contained while the
    global ``max_iterations`` cap still bounds runaway loops.
    """
    review = _get_review(state)
    state["review"] = {
        **review,
        "execution_retry_count": 0,
        "vlm_review_retry_count": 0,
        "last_error_fingerprint": None,
        "repeat_error_count": 0,
    }


def _format_agent_llm_error(error: AgentLLMError) -> str:
    return f"{error.agent_name} LLM failure during {error.operation}: {error.detail}"


def _make_structured_error(*, phase: str, error_type: str, summary: str, raw_message: str = "", operation: str | None = None, retryable: bool = True) -> StructuredError:
    return StructuredError.from_message(
        error_type=error_type,
        phase=phase,
        summary=summary,
        raw_message=raw_message or summary,
        operation=operation,
        retryable=retryable,
    )


def _build_planning_handoff(user_prompt: str, plan: Any) -> LLMHandoff:
    return LLMHandoff(
        task_intent="create_or_refine_simulation_plan",
        input_artifacts={
            "user_prompt": user_prompt,
            "plan_type": getattr(plan, "plan_type", None),
        },
        decisions={
            "visualization_mode": (getattr(plan, "visualization", None) or {}).get("mode") if getattr(plan, "visualization", None) else None,
        },
        constraints=[
            "Preserve approved plan decisions across later LLM calls.",
            "Treat future code generation as downstream of this plan, not a re-interpretation of the user prompt.",
        ],
        next_expected_action="approve_or_modify_plan" if getattr(plan, "clarifications_needed", None) else "generate_code",
        metadata={"source": "planning_node"},
    )


# ---------------------------------------------------------------------------
# Helper utilities (codegen)
# ---------------------------------------------------------------------------

def _get_review_feedback(state: Dict[str, Any]) -> Optional[str]:
    """Extract the most relevant feedback for codegen to act on.

    Priority: step_loop.step_feedback > review.execution_feedback
              > review.physics_feedback > review.vlm_feedback.

    Special case — review-failed → patch → patch CRASHED at runtime:
    when both ``step_feedback`` (from a prior step_review FAIL) AND
    ``review.execution_feedback`` (from the just-completed retry's
    execution) are set, COMBINE them. The execution failure is the
    most recent signal — codegen MUST see it or it will keep editing
    against a stale review verdict and never realise its previous
    patch broke compilation. Without this combine, the retry loop
    burns its entire 6-attempt budget feeding the same review-failure
    message to codegen while every retry crashes 0.3s into execution
    on a TypeError / AttributeError that codegen never gets to see.
    """
    # Step-by-step mode feedback (top-level step_loop state)
    step_loop = state.get("step_loop")
    review = state.get("review") if isinstance(state.get("review"), dict) else {}

    if isinstance(step_loop, dict):
        step_feedback = step_loop.get("step_feedback")
        if step_feedback:
            new_exec_fb = review.get("execution_feedback")
            if new_exec_fb:
                return (
                    "PRIOR REVIEW FEEDBACK (still unresolved — original reason "
                    "this retry loop started):\n"
                    f"{step_feedback}\n"
                    "\n"
                    "=== HOWEVER, YOUR LAST PATCH BROKE THE BUILD ===\n"
                    "The simulation FAILED to run after your previous edit. "
                    "Fix this NEW execution error FIRST — until the script "
                    "executes cleanly, no review will even get to see whether "
                    "the original visual issue is solved. After the run is "
                    "green again, also revisit the review feedback above.\n"
                    "\n"
                    f"NEW EXECUTION FAILURE:\n{new_exec_fb}"
                )
            return step_feedback

    # Review-stage feedback
    if not review:
        return None

    last_failure_source = review.get("last_failure_source")
    if last_failure_source == "execution":
        return review.get("execution_feedback")
    elif last_failure_source == "physics":
        return review.get("physics_feedback")
    elif last_failure_source == "vlm":
        return review.get("vlm_feedback")

    return None


def _make_code_artifact(code_dict: Dict[str, Any]) -> CodeArtifact:
    """Build a CodeArtifact from a GeneratedCode dict."""
    return CodeArtifact(
        file_name=code_dict.get("file_name", "simulation.py"),
        base_code=str(code_dict.get("base_code") or ""),
        full_code=str(
            code_dict.get("applied_code")
            or code_dict.get("code")
            or ""
        ),
        patch=str(code_dict.get("patch") or ""),
        hunks=code_dict.get("hunks"),
    )


# ---------------------------------------------------------------------------
# Helper utilities (execution / review)
# ---------------------------------------------------------------------------

def _build_execution_trace_snapshot(
    *,
    plan: Dict[str, Any] | None,
    code: Dict[str, Any] | None,
    review: Dict[str, Any],
    execution: ExecutionResult | None = None,
) -> Dict[str, Any]:
    """Summarize run_simulation input/output for trace UIs."""
    code_dict = code or {}
    plan_for_trace = SimulationPlan(**(plan or {"plan_type": "mbs"})) if plan else None
    snapshot: Dict[str, Any] = {
        "input": {
            "plan_type": plan_for_trace.plan_type if plan_for_trace else None,
            "implementation_steps": len(plan_for_trace.implementation_steps) if plan_for_trace else 0,
            "validation_status": code_dict.get("validation_status"),
            "code_chars": len(str(code_dict.get("code") or "")),
            "has_patch": bool(code_dict.get("patch")),
            "execution_retry_count": review.get("execution_retry_count", 0),
        }
    }
    if execution is not None:
        snapshot["output"] = {
            "success": execution.success,
            "return_code": execution.return_code,
            "runtime_seconds": execution.runtime_seconds,
            "output_files": execution.output_files,
            "error_messages": execution.error_messages,
            "error_message": execution.error_message,
        }
    return snapshot


def _merge_unique_messages(existing: Any, new: Any) -> List[str]:
    seen = set()
    merged: List[str] = []
    for group in (existing, new):
        candidates = []
        if isinstance(group, str):
            candidates = group.splitlines()
        elif isinstance(group, (list, tuple)):
            candidates = [str(item) for item in group]
        elif group:
            candidates = [str(group)]

        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            normalized = " ".join(text.split())
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(text)
    return merged


def _sanitize_error_message(text: Any) -> str:
    """Remove legacy heuristic prefixes and keep only the raw factual message."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^\[RELATED to prior error\]\s*", "", cleaned)
    return cleaned


def _current_error_messages(new_messages: Any, combined: str) -> List[str]:
    """Build a current-attempt-only error list for the next fix prompt."""
    merged = _merge_unique_messages([], new_messages)
    sanitized: List[str] = []
    for msg in merged:
        cleaned = _sanitize_error_message(msg)
        if cleaned:
            sanitized.append(cleaned)
    if sanitized:
        return sanitized

    fallback = _sanitize_error_message(combined)
    return [fallback] if fallback else []


def _format_structured_error_block(structured_error: Optional[Dict[str, Any]]) -> str:
    """Render ExecutionResult.structured_error into a compact header lines block
    so the code agent sees targeted fields (failing_symbol, introspection_hint)
    before the raw error list.
    """
    if not isinstance(structured_error, dict):
        return ""
    parts = []
    for key in ("error_type", "failing_symbol", "failing_line", "introspection_hint"):
        val = structured_error.get(key)
        if val not in (None, "", []):
            parts.append(f"{key}: {val}")
    excerpt = structured_error.get("file_line_excerpt")
    if excerpt:
        parts.append(f"file_line_excerpt:\n{excerpt}")
    if not parts:
        return ""
    return "Structured runtime error:\n" + "\n".join(parts) + "\n\n"


def _format_execution_feedback(
    *,
    rc: int | None,
    runtime: float,
    error_messages: List[str],
    combined: str,
    structured_error: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the next-turn feedback from current-attempt errors only."""
    fallback = _sanitize_error_message(combined)
    current_items = error_messages or ([fallback] if fallback else [])
    error_list_text = "\n".join(f"- {item}" for item in current_items) if current_items else combined
    structured_block = _format_structured_error_block(structured_error)
    signal_codes = {-11, -6, -8, -7}
    if rc in signal_codes:
        return (
            f"CORE DUMP / SEGMENTATION FAULT (return_code={rc}, runtime: {runtime:.2f}s)\n\n"
            f"{structured_block}"
            f"Unique Error Messages:\n{error_list_text}\n\n"
            "Analyze and fix physics/API usage."
        )
    return (
        f"EXECUTION FAILED (return_code={rc}, runtime: {runtime:.2f}s)\n\n"
        f"{structured_block}"
        f"Unique Error Messages:\n{error_list_text}\n\n"
        "Fix the code to resolve this runtime issue."
    )


def _semantic_deduplicate_errors(
    existing_messages: List[str],
    new_messages: List[str],
    existing_fingerprints: Dict[str, str],
) -> Tuple[List[str], Dict[str, str]]:
    """Deduplicate errors using fingerprint matching.

    Returns ``(deduplicated_messages, updated_fingerprints)``.
    """
    seen: Dict[str, str] = dict(existing_fingerprints)
    merged: List[str] = [_sanitize_error_message(msg) for msg in list(existing_messages)]

    if isinstance(new_messages, str):
        candidates = [new_messages]
    elif isinstance(new_messages, (list, tuple)):
        candidates = [str(item) for item in new_messages]
    elif new_messages:
        candidates = [str(new_messages)]
    else:
        candidates = []

    for item in candidates:
        text = _sanitize_error_message(item)
        if not text:
            continue

        fp = fingerprint_error(text)
        if fp in seen:
            continue

        merged.append(text)
        seen[fp] = text

    return merged, seen


# ---------------------------------------------------------------------------
# Pre-execute Chrono-API validation gate
# ---------------------------------------------------------------------------

def _pre_execute_api_validation_gate(
    state: WorkflowState,
    code: GeneratedCode,
    trace: Dict[str, Any],
) -> Optional[WorkflowState]:
    """Hard static API-validation check before subprocess execution.

    The in-codegen post-edit validator hook (``_run_all_post_edit_validators``
    in code_agent_tools.py) only surfaces FAIL as a tool_result string the
    LLM may ignore. The "invented-API guardrail" inside CodeGenerationAgent
    only triggers AFTER the prior execution already crashed with an
    AttributeError / NameError pattern, so the FIRST execution of any new
    code has no protection.

    This gate runs ``scripts/validate_chrono_apis.py`` (via the cached
    in-process loader) and, on FAIL, mutates ``state`` to look like a normal
    execution failure — synthetic ExecutionResult, execution_feedback,
    bumped retry counter, backfilled previous_code. The caller in
    ``execution_node`` MUST return the mutated state immediately so the
    subprocess never starts; ``route_after_execution`` will then route back
    to codegen exactly like a real runtime crash.

    Returns ``None`` on PASS or UNKNOWN — caller proceeds with subprocess.
    UNKNOWN (validator couldn't run, e.g. missing pychrono) is treated as
    PASS to avoid hard-blocking unusual environments; this matches the
    posture of the in-codegen hook.
    """
    code_text = code.code if hasattr(code, "code") else ""
    if not code_text or not code_text.strip():
        return None

    try:
        from chrono_code.tools.code_agent_tools import (
            _classify_validator_status,
            _extract_chrono_api_violations,
            _run_chrono_api_validation,
        )
    except Exception as exc:  # pragma: no cover - import safety
        logger.warning(
            "[pre-execute gate] validator import failed (%s); skipping gate",
            exc,
        )
        return None

    try:
        validator_out = _run_chrono_api_validation(code_text)
    except Exception as exc:
        logger.warning(
            "[pre-execute gate] validator raised (%s); treating as UNKNOWN, "
            "proceeding to subprocess",
            exc,
        )
        return None

    status = _classify_validator_status(validator_out)
    if status != "FAIL":
        # PASS or UNKNOWN — let the subprocess run.
        logger.info("[pre-execute gate] chrono_apis %s — proceeding to subprocess", status)
        return None

    violations = _extract_chrono_api_violations(validator_out) or []
    n = len(violations)
    logger.warning(
        "[pre-execute gate] chrono_apis FAIL (%d violations) — skipping subprocess",
        n,
    )

    # Build per-violation error messages with a unique prefix so the
    # error-fingerprint logic in conditions.route_after_execution doesn't
    # mistake a static-validation failure for a repeating runtime crash.
    err_messages = [
        f"PreExecuteAPIValidationError: {v}" for v in violations[:10]
    ]
    if n > 10:
        err_messages.append(
            f"PreExecuteAPIValidationError: ...and {n - 10} more violations "
            "(see scripts/validate_chrono_apis.py output)"
        )

    primary = (
        f"[pre-execute gate] chrono_apis FAIL — {n} violations. The "
        "simulation subprocess was NOT started. Fix the listed APIs and "
        "try again."
    )

    synthetic = ExecutionResult(
        success=False,
        output_files=[],
        execution_log="",
        runtime_seconds=0.0,
        return_code=None,
        error_message=primary,
        error_messages=err_messages,
    )

    # Mirror the failure path at execution_node:1140-1175.
    review = _get_review(state)
    existing_fps = review.get("error_fingerprints", {})
    existing_msgs = review.get("execution_error_messages", [])
    current_error_messages = _current_error_messages(err_messages, primary)

    error_messages, updated_fingerprints = _semantic_deduplicate_errors(
        existing_msgs,
        current_error_messages,
        existing_fps,
    )

    # Custom feedback block that's unambiguous about WHY no subprocess ran.
    # We don't reuse _format_execution_feedback here because its phrasing
    # ("EXECUTION FAILED ... runtime issue") would mislead codegen into
    # debugging a stack trace that doesn't exist.
    violation_block = "\n".join(f"  - {v}" for v in violations[:10])
    if n > 10:
        violation_block += f"\n  ...and {n - 10} more"
    feedback = (
        f"PRE-EXECUTE API VALIDATION FAILED ({n} violations). The simulation "
        "subprocess was NOT started — these are static-analysis errors found "
        "BEFORE any code ran.\n\n"
        f"Violations (from scripts/validate_chrono_apis.py):\n{violation_block}\n\n"
        "Fix every listed API. Use read_skill / read_skill_section to look up "
        "the correct signatures, then call edit_file or write_file. Do NOT "
        "treat this as a runtime crash — there is no stack trace to debug; "
        "the validator scanned the source and refused to launch."
    )

    state["execution"] = synthetic.model_dump()
    state["review"] = {
        **review,
        "execution": synthetic.model_dump(),
        "trace": trace,
        "execution_feedback": feedback,
        "current_execution_error_messages": current_error_messages,
        "execution_error_messages": error_messages,
        "error_fingerprints": updated_fingerprints,
        "vlm_feedback": None,
    }
    _bump_execution_retry(state, "execution")
    _backfill_previous_code(state)
    return state


def _append_progress_message(
    messages: list,
    agent: str,
    content: str,
    phase: str,
    step: str,
    progress_pct: int,
    extra: dict | None = None,
) -> None:
    metadata = {"phase": phase, "step": step, "progress_pct": progress_pct}
    if extra:
        metadata.update(extra)
    messages.append(
        make_agent_message(
            agent=agent,
            content=content,
            metadata=metadata,
        )
    )
    emit_progress_event(
        phase=phase,
        step=step,
        progress_pct=progress_pct,
        message=content,
        agent=agent,
        **(extra or {}),
    )


# =========================================================================
# Node: planning_node (Agent 1)
# =========================================================================

async def planning_node(state: WorkflowState) -> WorkflowState:
    """
    Planning node - Agent 1.
    Creates a simulation plan from user prompt.
    """
    logger.info("[Planning]")

    planning = _get_planning(state)
    plan_mode = planning.get("plan_mode") or state.get("plan_mode", "auto")
    # Clear regeneration flag when (re-)entering planning
    planning["plan_needs_regeneration"] = False

    user_prompt = state.get("user_prompt", "")
    if not user_prompt:
        logger.error("planning_node: 'user_prompt' is missing from state")

    try:
        clarification_callback = state.get("clarification_callback")
        agent = PlanningAgent(clarification_callback=clarification_callback)
        images = state.get("images")
        plan = await agent.execute(
            user_prompt=user_prompt,
            plan_mode=plan_mode,
            images=images,
        )
    except AgentLLMError as exc:
        error_msg = _format_agent_llm_error(exc)
        messages = list(state.get("messages", []))
        messages.append(make_agent_message(
            agent="PlanningAgent",
            content=error_msg,
            metadata={
                "error_type": "llm_error",
                "operation": exc.operation,
            },
        ))
        structured_error = _make_structured_error(phase="planning", error_type="llm_error", summary=error_msg, raw_message=error_msg, operation=exc.operation)
        return {
            **state,
            "messages": messages,
            "current_step": "planning_failed",
            "phase": "planning",
            "progress": {
                "step": "planning_llm_error",
                "progress_pct": 100,
                "error_msg": error_msg[:500],
            },
            "error": error_msg,
            "structured_error": structured_error.model_dump(),
            "planning": _merged_planning_state(planning, plan_mode, state),
        }
    except Exception as exc:
        error_msg = f"PlanningAgent unexpected error: {type(exc).__name__}: {exc}"
        logger.error(error_msg, exc_info=True)
        structured_error = _make_structured_error(phase="planning", error_type="unexpected_error", summary=error_msg, raw_message=error_msg, retryable=False)
        return {
            **state,
            "current_step": "planning_failed",
            "phase": "planning",
            "progress": {"step": "planning_error", "progress_pct": 100, "error_msg": error_msg[:500]},
            "error": error_msg,
            "structured_error": structured_error.model_dump(),
            "planning": _merged_planning_state(planning, plan_mode, state),
        }

    messages = list(state.get("messages", []))
    messages.append(make_agent_message(
        agent="PlanningAgent",
        content=f"Created {plan.plan_type} plan with {len(plan.implementation_steps)} steps",
        metadata={},
    ))

    # Clear validated_api_plan when plan agent runs (first time or regenerate)
    build = dict(state.get("build") or {})
    build["validated_api_plan"] = []
    codegen_state = dict(build.get("codegen_state") or {})
    codegen_state["validated_api_plan"] = []
    build["codegen_state"] = codegen_state

    # Emit plan to activity feed
    emit_custom_event({
        "type": "plan_generated",
        "plan_json": json.dumps(plan.dump_all(), indent=2, default=str),
    })

    # The legacy ``output_requirements`` auto-injection (scene_placement.csv /
    # scene_contacts.csv) is gone — under the per-step motion-CSV contract
    # the only motion-related output is ``cam/motion_log.csv``, written by
    # codegen when a step declares ``motion_expectations``. dump_all()
    # includes exclude=True fields (implementation_steps, assets, topology).
    plan_dict = plan.dump_all()

    handoff = _build_planning_handoff(user_prompt, plan)
    # The new pipeline carries unanswered questions as inline ``<<ASK_*>>``
    # tokens inside ``plan_markdown`` rather than populating
    # ``clarifications_needed``. Treat both as "user still has work to do".
    has_inline_tokens = has_unresolved_tokens(plan.plan_markdown or "")
    return {
        **state,
        "plan": plan_dict,
        "messages": messages,
        "current_step": "planning_complete",
        "needs_user_input": bool(plan.clarifications_needed) or has_inline_tokens,
        "planning": _merged_planning_state(planning, plan_mode, state),
        "build": build,
        "llm_handoff": handoff.model_dump(),
        "structured_error": None,
    }


# =========================================================================
# Node: wait_for_approval_node
# =========================================================================

def wait_for_approval_node(state: WorkflowState) -> WorkflowState:
    """
    Plan approval node.

    When ``planning.auto_approve_on_continue`` is True, entering this node
    sets ``plan_approved`` so the workflow can proceed. In CLI mode the
    workflow engine handles plan approval via the ``on_plan_approval``
    callback before reaching this node.

    Returns a new ``planning`` dict (no in-place mutation of nested state).
    """
    planning = dict(state.get("planning") or {})

    # Default False when key missing.
    auto_approve_on_continue = planning.get("auto_approve_on_continue", False)
    if (
        auto_approve_on_continue
        and planning.get("plan_approved") is None
        and not planning.get("plan_rejected", False)
    ):
        planning["plan_approved"] = True

    return {**state, "planning": planning}


# =========================================================================
# Node: error_node
# =========================================================================

async def error_node(state: WorkflowState) -> WorkflowState:
    """Handle errors in the workflow."""
    error_msg = state.get("error")
    structured_error = state.get("structured_error") or {}
    if not error_msg:
        build = _get_build(state)
        review = _get_review(state)
        if build.get("fix_state") == "PATCH_APPLY_FAILED":
            error_msg = (
                "Unified diff patch application failed. "
                f"Reason: {build.get('fix_reason', 'unknown')}"
            )
        elif build.get("catalog_gate_failed"):
            error_msg = (
                "Catalog gate failed before patch application. "
                f"Reason: {build.get('fix_reason', 'unknown')}"
            )
        elif (
            review.get("execution_retry_count", 0) >= review.get("max_execution_retries", DEFAULT_MAX_EXECUTION_RETRIES)
            or review.get("vlm_review_retry_count", 0) >= review.get("max_vlm_review_retries", 5)
        ):
            # Graceful give-up: surface the last execution error and repeat
            # fingerprint so the user sees why we stopped rather than a bare
            # attempt count.
            retries = review.get("execution_retry_count", 0)
            cap = review.get("max_execution_retries", DEFAULT_MAX_EXECUTION_RETRIES)
            repeat = review.get("repeat_error_count", 0)
            vlm_retries = review.get("vlm_review_retry_count", 0)
            vlm_cap = review.get("max_vlm_review_retries", 5)
            # execution_node always writes review['execution'] before routing;
            # don't fall back to the top-level state key (which can be stale if
            # an earlier node updated review but not state['execution']).
            last_execution = review.get("execution") or {}
            last_stderr = (last_execution.get("stderr") or last_execution.get("error") or "").strip()
            last_frame = ""
            if last_stderr:
                for ln in reversed(last_stderr.splitlines()):
                    if ln.strip():
                        last_frame = ln.strip()[:300]
                        break
            if vlm_retries >= vlm_cap:
                reason = f"{vlm_retries}/{vlm_cap} review rejections exhausted"
            else:
                reason = f"{retries}/{cap} execution attempts exhausted"
            if repeat >= 1:
                reason += f"; same error fingerprint repeated {repeat + 1}x"
            if last_frame:
                reason += f". Last error: {last_frame}"
            error_msg = f"Execution gave up gracefully: {reason}"
        elif (state.get("code") or {}).get("validation_status") == "validation_failed":
            code_dict = state.get("code") or {}
            errs = code_dict.get("compilation_errors") or []
            feedback = build.get("code_validation_feedback") or ""
            detail = "; ".join(str(e)[:200] for e in errs[:5]) if errs else feedback[:500]
            error_msg = (
                "Code validation failed: generated code did not pass API validation. "
                f"Details: {detail}" if detail else
                "Code validation failed: generated code did not pass API validation."
            )
        else:
            error_msg = structured_error.get("summary") if isinstance(structured_error, dict) and structured_error.get("summary") else "Unknown error occurred"

    logger.error(f"[Error] {error_msg}")

    messages = list(state.get("messages", []))
    messages.append(make_agent_message(
        agent="ErrorHandler",
        content=f"Workflow error: {error_msg}",
        metadata={"error_type": "workflow_error"},
    ))

    return {
        **state,
        "messages": messages,
        "current_step": "error_handler",
        "phase": "error",
        "progress": {
            "step": "error_handler",
            "progress_pct": 100,
            "error_msg": error_msg[:500],
        },
        "execution_complete": True,
        "final_output": {"error": error_msg, "success": False, "structured_error": structured_error if structured_error else None},
    }


# =========================================================================
# Node: check_plan_approval (routing function)
# =========================================================================

def check_plan_approval(state: WorkflowState) -> str:
    """
    Route based on plan approval status.

    This only *reads* ``planning`` and returns a route name; it does not set
    ``plan_approved``. Auto-approve is applied in ``wait_for_approval_node``
    when ``auto_approve_on_continue`` is True.
    """
    if state.get("error"):
        return "error"

    planning = _get_planning(state)
    plan_approved = planning.get("plan_approved")
    plan_rejected = planning.get("plan_rejected", False)
    plan_needs_regeneration = planning.get("plan_needs_regeneration", False)
    auto_flag = planning.get("auto_approve_on_continue")

    logger.info(
        "Plan approval: approved=%s, rejected=%s, regen=%s, auto_approve_on_continue=%s",
        plan_approved,
        plan_rejected,
        plan_needs_regeneration,
        auto_flag,
    )

    if plan_rejected:
        logger.warning("Plan rejected by user")
        state["error"] = "Plan rejected by user"
        return "rejected"
    if plan_needs_regeneration:
        logger.info("Plan modifications requested - regenerating")
        state["planning"] = {**planning, "plan_needs_regeneration": False}
        return "regenerate"
    if plan_approved:
        logger.info("Plan approved - proceeding to build stage")
        return "approved"
    logger.info("Plan awaiting user approval")
    return "awaiting_approval"


# =========================================================================
# Node: step_router_node
# =========================================================================

async def step_router_node(state: WorkflowState) -> WorkflowState:
    """Prepare the next step context for scene / mbs_in_scene step loops.

    Iterates ``plan.implementation_steps`` (a list of ``SimulationStep``
    objects carrying description + assets + camera + constraints). On each
    invocation, advances ``current_step_index`` and rebuilds the
    ``step_context`` bundle the codegen / review nodes consume.

    For pure ``mbs`` plans (no external assets / per-step breakdown),
    marks ``all_steps_complete=True`` so the workflow skips directly to
    the full monolithic codegen → review path.
    """
    plan_dict = state.get("plan") or {}
    plan_obj = SimulationPlan(**plan_dict)
    step_loop = dict(state.get("step_loop") or {})

    # Pure MBS plans skip the step_loop entirely (no per-step cam review).
    if plan_obj.plan_type not in ("scene", "mbs_in_scene", "fsi_in_scene"):
        step_loop["all_steps_complete"] = True
        logger.info("[StepRouter] Non-scene plan type '%s' — skipping step loop", plan_obj.plan_type)
        return {**state, "step_loop": step_loop}

    # scene / mbs_in_scene / fsi_in_scene: iterate the structured implementation_steps list.
    steps = plan_obj.implementation_steps or []
    if not steps:
        step_loop["all_steps_complete"] = True
        logger.warning(
            "[StepRouter] %s plan has no implementation_steps — skipping step loop",
            plan_obj.plan_type,
        )
        return {**state, "step_loop": step_loop}

    # On first entry, seed step_loop with serialized step dicts so downstream
    # read-only nodes don't have to re-instantiate SimulationStep.
    if not step_loop.get("steps"):
        step_loop["steps"] = [s.model_dump() for s in steps]
        step_loop["current_step_index"] = 0
        step_loop["completed_steps"] = []

    idx = step_loop.get("current_step_index", 0)
    if idx >= len(steps):
        step_loop["all_steps_complete"] = True
        logger.info("[StepRouter] All %d steps complete", len(steps))
        return {**state, "step_loop": step_loop}

    current: "SimulationStep" = steps[idx]
    step_loop["current_step_description"] = current.description
    step_loop["step_retry_count"] = 0
    step_loop["step_no_progress_retry_count"] = 0
    step_loop["step_last_failed_code_sha"] = None
    step_loop["step_feedback"] = None
    _reset_per_step_retry_budget(state)

    # Completed-step descriptions (used by downstream prompts for display only)
    completed_descs = [s.description for s in steps[:idx]]
    step_ctx = plan_obj.build_step_context(idx, completed_descs)
    step_loop["step_context"] = step_ctx.model_dump()

    # relevant_bodies: current/prior assets + procedural scene objects + anchors
    relevant: set[str] = set()
    for a in step_ctx.step_assets:
        n = str(a.get("name") or "").lower()
        if n:
            relevant.add(n)
    for obj in step_ctx.step_scene_objects:
        n = str(obj.get("name") or "").lower()
        if n:
            relevant.add(n)
    for ca in step_ctx.completed_assets:
        n = str(ca.get("name") or "").lower()
        if n:
            relevant.add(n)
    for obj in step_ctx.completed_scene_objects:
        n = str(obj.get("name") or "").lower()
        if n:
            relevant.add(n)
    relevant.update({"floor", "ground", "terrain"})
    relevant.discard("")
    step_loop["relevant_bodies"] = list(relevant)

    logger.info(
        "[StepRouter] Step %d/%d: %s (assets=%s, n_cameras=%d, camera0_pos=%s)",
        idx + 1, len(steps), current.description[:80],
        current.assets, len(current.cameras),
        current.cameras[0].position if current.cameras else None,
    )
    return {**state, "step_loop": step_loop}


# =========================================================================
# Node: codegen_node (Agent 2)
# =========================================================================

async def codegen_node(state: WorkflowState) -> WorkflowState:
    """Generate or patch simulation code."""
    logger.info("[Build] Code generation")

    messages = list(state.get("messages", []))
    plan = state.get("plan")
    if isinstance(plan, dict):
        plan = SimulationPlan(**plan)

    if plan is None:
        messages.append(
            make_agent_message(
                agent="CodeGenerationAgent",
                content="Cannot generate code: no plan available (planning may have failed).",
            )
        )
        build = _get_build(state)
        return {
            **state,
            "messages": messages,
            "build": {
                **build,
                "code": build.get("previous_code") or build.get("code"),
                "validation_status": "codegen_failed",
            },
        }

    build = _get_build(state)
    previous_code = build.get("previous_code") or build.get("code")
    feedback = _get_review_feedback(state)

    # fix_mode=True when we have feedback from a previous failure (execution,
    # physics, or VLM review).  This tells the code agent to use edit_file
    # on existing code rather than write_file from scratch.
    is_fix = feedback is not None and previous_code is not None

    try:
        agent = build.get("codegen_agent")
        if agent is None:
            agent = CodeGenerationAgent()
            build["codegen_agent"] = agent
        result = await agent.execute(
            plan=plan,
            compilation_feedback=feedback,
            previous_code=previous_code,
            state=state,
            fix_mode=is_fix,
        )
    except AgentLLMError as e:
        messages.append(
            make_agent_message(
                agent="CodeGenerationAgent",
                content=f"LLM failure during codegen: {e}",
            )
        )
        return {
            **state,
            "messages": messages,
            "build": {
                **build,
                "code": previous_code,
                "validation_status": "codegen_failed",
            },
        }
    except BaseException as e:
        # BaseException to also catch asyncio.CancelledError (not a subclass
        # of Exception in Python 3.9+).
        error_type = type(e).__name__
        logger.error(f"CodeGenerationAgent unexpected error: {e}")
        messages.append(
            make_agent_message(
                agent="CodeGenerationAgent",
                content=f"Unexpected error: {error_type}: {e}",
            )
        )
        return {
            **state,
            "messages": messages,
            "build": {
                **build,
                "code": previous_code,
                "validation_status": "codegen_failed",
            },
        }

    # execute() returns (GeneratedCode, metadata_dict)
    if isinstance(result, tuple):
        generated_code, meta = result
    else:
        generated_code = result
        meta = {}

    # Propagate iteration_dir from codegen to build state so execution_agent reuses it
    if isinstance(meta, dict) and meta.get("iteration_dir"):
        build["iteration_dir"] = meta["iteration_dir"]

    if not isinstance(generated_code, GeneratedCode):
        code_dict = (
            generated_code
            if isinstance(generated_code, dict)
            else generated_code.model_dump()
        )
        generated_code = GeneratedCode(**code_dict)

    code_artifact = _make_code_artifact(generated_code.model_dump())
    validation_status = generated_code.validation_status or "unknown"

    if validation_status == "validation_failed":
        build_update = {
            **build,
            "code": code_artifact,
            "validation_status": "failed",
            "attempted_fixes": (build.get("attempted_fixes") or [])
            + [str(feedback or "validation_failed")],
        }
    else:
        detail = (
            "Generated simulation.py"
            if previous_code is None
            else f"Updated simulation.py from {feedback}"
        )
        messages.append(
            make_agent_message(
                agent="CodeGenerationAgent",
                content=detail,
            )
        )
        build_update = {
            **build,
            "code": code_artifact,
            "validation_status": str(validation_status),
            "build_success": True,
            "retry_count": build.get("retry_count", 0),
        }

    return {
        **state,
        "messages": messages,
        "build": build_update,
        "code": generated_code.model_dump(),
    }


# =========================================================================
# Node: execution_node (Agent 3)
# =========================================================================

async def execution_node(state: WorkflowState) -> WorkflowState:
    """Run simulation; set execution and execution_feedback on failure."""
    logger.info("[Review] Execution")
    messages = list(state.get("messages", []))
    state["phase"] = "review_execution"
    state["progress"] = {"step": "execution_start", "progress_pct": 0}
    _append_progress_message(
        messages,
        agent="ExecutionAgent",
        content="Execution started",
        phase="review_execution",
        step="execution_start",
        progress_pct=0,
    )

    settings = get_settings()
    execution_timeout = settings.execution_timeout
    agent = ExecutionAgent(timeout=execution_timeout)
    code = GeneratedCode(**state["code"])
    review = _get_review(state)
    build = _get_build(state)
    trace = dict(review.get("trace") or {})
    trace["run_simulation"] = _build_execution_trace_snapshot(
        plan=state.get("plan"),
        code=state.get("code"),
        review=review,
    )
    state["review"] = {
        **review,
        "trace": trace,
    }

    # Hard pre-execute Chrono-API validation gate. On FAIL, mutates state to
    # look like a normal execution failure (synthetic ExecutionResult, bumped
    # retry, populated execution_feedback) and returns early — the subprocess
    # is never started. See _pre_execute_api_validation_gate() docstring.
    gate_state = _pre_execute_api_validation_gate(state, code, trace)
    if gate_state is not None:
        n_violations = len(
            (gate_state.get("review") or {}).get("current_execution_error_messages") or []
        )
        messages.append(make_agent_message(
            agent="ExecutionAgent",
            content=(
                "Execution skipped — pre-execute Chrono API validation FAILED "
                f"({n_violations} violations); routing back to codegen."
            ),
            metadata={
                "source": "pre_execute_gate",
                "violations": n_violations,
                "success": False,
            },
        ))
        gate_state["messages"] = messages
        gate_state["current_step"] = "execution_complete"
        gate_state["phase"] = "review_execution"
        gate_state["progress"] = {
            "step": "execution_finish",
            "progress_pct": 100,
            "success": False,
        }
        emit_progress_event(
            phase="review_execution",
            step="execution_finish",
            progress_pct=100,
            message=f"Execution skipped (pre-execute gate FAIL, {n_violations} violations)",
            success=False,
        )
        return gate_state

    # Start display server for headless rendering (Xvfb + noVNC)
    display_server = None
    frame_capture = None
    if settings.enable_headless_mode:
        try:
            from chrono_code.utils.display_server import DisplayServer
            from chrono_code.utils.frame_capture import XvfbFrameCapture

            display_server = DisplayServer()
            novnc_url = await display_server.start()
            import os
            os.environ["DISPLAY"] = display_server.display
            emit_custom_event({"type": "simulation_viewer", "novnc_url": novnc_url})
            logger.info(f"Display server started: {novnc_url}")

            # Start periodic Xvfb screenshot capture for VLM review
            frame_capture = XvfbFrameCapture(
                display=display_server.display,
                output_dir=Path(settings.visualization_output_path) / "xvfb_captures",
            )
            await frame_capture.start()
        except Exception as e:
            logger.warning(f"Failed to start display server: {e}")
            display_server = None
            frame_capture = None

    iteration_dir_str = build.get("iteration_dir") if isinstance(build, dict) else None
    iteration_dir = Path(iteration_dir_str) if iteration_dir_str else None

    # Sync the VSG window-title's "Step N of M" with the workflow's actual
    # current step, regardless of what codegen may have hard-coded into the
    # SetWindowTitle literal. ``step_loop`` is absent for pure-MBS plans —
    # in that case we pass step_info=None and the title is only ASCII-sanitized.
    step_loop = state.get("step_loop") or {}
    step_info: Optional[Dict[str, Any]] = None
    steps_list = step_loop.get("steps") or []
    if steps_list:
        idx = int(step_loop.get("current_step_index") or 0)
        if 0 <= idx < len(steps_list):
            step_info = {
                "step_number": idx + 1,
                "total_steps": len(steps_list),
            }

    try:
        execution = await agent.execute(
            generated_code=code,
            iteration_dir=iteration_dir,
            step_info=step_info,
        )
    finally:
        if frame_capture is not None:
            await frame_capture.stop()
        if display_server is not None:
            await display_server.stop()

    if not execution.success and (
        getattr(execution, "execution_log", None) or getattr(execution, "error_message", None)
    ):
        raw = getattr(execution, "execution_log", "") or getattr(execution, "error_message", "")
        rc = getattr(execution, "return_code", None)
        signal_codes = {-11, -6, -8, -7}
        title = "Execution failure: stderr" if rc in signal_codes else "Execution failure"
        log_boxed(logger, logging.ERROR, title, raw[:5000])

    messages.append(make_agent_message(
        agent="ExecutionAgent",
        content=f"Execution {'SUCCESS' if execution.success else 'FAILED'} in {execution.runtime_seconds:.2f}s",
        metadata={
            "success": execution.success,
            "output_files": len(execution.output_files),
            "return_code": execution.return_code,
            "runtime_seconds": execution.runtime_seconds,
        },
    ))

    state["execution"] = execution.model_dump()
    state["messages"] = messages
    state["current_step"] = "execution_complete"
    state["phase"] = "review_execution"
    state["progress"] = {
        "step": "execution_finish",
        "progress_pct": 100,
        "success": execution.success,
    }
    emit_progress_event(
        phase="review_execution",
        step="execution_finish",
        progress_pct=100,
        message=f"Execution {'SUCCESS' if execution.success else 'FAILED'}",
        success=execution.success,
    )
    trace["run_simulation"] = _build_execution_trace_snapshot(
        plan=state.get("plan"),
        code=state.get("code"),
        review=review,
        execution=execution,
    )

    if not execution.success:
        err = execution.error_message or ""
        log = execution.execution_log or ""
        rc = execution.return_code
        runtime = execution.runtime_seconds
        combined = err or log
        existing_fps = review.get("error_fingerprints", {})
        existing_msgs = review.get("execution_error_messages", [])
        new_err_msgs = getattr(execution, "error_messages", [])
        current_error_messages = _current_error_messages(new_err_msgs, combined)

        error_messages, updated_fingerprints = _semantic_deduplicate_errors(
            existing_msgs,
            current_error_messages,
            existing_fps,
        )
        feedback = _format_execution_feedback(
            rc=rc,
            runtime=runtime,
            error_messages=current_error_messages,
            combined=combined,
            structured_error=getattr(execution, "structured_error", None),
        )
        state["review"] = {
            **review,
            "execution": execution.model_dump(),
            "trace": trace,
            "execution_feedback": feedback,
            "current_execution_error_messages": current_error_messages,
            "execution_error_messages": error_messages,
            "error_fingerprints": updated_fingerprints,
            "vlm_feedback": None,
        }
        _bump_execution_retry(state, "execution")
        _backfill_previous_code(state)
        logger.info("Set execution_feedback for next build")
    else:
        state["review"] = {
            **review,
            "execution": execution.model_dump(),
            "trace": trace,
            "execution_feedback": None,
            "current_execution_error_messages": [],
            "execution_error_messages": review.get("execution_error_messages", []),
            "vlm_feedback": None,
            "last_failure_source": None,
        }

    return state


# =========================================================================
# Node: vlm_review_node (Agent 4 - visual description)
# =========================================================================

async def vlm_review_node(state: WorkflowState) -> WorkflowState:
    """Generate a visual description only; physics analysis will interpret it later."""
    logger.info("[Review] VLM Visual Description")
    messages = list(state.get("messages", []))
    state["phase"] = "review_vlm"
    state["progress"] = {"step": "vlm_start", "progress_pct": 0}
    _append_progress_message(
        messages,
        agent="VLMReviewAgent",
        content="VLM visual description started",
        phase="review_vlm",
        step="vlm_start",
        progress_pct=0,
    )

    agent = ReviewAgent()
    plan = SimulationPlan(**state["plan"])
    execution = state["execution"]
    code_raw = state.get("code", {})
    generated_code = code_raw.get("code", "") if isinstance(code_raw, dict) else ""

    try:
        vlm_result = await agent.review_visual_output(plan, execution, generated_code=generated_code)
    except AgentLLMError as exc:
        error_msg = _format_agent_llm_error(exc)
        messages.append(make_agent_message(
            agent="VLMReviewAgent",
            content=error_msg,
            metadata={"error_type": "llm_error", "operation": exc.operation},
        ))
        state["messages"] = messages
        state["current_step"] = "vlm_review_failed"
        state["phase"] = "review_vlm"
        state["progress"] = {
            "step": "vlm_llm_error",
            "progress_pct": 100,
            "error_msg": error_msg[:500],
        }
        emit_progress_event(
            phase="review_vlm",
            step="vlm_llm_error",
            progress_pct=100,
            message=error_msg,
            error_msg=error_msg[:500],
        )
        state["error"] = error_msg
        return state

    review = _get_review(state)
    description = (vlm_result.video_description or vlm_result.feedback or "").strip()
    messages.append(make_agent_message(
        agent="VLMReviewAgent",
        content="VLM visual description generated",
        metadata={
            "description_length": len(description),
            "analyzed_frames": len(vlm_result.analyzed_frames or []),
        },
    ))
    if description:
        messages.append(make_agent_message(
            agent="VLMReviewAgent",
            content=description,
            metadata={"detail_type": "video_description"},
        ))

    state["vlm_review"] = vlm_result.model_dump()
    state["review"] = {
        **review,
        "vlm_review": vlm_result.model_dump(),
        "video_description": description,
        "vlm_feedback": None,
        "execution_feedback": None,
        "last_failure_source": review.get("last_failure_source"),
    }
    state["messages"] = messages
    state["current_step"] = "vlm_review_complete"
    state["phase"] = "review_vlm"
    state["progress"] = {
        "step": "vlm_finish",
        "progress_pct": 100,
        "description_generated": bool(description),
    }
    emit_progress_event(
        phase="review_vlm",
        step="vlm_finish",
        progress_pct=100,
        message="VLM visual description complete",
        description_generated=bool(description),
    )
    return state


# =========================================================================
# Node: physics_analysis_node (Agent 4 - physics validation)
# =========================================================================

async def physics_analysis_node(state: WorkflowState) -> WorkflowState:
    """Agent4 LLM physics validation of CSV results against plan physics laws.

    For scene plans: uses deterministic CSV validation first (if scene_placement.csv exists).
    For MBS plans: uses LLM-based analysis.

    Routes to:
      - physics_valid: CSV confirms physics; sets final_output and exits loop.
      - physics_invalid: CSV shows clear violations; sends feedback to codegen.
      - physics_uncertain: CSV data is missing or ambiguous; loops back to codegen.
    """
    logger.info("[Review] Physics Analysis")
    messages = list(state.get("messages", []))
    _append_progress_message(
        messages,
        agent="ExecutionAgent",
        content="Physics analysis started",
        phase="review_physics",
        step="physics_start",
        progress_pct=0,
    )
    state["phase"] = "review_physics"
    state["progress"] = {"step": "physics_start", "progress_pct": 0}

    settings = get_settings()
    plan_dict = state.get("plan") or {}
    plan_obj = SimulationPlan(**plan_dict)

    # --- Scene plans: deterministic CSV validation (all predicates) ---
    output_dir = Path(settings.visualization_output_path).resolve() if hasattr(settings, "visualization_output_path") else Path("outputs")
    placement_csv = output_dir / "scene_placement.csv"
    contacts_csv = output_dir / "scene_contacts.csv"

    if plan_obj.plan_type == "scene" and placement_csv.exists():
        try:
            from chrono_code.validators.scene_placement import validate_scene_placement

            topo = plan_obj.topology
            physical_preds = [p.model_dump() for p in (topo.physical_predicates or [])] if topo else []
            scene_preds = [p.model_dump() for p in (topo.scene_predicates or [])] if topo else []
            gravity_axis = topo.gravity_axis if topo else "-z"

            csv_result = validate_scene_placement(
                placement_csv_path=str(placement_csv),
                contacts_csv_path=str(contacts_csv) if contacts_csv.exists() else None,
                physical_predicates=physical_preds,
                scene_predicates=scene_preds,
                gravity_axis=gravity_axis,
                dynamic_bodies=plan_obj.derive_dynamic_bodies(),
            )
            logger.info("[PhysicsAnalysis] CSV validation: %s — %s", csv_result.verdict, csv_result.summary)

            review = _get_review(state)
            physics_result = {
                "verdict": csv_result.verdict,
                "reasoning": csv_result.summary,
                "violations": [r.reason for r in csv_result.predicate_results if not r.passed],
                "suggested_fix": None,
                "source": "csv_deterministic",
            }
            if not csv_result.stability_passed:
                physics_result["violations"].append(csv_result.stability_detail)

            messages.append(make_agent_message(
                agent="ExecutionAgent",
                content=f"Physics analysis (CSV): verdict={csv_result.verdict}. {csv_result.summary[:200]}",
                metadata={"verdict": csv_result.verdict, "violations": physics_result["violations"], "source": "csv"},
            ))

            review["physics_analysis"] = physics_result

            if csv_result.verdict == "physics_valid":
                review["last_failure_source"] = None
                review["physics_feedback"] = "Physics validation passed (CSV deterministic)."
                state["current_step"] = "physics_valid"
                state["execution_complete"] = True
                state["final_output"] = {
                    "success": True,
                    "physics_analysis": physics_result,
                    "video_description": review.get("video_description"),
                }
            else:
                violations_text = "\n".join(f"- {v}" for v in physics_result["violations"])
                feedback = f"PHYSICS VALIDATION FAILED (CSV deterministic):\n{csv_result.summary}"
                if violations_text:
                    feedback += f"\n\nViolations:\n{violations_text}"
                review["physics_feedback"] = feedback
                review["execution_feedback"] = None
                review["vlm_feedback"] = None
                state["review"] = review
                _backfill_previous_code(state)
                _bump_execution_retry(state, "physics")
                review = _get_review(state)  # refresh for any follow-on tweaks
                state["current_step"] = "physics_invalid"

            state["review"] = review
            state["messages"] = messages
            state["phase"] = "review_physics"
            state["progress"] = {"step": "physics_finish", "progress_pct": 100, "verdict": csv_result.verdict}
            emit_progress_event(
                phase="review_physics",
                step="physics_finish",
                progress_pct=100,
                message=f"Physics analysis (CSV) verdict={csv_result.verdict}",
                verdict=csv_result.verdict,
            )
            return state
        except Exception as exc:
            csv_fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[PhysicsAnalysis] CSV validation failed, falling back to LLM: %s",
                csv_fallback_reason,
            )
            messages.append(make_agent_message(
                agent="ExecutionAgent",
                content=(
                    f"CSV physics validation unavailable ({csv_fallback_reason}) — "
                    "falling back to LLM physics analysis."
                ),
                metadata={
                    "csv_fallback_reason": csv_fallback_reason,
                    "source": "csv_fallback",
                },
            ))
            # Persist the reason so downstream LLM result can tag it for the user
            state.setdefault("review", {})
            state["review"]["csv_fallback_reason"] = csv_fallback_reason

    # --- MBS / fallback: LLM-based physics analysis ---
    results_dir = Path(settings.results_path).resolve() / "agent"
    agent = ExecutionAgent(timeout=settings.execution_timeout)
    plan = SimulationPlan(**plan_dict)
    execution = ExecutionResult(**state["execution"])

    try:
        physics_result = await agent.analyze_physics_with_llm(
            plan,
            execution,
            results_dir,
            video_description=str((_get_review(state).get("video_description") or "")),
        )
    except AgentLLMError as exc:
        error_msg = _format_agent_llm_error(exc)
        messages.append(make_agent_message(
            agent="ExecutionAgent",
            content=error_msg,
            metadata={"error_type": "llm_error", "operation": exc.operation},
        ))
        state["messages"] = messages
        state["current_step"] = "physics_analysis_failed"
        state["phase"] = "review_physics"
        state["progress"] = {
            "step": "physics_llm_error",
            "progress_pct": 100,
            "error_msg": error_msg[:500],
        }
        emit_progress_event(
            phase="review_physics",
            step="physics_llm_error",
            progress_pct=100,
            message=error_msg,
            error_msg=error_msg[:500],
        )
        state["error"] = error_msg
        return state
    verdict = physics_result.get("verdict", "physics_uncertain")

    messages.append(make_agent_message(
        agent="ExecutionAgent",
        content=f"Physics analysis: verdict={verdict}. {physics_result.get('reasoning', '')[:200]}",
        metadata={
            "verdict": verdict,
            "violations": physics_result.get("violations", []),
            "saved_csv_paths": physics_result.get("saved_csv_paths", []),
        },
    ))

    review = _get_review(state)
    if review.get("csv_fallback_reason") and isinstance(physics_result, dict):
        physics_result["csv_fallback_reason"] = review.get("csv_fallback_reason")
        physics_result.setdefault("source", "llm_after_csv_fallback")
    review["physics_analysis"] = physics_result

    if verdict == "physics_valid":
        review["last_failure_source"] = None
        review["physics_feedback"] = "Physics validation passed."
        state["current_step"] = "physics_valid"
        state["execution_complete"] = True
        state["final_output"] = {
            "success": True,
            "physics_analysis": physics_result,
            "video_description": review.get("video_description"),
        }

    elif verdict == "physics_invalid":
        r = physics_result.get("reasoning", "")
        violations = "\n".join(f"- {v}" for v in physics_result.get("violations", []))
        fix = physics_result.get("suggested_fix") or ""
        feedback = f"PHYSICS VALIDATION FAILED:\n{r}"
        if violations:
            feedback += f"\n\nViolations:\n{violations}"
        if fix:
            feedback += f"\n\nSuggested Fix: {fix}"
        review["physics_feedback"] = feedback
        review["execution_feedback"] = None
        review["vlm_feedback"] = None
        state["review"] = review
        _backfill_previous_code(state)
        # Also bump execution_retry_count here (the CSV-physics path already
        # does; the LLM path historically didn't, which let physics_invalid
        # loop forever without tripping the retry cap).
        _bump_execution_retry(state, "physics")
        review = _get_review(state)
        state["current_step"] = "physics_invalid"

    else:  # physics_uncertain — continue to visual_review
        r = physics_result.get("reasoning") or ""
        violations = "\n".join(f"- {v}" for v in physics_result.get("violations", []))
        fix = physics_result.get("suggested_fix") or ""
        feedback = f"PHYSICS UNCERTAIN: {r}"
        if violations:
            feedback += f"\n\nViolations:\n{violations}"
        if fix:
            feedback += f"\n\nSuggested Fix: {fix}"
        review["physics_feedback"] = feedback
        review["execution_feedback"] = None
        review["vlm_feedback"] = None
        state["review"] = review
        _backfill_previous_code(state)
        _bump_execution_retry(state, "physics")
        review = _get_review(state)
        state["current_step"] = "physics_uncertain"

    state["review"] = review
    state["messages"] = messages
    state["phase"] = "review_physics"
    state["progress"] = {"step": "physics_finish", "progress_pct": 100, "verdict": verdict}
    emit_progress_event(
        phase="review_physics",
        step="physics_finish",
        progress_pct=100,
        message=f"Physics analysis verdict={verdict}",
        verdict=verdict,
    )
    return state


# =========================================================================
# Node: step_review_node (step-by-step scene review)
# =========================================================================

# Maximum VLM retries on transient SDK/network failures. Set to 2 so a
# flaky API pass still has two bites at the apple before we give up and
# surface ``vlm_hard_failed`` to the caller. Set to 0 via VLM_MAX_RETRIES
# env if you want the old one-shot behavior.
_VLM_MAX_RETRIES = int(__import__("os").environ.get("VLM_MAX_RETRIES", "2"))


async def _get_vlm_description(
    agent: ReviewAgent,
    step_desc: str,
    step_idx: int,
    steps: list,
    completed: list,
    plan: Optional[SimulationPlan] = None,
    scene_objects_manifest: str = "(no procedural scene objects)",
    plan_assets_manifest: str = "(no plan-level external assets)",
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Call VLM to describe the current scene.

    Returns a dict:
      * ``description`` (str): rendered description (or a fallback string)
      * ``vlm_hard_failed`` (bool): True iff every retry raised an SDK
        exception — this means the observer is broken, NOT that the
        scene is empty. Upstream should NOT blame codegen.

    Retries up to ``_VLM_MAX_RETRIES`` times on transient errors (network,
    rate limit, malformed response). Non-transient "scene has nothing"
    cases are returned normally as a description (not a hard-fail).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_VLM_MAX_RETRIES + 1):
        try:
            vlm_result = await agent.describe_step_scene(
                step_description=step_desc,
                step_number=step_idx + 1,
                total_steps=len(steps),
                completed_steps=completed,
                plan=plan,
                scene_objects_manifest=scene_objects_manifest,
                plan_assets_manifest=plan_assets_manifest,
                image_paths=image_paths,
            )
            desc = vlm_result.get("description", "")
            visible_objects = vlm_result.get("visible_objects", [])
            objects_structured = vlm_result.get("objects", [])
            observations = vlm_result.get("spatial_observations", [])

            parts = [desc]
            if visible_objects:
                parts.append(
                    "Visible objects: "
                    + ", ".join(str(o) for o in visible_objects)
                )
            # New schema: render the structured per-object table the
            # decision prompt expects (presence + motion_state + location).
            if objects_structured:
                table = ["Per-object report (canonical name | present | motion_state | location):"]
                for entry in objects_structured:
                    name = entry.get("name") or "?"
                    present = "yes" if entry.get("present") else "no"
                    motion = entry.get("motion_state") or "unclear"
                    location = entry.get("location") or ""
                    suffix = f" — {location}" if location else ""
                    table.append(f"  - {name} | {present} | {motion}{suffix}")
                parts.append("\n".join(table))
            # Legacy fallback for older describer responses that still emit
            # spatial_observations (kept so a mid-flight prompt swap doesn't
            # silently drop signal).
            if observations:
                parts.append(
                    "Spatial observations (legacy): "
                    + "; ".join(str(o) for o in observations)
                )
            return {"description": "\n".join(parts), "vlm_hard_failed": False}
        except Exception as exc:
            last_exc = exc
            remaining = _VLM_MAX_RETRIES - attempt
            if remaining > 0:
                logger.warning(
                    "[StepReview] VLM describe failed (attempt %d/%d), retrying: %s",
                    attempt + 1, _VLM_MAX_RETRIES + 1, exc,
                )
            else:
                logger.error(
                    "[StepReview] VLM describe failed after %d attempts, giving up: %s",
                    _VLM_MAX_RETRIES + 1, exc,
                )
    return {
        "description": (
            f"No visual description available (VLM call failed after "
            f"{_VLM_MAX_RETRIES + 1} attempts: {last_exc})."
        ),
        "vlm_hard_failed": True,
    }


def _build_scene_objects_manifest(plan_dict: Dict[str, Any]) -> str:
    """Render plan.scene_objects[] as a markdown bullet list keyed on name.

    Each entry exposes the fields the review LLM uses to map free-text
    step descriptions to canonical names: role (e.g.
    ``static_support_platform``), construction_source, primitive +
    size (or domain_type), and the fixed flag. The format intentionally
    mirrors ``plan_assets_manifest`` so both manifests sit side-by-side
    in the same review prompt.
    """
    objs = plan_dict.get("scene_objects") or []
    lines: List[str] = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        attrs: List[str] = []
        for key in ("role", "construction_source", "primitive",
                    "domain_type", "fsi_registration"):
            val = obj.get(key)
            if val:
                attrs.append(f"{key}={val}")
        size = obj.get("size")
        if size:
            attrs.append(f"size={size}")
        if obj.get("fixed") is True:
            attrs.append("fixed=true")
        elif obj.get("fixed") is False:
            attrs.append("fixed=false")
        desc = obj.get("description")
        suffix = f" — {desc}" if isinstance(desc, str) and desc.strip() else ""
        lines.append(f"- {name}: {', '.join(attrs)}{suffix}")
    return "\n".join(lines) or "(no procedural scene objects)"


async def _get_review_decision(
    agent: ReviewAgent,
    step_desc: str,
    step_idx: int,
    steps: list,
    completed: list,
    csv_summary: str,
    vlm_description: str,
    csv_passed: bool,
    codegen_rebuttal: str = "",
    plan_assets_manifest: str = "(no plan-level external assets)",
    scene_objects_manifest: str = "(no procedural scene objects)",
    step_motion_expectations: list | None = None,
) -> dict:
    """Call review agent LLM to make final pass/fail decision based on CSV + VLM."""
    try:
        decision = await agent.review_step_decision(
            step_description=step_desc,
            step_number=step_idx + 1,
            total_steps=len(steps),
            completed_steps=completed,
            csv_summary=csv_summary,
            vlm_description=vlm_description,
            codegen_rebuttal=codegen_rebuttal,
            plan_assets_manifest=plan_assets_manifest,
            scene_objects_manifest=scene_objects_manifest,
            step_motion_expectations=step_motion_expectations,
        )
        return {
            "pass": bool(decision.get("pass", False)),
            "reasoning": decision.get("reasoning", ""),
            "issues": decision.get("issues", []),
        }
    except Exception as exc:
        logger.warning("[StepReview] Review decision LLM failed, using CSV result: %s", exc)
        return {
            "pass": csv_passed,
            "reasoning": f"Review agent unavailable, using CSV result (passed={csv_passed})",
            "issues": [],
        }


def _format_deterministic_findings(csv_result: Any) -> str:
    """Render a ``SceneValidationResult`` as a structured findings block.

    Output shape (groups predicates by ``kind``; FAILed kinds first so the
    judge sees them at the top of the LLM context window):

    ```
    ## Deterministic Findings (CSV-backed, authoritative for the listed checks)

    WHEEL_LANDING: FAIL
      - spindle_RR bottom z=1.45m is 0.12m ABOVE left_platform top z=1.30m ...
      - spindle_RL bottom z=1.31m sits on left_platform top — landed

    INTERPENETRATION: PASS
      - chassis & left_platform AABBs disjoint or within tolerance ...

    FLUID_CONTAINMENT: FAIL
      - 247 of 5000 SPH particles escape water_tank_boundary AABB ...

    OTHER PREDICATES: PASS
      - {...rests_on...}: gap=0.012m, footprint_overlap=yes
    ```
    """
    if csv_result is None:
        return ""

    # Group results by kind. Empty kind → "OTHER PREDICATES" bucket.
    by_kind: Dict[str, List[Any]] = {}
    for r in csv_result.predicate_results:
        key = (r.kind or "other_predicates").upper()
        by_kind.setdefault(key, []).append(r)

    # Order: hard-override kinds first (FAILed before PASSed within a kind),
    # then OTHER_PREDICATES, then stability.
    order = [
        "SUPPORT_SURFACE_COLLISION",
        "WHEEL_LANDING",
        "NO_INTERPENETRATION",
        "FLUID_CONTAINMENT",
        "OTHER_PREDICATES",
    ]
    keys = [k for k in order if k in by_kind] + [k for k in by_kind if k not in order]

    lines: List[str] = [
        "## Deterministic Findings (CSV-backed, authoritative for the listed checks)",
    ]
    label_map = {
        "SUPPORT_SURFACE_COLLISION": "SUPPORT_SURFACE",
        "WHEEL_LANDING": "WHEEL_LANDING",
        "NO_INTERPENETRATION": "INTERPENETRATION",
        "FLUID_CONTAINMENT": "FLUID_CONTAINMENT",
        "OTHER_PREDICATES": "OTHER PREDICATES",
    }
    for key in keys:
        rs = by_kind[key]
        any_fail = any(not r.passed for r in rs)
        lines.append("")
        lines.append(f"{label_map.get(key, key)}: {'FAIL' if any_fail else 'PASS'}")
        # Within a kind, FAIL items first.
        rs_sorted = sorted(rs, key=lambda r: r.passed)
        for r in rs_sorted:
            lines.append(f"  - {r.reason}")

    # Stability sits outside the predicate kinds.
    if not csv_result.stability_passed:
        lines.append("")
        lines.append(f"STABILITY: FAIL")
        lines.append(f"  - {csv_result.stability_detail}")
    return "\n".join(lines)


async def step_review_node(state: WorkflowState) -> WorkflowState:
    """Review a single scene-building step: CSV validation -> VLM description -> review agent decision.

    Flow:
    1. CSV deterministic validation (if available)
    2. VLM describes the scene (no pass/fail judgment)
    3. Review agent LLM makes final blocking decision based on CSV + VLM description

    On pass: advances step_loop.current_step_index.
    On fail: sets step_loop.step_feedback for codegen retry.
    """
    step_loop = dict(state.get("step_loop") or {})
    step_desc = step_loop.get("current_step_description", "")
    step_idx = step_loop.get("current_step_index", 0)
    steps = step_loop.get("steps", [])
    completed = list(step_loop.get("completed_steps", []))
    messages = list(state.get("messages", []))

    logger.info("[StepReview] Reviewing step %d/%d: %s", step_idx + 1, len(steps), step_desc)
    emit_progress_event(
        phase="step_review",
        step="step_review_start",
        progress_pct=0,
        message=f"Reviewing step {step_idx + 1}/{len(steps)}",
    )

    plan_dict = state.get("plan") or {}
    step_plan_obj = SimulationPlan(**plan_dict)
    settings = get_settings()
    output_dir = Path(settings.visualization_output_path).resolve() if hasattr(settings, "visualization_output_path") else Path("outputs")
    placement_csv = output_dir / "scene_placement.csv"
    contacts_csv = output_dir / "scene_contacts.csv"
    particles_csv = output_dir / "particles.csv"
    links_csv = output_dir / "scene_links.csv"
    motion_csv = output_dir / "cam" / "motion_log.csv"

    agent = ReviewAgent()

    # --- Phase 1: CSV deterministic validation (legacy scene placement) ---
    csv_passed = None
    csv_description = ""
    csv_issues: list = []
    # ``hard_override_findings`` carries the structured PredicateResult for any
    # ``kind in {wheel_landing, no_interpenetration, fluid_containment}`` that
    # FAILed. When non-empty, we short-circuit Phase 2/3 (skip both LLM calls)
    # because deterministic CSV-backed evidence outranks visual interpretation.
    hard_override_findings: list = []
    structured_findings_block = ""

    if step_plan_obj.plan_type in ("scene", "mbs_in_scene", "fsi_in_scene") and placement_csv.exists():
        try:
            from chrono_code.validators.scene_placement import validate_scene_placement

            topo = step_plan_obj.topology
            physical_preds = [p.model_dump() for p in (topo.physical_predicates or [])] if topo else []
            scene_preds = [p.model_dump() for p in (topo.scene_predicates or [])] if topo else []
            gravity_axis = topo.gravity_axis if topo else "-z"
            relevant = set(step_loop.get("relevant_bodies", []))

            csv_result = validate_scene_placement(
                placement_csv_path=str(placement_csv),
                contacts_csv_path=str(contacts_csv) if contacts_csv.exists() else None,
                physical_predicates=physical_preds,
                scene_predicates=scene_preds,
                gravity_axis=gravity_axis,
                relevant_bodies=relevant if relevant else None,
                dynamic_bodies=step_plan_obj.derive_dynamic_bodies(),
                scene_objects=list(plan_dict.get("scene_objects") or []),
                particles_csv_path=str(particles_csv) if particles_csv.exists() else None,
                plan_type=step_plan_obj.plan_type,
                links_csv_path=str(links_csv) if links_csv.exists() else None,
            )
            logger.info("[StepReview] CSV validation: %s — %s", csv_result.verdict, csv_result.summary)

            csv_passed = csv_result.verdict == "physics_valid"
            csv_description = csv_result.summary
            csv_issues = [r.reason for r in csv_result.predicate_results if not r.passed]
            if not csv_result.stability_passed:
                csv_issues.append(csv_result.stability_detail)

            # Identify hard-override findings (the three deterministic
            # predicates added per session 103931). Their FAIL skips the
            # downstream LLM calls entirely.
            hard_override_findings = [
                r for r in csv_result.predicate_results
                if not r.passed and r.kind in
                {
                    "wheel_landing",
                    "no_interpenetration",
                    "fluid_containment",
                    "support_surface_collision",
                }
            ]

            # Build the structured findings block fed into the review prompt.
            # Even when no hard-override fires, the block shows numerical
            # PASS evidence so the LLM knows what was already verified.
            structured_findings_block = _format_deterministic_findings(csv_result)

            # Surface concrete CSV issues to the user log so they see exactly
            # which predicate / stability check failed, not just "FAIL".
            if csv_issues:
                logger.info(
                    "[StepReview] CSV issues:\n%s",
                    "\n".join(f"  - {i}" for i in csv_issues),
                )

            messages.append(make_agent_message(
                agent="StepReviewAgent",
                content=f"Step {step_idx + 1} CSV validation: {'PASS' if csv_passed else 'FAIL'} — {csv_description}",
                metadata={"step_index": step_idx, "passed": csv_passed, "issues": csv_issues, "source": "csv"},
            ))
        except Exception as exc:
            logger.warning("[StepReview] CSV validation error, continuing to VLM: %s", exc)

    # --- Hard-override short-circuit ---
    # Any deterministic FAIL on the three CSV-backed predicates outranks LLM
    # judgment. Bypass the describe_step_scene + review_step_decision LLM
    # calls; the step is failed with the structured findings as feedback.
    # Also saves two VLM calls per failed step.
    if hard_override_findings:
        issue_lines = [r.reason for r in hard_override_findings]
        kinds = sorted({r.kind for r in hard_override_findings})
        feedback = (
            "Hard-override deterministic findings failed: "
            + ", ".join(kinds)
            + ".\n"
            + "\n".join(f"  - {r}" for r in issue_lines)
        )
        logger.info(
            "[StepReview] Hard-override: %d deterministic finding(s) failed (%s); "
            "skipping LLM describe/decide.",
            len(hard_override_findings), kinds,
        )
        step_loop["step_feedback"] = feedback
        step_loop["review_issues"] = issue_lines
        step_loop["step_retry_count"] = int(step_loop.get("step_retry_count") or 0) + 1
        _backfill_previous_code(state)
        _track_step_no_progress(step_loop, state)
        messages.append(make_agent_message(
            agent="StepReviewAgent",
            content=(
                f"Step {step_idx + 1} FAILED (hard-override): "
                f"{', '.join(kinds)}"
            ),
            metadata={
                "step_index": step_idx,
                "passed": False,
                "source": "deterministic_hard_override",
                "kinds": kinds,
                "issues": issue_lines,
            },
        ))
        state["step_loop"] = step_loop
        state["messages"] = messages
        state["current_step"] = "step_review_complete"
        emit_progress_event(
            phase="step_review",
            step="step_review_finish",
            progress_pct=100,
            message=(
                f"Step {step_idx + 1} FAILED (hard-override: "
                f"{', '.join(kinds)})"
            ),
            passed=False,
        )
        return state

    # --- Phase 2a: Detect incomplete-recording *before* spending a VLM call.
    # When execution is killed mid-run (timeout / SIGKILL), recorder.release()
    # never runs and cam/ is left with only .inprogress.mp4 files — those are
    # unplayable (no moov atom). Opening them through VLM produces a bogus
    # "nothing to see" answer that step_review would otherwise blame on
    # codegen. The right recovery is to re-execute, NOT to mutate the code.
    cams_dir = output_dir / "cam"
    recording_info = ReviewAgent.detect_incomplete_recording(cams_dir)
    if recording_info.get("incomplete"):
        _names = recording_info.get("inprogress_names", [])
        environmental_msg = (
            f"ENVIRONMENT FAILURE (not a code bug): the simulation process "
            f"appears to have been terminated before its video recorders "
            f"could finalize — {len(_names)} unfinalized "
            f".inprogress.mp4 file(s) remain in cam/: {_names}. "
            f"The generated code is likely correct; re-executing should "
            f"produce valid video for visual review."
        )
        logger.warning("[StepReview] %s", environmental_msg)
        step_loop["step_feedback"] = environmental_msg
        # Do NOT bump step_retry_count — this retry does not count against
        # the step budget since the model isn't being asked to change code.
        step_loop["review_issues"] = [
            f"Unfinalized recording: {_names} (execution was interrupted)"
        ]
        step_loop["environmental_failure"] = True
        _backfill_previous_code(state)
        messages.append(make_agent_message(
            agent="StepReviewAgent",
            content=(
                f"Step {step_idx + 1} review deferred: {environmental_msg}"
            ),
            metadata={
                "step_index": step_idx,
                "passed": False,
                "environmental": True,
                "source": "environment_incomplete_recording",
            },
        ))
        state["step_loop"] = step_loop
        state["messages"] = messages
        state["current_step"] = "step_review_complete"
        emit_progress_event(
            phase="step_review",
            step="step_review_finish",
            progress_pct=100,
            message=f"Step {step_idx + 1} DEFERRED (environment: "
                    f"unfinalized video)",
            passed=False,
            environmental=True,
        )
        return state

    # Build the procedural-scene-object manifest. Lives alongside the
    # plan_assets_manifest below so both the VLM describe step and the
    # final review decision see the canonical names of every procedural
    # body the plan declares (left_platform, right_platform,
    # water_tank_boundary, sph_water, floating_plate, ...). Without this
    # the planner's free-text "concrete platform supports" never lines
    # up with the VLM's free-text "gray rectangular beams" or with the
    # actual scene_object names from the plan, and the review LLM has to
    # invent the mapping itself.
    scene_objects_manifest = _build_scene_objects_manifest(plan_dict)

    # Build the plan-level asset manifest (vehicles, robots, external assets)
    # EARLY so it can also flow into the VLM describer prompt — the describer
    # needs a per-asset row for every dynamic vehicle so its structured
    # ``objects`` array carries a ``motion_state`` for the vehicle, not just
    # narrative prose. Without this the judge sees "drives onto plate" in the
    # description but no ``motion_state=moving`` evidence to verify against,
    # and confabulated motion verbs slip through (session_20260429_071454
    # iter_010: stationary Polaris narrated as "drives onto floating_plate").
    _plan_assets_for_describe = plan_dict.get("assets") or []
    plan_assets_manifest = "\n".join(
        f"- {a.get('name', '')}: type={a.get('type', '')}, "
        f"is_dynamic={a.get('is_dynamic', False)}"
        for a in _plan_assets_for_describe
        if isinstance(a, dict) and a.get("name")
    ) or "(no plan-level external assets)"

    # --- Phase 2b: VLM scene description (no judgment) ---
    # Collect cam media exactly once per step. The VLM describer is the
    # only stage that gets pixels; the downstream judge runs text-only on
    # the describer's narration + csv_summary + manifests.
    _step_cam_images = agent._collect_cam_images(output_dir / "cam")
    vlm_result = await _get_vlm_description(
        agent, step_desc, step_idx, steps, completed, plan=step_plan_obj,
        scene_objects_manifest=scene_objects_manifest,
        plan_assets_manifest=plan_assets_manifest,
        image_paths=_step_cam_images,
    )
    vlm_description = vlm_result["description"]
    vlm_hard_failed = bool(vlm_result.get("vlm_hard_failed"))
    # Hard VLM failure (SDK/network): treat like environmental incomplete —
    # don't let step_review blame codegen when our observer is the thing
    # that's broken. step_retry_count still isn't bumped; the retry asks
    # for the same code re-executed so a fresh VLM attempt can run.
    if vlm_hard_failed:
        logger.warning(
            "[StepReview] VLM hard-failed; treating as environmental failure, "
            "not blaming codegen."
        )
        step_loop["step_feedback"] = (
            f"ENVIRONMENT FAILURE (VLM observer unavailable): the VLM "
            f"describe call failed after retries. The generated code is "
            f"likely correct; re-executing will retry the VLM on the "
            f"fresh output."
        )
        step_loop["review_issues"] = ["VLM call failed after retries"]
        step_loop["environmental_failure"] = True
        _backfill_previous_code(state)
        messages.append(make_agent_message(
            agent="StepReviewAgent",
            content=f"Step {step_idx + 1} review deferred: VLM hard-failed.",
            metadata={
                "step_index": step_idx,
                "passed": False,
                "environmental": True,
                "source": "environment_vlm_hard_fail",
            },
        ))
        state["step_loop"] = step_loop
        state["messages"] = messages
        state["current_step"] = "step_review_complete"
        emit_progress_event(
            phase="step_review",
            step="step_review_finish",
            progress_pct=100,
            message=f"Step {step_idx + 1} DEFERRED (environment: VLM down)",
            passed=False,
            environmental=True,
        )
        return state

    messages.append(make_agent_message(
        agent="ReviewAgent",
        content=f"Step {step_idx + 1} VLM description: {vlm_description}",
        metadata={"source": "vlm_describe", "step_index": step_idx},
    ))
    # Echo the VLM's description to the user-facing log so they can see what
    # the review is actually grounded on BEFORE the pass/fail decision fires.
    # Truncate long descriptions to keep the terminal readable; the full text
    # still lives in the dialog transcript and in ``messages``.
    _vlm_preview = (vlm_description or "").strip().replace("\n", " ")
    if len(_vlm_preview) > 500:
        _vlm_preview = _vlm_preview[:497] + "..."
    logger.info(
        "[StepReview] Step %d VLM says: %s",
        step_idx + 1, _vlm_preview or "(no description returned)",
    )

    # --- Phase 3: Review agent final decision ---
    # Pull this step's motion_expectations once — used both here (to decide
    # whether a missing CSV is actually a fail signal) and by the
    # per-body motion-summary block further down.
    _curr_step_dict: Dict[str, Any] = {}
    try:
        _impl = plan_dict.get("implementation_steps") or []
        if 0 <= step_idx < len(_impl) and isinstance(_impl[step_idx], dict):
            _curr_step_dict = _impl[step_idx]
    except Exception:
        _curr_step_dict = {}
    _step_motion_expectations = [
        str(n).strip() for n in (_curr_step_dict.get("motion_expectations") or [])
        if str(n).strip()
    ]

    # Prefer the structured deterministic-findings block when CSV ran. This
    # gives the LLM judge per-predicate numerical evidence (gaps, overlap
    # volumes, particle counts, contact pairs) instead of a flat one-liner,
    # so its visual reasoning has quantitative anchors.
    if structured_findings_block:
        csv_summary = structured_findings_block
    elif csv_passed is not None:
        csv_summary = f"{'PASS' if csv_passed else 'FAIL'} — {csv_description}"
        if csv_issues:
            csv_summary += "\nIssues: " + "; ".join(csv_issues)
    elif _step_motion_expectations:
        # The step declared motion bodies but no CSV showed up. Distinguish
        # "wall-clock timeout killed the subprocess mid-loop" from "codegen
        # forgot the trajectory dump". The motion-summary block below
        # surfaces per-body "absent from CSV" rows; this is just the
        # header.
        _exec_log = ""
        try:
            _exec_log = (state.get("execution_result") or {}).get("execution_log", "") or ""
        except Exception:
            _exec_log = ""
        if "reached timeout" in _exec_log:
            csv_summary = (
                "motion_log.csv MISSING — the simulation subprocess was "
                "KILLED BY THE WALL-CLOCK TIMEOUT before the loop's "
                "finally-block could flush it, but this step declared "
                f"motion_expectations={_step_motion_expectations}. "
                "Prefer FAIL — codegen needs to lower particle count, "
                "coarsen dT, shorten t_end, or reduce render_fps so the "
                "loop completes."
            )
        else:
            csv_summary = (
                f"motion_log.csv MISSING despite motion_expectations="
                f"{_step_motion_expectations}. See the per-step motion "
                "summary below for per-body details."
            )
    else:
        # Step declared no motion_expectations: under the per-step motion
        # contract, no CSV is required. A missing CSV here is the
        # expected, non-failing condition — do NOT echo any timeout-doom
        # text the LLM could pick up as an issue.
        csv_summary = "No CSV requirements for this step."

    # Append the raw scene_placement.csv body table so the review LLM sees
    # per-body physics evidence (pos / |v| / |w| / dynamic flag) alongside
    # the VLM scene description. This is structured ground-truth data that
    # outweighs perception of overlay markers in the rendered mp4 — see
    # session_20260429_060447 where the VLM failed step 2 on "tank invisible"
    # because BCE marker dots dominated its frame interpretation. The table
    # is small (one row per body, typically < 20 bodies) and surfaces motion
    # evidence directly: a dynamic body with |v|=0 and pos == plan-predicate
    # position is a "physics-never-advanced" smell the LLM can spot.
    if placement_csv.exists():
        try:
            import csv as _csv
            import math as _math
            _dynamic_names = {
                a.get("name") for a in (plan_dict.get("assets") or [])
                if isinstance(a, dict) and a.get("is_dynamic")
            }
            _table_rows: list = []
            with placement_csv.open() as _f:
                for _r in _csv.DictReader(_f):
                    _v = _math.sqrt(
                        float(_r.get("vel_x") or 0) ** 2
                        + float(_r.get("vel_y") or 0) ** 2
                        + float(_r.get("vel_z") or 0) ** 2
                    )
                    _w = _math.sqrt(
                        float(_r.get("ang_vel_x") or 0) ** 2
                        + float(_r.get("ang_vel_y") or 0) ** 2
                        + float(_r.get("ang_vel_z") or 0) ** 2
                    )
                    _pos = (
                        f"({float(_r.get('pos_x') or 0):+.2f}, "
                        f"{float(_r.get('pos_y') or 0):+.2f}, "
                        f"{float(_r.get('pos_z') or 0):+.2f})"
                    )
                    _is_dyn = "✓" if _r.get("body_name") in _dynamic_names else ""
                    _table_rows.append(
                        f"| {_r.get('body_name', '?')} | {_pos} | {_v:.3f} | {_w:.3f} | {_is_dyn} |"
                    )
            if _table_rows:
                csv_summary += (
                    "\n\n## Body end-states (from scene_placement.csv)\n"
                    "| body | pos (x, y, z) | |v| (m/s) | |ω| (rad/s) | is_dynamic |\n"
                    "|---|---|---|---|---|\n"
                    + "\n".join(_table_rows)
                )
        except Exception as _exc:
            logger.warning(
                "[StepReview] could not inline scene_placement.csv into "
                "csv_summary: %s",
                _exc,
            )

    # Per-step motion summary for declared-moving bodies (cam/motion_log.csv).
    # ``_step_motion_expectations`` was hoisted above the csv_summary build
    # so both blocks share the same value. Codegen writes a per-render-frame
    # log; we surface Δp (start→end displacement) and peak |v| per body so
    # the review LLM can fail steps where a declared-moving body is stuck —
    # the canonical orphan-ChSystem / brake-stuck failure mode.
    if _step_motion_expectations and motion_csv.exists():
        try:
            import csv as _csv
            import math as _math
            # Group rows per body across the entire CSV — the simulation
            # writes one row per body per render frame, ordered by time.
            _by_body: Dict[str, list] = {}
            with motion_csv.open() as _mf:
                for _r in _csv.DictReader(_mf):
                    _name = (_r.get("body_name") or "").strip()
                    if not _name:
                        continue
                    try:
                        _by_body.setdefault(_name, []).append({
                            "t": float(_r.get("time") or 0.0),
                            "px": float(_r.get("pos_x") or 0.0),
                            "py": float(_r.get("pos_y") or 0.0),
                            "pz": float(_r.get("pos_z") or 0.0),
                            "vx": float(_r.get("vel_x") or 0.0),
                            "vy": float(_r.get("vel_y") or 0.0),
                            "vz": float(_r.get("vel_z") or 0.0),
                        })
                    except (ValueError, TypeError):
                        continue
            _motion_rows: list = []
            for _name in _step_motion_expectations:
                _rows = _by_body.get(_name) or []
                if not _rows:
                    _motion_rows.append(
                        f"| {_name} | absent from CSV | absent | 0 |"
                    )
                    continue
                _first = _rows[0]
                _last = _rows[-1]
                _dp = _math.sqrt(
                    (_last["px"] - _first["px"]) ** 2
                    + (_last["py"] - _first["py"]) ** 2
                    + (_last["pz"] - _first["pz"]) ** 2
                )
                _peak_v = max(
                    _math.sqrt(r["vx"] ** 2 + r["vy"] ** 2 + r["vz"] ** 2)
                    for r in _rows
                )
                _motion_rows.append(
                    f"| {_name} | {_dp:.3f} | {_peak_v:.3f} | {len(_rows)} |"
                )
            if _motion_rows:
                csv_summary += (
                    "\n\n## Per-step motion summary (declared-moving bodies — "
                    "from cam/motion_log.csv)\n"
                    "| body | Δp (m) | peak |v| (m/s) | samples |\n"
                    "|---|---|---|---|\n"
                    + "\n".join(_motion_rows)
                )
        except Exception as _exc:
            logger.warning(
                "[StepReview] could not inline motion_log.csv into "
                "csv_summary: %s",
                _exc,
            )
    # The "motion_log.csv MISSING" header is already injected into
    # csv_summary further up when the step had motion_expectations but no
    # CSV showed up (timeout or codegen omission). No need to duplicate it
    # here.

    # Pull codegen rebuttal (if any) from build state. Cleared after use so
    # a fresh review round is not biased by a stale rebuttal from a prior step.
    build_state = dict(state.get("build") or {})
    codegen_rebuttal = str(build_state.get("codegen_rebuttal") or "").strip()
    if codegen_rebuttal:
        logger.info(
            "[StepReview] Received codegen rebuttal — re-evaluating review decision "
            "with rebuttal context: %s",
            codegen_rebuttal[:200],
        )

    # ``plan_assets_manifest`` was built earlier (before the VLM describe
    # call) so the describer prompt can include plan-level dynamic assets.
    # Reused here unchanged.

    # Hard short-circuit: if describe_step_scene flagged a structural pipeline
    # failure (no rendered output despite recording_mode requiring it), skip
    # the LLM decision call entirely and FAIL outright. The text-only LLM
    # decision was the silent-pass machine here — given a manifest of declared
    # objects + the description "no images, but the manifest looks fine", it
    # would PASS on manifest grounds even though zero evidence existed that
    # the code actually built or rendered the scene. The PIPELINE FAILURE
    # marker injected by review_agent.describe_step_scene encodes a definitive
    # failure that does not need (and must not be subjected to) LLM second-
    # guessing.
    if "PIPELINE FAILURE" in (vlm_description or ""):
        logger.error(
            "[StepReview] Step %d: PIPELINE FAILURE detected in VLM description "
            "— short-circuit FAIL (skipping LLM decision call).",
            step_idx + 1,
        )
        decision = {
            "pass": False,
            "reasoning": (
                "Step failed structurally before any visual judgment was "
                "possible — the simulation did not produce the rendered "
                "output the review pipeline reads. See the description "
                "above for the canonical fix path."
            ),
            "issues": [
                "PIPELINE FAILURE: rendered output missing at the expected "
                "<iteration_dir>/cam/ location.",
                "Codegen must call setup_vsg_recording(vis, 'cam/vsg.mp4', "
                "fps=50.0) (or setup_preview_camera with output_root='cam') "
                "and ensure finalize() runs in a try/finally.",
            ],
        }
    else:
        decision = await _get_review_decision(
            agent, step_desc, step_idx, steps, completed,
            csv_summary=csv_summary,
            vlm_description=vlm_description,
            csv_passed=csv_passed if csv_passed is not None else True,
            codegen_rebuttal=codegen_rebuttal,
            plan_assets_manifest=plan_assets_manifest,
            scene_objects_manifest=scene_objects_manifest,
            step_motion_expectations=_step_motion_expectations,
        )

    # One-shot: always clear the rebuttal after the review has re-evaluated,
    # regardless of whether it was accepted or rejected. The code agent must
    # re-rebut explicitly if the review rejects and it still disagrees.
    if codegen_rebuttal:
        build_state.pop("codegen_rebuttal", None)
        state["build"] = build_state

    final_pass = decision["pass"]
    final_reasoning = decision["reasoning"]
    final_issues = decision["issues"]

    messages.append(make_agent_message(
        agent="StepReviewAgent",
        content=f"Step {step_idx + 1} review decision: {'PASS' if final_pass else 'FAIL'} — {final_reasoning}",
        metadata={"step_index": step_idx, "passed": final_pass, "issues": final_issues, "source": "review_decision"},
    ))

    # --- Apply decision ---
    # Build a human-readable reasoning block shared by the log line and the
    # progress event so the user always sees WHY the review voted this way —
    # not just "PASSED" / "FAILED" with the reasoning stuffed only into the
    # next codegen's feedback.
    _reasoning_display = (final_reasoning or "").strip() or "(no reasoning provided)"
    _issues_display = "\n".join(f"  - {i}" for i in final_issues) if final_issues else ""

    if final_pass:
        completed.append(step_desc)
        step_loop["completed_steps"] = completed
        step_loop["current_step_index"] = step_idx + 1
        step_loop["step_feedback"] = None
        step_loop["step_retry_count"] = 0
        step_loop["step_no_progress_retry_count"] = 0
        step_loop["step_last_failed_code_sha"] = None
        # Persist the successful code into build["previous_code"] so step N+1's
        # codegen (which lands in a fresh iteration_NNN dir) can seed its new
        # dir from step N's simulation.py. Without this, the per-step fresh
        # directory model loses continuity.
        _backfill_previous_code(state)
        logger.info(
            "[StepReview] Step %d PASSED — %s",
            step_idx + 1, _reasoning_display,
        )
        if _issues_display:
            logger.info("[StepReview] Non-blocking observations:\n%s", _issues_display)
    else:
        feedback = f"Step {step_idx + 1} review failed: {final_reasoning}"
        if final_issues:
            feedback += "\nIssues:\n" + "\n".join(f"- {issue}" for issue in final_issues)
        step_loop["step_feedback"] = feedback
        step_loop["step_retry_count"] = step_loop.get("step_retry_count", 0) + 1
        step_loop["review_issues"] = list(final_issues or [])

        _backfill_previous_code(state)
        _track_step_no_progress(step_loop, state)
        logger.warning(
            "[StepReview] Step %d FAILED (retry %d) — %s",
            step_idx + 1, step_loop["step_retry_count"], _reasoning_display,
        )
        if _issues_display:
            logger.warning("[StepReview] Issues:\n%s", _issues_display)

    state["step_loop"] = step_loop
    state["messages"] = messages
    state["current_step"] = "step_review_complete"

    _pass_label = "PASSED" if final_pass else "FAILED"
    _progress_msg = f"Step {step_idx + 1} {_pass_label} — {_reasoning_display}"
    if final_issues:
        _progress_msg += "\n" + _issues_display
    emit_progress_event(
        phase="step_review",
        step="step_review_finish",
        progress_pct=100,
        message=_progress_msg,
        passed=final_pass,
        reasoning=_reasoning_display,
        issues=list(final_issues or []),
    )
    return state
