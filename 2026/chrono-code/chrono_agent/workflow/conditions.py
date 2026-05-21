"""
Workflow conditional routing functions.

Merged from the original workflow_conditions.py and review/conditions.py
into a single module for the flat (non-subgraph) workflow.
"""

import functools
import hashlib
import logging
import re
from typing import Callable, Optional

from chrono_agent.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def _short_circuit_on_error(fn: Callable[[WorkflowState], str]) -> Callable[[WorkflowState], str]:
    """Decorator: return ``"error"`` immediately when ``state["error"]`` is set.

    Routing functions all share the same upstream-error short-circuit; lift it
    here so the per-function bodies focus on routing logic, not boilerplate.
    """

    @functools.wraps(fn)
    def wrapper(state: WorkflowState) -> str:
        if state.get("error"):
            return "error"
        return fn(state)

    return wrapper

DEFAULT_MAX_EXECUTION_RETRIES = 6
DEFAULT_MAX_VLM_REVIEW_RETRIES = 5
FINGERPRINT_FAST_FAIL_STREAK = 3
# Scoped-repair (Issue 2). After this many consecutive repair-mode attempts
# without a successful execution, give up on the cheap path and route the
# next codegen call through the full skill bundle + tool set.
DEFAULT_MAX_REPAIR_ATTEMPTS = 2


def _extract_error_fingerprint(execution: dict) -> Optional[str]:
    """Compute a stable hash of the current execution failure so repeat failures
    can be detected across retries. Uses error_type + last traceback frame +
    last exception line. Returns None when execution succeeded or stderr is empty.
    """
    if not execution or execution.get("success"):
        return None
    stderr = execution.get("stderr") or execution.get("error") or ""
    if not isinstance(stderr, str) or not stderr.strip():
        return None

    return_code = execution.get("return_code")
    if return_code == -11:
        return "SIGSEGV"  # coarse grouping — segfaults look the same to the agent

    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return None

    # Last non-empty line is typically the exception (e.g. 'AttributeError: ...')
    last_line = lines[-1]

    # Last file reference tells us where it blew up
    file_frame = ""
    for ln in reversed(lines):
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m:
            file_frame = f"{m.group(1).split('/')[-1]}:{m.group(2)}"
            break

    key = f"{last_line[:200]}|{file_frame}"
    return hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()[:16]

# The review conditions originally used a separate ReviewState type alias.
# In the flat graph both are the same dict shape.
ReviewState = WorkflowState


# ---------------------------------------------------------------------------
# Review-stage routing (formerly review/conditions.py)
# ---------------------------------------------------------------------------

@_short_circuit_on_error
def check_execution_status(state: ReviewState) -> str:
    """Route after execution success/failure."""
    review = state.get("review") or {}
    execution = review.get("execution", state.get("execution", {}))
    success = execution.get("success", False)
    logger.info(f"Execution routing: success={success}")
    if success:
        # Successful execution clears repair bookkeeping so a future failure
        # gets a fresh repair budget (Issue 2).
        build = state.get("build")
        if isinstance(build, dict) and (build.get("repair_mode") or build.get("repair_attempts")):
            build["repair_mode"] = False
            build["repair_attempts"] = 0
    return "success" if success else "failed"


@_short_circuit_on_error
def check_vlm_review_status(state: ReviewState) -> str:
    """Route after visual description generation."""
    return "described"


@_short_circuit_on_error
def check_physics_analysis_status(state: ReviewState) -> str:
    """Route after physics_analysis based on its final verdict."""
    review = state.get("review") or {}
    physics = review.get("physics_analysis") or {}
    verdict = physics.get("verdict", "physics_uncertain")
    logger.info(f"Physics analysis routing: verdict={verdict}")

    if verdict == "physics_valid":
        return "physics_valid"
    if verdict == "physics_invalid":
        return "physics_invalid"

    retry = review.get("execution_retry_count", 0)
    max_retries = review.get("max_execution_retries", DEFAULT_MAX_EXECUTION_RETRIES)
    vlm_retry = review.get("vlm_review_retry_count", 0)
    vlm_max = review.get("max_vlm_review_retries", DEFAULT_MAX_VLM_REVIEW_RETRIES)
    if retry >= max_retries or vlm_retry >= vlm_max:
        return "max_execution_retries"
    return "physics_uncertain"


# ---------------------------------------------------------------------------
# Workflow-level routing (formerly workflow_conditions.py)
# ---------------------------------------------------------------------------

@_short_circuit_on_error
def check_codegen_status(state: WorkflowState) -> str:
    """
    Route after codegen.

    Returns:
        "codegen_complete": safe to proceed to run_simulation
        "codegen_retry": codegen produced retryable issues — loop back to codegen with feedback
        "codegen_failed": codegen produced unrecoverable failure
    """
    build = state.get("build") or {}

    build_success = bool(build.get("build_success", False))
    fix_state = str(build.get("fix_state") or "")
    retries_exceeded = bool(build.get("build_max_retries_exceeded", False))

    # Rebuttal takes priority: the LLM intentionally did not edit because it
    # disagrees with the review agent. Route to run_simulation as normal —
    # the simulation re-runs (idempotent on unchanged code) and step_review
    # will see the rebuttal and re-run the review decision with it.
    if build.get("codegen_rebuttal"):
        logger.info(
            "[check_codegen_status] codegen_complete (rebuttal submitted): %s",
            str(build.get("codegen_rebuttal"))[:120],
        )
        return "codegen_complete"

    if build_success:
        return "codegen_complete"

    # TOOL_LOOP_NO_EDIT is retryable: the LLM didn't make edits, so retry
    # with feedback forcing it to use edit tools.
    if fix_state == "TOOL_LOOP_NO_EDIT" and not retries_exceeded:
        logger.warning(
            "[check_codegen_status] codegen_retry: fix_state=%s, fix_reason=%s. "
            "Retrying — LLM must produce edit tool calls.",
            fix_state,
            build.get("fix_reason", ""),
        )
        return "codegen_retry"

    reason_parts = []
    if fix_state in ("PATCH_APPLY_FAILED", "TOOL_LOOP_NO_EDIT"):
        reason_parts.append(f"fix_state={fix_state}")
    if retries_exceeded:
        reason_parts.append("retries_exceeded")
    if not reason_parts:
        reason_parts.append("build_success=False")

    logger.warning(
        "[check_codegen_status] codegen_failed: %s. fix_reason=%s.",
        ", ".join(reason_parts),
        build.get("fix_reason", ""),
    )
    return "codegen_failed"


@_short_circuit_on_error
def route_after_execution(state: WorkflowState) -> str:
    """
    Route after run_simulation.

    Returns:
        "success_step": execution succeeded in step-by-step scene mode → step_review
        "success_final": execution succeeded in mbs/final mode → visual_review
        "failed": execution failed, go back to codegen (with possible escalation)
        "max_execution_retries": execution retry cap reached, terminate
        "graceful_give_up": repeat-failure detected or cap reached, surface a
            rich final error to the user rather than spinning further
    """
    result = check_execution_status(state)
    if result == "success":
        step_loop = state.get("step_loop") or {}
        if step_loop.get("steps") and not step_loop.get("all_steps_complete", False):
            return "success_step"
        return "success_final"

    # Execution failed — update fingerprint / escalation flags on state.
    review = dict(state.get("review") or {})
    execution = review.get("execution") or state.get("execution") or {}
    retry = review.get("execution_retry_count", 0)
    max_retries = review.get("max_execution_retries", DEFAULT_MAX_EXECUTION_RETRIES)
    vlm_retry = review.get("vlm_review_retry_count", 0)
    vlm_max = review.get("max_vlm_review_retries", DEFAULT_MAX_VLM_REVIEW_RETRIES)

    fp = _extract_error_fingerprint(execution)
    last_fp = review.get("last_error_fingerprint")
    repeat_count = int(review.get("repeat_error_count") or 0)
    if fp and fp == last_fp:
        repeat_count += 1
    else:
        repeat_count = 0
    review["last_error_fingerprint"] = fp
    review["repeat_error_count"] = repeat_count

    # Strategy escalation ladder (per-step after P0-1a reset):
    #   retry 1: normal replay (no flags)
    #   retry 2: structured error feedback naturally nudges toward bash
    #            introspection (see Step 4 execution_agent changes)
    #   retry 3: one more fix attempt that preserves edit_file + cached
    #            prompt cache (Aider/Cline-style continuation)
    #   retry 4+: full-rewrite mode — disable edit_file, force write_file
    #   retry >= max_execution_retries (DEFAULT 6): graceful_give_up
    #   fingerprint repeat ≥ FINGERPRINT_FAST_FAIL_STREAK: fast-fail before cap
    #   fingerprint repeat ≥ 2: immediately jump to full-rewrite
    #
    # Threshold raised from 2 → 3 on 2026-04-22 as part of CodeGen cost
    # reduction: flipping to full-rewrite rebuilds the system prompt and
    # invalidates the 20K+ token Anthropic prompt-cache block, wasting
    # cache_creation every escalation. One extra fix attempt typically
    # costs ~3-5K tokens; a premature full-rewrite costs ~15-25K.
    build = dict(state.get("build") or {})
    force_full_rewrite = retry >= 3 or repeat_count >= 2
    if force_full_rewrite and not build.get("force_full_rewrite"):
        logger.warning(
            "[route_after_execution] Escalating to full-rewrite mode "
            "(retry=%d, repeat_error_count=%d).",
            retry, repeat_count,
        )
    build["force_full_rewrite"] = force_full_rewrite

    # Scoped-repair decision (Issue 2). Only fire when:
    #   - escalation hasn't kicked in (full rewrite wins),
    #   - the failure has a narrow, non-segfault fingerprint we can pinpoint,
    #   - we have prior code to patch,
    #   - this isn't a repeat we've already burned through repair attempts on,
    #   - we're under the repair-attempt cap.
    # If a previous repair attempt didn't fix the bug, clear the flag so the
    # next codegen runs in full mode.
    repair_attempts = int(build.get("repair_attempts") or 0)
    repair_max = int(build.get("repair_max_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS)
    was_in_repair = bool(build.get("repair_mode"))
    has_prior_code = bool(build.get("code") or build.get("previous_code"))
    eligible = (
        not force_full_rewrite
        and fp is not None
        and fp != "SIGSEGV"
        and has_prior_code
    )
    if was_in_repair:
        repair_attempts += 1
    if eligible and repair_attempts < repair_max:
        build["repair_mode"] = True
        build["repair_attempts"] = repair_attempts
        logger.info(
            "[route_after_execution] repair_mode=True (attempt %d/%d, fp=%s).",
            repair_attempts + 1, repair_max, fp,
        )
    else:
        if was_in_repair:
            logger.info(
                "[route_after_execution] Exiting repair_mode (attempts=%d, "
                "force_full_rewrite=%s) — next codegen runs full.",
                repair_attempts, force_full_rewrite,
            )
        build["repair_mode"] = False
        build["repair_attempts"] = repair_attempts

    state["review"] = review
    state["build"] = build

    # Fast-fail when the same fingerprint repeats beyond the streak threshold
    # even if the hard retry cap hasn't been hit — there's no replan path
    # to recover from a stuck step, so sitting on the same error wastes budget.
    if repeat_count + 1 >= FINGERPRINT_FAST_FAIL_STREAK:
        logger.warning(
            "[route_after_execution] graceful_give_up: identical error "
            "fingerprint repeated %d times (fp=%s, cap=%d) — freeing budget "
            "for upstream decisions.",
            repeat_count + 1, fp, FINGERPRINT_FAST_FAIL_STREAK,
        )
        return "graceful_give_up"

    if retry >= max_retries:
        logger.warning(
            "[route_after_execution] graceful_give_up triggered (retry=%d, cap=%d).",
            retry, max_retries,
        )
        return "graceful_give_up"
    if vlm_retry >= vlm_max:
        logger.warning(
            "[route_after_execution] graceful_give_up: VLM/physics review "
            "budget exhausted (vlm_retry=%d, cap=%d).",
            vlm_retry, vlm_max,
        )
        return "graceful_give_up"
    return "failed"


def route_after_step_router(state: WorkflowState) -> str:
    """Route after step_router: to codegen (next step) or bypass codegen (all done).

    Routing logic:
    - Step loop still in progress → "codegen" (per-step build)
    - All steps complete AND plan is scene / mbs_in_scene / fsi_in_scene
      → "complete". The last step has already run execution + step_review.
      Do not run a final whole-simulation pass; that only duplicates the
      last execution.
    - All steps complete AND plan is non-scene (mbs, generic) → "codegen",
      which runs a full-file generation pass. Non-scene plans never exercised
      the step loop, so there is nothing in simulation.py yet.
    """
    step_loop = state.get("step_loop") or {}
    if not step_loop.get("all_steps_complete", False):
        return "codegen"

    plan_dict = state.get("plan") or {}
    plan_type = str(plan_dict.get("plan_type") or "").lower()
    if plan_type in {"scene", "mbs_in_scene", "fsi_in_scene"}:
        logger.info(
            "[route_after_step_router] All %s steps complete — finishing "
            "without final simulation rerun.",
            plan_type,
        )
        return "complete"
    return "codegen"


@_short_circuit_on_error
def route_after_step_review(state: WorkflowState) -> str:
    """Route after step_review: next step, retry, or terminate.

    * ``next_step`` — step passed (no feedback).
    * ``retry_step`` — step failed, retry budget available.
    * ``step_max_retries`` — retry budget exhausted (either the absolute
      ``max_step_retries`` cap or the no-progress
      ``max_step_no_progress_retries`` cap), terminate.

    Two retry budgets are checked. The absolute ``max_step_retries`` cap
    bounds total attempts. The ``max_step_no_progress_retries`` cap aborts
    early when consecutive failed retries produce a byte-identical
    ``simulation.py`` (codegen is stuck on the same broken output —
    typically because review feedback is misdiagnosed and codegen
    re-applies the same non-fix). The no-progress counter is updated by
    ``_track_step_no_progress`` in the step_review failure paths.
    """
    step_loop = state.get("step_loop") or {}
    if not step_loop.get("step_feedback"):
        return "next_step"

    retry = step_loop.get("step_retry_count", 0)
    max_retries = step_loop.get("max_step_retries", 6)
    if retry >= max_retries:
        logger.warning(
            "[StepReview] Max retries (%d) for step %d",
            max_retries, step_loop.get("current_step_index", 0) + 1,
        )
        return "step_max_retries"

    no_progress = step_loop.get("step_no_progress_retry_count", 0)
    no_progress_cap = step_loop.get("max_step_no_progress_retries", 2)
    if no_progress >= no_progress_cap:
        logger.warning(
            "[StepReview] Aborting step %d: %d consecutive retries produced "
            "byte-identical simulation.py — codegen is stuck on the same "
            "output. (step_retry_count=%d, max_step_retries=%d.)",
            step_loop.get("current_step_index", 0) + 1,
            no_progress, retry, max_retries,
        )
        return "step_max_retries"

    return "retry_step"
