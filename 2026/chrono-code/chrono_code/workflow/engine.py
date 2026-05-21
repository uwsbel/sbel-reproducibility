"""
Async workflow engine replacing LangGraph StateGraph.

Implements the same routing logic as the original workflow.py conditional edges,
but as explicit Python control flow.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Awaitable

from chrono_code.workflow.state import WorkflowState
from chrono_code.workflow.nodes import (
    planning_node,
    wait_for_approval_node,
    error_node,
    check_plan_approval,
    step_router_node,
    codegen_node,
    execution_node,
    vlm_review_node,
    physics_analysis_node,
    step_review_node,
)
from chrono_code.workflow.conditions import (
    DEFAULT_MAX_EXECUTION_RETRIES,
    check_codegen_status,
    route_after_execution,
    route_after_step_router,
    route_after_step_review,
    check_vlm_review_status,
    check_physics_analysis_status,
)
from chrono_code.workflow import events
from chrono_code.config import get_settings

logger = logging.getLogger(__name__)

# Independent cap for plan regeneration cycles. Prevents CLI-driven
# "reject → regenerate" loops from silently consuming the global
# max_iterations budget (and then surfacing a misleading "exceeded max
# iterations" error from the execution phase).
MAX_PLAN_REGENERATIONS = 3

# Type for the plan approval callback
PlanApprovalCallback = Callable[[WorkflowState], Awaitable[WorkflowState]]
# Phase 4 batch UI callback: List[StructuredClarification] → Dict[target_field, answer].
# Async; plumbed through WorkflowState so planning_node hands it to PlanningAgent.
ClarificationCallback = Callable[[list], Awaitable[dict]]


_USAGE_KEYS = ("input", "output", "cache_read", "cache_creation")


class _PipelineStatsCollector:
    """Aggregate per-agent and total token / time stats over a workflow run.

    Subscribes to ``agent_lifecycle`` "finished" events emitted by every
    ``invoke_llm`` and ``run_tool_loop`` session and accumulates per-agent
    and global totals. The collector is read at workflow end to populate
    the ``pipeline_stats`` event and the on-disk ``pipeline_stats.json``.
    """

    def __init__(self) -> None:
        self.start_time = time.time()
        self.per_agent: Dict[str, Dict[str, Any]] = {}
        self.total_usage: Dict[str, int] = {k: 0 for k in _USAGE_KEYS}
        self.total_calls: int = 0
        self.total_sessions: int = 0

    def observe(self, event: Dict[str, Any]) -> None:
        if event.get("type") != "agent_lifecycle":
            return
        if event.get("state") != "finished":
            return
        agent = str(event.get("agent") or "")
        if not agent:
            return
        usage = event.get("usage") or {}
        elapsed = float(event.get("elapsed") or 0.0)
        calls = int(event.get("calls") or 0)
        turns = int(event.get("turns") or 0)
        kind = str(event.get("session_kind") or "")

        entry = self.per_agent.setdefault(agent, {
            "elapsed": 0.0,
            "usage": {k: 0 for k in _USAGE_KEYS},
            "calls": 0,
            "sessions": 0,
            "turns": 0,
            "by_kind": {},
        })
        entry["elapsed"] += elapsed
        for k in _USAGE_KEYS:
            v = int(usage.get(k, 0) or 0)
            entry["usage"][k] += v
            self.total_usage[k] += v
        entry["calls"] += calls
        entry["sessions"] += 1
        entry["turns"] += turns
        if kind:
            kind_entry = entry["by_kind"].setdefault(kind, {"sessions": 0, "calls": 0})
            kind_entry["sessions"] += 1
            kind_entry["calls"] += calls

        self.total_calls += calls
        self.total_sessions += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "elapsed": time.time() - self.start_time,
            "usage": dict(self.total_usage),
            "per_agent": {
                name: {
                    "elapsed": e["elapsed"],
                    "usage": dict(e["usage"]),
                    "calls": e["calls"],
                    "sessions": e["sessions"],
                    "turns": e["turns"],
                    "by_kind": dict(e["by_kind"]),
                }
                for name, e in self.per_agent.items()
            },
            "sessions": self.total_sessions,
            "calls": self.total_calls,
        }


async def run_workflow(
    user_prompt: str,
    plan_mode: str = "auto",
    images: Optional[list] = None,
    on_plan_approval: Optional[PlanApprovalCallback] = None,
    clarification_callback: Optional[ClarificationCallback] = None,
    event_callback: Optional[events.EventCallback] = None,
    max_retries: int = 3,
    preloaded_plan: Optional[dict] = None,
) -> dict:
    """Run the complete 4-agent pipeline.

    Args:
        user_prompt: The simulation description
        plan_mode: Planning mode (simple, detailed, auto)
        images: Optional list of image paths for multimodal planning
        on_plan_approval: Callback for interactive plan approval (CLI)
        event_callback: Optional callback for progress events
        max_retries: Maximum retries for LLM calls
        preloaded_plan: Optional fully-formed plan dict (e.g. loaded from a
            fixture). When provided, planning is skipped and the engine
            enters at ``step_router`` with the plan pre-populated. Used for
            fixture-driven end-to-end tests / demos.
    """
    logger.info(f"Starting workflow for prompt: {user_prompt[:100]}...")

    # Pipeline-level stats collector. Subscribes to every agent_lifecycle
    # "finished" event so per-agent and total token/time numbers can be
    # emitted (and persisted) at workflow end without changing any of the
    # agent code paths.
    stats_collector = _PipelineStatsCollector()

    def _fanout_callback(event: Dict[str, Any]) -> None:
        try:
            stats_collector.observe(event)
        except Exception:  # pragma: no cover - defensive
            logger.exception("stats_collector raised on event")
        if event_callback is not None:
            event_callback(event)

    events.set_event_callback(_fanout_callback)

    state = _build_initial_state(user_prompt, plan_mode, images)
    # Stash the synchronous mid-loop clarification callback so planning_node
    # can hand it to PlanningAgent without depending on a CLI module.
    state["clarification_callback"] = clarification_callback

    if preloaded_plan is not None:
        state["plan"] = preloaded_plan
        # Mark planning as approved so the planning branch isn't entered
        planning_flags = dict(state.get("planning") or {})
        planning_flags.update({
            "plan_approved": True,
            "plan_rejected": False,
            "plan_needs_regeneration": False,
        })
        state["planning"] = planning_flags
        # Seed step_loop.steps from the preloaded plan so step_router_node
        # doesn't have to derive them implicitly. Avoids surprises when the
        # router's discovery logic and the engine's initial state disagree.
        preloaded_steps = preloaded_plan.get("implementation_steps") or []
        if preloaded_steps:
            step_loop = dict(state.get("step_loop") or {})
            step_loop["steps"] = list(preloaded_steps)
            state["step_loop"] = step_loop
        # Enter the engine loop directly at step_router
        state["current_step"] = "step_router"
        logger.info(
            "[run_workflow] preloaded_plan provided (plan_type=%s, steps=%d) — "
            "skipping planning phase.",
            preloaded_plan.get("plan_type"),
            len(preloaded_steps),
        )

    # Recursion limit.
    # Scales with the plan's actual shape:
    #   - Base 80 — legacy floor for simple mbs plans.
    #   - steps * 20 — each step gets room for a retry cycle at codegen + review.
    #     Without this, plans with many steps (e.g. 15-step mbs_in_scene scenes)
    #     would hit the global cap mid-flight even though per-step budgets
    #     (reset on advance) are fresh.
    # The fingerprint-fast-fail streak (conditions.py:FINGERPRINT_FAST_FAIL_STREAK)
    # remains the safety net that prevents the widened headroom from turning
    # into wasted iterations on a truly stuck step.
    plan_dict = state.get("plan") or {}
    steps_count = len(plan_dict.get("implementation_steps") or [])
    max_iterations = max(80, steps_count * 20)
    iteration = 0

    try:
        while iteration < max_iterations:
            iteration += 1
            current = state.get("current_step", "start")

            # Fresh request → planning. Clarifications are batched inside
            # the 6-phase pipeline (Phase 4); routing-only here.
            if current == "start":
                state["current_step"] = "planning_regenerate"
                continue

            # === PLANNING PHASE ===
            if current in ("planning_regenerate",):
                state = await planning_node(state)
                route = check_plan_approval(state)

                if route == "error" or route == "rejected":
                    state = await error_node(state)
                    break
                elif route == "regenerate":
                    if _bump_and_check_regen_cap(state):
                        state = await error_node(state)
                        break
                    state["current_step"] = "planning_regenerate"
                    continue
                elif route == "awaiting_approval":
                    if on_plan_approval:
                        state = await on_plan_approval(state)
                        route = check_plan_approval(state)
                        if route == "approved":
                            pass  # fall through to step_router
                        elif route == "regenerate":
                            if _bump_and_check_regen_cap(state):
                                state = await error_node(state)
                                break
                            state["current_step"] = "planning_regenerate"
                            continue
                        else:
                            state = await error_node(state)
                            break
                    else:
                        # Auto-approve (no callback)
                        state = wait_for_approval_node(state)
                        route = check_plan_approval(state)
                        if route != "approved":
                            state = await error_node(state)
                            break
                # approved - fall through to step_router
                state["current_step"] = "step_router"
                continue

            # === STEP ROUTER ===
            elif current == "step_router":
                state = await step_router_node(state)
                route = route_after_step_router(state)
                if route == "codegen":
                    state["current_step"] = "codegen"
                elif route == "complete":
                    state = _finalize_step_loop_success(state)
                    break
                continue

            # === CODE GENERATION ===
            elif current == "codegen":
                state = await codegen_node(state)
                route = check_codegen_status(state)
                if route == "codegen_complete":
                    state["current_step"] = "run_simulation"
                elif route == "codegen_retry":
                    state["current_step"] = "codegen"
                elif route in ("codegen_failed", "error"):
                    state = await error_node(state)
                    break
                continue

            # === EXECUTION ===
            elif current == "run_simulation":
                state = await execution_node(state)
                route = route_after_execution(state)

                if route == "success_step":
                    state["current_step"] = "step_review"
                elif route == "success_final":
                    state["current_step"] = "visual_review"
                elif route == "failed":
                    state["current_step"] = "codegen"
                elif route in ("max_execution_retries", "graceful_give_up", "error"):
                    state = await error_node(state)
                    break
                continue

            # === STEP REVIEW ===
            elif current == "step_review":
                state = await step_review_node(state)
                route = route_after_step_review(state)
                if route == "next_step":
                    state["current_step"] = "step_router"
                elif route == "retry_step":
                    state["current_step"] = "codegen"
                elif route in ("step_max_retries", "graceful_give_up", "error"):
                    state = await error_node(state)
                    break
                continue

            # === VISUAL REVIEW ===
            elif current == "visual_review":
                state = await vlm_review_node(state)
                route = check_vlm_review_status(state)
                if route == "described":
                    state["current_step"] = "physics_analysis"
                elif route == "error":
                    state = await error_node(state)
                    break
                continue

            # === PHYSICS ANALYSIS ===
            elif current == "physics_analysis":
                state = await physics_analysis_node(state)
                route = check_physics_analysis_status(state)
                if route == "physics_valid":
                    logger.info("Workflow complete: physics_valid")
                    break
                elif route in ("physics_invalid", "physics_uncertain"):
                    state["current_step"] = "codegen"
                elif route in ("max_execution_retries", "graceful_give_up", "error"):
                    state = await error_node(state)
                    break
                continue

            else:
                logger.error(f"Unknown step: {current}")
                state = await error_node(state)
                break

        if iteration >= max_iterations:
            step_loop = state.get("step_loop") or {}
            current_desc = step_loop.get("current_step_description") or ""
            stuck_step = state.get("current_step", "?")
            logger.error(
                "Workflow hit recursion limit (max=%d) at step=%s, desc=%s",
                max_iterations, stuck_step, current_desc,
            )
            state["error"] = (
                f"Workflow exceeded max iterations ({max_iterations}) — "
                f"stuck at step={stuck_step}, desc={current_desc or 'n/a'}"
            )
            state = await error_node(state)

        return state

    except Exception as e:
        import traceback
        logger.error(f"Workflow execution error: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        snap = stats_collector.snapshot()
        # Emit BEFORE clearing the callback so the renderer (still
        # subscribed via _fanout_callback) sees the pipeline summary.
        try:
            events.emit_pipeline_stats_event(
                elapsed=snap["elapsed"],
                usage=snap["usage"],
                per_agent=snap["per_agent"],
                sessions=snap["sessions"],
                calls=snap["calls"],
            )
        except Exception:
            logger.exception("Failed to emit pipeline_stats event")
        # Persist to dialog. Resolve the manager via BaseAgent's shared
        # singleton so we don't have to thread it through the workflow.
        try:
            from chrono_code.agents.base import BaseAgent
            dm = BaseAgent._shared_dialog_manager
            if dm is not None and getattr(dm, "current_session", None) is not None:
                dm.log_pipeline_stats(
                    elapsed=snap["elapsed"],
                    usage=snap["usage"],
                    per_agent=snap["per_agent"],
                    sessions=snap["sessions"],
                    calls=snap["calls"],
                )
        except Exception:
            logger.exception("Failed to persist pipeline stats to dialog")
        # Surface the snapshot on the returned state for downstream tools
        # (CLI, tests) that don't subscribe to events. ``state`` may not
        # be in scope on early-error paths; guard with locals().
        try:
            local_state = locals().get("state")
            if isinstance(local_state, dict):
                local_state["pipeline_stats"] = snap
        except Exception:
            pass
        events.set_event_callback(None)


def _bump_and_check_regen_cap(state: WorkflowState) -> bool:
    """Increment planning.regeneration_count; return True if cap reached.

    When True, the caller should route to error_node with a dedicated message
    so the user sees a plan-phase failure rather than a generic "max iterations"
    error from the execution phase later.
    """
    planning = dict(state.get("planning") or {})
    count = int(planning.get("regeneration_count") or 0) + 1
    planning["regeneration_count"] = count
    state["planning"] = planning
    if count > MAX_PLAN_REGENERATIONS:
        logger.error(
            "Plan regeneration cap reached (%d > %d).",
            count, MAX_PLAN_REGENERATIONS,
        )
        state["error"] = (
            f"Plan regeneration exceeded limit ({MAX_PLAN_REGENERATIONS}). "
            "Aborting before consuming the global iteration budget."
        )
        return True
    return False


def _finalize_step_loop_success(state: WorkflowState) -> WorkflowState:
    """Finish a scene step-loop without re-running the whole simulation."""
    review = state.get("review") or {}
    build = state.get("build") or {}
    artifact = build.get("code")
    # build["code"] is a CodeArtifact; consumers (CLI, tests) expect the
    # final source as a string. Prefer the patched result, fall back to
    # the freshly-generated full_code.
    if hasattr(artifact, "applied_code") or hasattr(artifact, "full_code"):
        code_str = (
            getattr(artifact, "applied_code", None)
            or getattr(artifact, "full_code", None)
            or ""
        )
    elif isinstance(artifact, dict):
        code_str = artifact.get("applied_code") or artifact.get("full_code") or ""
    else:
        code_str = artifact or ""
    final_output = {
        "success": True,
        "code": code_str,
        "execution": review.get("execution") or state.get("execution"),
        "step_loop": state.get("step_loop"),
        "message": "All implementation steps completed and reviewed.",
    }
    return {
        **state,
        "current_step": "workflow_complete",
        "phase": "complete",
        "progress": {
            "step": "workflow_complete",
            "progress_pct": 100,
            "message": "All implementation steps completed and reviewed.",
        },
        "execution_complete": True,
        "final_output": final_output,
    }


def _build_initial_state(user_prompt: str, plan_mode: str, images: list | None) -> WorkflowState:
    """Build the initial workflow state dict."""
    return {
        "user_prompt": user_prompt,
        "images": images or [],
        "messages": [],
        "code_artifacts": [],
        "llm_handoff": None,
        "structured_error": None,
        "current_step": "start",
        "phase": "root_start",
        "progress": {"step": "workflow_init", "progress_pct": 0},
        "needs_user_input": False,
        "planning": {
            "plan_mode": plan_mode,
            "plan_approved": None,
            "plan_rejected": False,
            "plan_needs_regeneration": False,
            "auto_approve_on_continue": False,
            "regeneration_count": 0,
        },
        "build": {
            "previous_code": None,
            "codegen_state": {},
            "attempted_fixes": [],
            "error_history": {},
            "build_success": False,
            "build_max_retries_exceeded": False,
            "needs_code_approval": False,
        },
        "review": {
            "execution_retry_count": 0,
            "max_execution_retries": DEFAULT_MAX_EXECUTION_RETRIES,
            "repeat_error_count": 0,
            "last_error_fingerprint": None,
            "vlm_review_retry_count": 0,
            "max_vlm_review_retries": 5,
            "execution_feedback": None,
            "execution_error_messages": [],
            "vlm_feedback": None,
            "video_description": None,
            "physics_feedback": None,
            "physics_analysis": None,
            "saved_csv_paths": [],
            "last_failure_source": None,
        },
        "step_loop": {
            "steps": [],
            "current_step_index": 0,
            "current_step_description": "",
            "step_retry_count": 0,
            "max_step_retries": 6,
            "step_no_progress_retry_count": 0,
            "max_step_no_progress_retries": 2,
            "step_last_failed_code_sha": None,
            "completed_steps": [],
            "step_feedback": None,
            "all_steps_complete": False,
            "review_issues": [],
        },
        "execution_complete": False,
    }
