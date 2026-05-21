"""
Workflow state definition.

Root state uses nested structure (planning / build / review) so that
each sub-state group can be inspected independently.
"""

from datetime import datetime
from typing import List, Optional, Literal, Any, Dict

from typing_extensions import TypedDict, NotRequired


class AgentMessage(TypedDict):
    """Trace-friendly message passed between agents."""
    type: Literal["ai"]
    role: Literal["assistant"]
    name: str
    agent: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


def make_agent_message(
    *,
    agent: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> AgentMessage:
    """Build a trace-friendly message record for workflow state.

    Returns a dict message with stable ``type`` / ``role`` / ``name`` fields
    plus the existing ``agent`` field for local consumers.
    """
    return AgentMessage(
        type="ai",
        role="assistant",
        name=agent,
        agent=agent,
        content=content,
        timestamp=timestamp or datetime.now().isoformat(),
        metadata=metadata or {},
    )


class PlanningState(TypedDict, total=False):
    """Plan approval and planning-related fields."""
    plan_mode: Literal["simple", "detailed", "auto"]
    plan_approved: Optional[bool]
    plan_rejected: bool
    plan_needs_regeneration: bool
    auto_approve_on_continue: bool
    cli_manual_plan_approval: NotRequired[bool]
    user_modifications: Optional[str]
    regeneration_count: int


class BuildResultState(TypedDict, total=False):
    """Build-stage control/runtime state (canonical under state['build'])."""
    code: Optional[Dict[str, Any]]
    code_artifacts: List[Dict[str, Any]]
    trace: Optional[Dict[str, Any]]
    previous_code: Optional[str]
    codegen_state: Optional[Dict[str, Any]]
    build_success: bool
    build_max_retries_exceeded: bool
    needs_code_approval: bool
    attempted_fixes: List[str]
    error_history: Optional[Dict[str, Any]]
    last_feedback_fingerprint: Optional[str]
    codegen_tool_calls: List[Dict[str, Any]]

    # Persisted CodeGenerationAgent instance, reused across retries so the
    # plan-hash-gated skill router + bundle cache survives between codegen
    # invocations. Anthropic prompt-cache hits depend on the system prompt
    # being byte-identical across retries.
    codegen_agent: NotRequired[Any]
    # Repair-mode bookkeeping (Issue 2). Counts consecutive scoped-repair
    # attempts; reset on success, escalates to full codegen at the cap.
    repair_attempts: NotRequired[int]
    force_full_rewrite: NotRequired[bool]


class StepLoopState(TypedDict, total=False):
    """Step-by-step execution state for scene plans."""
    steps: List[str]
    current_step_index: int
    current_step_description: str
    step_retry_count: int
    max_step_retries: int
    # Byte-identical-retry guard: counts consecutive failed retries that
    # produced a simulation.py with the same SHA-256 as the prior failed
    # retry. Reset to 0 whenever the code differs (i.e. forward progress) or
    # the step passes. When this hits ``max_step_no_progress_retries`` the
    # step is aborted early — the loop is clearly stuck on the same broken
    # output. Independent of ``step_retry_count``.
    step_no_progress_retry_count: int
    max_step_no_progress_retries: int
    step_last_failed_code_sha: Optional[str]
    completed_steps: List[str]
    step_feedback: Optional[str]
    all_steps_complete: bool
    relevant_bodies: Optional[List[str]]
    step_context: Optional[Dict[str, Any]]

    # --- Step-review side-channel (updated by step_review_node) ---
    review_issues: List[str]                    # latest review-decision issues
    steps: List[Dict[str, Any]]                 # serialized SimulationStep dicts


class ReviewResultState(TypedDict, total=False):
    """Review-stage control/runtime state (canonical under state['review'])."""
    execution: Optional[Dict[str, Any]]
    vlm_review: Optional[Dict[str, Any]]
    trace: Optional[Dict[str, Any]]
    review_approved: bool
    review_visual_issues: bool
    review_max_retries_exceeded: bool
    vlm_feedback: Optional[str]
    video_description: Optional[str]
    execution_feedback: Optional[str]
    current_execution_error_messages: List[str]
    execution_error_messages: List[str]
    physics_feedback: Optional[str]
    physics_analysis: Optional[Dict[str, Any]]
    saved_csv_paths: List[str]
    last_failure_source: Optional[Literal["execution", "vlm", "physics"]]
    execution_retry_count: int
    max_execution_retries: int
    repeat_error_count: int
    last_error_fingerprint: Optional[str]
    vlm_review_retry_count: int
    max_vlm_review_retries: int


class WorkflowState(TypedDict, total=False):
    """
    Root state for the main workflow graph.

    Top-level keys only; planning/build/review details live in nested objects.
    """
    user_prompt: str
    messages: List[AgentMessage]
    plan: Optional[Dict[str, Any]]
    code: Optional[Dict[str, Any]]
    execution: Optional[Dict[str, Any]]
    code_artifacts: List[Dict[str, Any]]
    llm_handoff: Optional[Dict[str, Any]]
    structured_error: Optional[Dict[str, Any]]
    images: Optional[List[str]]  # Image paths for multimodal planning

    current_step: str
    phase: Optional[str]
    progress: Optional[Dict[str, Any]]
    needs_user_input: bool
    user_response: Optional[str]
    cli_manual_plan_approval: NotRequired[bool]

    planning: PlanningState
    build: BuildResultState
    review: ReviewResultState
    step_loop: StepLoopState

    final_output: Optional[Dict[str, Any]]
    execution_complete: bool
    error: Optional[str]
