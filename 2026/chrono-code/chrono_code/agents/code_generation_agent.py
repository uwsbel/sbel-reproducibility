"""
Code Generation Agent (Agent 2).

Adapted for direct Anthropic/OpenAI SDK usage (no LangChain).
"""

import logging
import asyncio
import hashlib
import json
import re
from typing import List, Optional, Dict, Any, Tuple, Callable, Set
from datetime import datetime
from pathlib import Path

from chrono_code.agents.base import BaseAgent, _diff_usage
from chrono_code.agents.exceptions import AgentLLMError
from chrono_code.workflow.events import emit_agent_lifecycle_event
from chrono_code.config import get_settings
from chrono_code.agents.prompts import codegen_prompts
from chrono_code.models.plan import SimulationPlan
from chrono_code.models.code import GeneratedCode
from chrono_code.utils.response_parser import ResponseParser
from chrono_code.models.error_context import ErrorSignature
from chrono_code.models.handoff import FailureContext, HistoryContext, LLMHandoff, SkillBundle, StructuredError, PriorError
from chrono_code.skills import SkillRegistry
from chrono_code.tools import make_code_agent_tools
from chrono_code.utils.error_utils import (
    classify_error,
    compact_tool_output,
    elide_middle,
    extract_code_snippet_at_frame,
    get_user_frame_from_feedback,
    extract_error_details,
)
from chrono_code.utils.diff_utils import (
    compute_unified_diff,
    parse_hunks,
)
from chrono_code.utils._signature_digest import build_reference_block as build_utils_reference_block
from chrono_code.validators import (
    format_issues_for_feedback as format_utils_call_issues,
    validate_utils_calls,
)

logger = logging.getLogger(__name__)

FIX_STATE_PATCH_TRY_1 = "PATCH_TRY_1"
FIX_STATE_PATCH_TRY_2 = "PATCH_TRY_2"
FIX_STATE_PATCH_APPLY_FAILED = "PATCH_APPLY_FAILED"
FIX_STATE_TOOL_LOOP_NO_EDIT = "TOOL_LOOP_NO_EDIT"
FIX_STATE_SUCCESS = "SUCCESS"


def _format_skill_truncation_footer(
    skill_name: str,
    tier: str,
    included_sections: Optional[List[str]] = None,
) -> str:
    """Footer that lists section headings the agent does NOT see at this tier.

    Pre-injection at MEDIUM/COMPACT tier delivers a condensed summary, but
    historically the prompt didn't tell the agent which sections were
    omitted — so it treated the summary as exhaustive (the iter_005/006
    SIGSEGV @ InitializeTire chain in session_20260429_112754: agent never
    called read_skill_section('veh/wheeled_vehicle', 'FSI Coupling') even
    though the summary explicitly pointed at it). This footer surfaces
    the gap so the agent can self-route.

    Returns ``""`` when:
      - tier is FULL (everything is loaded; nothing to disclose)
      - the skill has no sections (single-section skills)
      - SECTIONED tier already covers every section
    """
    if tier == "FULL":
        return ""
    all_sections = SkillRegistry.list_sections(skill_name) or []
    if not all_sections:
        return ""
    included = set(included_sections or [])
    missing = [s for s in all_sections if s not in included]
    if not missing:
        return ""

    if tier == "SECTIONED":
        intro = (
            f"Tier=SECTIONED — only the sections above were inlined. "
            f"Other sections in '{skill_name}' available via "
            f"read_skill_section(name='{skill_name}', heading=...):"
        )
    else:  # MEDIUM / COMPACT
        intro = (
            f"Tier={tier} — content above is a condensed summary of "
            f"'{skill_name}'. For full text of any section call "
            f"read_skill_section(name='{skill_name}', heading=...):"
        )

    bullet_lines = [f"  - '{s}'" for s in missing[:10]]
    if len(missing) > 10:
        bullet_lines.append(f"  ...and {len(missing) - 10} more sections")
    return "\n\n[" + intro + "\n" + "\n".join(bullet_lines) + "]"


class ErrorSignatureExtractor:
    """
    Generic error signature extractor.

    Extracts comparable signatures from ANY error type.
    Does NOT hardcode specific error types like "SIGSEGV" - instead uses
    pattern matching to extract location and core message.
    """

    @staticmethod
    def extract(error_info: dict, iteration: int) -> ErrorSignature:
        """
        Extract error signature from error information.

        Args:
            error_info: Dict with error_log, backtrace, return_code
            iteration: Current iteration number

        Returns:
            ErrorSignature for pattern detection
        """
        error_log = error_info.get("error_log", "")
        backtrace = error_info.get("backtrace", "")
        return_code = error_info.get("return_code", 0)

        # 1. Extract error location (generic patterns)
        location = ErrorSignatureExtractor._extract_location(error_log, backtrace)

        # 2. Extract core message
        core_message = ErrorSignatureExtractor._extract_core_message(error_log)

        # 3. Infer category (not hardcoded types)
        category = ErrorSignatureExtractor._infer_category(error_log, return_code)

        return ErrorSignature(
            location=location,
            core_message=core_message,
            category=category,
            iteration=iteration
        )

    @staticmethod
    def _extract_location(error_log: str, backtrace: str) -> Optional[str]:
        """Extract error location using generic patterns."""
        combined = f"{error_log}\n{backtrace}"

        # Try multiple patterns - generic, not error-type specific
        patterns = [
            # Python traceback: File "xxx.py", line N
            r'File "([^"]+)", line (\d+)',
            # C++ backtrace: in ClassName::Method
            r'in (\w+::\w+)',
            # Generic function call: in function_name(
            r'in (\w+)\s*\(',
            # Line number reference: at line N
            r'at line (\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, combined)
            if match:
                if len(match.groups()) == 2:
                    # File + line
                    return f"{match.group(1)}:{match.group(2)}"
                return match.group(1)

        return None

    @staticmethod
    def _extract_core_message(error_log: str) -> str:
        """Extract the core error message."""
        # Try multiple patterns to extract the error message
        patterns = [
            # Python exceptions: TypeName: message
            r'(\w+Error): (.+)',
            r'(\w+Exception): (.+)',
            # Generic error patterns
            r'Error: (.+)',
            r'error: (.+)',
            r'FATAL: (.+)',
            r'failed: (.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_log, re.IGNORECASE)
            if match:
                return match.group(0)[:200]

        # Fallback: first non-empty line
        for line in error_log.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                return line[:200]

        return "Unknown error"

    @staticmethod
    def _infer_category(error_log: str, return_code: Optional[int]) -> str:
        """
        Infer error category from content.

        Not hardcoding specific types - infers from content patterns.
        """
        error_lower = error_log.lower()

        # Syntax-related
        if any(x in error_lower for x in ["syntaxerror", "indentationerror", "invalid syntax"]):
            return "syntax"

        # Import-related
        if any(x in error_lower for x in ["importerror", "modulenotfounderror", "no module named"]):
            return "import"

        # Timeout
        if "timeout" in error_lower:
            return "timeout"

        # Signal crashes (negative return codes)
        if return_code is not None and return_code < 0:
            return "crash"

        # Type errors
        if "typeerror" in error_lower:
            return "type"

        # Attribute errors
        if "attributeerror" in error_lower:
            return "attribute"

        # Value errors
        if "valueerror" in error_lower:
            return "value"

        # Generic runtime
        if any(x in error_lower for x in ["error", "exception", "failed", "traceback"]):
            return "runtime"

        return "unknown"


class CodeGenerationAgent(BaseAgent):
    """
    Agent 2: Generates PyChrono code using web search for API lookup.

    Responsibilities:
    - Generate code from simulation plan
    - Validate syntax and compilation
    - Use web search (Tavily) to look up API documentation
    - Retry with corrections when errors occur
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        progress_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        tool_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(
            agent_name="CodeGenerationAgent",
            agent_number=2,
            llm_provider=llm_provider,
            model=model,
            temperature=temperature,
        )

        self.progress_callback = progress_callback
        self.tool_event_callback = tool_event_callback
        self._iteration = 1  # Track iteration for history saving

        settings = get_settings()
        # Initialize response parser for robust code extraction
        self.response_parser = ResponseParser()

        # NOTE: do NOT instantiate a fresh DialogManager here. BaseAgent.__init__
        # already attached the SHARED dialog_manager (the one with the active
        # session created by start_session()). Overriding it with a new instance
        # — as the previous code did — breaks two things at once:
        #   1. The new instance has no current_session, so log_response() short-
        #      circuits and writes nothing; log_prompt() falls back to creating
        #      a session named "unknown" outside the active session tree.
        #   2. Even if logging worked, codegen's writes would land in a parallel
        #      session dir disconnected from planner / review, defeating the
        #      whole "one session, all agents" trace model.
        # Just inherit ``self.dialog_manager`` from BaseAgent and use it.

        data_dir = Path(__file__).resolve().parents[1] / "data"
        self.method_pitfalls: Dict[str, str] = self._load_method_pitfalls(data_dir / "method_pitfalls.json")
        self.max_methods_per_class = 10

        self._routed_cache_plan_hash: Optional[str] = None
        self._cached_skill_bundle: Optional[SkillBundle] = None
        self._llm_routed_skills: Optional[List[str]] = None
        self._llm_routed_sections: Dict[str, List[str]] = {}

    def _load_method_pitfalls(self, path: Path) -> Dict[str, str]:
        """Load optional class.method pitfall hints from JSON."""
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            self.logger.warning(f"Failed to load method pitfalls from {path}: {exc}")
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(val).strip()
            for key, val in raw.items()
            if str(key).strip() and str(val).strip()
        }

    def _emit_tool_event(self, **kwargs: Any) -> None:
        """Emit optional tool event callback for activity feed."""
        if self.tool_event_callback:
            try:
                self.tool_event_callback(kwargs)
            except Exception:
                pass

    def _emit_progress(self, phase: str, step: str, **payload: Any) -> None:
        """Emit optional progress callback for Studio observability."""
        if not self.progress_callback:
            return
        event = {"phase": phase, "step": step, **payload}
        try:
            self.progress_callback(phase, step, event)
        except Exception as exc:
            self.logger.debug(f"Progress callback failed: {exc}")

    @staticmethod
    def _shorten_for_log(text: str, limit: int = 240) -> str:
        text = " ".join((text or "").split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _log_tool_loop_context(
        self,
        *,
        mode: str,
        effective_mode: str,
        system_prompt: str,
        user_prompt: str,
        max_steps: int,
    ) -> None:
        total_chars = len(system_prompt) + len(user_prompt)
        est_tokens = self._estimate_token_count(system_prompt + "\n" + user_prompt)
        self.logger.info(
            f"[ToolLoopDiag] start mode={mode} effective_mode={effective_mode} "
            f"total_chars={total_chars} est_tokens~={est_tokens} "
            f"max_steps={max_steps}"
        )
        self.logger.debug(
            f"[ToolLoopDiag] "
            f"configured_max_tokens={self.settings.get_agent_config(2).get('max_tokens')}; "
            f"context_window=provider_default_not_explicitly_set"
        )

    def _emit_tool_step_progress(
        self,
        *,
        step: str,
        progress_pct: int,
        message: str,
        **payload: Any,
    ) -> None:
        self._emit_progress(
            "build_codegen",
            step,
            progress_pct=progress_pct,
            message=message,
            **payload,
        )

    def _serialize_tool_feedback(
        self, feedback: Optional[Any], initial_code: Optional[str] = None
    ) -> str:
        """Serialize feedback for tool-loop prompts as raw factual text only."""
        if feedback is None:
            return "None"

        if isinstance(feedback, str):
            result = feedback
        elif isinstance(feedback, dict):
            parts: List[str] = []
            for key in ("feedback_text", "error_text", "message"):
                val = feedback.get(key)
                if val:
                    parts.append(str(val))
            for issue in (feedback.get("issues") or []):
                if isinstance(issue, dict):
                    desc = issue.get("description") or issue.get("message") or ""
                    if desc:
                        parts.append(str(desc))
                elif issue:
                    parts.append(str(issue))
            backtrace = feedback.get("backtrace") or feedback.get("traceback")
            if backtrace:
                parts.append(f"\nTraceback:\n{backtrace}")
            result = "\n".join(p for p in parts if p)
            if not result:
                try:
                    result = json.dumps(feedback, ensure_ascii=True, indent=2)
                except TypeError:
                    result = str(feedback)
        else:
            try:
                result = json.dumps(feedback, ensure_ascii=True, indent=2)
            except TypeError:
                result = str(feedback)

        if initial_code and result:
            user_frame = get_user_frame_from_feedback(feedback)
            if user_frame and user_frame != "unknown":
                snippet = extract_code_snippet_at_frame(
                    initial_code, user_frame, context_lines=5
                )
                if snippet:
                    result += "\n\nCode around error location:\n" + snippet

        # Head+tail elision: fix-mode prompts frequently carry 5-10KB
        # backtraces that get re-sent every retry. Keep the head (exception
        # type + first frames) and the tail (user frame where exception
        # actually surfaced).
        settings = getattr(self, "settings", None) or get_settings()
        if bool(getattr(settings, "tool_output_truncate_enabled", True)):
            head = int(getattr(settings, "tool_output_head_chars", 1500))
            tail = int(getattr(settings, "tool_output_tail_chars", 500))
            # Feedback injection runs along the "prompt" axis, so use a
            # generous budget (4x the per-tool-result cap). Most real
            # compilation_feedback is under this anyway; we only bite when
            # the backtrace is pathological.
            result = elide_middle(result, head_chars=head * 4, tail_chars=tail * 2)
        return result

    def _check_python_compile(self, code: str) -> List[str]:
        if not (code or "").strip():
            return ["[CompileError] Code is empty."]
        try:
            compile(code, "<generated_simulation>", "exec")
            return []
        except SyntaxError as exc:
            lineno = getattr(exc, "lineno", None)
            msg = str(exc.msg) if hasattr(exc, "msg") else str(exc)
            return [f"[SyntaxError] line {lineno or '?'}: {msg}"]
        except Exception as exc:
            return [f"[CompileError] {exc}"]

    def _check_utils_calls(self, generated_code: Optional[GeneratedCode]) -> str:
        """Static AST check for misuse of chrono_code.utils.* APIs.

        Returns a non-empty feedback string iff at least one call site has
        arg/kwarg mismatches against the real `inspect.Signature`. Empty
        string means "no utils-call issues detected" and the caller should
        proceed to the usual success path.
        """
        code = getattr(generated_code, "code", "") if generated_code is not None else ""
        if not code:
            return ""
        try:
            issues = validate_utils_calls(code)
        except Exception as exc:
            self.logger.warning(f"[UtilsCallValidator] validator raised {exc!r}; skipping")
            return ""
        if not issues:
            return ""
        return format_utils_call_issues(issues)

    # ------------------------------------------------------------------
    # Helper: extract text from an Anthropic response
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response_text(response) -> str:
        """Extract text content from an Anthropic messages response."""
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helper: call LLM (Anthropic or OpenAI-compat) with tool definitions
    # ------------------------------------------------------------------

    async def _call_llm_with_tools(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tool_defs: List[Dict[str, Any]],
    ) -> Any:
        """
        Invoke the LLM with tool definitions.

        For Anthropic: uses client.messages.create with system= and tools=.
        For OpenAI-compat: converts tool_defs to OpenAI function format.

        Returns the raw response object. When the provider streams
        ``thinking_delta`` events (Anthropic extended-thinking models),
        the concatenated thinking text is stashed on the returned object
        as ``response._codegen_thinking_text`` so the caller can persist
        it via ``_log_tool_turn_response(thinking=...)``. Without this
        capture the entire chain-of-thought is dropped at the stream
        boundary and never reaches the dialog log.
        """
        if self.provider == "anthropic":
            # MUST use streaming: with claude-sonnet-4-6 + long tool schemas +
            # max_tokens above ~4k, the Anthropic SDK refuses non-streaming
            # calls up front with "Streaming is required for operations that
            # may take longer than 10 minutes." We consume the stream fully
            # and return the aggregated final Message, which has the same
            # shape downstream callers (_parse_tool_calls_from_response) expect.
            thinking_chunks: List[str] = []
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=messages,
                tools=tool_defs,
            ) as stream:
                async for sevent in stream:
                    if getattr(sevent, "type", "") == "content_block_delta":
                        delta = getattr(sevent, "delta", None)
                        if delta and getattr(delta, "type", "") == "thinking_delta":
                            chunk = getattr(delta, "thinking", "") or ""
                            if chunk:
                                thinking_chunks.append(chunk)
                final_msg = await stream.get_final_message()
                try:
                    final_msg._codegen_thinking_text = "".join(thinking_chunks)
                except (AttributeError, TypeError):
                    pass
                # Telemetry: codegen overrides ``_run_tool_loop`` instead of
                # going through ``BaseAgent.run_tool_loop``, so we log usage
                # per turn here. Without this, every codegen LLM call is
                # invisible to the pipeline_stats collector and the panel
                # under-reports total tokens by an order of magnitude.
                self._log_llm_usage(final_msg, where="codegen_tool_turn")
                return final_msg
        else:
            # OpenAI-compatible path
            openai_tools = []
            for td in tool_defs:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": td["name"],
                        "description": td.get("description", ""),
                        "parameters": td.get("input_schema", {}),
                    },
                })
            oai_messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt}
            ]
            oai_messages.extend(messages)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                tools=openai_tools if openai_tools else None,
            )
            # Capture provider-side reasoning content when present (DeepSeek-R1,
            # Moonshot, OpenAI o1+ via reasoning_content). Always optional —
            # vanilla gpt-4 / minimax non-reasoning models leave it None and
            # we just stash an empty string.
            try:
                msg = response.choices[0].message
                reasoning = getattr(msg, "reasoning_content", None) or ""
                response._codegen_thinking_text = reasoning
            except (AttributeError, IndexError, TypeError):
                pass
            # Telemetry: see anthropic branch above. Per-turn logging is the
            # only place codegen surfaces token usage; without it the
            # pipeline_stats collector misses every codegen LLM call.
            self._log_llm_usage(response, where="codegen_tool_turn")
            return response

    def _parse_tool_calls_from_response(self, response: Any) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse tool calls and text from a response (Anthropic or OpenAI).

        Returns:
            (tool_calls_list, response_text)
            Each tool call is a dict with keys: id, name, args
        """
        if self.provider == "anthropic":
            tool_calls = []
            text_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "args": block.input or {},
                    })
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            return tool_calls, "\n".join(text_parts)
        else:
            # OpenAI-compat
            choice = response.choices[0]
            text = choice.message.content or ""
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        args = {}
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": args,
                    })
            return tool_calls, text

    def _build_assistant_message(self, response: Any) -> Dict[str, Any]:
        """Build an assistant message dict from the raw response for appending to messages."""
        if self.provider == "anthropic":
            # For Anthropic, we pass the content blocks directly
            return {"role": "assistant", "content": response.content}
        else:
            # For OpenAI, append the raw message object
            return response.choices[0].message

    def _build_tool_result_message(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        """Build a tool result message for the conversation."""
        if self.provider == "anthropic":
            return {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
            }
        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }

    def _append_tool_results_to_messages(
        self,
        messages: List[Any],
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """Append tool results to the messages list in the correct format."""
        if self.provider == "anthropic":
            # Anthropic: all tool_result blocks go in a single user message
            messages.append({"role": "user", "content": tool_results})
        else:
            # OpenAI: each tool result is a separate message
            messages.extend(tool_results)

    def _append_user_nudge(self, messages: List[Any], text: str) -> None:
        """Append a user nudge message to the conversation."""
        if self.provider == "anthropic":
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        else:
            messages.append({"role": "user", "content": text})

    async def _run_tool_loop(
        self,
        plan: SimulationPlan,
        compilation_feedback: Optional[Any],
        initial_code: str,
        max_steps: int = 40,
        attempted_fixes: Optional[List[str]] = None,
        fix_mode: bool = False,
        prior_handoff: Optional[dict] = None,
        feedback_source: Optional[str] = None,
        error_history: Optional[dict] = None,
        step_loop: Optional[dict] = None,
        force_full_rewrite: bool = False,
        repair_mode: bool = False,
    ) -> Tuple[str, int, int, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Run a tool-calling loop with code + web-search tools.

        Mode behavior:
        - fix_mode=True (simulation failure): allow edit_file, multiple iterations
        - fix_mode=False (generation/compilation error): one-shot write_file, no re-read after write

        Returns:
            Tuple of (final_code, total_tool_calls, edit_tool_calls_count, llm_handoff_dict, structured_error_dict)
        """
        mode = "fix" if (compilation_feedback is not None and fix_mode) else "generate"
        effective_mode = (
            "generate" if (mode == "fix" and not (initial_code or "").strip()) else mode
        )
        # Escalation: when repeat failures are detected, the workflow forces a
        # full-file rewrite — treat this run as generate-mode so edit_file is
        # not the primary expected mutation; the model emits write_file.
        if force_full_rewrite:
            self.logger.warning(
                "[ToolLoop] force_full_rewrite=True — switching effective_mode to 'generate'"
            )
            effective_mode = "generate"

        # Scoped-repair mode (Issue 2): the workflow has identified a narrow,
        # likely-single-line failure (e.g. NameError with a clean traceback
        # frame). Run the smallest possible loop — no skill bundle, fewer
        # steps, edit_file-only — to avoid paying the full ~50K-char system
        # prompt + tool-set on what is usually a 1–3 turn fix. If the repair
        # doesn't converge, the workflow clears the flag and the next codegen
        # invocation runs in full mode. force_full_rewrite wins if both are
        # set (escalation overrides the optimistic shortcut).
        if force_full_rewrite:
            repair_mode = False
        if repair_mode:
            if max_steps > 8:
                max_steps = 8
            effective_mode = "fix"
            self.logger.info(
                "[ToolLoop] repair_mode=True — skill bundle skipped, "
                "tools restricted, max_steps capped to %d", max_steps,
            )

        is_active_step_mode = bool(
            step_loop and step_loop.get("steps")
            and not step_loop.get("all_steps_complete")
        )
        is_step_continuation = (
            is_active_step_mode
            and step_loop.get("current_step_index", 0) > 0
        )

        feedback_text = self._serialize_tool_feedback(
            compilation_feedback,
            initial_code=initial_code if mode == "fix" else None,
        )

        # LLM-based skill routing. One router call per plan-content hash —
        # subsequent retries with the same plan reuse the cached decision and
        # skill bundle, so the resulting system prompt is byte-identical
        # across retries and Anthropic prompt-cache hits land.
        _plan_hash = self._compute_plan_hash(plan)
        if (
            self._routed_cache_plan_hash == _plan_hash
            and self._cached_skill_bundle is not None
        ):
            self.logger.info(
                f"[ToolLoop] reusing cached skill routing + bundle (plan_hash={_plan_hash[:12]})"
            )
            skill_bundle = self._cached_skill_bundle
        else:
            _routed_decision = await self._route_skills_via_llm(plan)
            if _routed_decision is None:
                self._llm_routed_skills = None
                self._llm_routed_sections = {}
            else:
                self._llm_routed_skills = list(_routed_decision.skills)
                self._llm_routed_sections = dict(_routed_decision.sections)
            skill_bundle = self._build_skill_bundle(plan, compilation_feedback)
            self._routed_cache_plan_hash = _plan_hash
            self._cached_skill_bundle = skill_bundle
        handoff = self._build_handoff_packet(
            plan=plan,
            bundle=skill_bundle,
            effective_mode=effective_mode,
            feedback=compilation_feedback,
            feedback_source=feedback_source or "",
            previous_code=initial_code or "",
            prior_handoff=prior_handoff,
            error_history=error_history,
            is_step_continuation=is_step_continuation,
        )
        # If this is a retry driven by a step_review failure, expose the review
        # feedback in context so rebut_review() becomes available to the LLM.
        _sf = (step_loop or {}).get("step_feedback") if step_loop else None
        _review_feedback_text = None
        if _sf:
            if isinstance(_sf, str):
                _review_feedback_text = _sf
            elif isinstance(_sf, dict):
                _review_feedback_text = (
                    _sf.get("reasoning") or _sf.get("feedback") or json.dumps(_sf)
                )

        # Pull step_context early (before make_code_agent_tools) so the
        # skill gate can see step-level signals like ``step_assets`` and
        # mark ``scene/custom_assets_scene_convex_decomp`` as required on
        # asset-heavy steps. Without this, the gate only sees the plan-level
        # rules and the codegen burns its first turn voluntarily reading
        # the asset skill.
        _step_ctx_for_context: Optional[Dict[str, Any]] = None
        if step_loop:
            _raw_sc = step_loop.get("step_context")
            if isinstance(_raw_sc, dict):
                _step_ctx_for_context = _raw_sc
            elif hasattr(_raw_sc, "model_dump"):
                try:
                    _step_ctx_for_context = _raw_sc.model_dump()
                except Exception:
                    _step_ctx_for_context = None

        context: Dict[str, Any] = {
            "current_code": initial_code or "",
            "skill_bundle": skill_bundle.model_dump(),
            "review_feedback": _review_feedback_text,
            # Stash the serialized plan so skill-gate / validator helpers
            # can read plan.simulation_parameters without being wired
            # through yet another parameter chain.
            "plan": plan.dump_all() if hasattr(plan, "dump_all") else dict(plan or {}),
            # Per-step context for the skill gate's step-level rules (e.g.
            # step_assets → scene/custom_assets_scene_convex_decomp). Stays
            # None outside step mode and the gate just skips the step-level
            # checks in that case.
            "step_context": _step_ctx_for_context,
            # LLM-routed skill set (see _route_skills_via_llm). When non-None,
            # the skill gate uses this list as the required set instead of
            # falling back to keyword detection in _derive_required_skills.
            "llm_routed_skills": (
                list(self._llm_routed_skills)
                if getattr(self, "_llm_routed_skills", None)
                else None
            ),
            # Factory for spawn_subagent: produces a fresh CodeGenerationAgent
            # instance with its own LLM client. The factory is lazy so we only
            # pay the client setup cost when the LLM actually delegates.
            "subagent_factory": lambda: CodeGenerationAgent(),
        }

        # make_code_agent_tools now returns (tool_definitions, tool_executors)
        tool_defs, tool_executors = make_code_agent_tools(context=context)

        # File explorer tools fall into two buckets:
        #   * Broad discovery (list_directory / find_files / read_file_content)
        #     — still disabled in generate mode; they encourage exploratory
        #     loops instead of writing code.
        #   * Asset verification (check_asset_path / find_assets /
        #     list_chrono_assets) — KEPT in generate mode. Session logs show
        #     the code agent regularly invents project-relative paths
        #     (``data/models/fixedterrain.obj`` etc.) that crash tiny_obj →
        #     SIGSEGV at ChSensorManager.Update(). A cheap path-check call
        #     right before write_file catches this class of bug at codegen
        #     time instead of after a full simulation run.
        _BROAD_EXPLORER_TOOLS = {"list_directory", "find_files", "read_file_content"}
        if effective_mode == "fix":
            tool_defs = [t for t in tool_defs if t["name"] != "write_file"]
            tool_executors = {k: v for k, v in tool_executors.items() if k != "write_file"}
        else:
            # Generation mode: disable broad exploration. Step 0 uses write_file
            # (initial creation); step 2+ uses edit_file for targeted edits to
            # the existing file (no full rewrite). apply_patch was removed —
            # multi-substring sequences just become multiple edit_file calls.
            _filter_write_file = is_step_continuation
            tool_defs = [
                t for t in tool_defs
                if (t["name"] != "write_file" or not _filter_write_file)
                and t["name"] not in _BROAD_EXPLORER_TOOLS
            ]
            tool_executors = {
                k: v for k, v in tool_executors.items()
                if (k != "write_file" or not _filter_write_file)
                and k not in _BROAD_EXPLORER_TOOLS
            }
        # Repair mode: restrict to the smallest set of tools needed to apply
        # a targeted fix. No skill discovery (no skill bundle is injected),
        # no broad exploration. read_file is kept so the LLM can re-read the
        # failing line; edit_file is the only mutation path; validate is the
        # only safety check.
        if repair_mode:
            _REPAIR_ALLOWED = {"edit_file", "read_file", "read_file_content", "validate_chrono_apis"}
            tool_defs = [t for t in tool_defs if t["name"] in _REPAIR_ALLOWED]
            tool_executors = {k: v for k, v in tool_executors.items() if k in _REPAIR_ALLOWED}

        # Skills are always compact-only now (summaries, no full text).
        # Never remove search_skills -- the LLM needs it to discover and read
        # skills on demand via read_skill().
        tools_by_name = {t["name"]: t for t in tool_defs}
        is_anthropic_provider = str(self.provider or "").lower() == "anthropic"
        _sl_summary = {k: v for k, v in (step_loop or {}).items() if k in ("steps", "current_step_index", "all_steps_complete", "relevant_bodies")} if step_loop else None
        self.logger.info(
            f"[ToolLoop] step_loop={_sl_summary}, is_active_step_mode={is_active_step_mode}"
        )
        if effective_mode == "fix":
            edit_tool_names = {"edit_file"}
        elif is_step_continuation:
            edit_tool_names = {"edit_file"}
        else:
            edit_tool_names = {"write_file", "edit_file"}
        validation_tool_name = "validate_chrono_apis"

        # Build extra virtual files the agent can read on demand
        extra_files: Dict[str, str] = {
            "handoff.json": json.dumps(handoff.model_dump(), ensure_ascii=True, indent=2),
        }

        # plan.md — the readable markdown the planner actually submitted.
        # This is a strictly-better orientation doc than plan.json: fewer
        # escapes, YAML blocks the LLM can scan, milestone `description`
        # rendered as prose instead of JSON-escaped strings. We still expose
        # plan.json for precise field access; plan.md is for the initial
        # "understand the intent" read.
        if getattr(plan, "plan_markdown", None):
            extra_files["plan.md"] = plan.plan_markdown

        if is_active_step_mode:
            step_ctx = step_loop.get("step_context")
            if not step_ctx:
                # Rebuild step_context from plan (e.g. after checkpoint restore)
                self.logger.warning("[ToolLoop] step_context missing in step_loop, rebuilding from plan")
                try:
                    step_ctx = plan.build_step_context(
                        step_loop.get("current_step_index", 0),
                        step_loop.get("completed_steps", []),
                    ).model_dump()
                except Exception as exc:
                    self.logger.error(f"[ToolLoop] Failed to rebuild step_context: {exc}")
                    step_ctx = None
            if step_ctx:
                # Legacy step mode: ONLY expose step_context.json
                extra_files["step_context.json"] = json.dumps(
                    step_ctx, ensure_ascii=True, indent=2
                )
            else:
                # Fallback: expose full plan.json if step_context rebuild failed
                self.logger.warning("[ToolLoop] Falling back to plan.json in step mode")
                extra_files["plan.json"] = json.dumps(
                    plan.dump_all(), ensure_ascii=True, indent=2
                )
        else:
            # Non-step mode: expose full plan.json
            extra_files["plan.json"] = json.dumps(
                plan.dump_all(), ensure_ascii=True, indent=2
            )

        context["extra_files"] = extra_files

        # -- Resolve required skills early so mode_constraints and skill_constraints can reference them --
        _plan_type = getattr(plan, "plan_type", None) or "mbs"
        _core_skill_name = f"core/{_plan_type}"
        _routed = getattr(self, "_llm_routed_skills", None)
        if _routed:
            # LLM router output is the authoritative required set. Preserve
            # the router's importance order for the children (it already
            # ranked core first, then domain skills, then utilities), so
            # the budget cascade below picks the right ones to keep at FULL.
            required_skills = set(_routed)
            required_skills.add(_core_skill_name)
            _required_children = [n for n in _routed if n != _core_skill_name]
        else:
            # Fallback: parse the core skill's static "Required Skills" table.
            _core_skill_obj = SkillRegistry._skills.get(_core_skill_name)
            required_skills: set[str] = set()
            if _core_skill_obj is not None:
                required_skills = set(_core_skill_obj.get_required_skills())
                required_skills.add(_core_skill_name)
            _required_children = sorted(required_skills - {_core_skill_name})

        # -- Pre-inject authoritative skill content so on-demand read_skill
        # calls become the narrow case rather than the norm. Budget total
        # size so the system prompt stays under Claude's practical limits.
        # Prioritize visualization / sensor skills because their bugs tend
        # to be native crashes (SIGSEGV) that are expensive to recover from.
        #
        # Compression cascade: when a required skill exceeds the remaining
        # budget, we DOWNGRADE the rendering tier rather than drop the
        # skill entirely:
        #   Tier A — FULL    (~entire SKILL.md)
        #   Tier B — MEDIUM  (~4 KB: API contract + pitfalls / rules)
        #   Tier C — COMPACT (~0.5 KB: 1-line desc + signature heads)
        # Whichever tier lands in the prompt counts as "gate-satisfied" —
        # the LLM can still call read_skill() to upgrade to FULL on demand,
        # but the gate won't refuse the first edit. This eliminates the
        # "edit → gate refusal → read_skill → re-edit" wasted-turn pattern
        # that used to fire whenever a required skill spilled the budget.
        # Repair mode skips skill pre-injection entirely — the failure
        # fingerprint is narrow enough that the existing code + traceback is
        # all the LLM needs. Anything broader is paid for by the upstream
        # full-codegen cache (Issue 1).
        PRE_INJECT_BUDGET = 0 if repair_mode else 50_000  # chars, roughly 12.5k tokens
        _HIGH_PRIORITY_PREFIXES = ("vsg", "sens/", "veh/")
        _pre_injected_chunks: List[str] = []
        _budget_remaining = PRE_INJECT_BUDGET
        if _routed:
            # The LLM router already ranked skills by importance for THIS
            # plan (core → domain → utility). Preserve that order so the
            # cascade keeps the FSI / domain skills at FULL rather than
            # demoting them in favor of generic veh/* / vsg / sens/* whose
            # prefix happens to match _HIGH_PRIORITY_PREFIXES.
            _ordered_skills = [_core_skill_name] + _required_children
        else:
            # Static-table fallback: prefix-priority is the only signal.
            _high_priority_children = [
                c for c in _required_children
                if any(c.startswith(p) for p in _HIGH_PRIORITY_PREFIXES)
            ]
            _other_children = [c for c in _required_children if c not in _high_priority_children]
            _ordered_skills = [_core_skill_name] + _high_priority_children + _other_children

        _routed_sections = getattr(self, "_llm_routed_sections", None) or {}

        _injected_names: List[str] = []
        _injected_at_tier: Dict[str, str] = {}  # for logging; SECTIONED / FULL / MEDIUM / COMPACT
        for _sn in _ordered_skills:
            _skill_obj = SkillRegistry._skills.get(_sn)
            if _skill_obj is None:
                continue

            # Section-level injection takes priority: when the router
            # picked specific sections for this skill, render only those
            # — no FULL/MEDIUM/COMPACT cascade. This is the only path
            # that reliably keeps "Pattern X — ..." code blocks in the
            # prompt for skills that wouldn't otherwise fit FULL.
            _sectioned: Optional[str] = None
            _sec_keys = _routed_sections.get(_sn) if _routed_sections else None
            if _sec_keys:
                _sectioned = SkillRegistry.render_sections(_sn, _sec_keys)
                if _sectioned is not None:
                    _sectioned = _sectioned.strip() or None

            # Tier cascade fallbacks (used when section injection is
            # absent or rendered empty).
            _full = (SkillRegistry.get_skill_fragment(_sn) or "").strip()
            _medium = SkillRegistry._render_medium_skill_fragment(_skill_obj).strip()
            _compact = SkillRegistry._render_compact_skill_fragment(_skill_obj).strip()
            _is_core = _sn == _core_skill_name

            # Candidate order: SECTIONED first when available, then the
            # legacy FULL → MEDIUM → COMPACT cascade.
            _candidates: List[Tuple[str, str]] = []
            if _sectioned:
                _candidates.append(("SECTIONED", _sectioned))
            _candidates.extend([
                ("FULL", _full),
                ("MEDIUM", _medium),
                ("COMPACT", _compact),
            ])

            _picked: Optional[Tuple[str, str]] = None
            for _tier, _text in _candidates:
                if not _text:
                    continue
                if len(_text) <= _budget_remaining:
                    _picked = (_tier, _text)
                    break
            # Core skill: never drop. Take COMPACT even if it overflows
            # (overflow is small; preserves the gate-satisfaction contract).
            if _picked is None and _is_core and _compact:
                _picked = ("COMPACT", _compact)

            if _picked is None:
                # Non-core skill that doesn't fit even in COMPACT form —
                # leave for on-demand read_skill. Vanishingly rare.
                continue

            _tier, _text = _picked
            # Honest footer: when this tier doesn't include the whole
            # skill, list the section headings the agent would gain by
            # calling read_skill_section(). Closes the "I already saw
            # everything" trap that pre-injection creates — see fix 2.A
            # in dialog-sessions-session-20260429-112754-glittery-pixel.md.
            _footer = _format_skill_truncation_footer(
                _sn,
                _tier,
                included_sections=(_sec_keys if _tier == "SECTIONED" else None),
            )
            _chunk_body = _text + _footer if _footer else _text
            _pre_injected_chunks.append(
                f"<skill name=\"{_sn}\" tier=\"{_tier}\">\n{_chunk_body}\n</skill>"
            )
            _injected_names.append(_sn)
            _injected_at_tier[_sn] = _tier
            _budget_remaining = max(0, _budget_remaining - len(_chunk_body))

        if _injected_at_tier:
            _tier_summary = ", ".join(
                f"{n}={_injected_at_tier[n]}" for n in _injected_names
            )
            self.logger.info(
                f"[skill-injection] {len(_injected_names)} skills injected "
                f"(budget {PRE_INJECT_BUDGET - _budget_remaining}/{PRE_INJECT_BUDGET} chars): "
                f"{_tier_summary}"
            )

        _pre_injected_block = "\n\n".join(_pre_injected_chunks)

        # Expose the set of pre-injected skill names to the tool context so
        # the skill-gate can treat them as "already read" — the authoritative
        # text is already in the system prompt that every turn pays for, so
        # forcing an additional read_skill() tool call for the same content
        # is one wasted loop turn per generation. See settings flag
        # skill_gate_treat_preinjected_as_read.
        #
        # Fix 2.B: only FULL / SECTIONED tier counts as "satisfies the gate".
        # MEDIUM / COMPACT inlines a condensed summary that the visit_005/006
        # SIGSEGV chain in session_20260429_112754 showed agents treat as
        # exhaustive — gating those would let the agent edit without ever
        # reading the authoritative section. Demoting them keeps the gate
        # honest while the truncation footer (fix 2.A) tells the agent
        # which heading to fetch.
        #
        # Match the normalization used by _skill_gate_check in
        # code_agent_tools.py: strip whitespace and surrounding quotes,
        # lowercase. Inlining here avoids creating a cross-module
        # dependency just for a trivial string op.
        _GATE_SATISFYING_TIERS = {"FULL", "SECTIONED"}
        context["preinjected_skill_names"] = {
            (n or "").strip().strip("'\"").lower()
            for n in _injected_names
            if _injected_at_tier.get(n) in _GATE_SATISFYING_TIERS
        }

        # Settings-gated: whether the *wording* of skill_constraints should
        # match the relaxed gate. Keeps the prompt aligned with the actual
        # harness behavior so we don't tell the model "harness REQUIRES read"
        # when the harness has in fact already accepted the preinjected content.
        _gate_relaxed = bool(
            getattr(get_settings(), "skill_gate_treat_preinjected_as_read", True)
        )
        # Single-line preamble: previously we listed `Core:`, `Children
        # (required):`, AND `Pre-injected (gate-satisfied):` — three lines for
        # what is almost always the same set of skills (Core ⊆ Children ⊆
        # Pre-injected, and the actual `<skill name="..." tier="...">` blocks
        # immediately below are self-documenting). Trimmed to one line; the
        # gate semantics ride on the relaxed-vs-strict wording only.
        if _gate_relaxed and _injected_names:
            skill_constraints = (
                f"\nPRE-INJECTED SKILLS (plan_type='{_plan_type}', gate-satisfied): "
                f"{_injected_names}. Use query_skill / read_skill only for "
                f"content NOT in the embedded blocks below.\n"
            )
        else:
            skill_constraints = (
                f"\nPRE-INJECTED SKILLS (plan_type='{_plan_type}'): "
                f"{_injected_names}. Embedded for quick reference, but the "
                f"harness still REQUIRES an explicit read_skill(name) call "
                f"for required skills before write_file / edit_file.\n"
            )
        # No more "deferred" tail: the compression cascade above guarantees
        # every required skill is injected at FULL / MEDIUM / COMPACT tier.
        # Skills genuinely outside the required set are still discoverable
        # via the skill directory below — read_skill(name) on demand.
        # Skill directory: name + one-line description for every registered
        # skill that is NOT already pre-injected above. The model is
        # responsible for routing — pick whatever is relevant, call
        # read_skill(name) on demand. This replaces an upstream keyword
        # router; missing a domain (e.g. fluid/SPH) silently is no longer
        # possible because the directory always lists every skill, and the
        # model decides from descriptions instead of regex matches.
        _preinjected_set = {n for n in _injected_names}
        _directory_text = SkillRegistry.format_skill_directory(
            exclude=_preinjected_set
        )
        skill_constraints += (
            "\nSKILL DIRECTORY (call read_skill(name) on demand; "
            "skills already pre-injected above are omitted):\n"
            f"{_directory_text}\n"
        )
        if _pre_injected_block:
            skill_constraints += (
                "\n=== BEGIN PRE-INJECTED SKILLS ===\n"
                f"{_pre_injected_block}\n"
                "=== END PRE-INJECTED SKILLS ===\n"
            )

        assets_reminder = ""
        if plan.assets:
            assets_reminder = (
                "\nASSET LOADING:\n"
                "- Before writing ANY OBJ / URDF / mesh filename into the simulation, "
                "call check_asset_path(path) to confirm it resolves. A hallucinated "
                "path causes tiny_obj to silently fail, then SIGSEGV inside "
                "ChSensorManager.Update() or ChVisualSystemVSG.\n"
                "- For Chrono built-in assets (paths under sensor/, vehicle/, etc.): "
                "load via chrono.GetChronoDataFile(relative_path).\n"
                "- For type=wrapper_vehicle (HMMWV / CityBus / FEDA / HMMWV_Reduced): "
                "the asset entry has NO `filename`; it carries a `factory` field "
                "like `veh.HMMWV_Full()`. Evaluate that expression directly — the "
                "wrapper owns its ChSystem and loads its internal JSON bundle "
                "automatically. Follow the `veh/wheeled_vehicle` skill's hard rules "
                "on ownership and construction order.\n"
                "- For type=vehicle_json assets (JSON-driven WheeledVehicle such as "
                "Polaris, Sedan): the `filename` is GetVehicleDataFile-relative "
                "(e.g. `Polaris/Polaris.json`). Instantiate via "
                "`veh.WheeledVehicle(sys, veh.GetVehicleDataFile(filename))` then "
                "load companion engine/transmission/tire JSONs from the same dir "
                "with `veh.ReadEngineJSON` / `veh.ReadTransmissionJSON` / "
                "`veh.ReadTireJSON` and wire them in via `InitializePowertrain` and "
                "`InitializeTire` per axle. Those companion JSONs are NOT listed in "
                "`plan.assets[]` — they ride along with the primary vehicle JSON.\n"
                "- For project-local assets (data/scene/... / data/robot/...): always "
                "construct an absolute path — e.g. "
                "os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', ...). "
                "Do NOT rely on the cwd at runtime; simulations run from "
                "history/iteration_NNN/ and bare 'data/...' paths will miss.\n"
                "- Mesh pattern (see demo/scene/demo_SEN_HMMWV_offroad_vsg.py for the "
                "canonical example): prefer ChTriangleMeshConnected.LoadWavefrontMesh "
                "+ ChVisualShapeTriangleMesh + body.AddVisualShape for scene meshes — "
                "that's what the scene/custom_assets_scene_convex_decomp skill teaches "
                "and what the demos use. ChBodyEasyMesh is a viable shortcut but carries "
                "a specific surprise: `ChBodyEasyMesh(filename, density, compute_mass, "
                "create_visualization, create_collision, material)` — note the 4th "
                "positional is create_visualization, not create_collision. Getting the "
                "order wrong silently disables rendering.\n"
                "\n"
                "## VSG CRASH TRAP (SIGSEGV at vsg::BindGraphicsPipeline::record)\n"
                "Do NOT call `vis.BindAll()` after `vis.Initialize()` when "
                "`vis.EnableShadows(True)` is also set and any body carries a mesh "
                "visual. This combination reproducibly SIGSEGVs on the first render. "
                "Either feature alone is fine; together they crash. The project's "
                "reference demo `demo/scene/demo_SEN_HMMWV_offroad_vsg.py` calls "
                "`vis.Initialize()` and nothing else — match that. `BindAll()` is for "
                "the Chrono collision system (e.g. `sys.GetCollisionSystem().BindAll()`) "
                "and ChSystem::Initialize() already triggers it; VSG does not need a "
                "separate `vis.BindAll()` call.\n"
            )
        assets_reminder += self._build_office_height_override(plan, compilation_feedback)

        # -- scope_rule: what to implement (Rule 7 in template) --
        if is_active_step_mode and step_loop.get("step_context"):
            step_ctx_obj = step_loop.get("step_context") or {}
            s_idx = step_ctx_obj.get("step_index", 0)
            s_total = step_ctx_obj.get("total_steps", 1)
            s_desc = step_ctx_obj.get("step_description", "")
            s_assets = [a.get("name", "") for a in step_ctx_obj.get("step_assets", []) if a.get("name")]
            s_constraints = step_ctx_obj.get("step_constraints", []) or []
            # Read the list form (new schema) with back-compat for the
            # singular ``step_camera`` key that older plan.json artifacts
            # may still carry. Pydantic's SimulationStep validator already
            # coerces singular→list at plan-load time; this branch only
            # guards against step_context.json dumps that pre-date the
            # migration.
            s_cameras = step_ctx_obj.get("step_cameras")
            if not s_cameras:
                legacy = step_ctx_obj.get("step_camera") or {}
                s_cameras = [legacy] if legacy else []
            prior_constraints_entries = step_ctx_obj.get("prior_constraints", []) or []

            # Hard-constraints block (current + prior, verbatim)
            constraint_lines: List[str] = []
            if s_constraints:
                constraint_lines.append(f"Constraints for step {s_idx + 1}:")
                constraint_lines.extend(f"  - {c}" for c in s_constraints)
            for entry in prior_constraints_entries:
                cs = entry.get("constraints") or []
                if cs:
                    constraint_lines.append(
                        f"Constraints carried from step {entry.get('step_index', '?') + 1 if isinstance(entry.get('step_index'), int) else '?'} "
                        "(must remain satisfied):"
                    )
                    constraint_lines.extend(f"  - {c}" for c in cs)
            constraints_block = "\n".join(constraint_lines) if constraint_lines else "(no hard constraints declared)"

            if not s_cameras:
                # Schema normally prevents this (cameras is min_length=1) but
                # the default dict fallback keeps codegen from crashing on
                # malformed step_context.json.
                s_cameras = [{"position": [0, 0, 5], "target": [0, 0, 0], "up": [0, 0, 1]}]

            # Resolve recording_mode early so the camera block can branch on
            # it. The full-fidelity assignment + comment block lives further
            # down (search '_recording_mode = getattr(plan, ...)') and is
            # left intact; this just hoists the read.
            _recording_mode_for_camblock = (
                getattr(plan, "recording_mode", "sensor_cams") or "sensor_cams"
            )

            if _recording_mode_for_camblock == "vsg_only":
                # vsg_only ≡ FSI/SPH (enforced by planning_prompts.py).
                # Planner-emitted cam_pos is currently unreliable: it copies
                # `lock_side_camera`'s docstring `-7 * fyDim` formula, which
                # only frames tank-only scenes — multi-object FSI scenes
                # (platforms + vehicle + tank) clip out the X edges. Until
                # we wire camera framing through a scene-AABB solver,
                # hardcode a deterministic default that fits the canonical
                # bounded FSI demos.
                cam_pos = [0.0, -14.0, 1.5]
                cam_tgt = [0.0, 0.0, 0.5]
                cam_up = [0.0, 0.0, 1.0]
                camera_block = (
                    f"CAMERA POSE for this step (vsg_only mode — exactly 1 camera):\n"
                    f"  cam_pos     = chrono.ChVector3d({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]})\n"
                    f"  target_pos  = chrono.ChVector3d({cam_tgt[0]}, {cam_tgt[1]}, {cam_tgt[2]})\n"
                    f"  up_direction= chrono.ChVector3d({cam_up[0]}, {cam_up[1]}, {cam_up[2]})\n"
                    f"This pose MUST be passed VERBATIM to "
                    f"`lock_side_camera(vis, cam_pos, target_pos)`. The recorded "
                    f"mp4 captures exactly this view. Do NOT call "
                    f"`vis.AddCamera(...)` (interactive-only, no effect on mp4) "
                    f"or `vis.SetChaseCamera(...)` (chase mode overrides "
                    f"lock_side_camera and produces a moving viewpoint that "
                    f"won't match the planned pose). Do NOT pass "
                    f"`recorders=[...]` to `run_recording_loop` — VSG-only mode "
                    f"uses `manager=None, recorders=[]`.\n"
                )
            else:
                # sensor_cams mode: 2-3 cameras → one setup_preview_camera
                # per entry → one mp4 per camera. This is the legacy path,
                # preserved unchanged.
                camera_lines: List[str] = [
                    f"CAMERA POSES for this step ({len(s_cameras)} sensor(s) — emit ONE "
                    f"setup_preview_camera(...) call per entry, with a unique `name=` per call):"
                ]
                for i, cam in enumerate(s_cameras):
                    cam_pos = cam.get("position") or [0, 0, 5]
                    cam_tgt = cam.get("target") or [0, 0, 0]
                    cam_up = cam.get("up") or [0, 0, 1]
                    camera_lines.append(
                        f"  camera {i} (suggested name=\"cam_{i}\"):\n"
                        f"    cam_pos     = chrono.ChVector3d({cam_pos[0]}, {cam_pos[1]}, {cam_pos[2]})\n"
                        f"    target_pos  = chrono.ChVector3d({cam_tgt[0]}, {cam_tgt[1]}, {cam_tgt[2]})\n"
                        f"    up_direction= chrono.ChVector3d({cam_up[0]}, {cam_up[1]}, {cam_up[2]})"
                    )
                camera_lines.append(
                    "Do NOT pass recorders=[...] to run_recording_loop — the process-wide "
                    "registry auto-collects every recorder setup_preview_camera created. "
                    "See sens/camera skill § 'Multi-camera per step (plan-driven)'."
                )
                camera_block = "\n".join(camera_lines) + "\n"

            # Build RESOLVED POSITIONS block. The planner has already computed
            # the world-frame (x, y, z) for every body via the predicate
            # algebra in `planning/scene_coordinate_system` and stored it
            # under `scene_predicates[].position`. Without this block,
            # codegen reads step_context.json and SEES the right numbers but
            # then re-derives positions via tutorial-style heuristics like
            # `-bxDim/2 - bxDim*chrono.CH_1_3` or `-FX/2 - 0.05`, producing
            # body coords that don't match the plan (the iter_001 disaster:
            # left_platform at -2.05 instead of -4.05, plate hovering 0.28m
            # above the water, fluid centred at +X+Y instead of origin).
            #
            # The CAMERA POSE block above already proves codegen reliably
            # uses verbatim ChVector3d values when they're rendered in the
            # prompt (it never re-derives camera positions from formulas).
            # This block applies the same treatment to body positions:
            # render literal ChVector3d values, with one entry per subject,
            # so codegen can paste them directly into SetPos / ChFramed.
            scene_preds = step_ctx_obj.get("scene_predicates") or []
            resolved_positions_block = ""
            if scene_preds:
                # Dedupe by subject — multiple predicates per body all
                # carry the same resolved position by plan invariant.
                seen_subjects: set = set()
                position_lines: List[str] = []
                for pred in scene_preds:
                    if not isinstance(pred, dict):
                        continue
                    subject = str(pred.get("subject") or "").strip()
                    pos = pred.get("position") or {}
                    if not subject or subject in seen_subjects:
                        continue
                    if not isinstance(pos, dict):
                        continue
                    px = pos.get("x")
                    py = pos.get("y")
                    pz = pos.get("z")
                    if px is None or py is None or pz is None:
                        continue
                    seen_subjects.add(subject)
                    position_lines.append(
                        f"  {subject:24s} → chrono.ChVector3d({float(px)}, {float(py)}, {float(pz)})"
                    )
                if position_lines:
                    resolved_positions_block = (
                        "RESOLVED POSITIONS (planner-computed; use these VERBATIM in SetPos / "
                        "ChFramed — DO NOT recompute via tutorial formulas like "
                        "`-FX/2 - 0.05` or `-bxDim/2 - bxDim*chrono.CH_1_3`; those are "
                        "calibrated to one specific tutorial geometry and produce wrong "
                        "world coords on any other layout):\n"
                        + "\n".join(position_lines)
                        + "\n\n"
                        "Notes for special subjects:\n"
                        "  - The position for the SPH-fluid subject (e.g. `sph_water`) "
                        "is the FREE-SURFACE point, NOT the fluid sampler's box centre. "
                        "If you need the fluid sampler centre, it is "
                        "(0, 0, free_surface_z / 2) for a fluid that fills the tank "
                        "from z=0 to z=free_surface_z.\n"
                        "  - FSI containers are generated boundaries, not ordinary "
                        "centered boxes. Use the plan's generated-boundary convention "
                        "(xy center, z floor, size.z rim height) to build the explicit "
                        "interior AABB, then pass that AABB to "
                        "`build_fsi_tank(world_extent=ChAABB(...), ...)` as shown in "
                        "fsi/sph Pattern C.\n"
                        "  - For all OTHER subjects (platforms, plates, vehicles, "
                        "robots), use the position directly as `body.SetPos(<position>)`.\n"
                        "Any deviation from these positions silently breaks the plan's "
                        "spatial layout — if you genuinely believe a position is wrong, "
                        "call `rebut_review` to escalate; do NOT silently substitute "
                        "your own value.\n\n"
                    )

            scope_rule = (
                f"STEP MODE (step {s_idx + 1}/{s_total})\n"
                f"Goal: {s_desc}\n"
                f"Assets introduced this step: {s_assets or '(none)'}\n"
                f"Procedural scene objects introduced this step: "
                f"{[o.get('name') for o in step_ctx_obj.get('step_scene_objects', [])] or '(none)'}\n"
                f"step_context.json = focused spec for this step (includes camera, assets, "
                f"scene_objects, constraints).\n\n"
                f"{camera_block}\n"
                f"{resolved_positions_block}"
                f"HARD CONSTRAINTS (must not be violated by the code you write):\n"
                f"{constraints_block}\n"
            )
        elif is_active_step_mode:
            _si = step_loop.get("current_step_index", 0)
            _st = len(step_loop["steps"])
            _sd = step_loop.get("current_step_description", "")
            _sr = step_loop.get("relevant_bodies", [])
            _completed = step_loop.get("completed_steps", [])
            scope_rule = (
                f"STEP MODE (step {_si + 1}/{_st}): Implement ONLY this step -- {_sd}\n"
                f"   ALLOWED bodies: {_sr}. Do NOT create/load/reference any other asset.\n"
                f"   step_context.json is the ONLY spec -- it contains this step's assets, scene_objects, predicates, and parameters.\n"
                f"   There is NO plan.json in step mode. Only use step_context.json."
            )
        else:
            scope_rule = (
                "Implement ALL implementation_steps from plan.json. "
                "Each step must be realized in code; do not omit any."
            )

        # -- mode_constraints: mode-specific rules --
        # Skill-reference reminder kept very short — Rule 11 in the system
        # prompt already enumerates the full grounding hierarchy (pre-injected
        # → query_skill → read_skill_section → read_skill → search_skills →
        # bash introspect). Repeating it here was 400 bytes of "see above".
        _skill_workflow = (
            f"Skill reference: pre-injected blocks below are authoritative for "
            f"plan_type='{_plan_type}'; use read_skill(name) for anything else "
            f"(see Rule 11 for the full grounding hierarchy).\n"
        )
        if is_active_step_mode:
            _si = step_loop.get("current_step_index", 0)
            _completed = step_loop.get("completed_steps", [])
            _is_fsi_plan = _plan_type == "fsi_in_scene"
            if _si == 0:
                if _is_fsi_plan:
                    _edit_instr = (
                        "Read step_context.json, then write_file with the complete simulation "
                        "for THIS STEP. STRUCTURE THE FILE INTO THREE SECTIONS using these EXACT "
                        "barrier comments (per fsi/sph Pattern G):\n"
                        "    # === SECTION 1: SYSTEMS ===\n"
                        "    # === SECTION 2: BODIES + FSI REGISTRATIONS ===\n"
                        "    # === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===\n"
                        "    sysFSI.Initialize()\n"
                        "    # === SECTION 3: VISUALIZATION + RUN LOOP ===\n"
                        "Subsequent steps will pattern-match the 'DO NOT ADD BODIES' line "
                        "to know where to insert new bodies, vehicles, and spindle FSI "
                        "registrations. Do NOT skip these comments — they are the contract."
                    )
                else:
                    _edit_instr = "Read step_context.json, then write_file with the complete simulation for THIS STEP."
            else:
                if _is_fsi_plan:
                    _edit_instr = (
                        "Read simulation.py first (has code from prior steps), then read "
                        "step_context.json, then use edit_file (one or more calls) to ADD "
                        "this step's new content. "
                        "The file is structured into three sections separated by barrier comments "
                        "(per fsi/sph Pattern G):\n"
                        "    # === SECTION 2: BODIES + FSI REGISTRATIONS ===\n"
                        "    # === DO NOT ADD BODIES OR FSI REGISTRATIONS BELOW THIS LINE ===\n"
                        "    # === SECTION 3: VISUALIZATION + RUN LOOP ===\n"
                        "Every new ChBody, vehicle.Initialize(), sysFSI.AddFsiBody(...), or "
                        "sysSPH.AddSPHParticle(...) call MUST be inserted ABOVE the "
                        "'DO NOT ADD BODIES' barrier. Code added below the barrier lands after "
                        "sysFSI.Initialize() and is silently ignored — that is the iteration_008 "
                        "chassis-missing bug from session_20260428_164422 (12 wasted iterations). "
                        "Visualization-only tweaks (camera, window title, recording fps) go in "
                        "SECTION 3. Preserve ALL prior steps' code exactly as-is."
                    )
                else:
                    _edit_instr = (
                        "Read simulation.py first (has code from prior steps), "
                        "then read step_context.json, then use edit_file (one or more calls) "
                        "to ADD this step's new content. Do NOT rewrite the entire file — use "
                        "targeted edit_file calls that insert new import lines at the top and "
                        "new code blocks after the existing content. Preserve ALL prior "
                        "steps' code exactly as-is."
                    )
            mode_constraints = (
                f"## STEP GENERATION MODE\n"
                f"1. {_edit_instr}\n"
                f"2. {_skill_workflow}"
                f"3. Write/patch code using ONLY APIs documented in the pre-injected skills or returned by read_skill().\n"
                f"4. validate_chrono_apis(). Fix invalid APIs by re-reading the relevant skill, not by guessing.\n"
            )
            if _completed:
                mode_constraints += "Completed steps (code exists in simulation.py):\n"
                mode_constraints += "\n".join(f"  {i}. {s}" for i, s in enumerate(_completed, 1)) + "\n"
        elif effective_mode == "generate":
            mode_constraints = (
                "## GENERATION MODE -- ACTION SEQUENCE\n"
                f"1. read_file('plan.json') -- authoritative specification.\n"
                f"2. {_skill_workflow}"
                "3. write_file -- write the ENTIRE simulation in one call, preferring APIs you saw in the pre-injected skills.\n"
                "4. validate_chrono_apis(). If it fails, use bash('python -c \"...\"') or read_skill() to find the correct form, then fix in one edit.\n"
                "\n"
                "In generate mode, prefer not to browse the filesystem unless plan.json points you there.\n"
                "Prefer APIs documented in the pre-injected skills; when uncertain, introspect with bash before guessing.\n"
            )
        else:
            mode_constraints = (
                "## FIX MODE -- ACTION SEQUENCE\n"
                "1. Read handoff.json (failure_context, including any structured runtime error).\n"
                "2. Read simulation.py to locate the broken code; use the failing_line from the structured error if present.\n"
                f"3. {_skill_workflow}"
                "4. If the error references a specific API, prefer bash('python -c \"import pychrono as c; help(c.ChXxx)\"') to verify its actual shape before editing.\n"
                "5. edit_file to fix the issues. Address known issues together when you can; "
                "for several independent edits, emit several edit_file calls in the same turn.\n"
                "6. validate_chrono_apis(). If it fails, consult pre-injected skills or bash introspection, then fix.\n"
            )
            # When the failure feedback originated from a step_review VLM/CSV
            # rejection, expose the rebut_review escape hatch. The LLM should
            # first try to understand the review complaint and patch the code;
            # only if the code is genuinely already correct should it push
            # back with rebut_review.
            if _review_feedback_text:
                mode_constraints += (
                    "\n## REVIEW REBUTTAL (use sparingly)\n"
                    "A step_review rejection triggered this run. If after reading "
                    "simulation.py and the relevant skill you conclude that the "
                    "code ALREADY correctly implements the step and the review's "
                    "complaint is a false positive (e.g. it misread the image, "
                    "misidentified the asset's canonical axis, or flagged a "
                    "predicate that actually holds), you may call rebut_review("
                    "reasoning=...) instead of making any code edit. The review "
                    "will be re-run with your reasoning as additional context.\n"
                    "Do NOT use rebut_review to dodge work -- if even part of the "
                    "complaint is valid, patch the code. Rebuttals are for full, "
                    "evidence-backed disagreement only.\n"
                )

        # -- available_files_hint --
        # plan.md (the planner's readable markdown) is always listed when
        # available so the code agent can orient itself before diving into
        # plan.json / step_context.json. It's a strictly-additive hint.
        _plan_md_hint = " plan.md (readable overview of the planner's intent)," if "plan.md" in extra_files else ""
        if is_active_step_mode:
            available_files_hint = (
                "Available files: simulation.py (code), step_context.json (step spec -- primary),"
                f"{_plan_md_hint}"
                " handoff.json (secondary). There is NO plan.json in step mode.\n"
            )
        elif effective_mode == "generate":
            available_files_hint = (
                "Available files: simulation.py (code), plan.json (spec -- primary),"
                f"{_plan_md_hint}"
                " handoff.json (secondary).\n"
            )
        else:
            available_files_hint = (
                "Available files: simulation.py (code), handoff.json (failure_context -- primary),"
                f"{_plan_md_hint}"
                " plan.json (secondary).\n"
            )

        # -- Assemble system prompt --
        if is_active_step_mode:
            spec_file = "step_context.json"
        else:
            spec_file = "plan.json"
        try:
            utils_reference = build_utils_reference_block()
        except Exception as _utils_ref_exc:
            self.logger.warning(f"[ToolLoop] utils signature digest failed: {_utils_ref_exc}")
            utils_reference = ""
        # Plan-level switch — taken VERBATIM from plan.recording_mode (set by
        # the planner per the planning_prompts.py template + plan_format skill
        # rule: FSI / SPH scenes → 'vsg_only', everything else → 'sensor_cams').
        # No validator override; if the planner picked wrong, the wrong
        # recording mode propagates here and the resulting mp4 will visibly
        # demonstrate the mistake. Default to 'sensor_cams' for legacy plans
        # created before the field existed.
        _recording_mode = getattr(plan, "recording_mode", "sensor_cams") or "sensor_cams"
        # Read motion_expectations from the step_context so rule 6 of the
        # codegen prompt can render the per-step CSV-output contract.
        # Outside step mode (monolithic mbs plans) the list is empty and
        # the rule degrades to the "no motion CSV required" notice.
        _motion_expectations: List[str] = []
        if is_active_step_mode:
            _step_ctx_for_motion = step_loop.get("step_context") or {}
            if isinstance(_step_ctx_for_motion, dict):
                _raw_me = _step_ctx_for_motion.get("step_motion_expectations") or []
                if isinstance(_raw_me, list):
                    _motion_expectations = [
                        str(n).strip() for n in _raw_me if str(n).strip()
                    ]
        # Forward plan.geometry_relations into the codegen prompt so that the
        # geometry_relations_rule is injected only when the plan actually has
        # entries. Empty list → empty string in the prompt (no extra tokens).
        _geometry_relations = list(getattr(plan, "geometry_relations", None) or [])

        # Unified objects[] view for the new placement rule. Read step_objects
        # and scene_predicates straight from the step_context so the rule
        # only fires when the active step actually introduces objects.
        _step_objects: List[Any] = []
        _scene_predicates: List[Any] = []
        if is_active_step_mode:
            _ctx = step_loop.get("step_context") or {}
            if isinstance(_ctx, dict):
                _step_objects = list(_ctx.get("step_objects") or [])
                _scene_predicates = list(_ctx.get("scene_predicates") or [])

        system_prompt = codegen_prompts.build_tool_loop_system_prompt(
            skill_constraints=skill_constraints,
            mode_constraints=mode_constraints,
            scope_rule=scope_rule,
            available_files_hint=available_files_hint,
            assets_reminder=assets_reminder,
            spec_file=spec_file,
            utils_reference=utils_reference,
            recording_mode=_recording_mode,
            motion_expectations=_motion_expectations,
            geometry_relations=_geometry_relations,
            step_objects=_step_objects,
            scene_predicates=_scene_predicates,
        )

        # Dump the realized system prompt to the iteration dir so that
        # offline debugging can inspect exactly what the model saw
        # (placeholders already rendered, pre-injected skills included).
        try:
            from chrono_code.tools.code_agent_tools import _iteration_dir as _iter_dir
            if _iter_dir is not None:
                (_iter_dir / "codegen").mkdir(parents=True, exist_ok=True)
                realized_path = _iter_dir / "codegen" / "system_prompt_realized.txt"
                realized_path.write_text(system_prompt, encoding="utf-8")
                self.logger.debug(
                    f"[ToolLoop] wrote realized system prompt to {realized_path} "
                    f"({len(system_prompt)} chars)"
                )
        except Exception as _dump_exc:
            # Never let prompt-dumping break codegen — it's an observability nicety.
            self.logger.warning(f"[ToolLoop] could not write realized system prompt: {_dump_exc}")

        # -- User prompt --
        if is_active_step_mode:
            _si = step_loop.get("current_step_index", 0)
            _st = len(step_loop["steps"])
            _sd = step_loop.get("current_step_description", "")
            _sr = step_loop.get("relevant_bodies", [])
            if _si == 0:
                _action = "Read step_context.json, then write_file."
            else:
                _action = "Read simulation.py, then step_context.json, then use edit_file (one or more calls) to add this step's code."
            user_prompt = (
                f"Mode: step {_si + 1}/{_st}\n"
                f"Step: {_sd}\n"
                f"Allowed: {_sr}\n"
                f"Action: {_action}\n"
                f"Feedback:\n{feedback_text}\n"
            )
        else:
            mode_display = (
                "fix (bootstrap -- file empty, use write_file)"
                if effective_mode == "generate" and mode == "fix"
                else effective_mode
            )
            user_prompt = (
                f"Mode: {mode_display}\n"
                f"Feedback:\n{feedback_text}\n\n"
                + (
                    "Start now: read_file('plan.json'), then write_file with the complete simulation code.\n"
                    if effective_mode == "generate"
                    else
                    "Read handoff.json first. Patch directly when clear from traceback.\n"
                )
                + (
                    (
                        "TOPOLOGY CHECK: For each joint in topology.joints, state bodies + type, "
                        "cite topology skill rule (Rule 1-6).\n"
                    )
                    if effective_mode == "generate"
                    and plan.topology is not None
                    and getattr(plan.topology, "joints", None)
                    else (
                        "JOINT DESIGN: topology.joints not pre-specified. Design from plan + topology skill rules.\n"
                        if effective_mode == "generate" else ""
                    )
                )
            )

        # Build initial messages list (provider-specific format)
        if self.provider == "anthropic":
            messages: List[Any] = [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
        else:
            messages = [
                {"role": "user", "content": user_prompt},
            ]

        self._log_tool_loop_context(
            mode=mode,
            effective_mode=effective_mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_steps=max_steps,
        )
        total_tool_calls = 0
        no_tool_retries = 0
        fix_mode_requires_edit = effective_mode == "fix" and bool(initial_code)
        edit_tool_call_count = 0
        rebuttal_submitted = False

        # Anti-burn guard. Counts consecutive LLM turns in this loop where the
        # model called tools but NONE of them were edit tools (write_file /
        # edit_file). Reset on any successful edit. Triggers a
        # hard "STOP READING, EDIT NOW" nudge at NO_EDIT_NUDGE_THRESHOLD turns,
        # and ABORTS the loop at NO_EDIT_ABORT_THRESHOLD turns. Without this,
        # session_20260426_204448 burned 80 turns across 2 retries doing pure
        # exploration (read_file_content / list_directory / glob / bash) and
        # never produced a single new write — every "Updated simulation.py
        # from review fail" event in the transcript was a no-op rerun of the
        # iter_001 file.
        consecutive_turns_without_edit = 0
        NO_EDIT_NUDGE_THRESHOLD = 8
        NO_EDIT_ABORT_THRESHOLD = 16

        # Detect "invented API" prior-failure pattern. When the previous
        # execution crashed with AttributeError / NameError / "has no attribute"
        # / "is not callable", the most likely cause is that the LLM hallucinated
        # a method name (e.g. ``ChWheel.GetSpindleLocal`` — does not exist).
        # In this state we force a ``validate_chrono_apis`` call after the next
        # successful edit, BEFORE another edit is allowed, so the validator
        # catches re-hallucinations immediately instead of paying for another
        # full execution + 6-minute retry cycle to re-discover the same class
        # of bug. ``feedback_text`` carries the structured prior-failure body
        # whenever the workflow loops back after a crash. We do NOT gate on
        # ``effective_mode == "fix"`` because step-mode regenerations after a
        # crash flip ``effective_mode`` to "generate" (initial_code is empty
        # for the rebuilt step), which silently disabled this guardrail and
        # let invented-API edits ship to execution unvalidated -- see
        # session_20260428_150327 iter_5 ``ChBodyEasyBox.SetBodyFixed``.
        _prior_failure_blob = (feedback_text or "").lower()
        prior_failure_was_invented_api = any(
            sig in _prior_failure_blob for sig in (
                "attributeerror",
                "nameerror",
                "has no attribute",
                "is not callable",
                "object has no attribute",
                # TypeError on a Chrono symbol with a bogus kwarg / arg
                # signature is also "invented API" -- the model fabricated
                # a constructor signature. iter_4 of the same session hit
                # ``ChBodyEasyBox(material=...)`` here.
                "got an unexpected keyword argument",
                "takes no keyword arguments",
                "got multiple values for argument",
                "missing 1 required positional argument",
            )
        )
        if prior_failure_was_invented_api:
            self.logger.info(
                "[ToolLoop] prior_failure_was_invented_api=True — will require "
                "validate_chrono_apis between successive edits this run."
            )

        # Detect "prior failure was SPH NaN / SIGABRT" — the failure mode that
        # tempts codegen to delete spindle FSI registration to silence the
        # crash. Per fsi/sph HR-5 anti-fix warning, that's the wrong fix (silently breaks
        # wheel-fluid coupling, mp4 still looks plausible, review PASSes).
        # The right fix is HR-11: expand the computational domain. We can't
        # tell from a diff alone whether codegen ADDED or REMOVED a spindle
        # registration, so on every successful edit during a NaN-prior fix
        # run we inject a strong reminder.
        prior_failure_was_sph_nan = any(
            sig in _prior_failure_blob for sig in (
                "sigabrt",
                "nan in sph",
                "nan particle",
                "nan at particle",
                "out of min boundary",
                "out of max boundary",
                "calchashd",
            )
        )
        if prior_failure_was_sph_nan:
            self.logger.info(
                "[ToolLoop] prior_failure_was_sph_nan=True — will warn against "
                "deleting spindle FSI registration to silence the crash "
                "(fsi/sph HR-5 anti-fix warning)."
            )


        # Generic self-critique nudge fired after EVERY successful edit. The
        # bug class this targets: codegen reads a skill (gate-enforced) and
        # then writes code that violates one of the skill's hard rules anyway
        # — typically a default-is-wrong VSG / vehicle setup omission
        # (`vis.AttachSystem(sysMBS)`, `SetChassisVisualizationType`, FSI
        # step function ordering). Pure-prompt mitigation: force the model to
        # enumerate the hard rules its patch should satisfy, cite the line
        # that satisfies each, and surface gaps. No grep templates, no
        # validator code; the skill prose stays the source of truth and the
        # model's own reasoning closes the loop. Coexists with the
        # invented-API nudge above — that one wins on conflict via the
        # `not pending_post_batch_nudge` guard at the fire site.
        SKILL_SELFCHECK_NUDGE = (
            "Your edit landed. Before doing anything else, scan the "
            "pre-injected skill blocks above for any 'Hard Rule' / 'HR-N' / "
            "'HARD RULE' that applies to the area you just touched. Reply "
            "with a short audit:\n"
            "\n"
            "  - HR-X (skill_name): satisfied by line N — `<one-line code excerpt>`\n"
            "  - HR-Y (skill_name): MISSING — needs `<what to add>`\n"
            "\n"
            "Rules:\n"
            "  - Only list rules whose subject (a class, function, or "
            "pattern) appears in your latest patch. Do NOT list rules "
            "unrelated to your edit.\n"
            "  - If a rule is MISSING, your next action MUST be one "
            "or more edit_file calls in the same turn that fix ALL "
            "missing rules. Do "
            "NOT continue to validation / execution / further reads until "
            "the audit shows zero MISSING entries.\n"
            "  - If every applicable rule is satisfied, write 'audit clean' "
            "on its own line and proceed."
        )

        self.logger.info(
            f"[ToolLoop] plan_type={_plan_type} mode={effective_mode} "
            f"max_steps={max_steps} model={self.model} provider={self.provider}"
        )
        diagnostic_tool_names = {
            "grep_code",
            "search_skills",
            "read_skill",
            "read_skill_section",
        }
        failure_ctx = handoff.failure_context
        diagnostic_budget = dict((failure_ctx.metadata or {}).get("diagnostic_budget") or {}) if failure_ctx is not None else {}
        max_total_diagnostic_queries = int(diagnostic_budget.get("max_total_diagnostic_queries") or 3)
        max_same_family_queries = int(diagnostic_budget.get("max_same_family_queries") or 1)
        diagnostic_total_queries = 0
        diagnostic_query_fingerprints: Dict[str, int] = dict((((handoff.metadata or {}).get("diagnostic_memory") or {}).get("query_fingerprints") or {}))
        diagnostic_family_counts: Dict[str, int] = {}

        # One Read One Patch enforcement: track reads and patches per LLM response
        reads_since_last_patch = 0
        patches_since_last_read = 0

        for loop_idx in range(max_steps):
            # Persist the prompt for THIS turn before firing the LLM, so the
            # dialog log captures intent even if the call raises (timeout /
            # cancel / API error). Without this, the only on-disk evidence of
            # codegen activity is the workflow's "Generated simulation.py"
            # transcript line — every read_skill / write_file / edit_file
            # arg, and every preceding system prompt, is invisible.
            await self._log_tool_turn_prompt(
                system_prompt=system_prompt,
                messages=messages,
                step=loop_idx,
                terminal_tool_name=None,
            )
            try:
                response = await self._call_llm_with_tools(
                    system_prompt=system_prompt,
                    messages=messages,
                    tool_defs=tool_defs,
                )
            except BaseException as exc:
                # LLM call failed (CancelledError, timeout, API error, etc.).
                # Preserve any code that was already patched in context rather
                # than letting the exception propagate and losing the work.
                _salvaged = str(context.get("current_code") or "")
                _changed = _salvaged and _salvaged != initial_code
                self.logger.error(
                    f"[ToolLoop] LLM call failed at iter {loop_idx + 1}: "
                    f"{type(exc).__name__}: {exc}; "
                    f"code_changed={_changed}, salvaging current_code"
                )
                break

            # Parse tool calls and text from the response
            tool_calls, response_text = self._parse_tool_calls_from_response(response)
            tool_names = [c.get("name", "?") for c in tool_calls]
            self.logger.info(f"[ToolLoop] iter={loop_idx + 1} tools={tool_names}")

            # Persist the response artifact (text + tool calls + thinking).
            # ``_codegen_thinking_text`` is set by ``_call_llm_with_tools`` —
            # Anthropic streams thinking_delta events and concatenates them;
            # OpenAI-compat reads ``message.reasoning_content`` when present.
            _turn_thinking = getattr(response, "_codegen_thinking_text", "") or ""
            await self._log_tool_turn_response(
                text=response_text,
                tool_calls=tool_calls,
                step=loop_idx,
                thinking=_turn_thinking,
            )

            # Append assistant response to messages
            assistant_msg = self._build_assistant_message(response)
            messages.append(assistant_msg)

            # Emit agent thinking if LLM returned text alongside tool calls
            if tool_calls:
                if response_text.strip():
                    self._emit_tool_event(
                        type="agent_thinking",
                        agent=self.agent_name,
                        content=response_text,
                        loop_iter=loop_idx,
                    )

            if not tool_calls:
                if rebuttal_submitted:
                    # A rebuttal was accepted as a terminal action -- exit cleanly
                    # without forcing an edit. The outer workflow will re-run the
                    # review with the rebuttal as additional context.
                    break
                if fix_mode_requires_edit and edit_tool_call_count == 0 and no_tool_retries < 3:
                    no_tool_retries += 1
                    self._append_user_nudge(
                        messages,
                        "Fix mode is active and no edit tool was called. "
                        "You MUST call edit_file now to modify simulation.py. "
                        "Prose-only responses are not accepted -- use tools.",
                    )
                    continue
                # Catch-all: if no edit tool was ever called, force the LLM to edit.
                # This covers both generation mode (write_file) and fix mode (edit_file).
                if edit_tool_call_count == 0 and no_tool_retries < 3:
                    no_tool_retries += 1
                    _edit_names = ", ".join(sorted(edit_tool_names))
                    self._append_user_nudge(
                        messages,
                        f"You have not made any code edits yet. "
                        f"You MUST call {_edit_names} to produce or modify simulation.py. "
                        f"Prose-only or read-only responses are not accepted. "
                        f"Use {_edit_names} NOW.",
                    )
                    continue
                # Clean exit — LLM produced text-only response (Claude Code pattern).
                break
            no_tool_retries = 0

            # Anti-burn: detect "tools called but no edit was among them" turns
            # and escalate. ``edit_tool_call_count`` is the cumulative counter;
            # we capture it BEFORE this batch so the post-batch comparison
            # tells us whether any edit landed on this turn.
            edit_count_before_batch = edit_tool_call_count

            # One Read One Patch: reset counters at start of each LLM response batch
            reads_since_last_patch = 0
            patches_since_last_read = 0

            # Collect all tool results for this batch. tool_results MUST flush
            # as a single contiguous block immediately after the assistant
            # message carrying the matching tool_calls — OpenAI-compat
            # providers (MiniMax in particular) reject the conversation with
            # `invalid params, tool call result does not follow tool call
            # (2013)` if any non-tool message slips between a tool_call and
            # its tool_result. Anthropic is more forgiving because it collects
            # tool_result blocks inside one user message, but the same
            # contract applies. Any user-nudge that needs to run this
            # iteration must therefore wait until AFTER the batch flush.
            tool_results_batch: List[Dict[str, Any]] = []
            pending_post_batch_nudge: Optional[str] = None

            # Anti-burn progress tracking. The cumulative ``edit_tool_call_count``
            # only ticks on edits that succeeded (write_file / edit_file
            # returned a "successfully" / "Replaced … occurrence(s)" result).
            # Track per-batch whether the LLM made any edit attempt or
            # called the validator, so the downstream no-edit-streak guard
            # can distinguish "agent is doing something" from "agent is
            # only reading".
            edit_attempt_in_batch = False
            validator_call_in_batch = False

            for call_idx, call in enumerate(tool_calls):
                total_tool_calls += 1
                tool_name = call.get("name")
                tool_args = call.get("args", {}) or {}
                tool_call_id = call.get("id", "")
                tool_exists = str(tool_name) in tools_by_name
                if str(tool_name) in {"write_file", "edit_file"}:
                    edit_attempt_in_batch = True
                if str(tool_name) == "validate_chrono_apis":
                    validator_call_in_batch = True

                # Emit tool call start event for activity feed
                self._emit_tool_event(
                    type="tool_call",
                    tool_name=str(tool_name),
                    tool_args=tool_args,
                    status="start",
                    call_index=call_idx,
                    loop_iter=loop_idx,
                )

                # Track read/edit for One-Read-One-Edit enforcement.
                if str(tool_name) == "read_file":
                    reads_since_last_patch += 1
                    patches_since_last_read = 0

                # One-Read-One-Edit violation detection: edit_file called
                # repeatedly with no read_file in between. (The legacy name
                # "patches" is preserved on the counter variables for
                # minimal churn — semantics are identical for edit_file.)
                if str(tool_name) == "edit_file":
                    patches_since_last_read += 1
                    if patches_since_last_read > 1 and reads_since_last_patch == 0:
                        # Allow consecutive edits in the validate-fix cycle:
                        # after a successful edit -> validate failure, the validator output
                        # serves as equivalent context to a read_file call.
                        recent_has_validator = any(
                            "chrono_api_validation" in str(tr.get("content", ""))
                            for tr in tool_results_batch[-4:]
                        )
                        if not recent_has_validator:
                            self.logger.warning(
                                f"[ToolLoop] ONE-READ-ONE-EDIT VIOLATION at iter {loop_idx + 1}: "
                                f"{patches_since_last_read} edit_file calls without read_file"
                            )
                            tool_results_batch.append(
                                self._build_tool_result_message(
                                    tool_call_id,
                                    "ONE-READ-ONE-EDIT VIOLATION: You called edit_file multiple times "
                                    "after a single read_file. You MUST call read_file again before "
                                    "any subsequent edit_file call. Re-read the file now.",
                                )
                            )
                            # Reset to allow correction / next cycle
                            patches_since_last_read = 0
                            reads_since_last_patch = 1  # Pretend we just read to allow next edit

                result = None
                if not tool_exists:
                    result = f"Unknown tool: {tool_name}"
                    self._emit_tool_event(type="tool_call", tool_name=str(tool_name), tool_args=tool_args, status="error", result_preview=result, call_index=call_idx, loop_iter=loop_idx)
                else:
                    should_block_diagnostic = False
                    if effective_mode == "fix" and str(tool_name) in diagnostic_tool_names:
                        fingerprint = self._build_query_fingerprint(str(tool_name), tool_args)
                        family_key = self._build_query_family(str(tool_name), tool_args, failure_ctx)
                        if fingerprint in diagnostic_query_fingerprints:
                            should_block_diagnostic = True
                            result = (
                                "Duplicate diagnostic query blocked. This semantic grep/search/skill query "
                                "was already used for the current question. Change strategy instead of repeating it."
                            )
                        elif diagnostic_total_queries >= max_total_diagnostic_queries:
                            should_block_diagnostic = True
                            result = (
                                "Diagnostic query budget exhausted for the current question. Stop searching and "
                                "either apply a safe validated fallback or surface a structured failure."
                            )
                        elif diagnostic_family_counts.get(family_key, 0) >= max_same_family_queries:
                            should_block_diagnostic = True
                            result = (
                                "Same-family diagnostic query budget exhausted for the current question. "
                                "Do not keep probing the same hypothesis family; change strategy now."
                            )
                        if not should_block_diagnostic:
                            diagnostic_total_queries += 1
                            diagnostic_query_fingerprints[fingerprint] = diagnostic_query_fingerprints.get(fingerprint, 0) + 1
                            diagnostic_family_counts[family_key] = diagnostic_family_counts.get(family_key, 0) + 1
                    if should_block_diagnostic:
                        self._emit_tool_event(type="tool_call", tool_name=str(tool_name), tool_args=tool_args, status="blocked", result_preview=str(result), call_index=call_idx, loop_iter=loop_idx)
                    else:
                        executor = tool_executors.get(str(tool_name))
                        if executor is None:
                            result = f"No executor for tool: {tool_name}"
                            self._emit_tool_event(type="tool_call", tool_name=str(tool_name), tool_args=tool_args, status="error", result_preview=result, call_index=call_idx, loop_iter=loop_idx)
                        else:
                            try:
                                if asyncio.iscoroutinefunction(executor):
                                    result = await executor(**tool_args)
                                else:
                                    result = await asyncio.to_thread(executor, **tool_args)
                                self._emit_tool_event(type="tool_call", tool_name=str(tool_name), tool_args=tool_args, status="complete", result_preview=str(result), call_index=call_idx, loop_iter=loop_idx)
                            except Exception as exc:
                                result = f"Tool {tool_name} failed: {type(exc).__name__}: {exc}"
                                self._emit_tool_event(type="tool_call", tool_name=str(tool_name), tool_args=tool_args, status="error", result_preview=result, call_index=call_idx, loop_iter=loop_idx)

                if str(tool_name) in edit_tool_names:
                    edit_tool_call_count += 1

                _result_text = str(result).lower()
                # edit tools: inline auto-validators already ran inside the tool
                # and their output is in str(result) — no forced validate call needed.

                if str(tool_name) == validation_tool_name:
                    # validate_chrono_apis is now ADVISORY — its result is fed back
                    # to the LLM which decides what to do next. It is NOT a
                    # termination gate. Loop exit happens only when the LLM produces
                    # a text-only response that also passes the semantic completion
                    # check in the no-tool-calls handler below.
                    tool_results_batch.append(
                        self._build_tool_result_message(tool_call_id, str(result))
                    )
                    if "chrono_api_validation: pass" in _result_text:
                        self.logger.info(
                            f"[ToolLoop] validate_chrono_apis passed at iter {loop_idx + 1} "
                            f"(edit_count={edit_tool_call_count}) — continuing, LLM decides next."
                        )
                        if edit_tool_call_count == 0 and not rebuttal_submitted:
                            _edit_names = ", ".join(sorted(edit_tool_names))
                            pending_post_batch_nudge = (
                                "validate_chrono_apis passed, but you have NOT "
                                "edited simulation.py yet in this run. The plan step "
                                "still needs to be implemented. Call "
                                f"{_edit_names} now to apply the required changes."
                            )
                    if "chrono_api_validation: fail" in _result_text:
                        _repair_hint = (
                            "Validator found invalid PyChrono APIs. "
                            "The '--- AUTO-RETRIEVED SKILL API CONTRACTS ---' block above "
                            "lists skills matching the failing symbols. Call read_skill(name) "
                            "or read_skill_section(name, heading) to fetch correct signatures, "
                            "then fix ALL invalid APIs (one or more edit_file calls in the same turn) and validate again."
                        )
                        pending_post_batch_nudge = _repair_hint
                    continue

                _edit_succeeded_now = (
                    (str(tool_name) == "write_file" and "wrote " in _result_text)
                    or (str(tool_name) == "edit_file" and "replaced " in _result_text)
                )
                if str(tool_name) == "write_file" and "wrote " in _result_text:
                    self.logger.info(f"[ToolLoop] write_file succeeded at iter {loop_idx + 1}")
                if str(tool_name) == "edit_file" and "replaced " in _result_text:
                    self.logger.info(f"[ToolLoop] edit_file succeeded at iter {loop_idx + 1}")

                # Bug C — invented-API guardrail. When the prior failure was
                # an AttributeError / NameError / "has no attribute" pattern
                # and the model just made a successful edit, force it to
                # validate_chrono_apis BEFORE the next edit. Without this
                # gate the loop happily re-invents APIs each retry: edit →
                # crash 0.3s into execution → workflow re-feeds same review
                # text → edit again with a different invented method → crash
                # again, until the 6-attempt budget burns out.
                if _edit_succeeded_now and prior_failure_was_invented_api:
                    # Auto-run the validator HERE rather than nudging the LLM
                    # to call it. The nudge alone is unreliable -- iter_5 of
                    # session_20260428_150327 made one successful edit and
                    # then exited the loop with text-only, never validating,
                    # shipping a fresh ``ChBodyEasyBox.SetBodyFixed`` (a
                    # different invented API than the prior crash) straight
                    # to execution. Running the validator in-harness against
                    # the just-saved code closes that hole: the LLM cannot
                    # ignore the FAIL because the FAIL block lands in the
                    # same turn's transcript, and on PASS we drop the gate
                    # entirely so a subsequent edit is not blocked.
                    from chrono_code.tools.code_agent_tools import (
                        _run_chrono_api_validation,
                    )
                    _post_edit_code = str(context.get("current_code") or "")
                    try:
                        _post_edit_validator_out = _run_chrono_api_validation(
                            _post_edit_code
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning(
                            "[ToolLoop] post-edit auto-validate raised %s; "
                            "falling back to LLM-driven validate-before-edit",
                            exc,
                        )
                        _post_edit_validator_out = ""

                    _post_edit_passed = (
                        "chrono_api_validation: pass"
                        in _post_edit_validator_out.lower()
                    )

                    if _post_edit_passed:
                        # Validator confirms the patch only used real APIs.
                        pending_post_batch_nudge = (
                            "Auto-validation ran on your edit and PASSED -- "
                            "all PyChrono symbols you wrote exist. You may "
                            "continue with the next required change or end "
                            "the turn if the step is complete.\n\n"
                            f"{_post_edit_validator_out}"
                        )
                    else:
                        # FAIL (or unknown) -- surface the validator output to
                        # the LLM via post-batch nudge. No hard gate: the LLM
                        # decides whether to fix syntax, re-read a skill, or
                        # try a different edit. Hard gating produced the
                        # session_20260428_160344 chicken-and-egg deadlock
                        # where IndentationError was misclassified as an
                        # invented-API failure.
                        pending_post_batch_nudge = (
                            "Your previous run failed with AttributeError / "
                            "NameError / TypeError -- the kind of error that "
                            "signals an INVENTED PyChrono API (a symbol or "
                            "kwarg the model wrote from memory that does not "
                            "exist in the installed bindings). Your edit just "
                            "landed and the harness auto-validated it. "
                            "Validator output below -- if it reports FAIL, "
                            "fix ALL invalid symbols (one or more edit_file calls in the same turn) using "
                            "the AUTO-RETRIEVED SKILL API CONTRACTS as the "
                            "source of truth, then re-run validate_chrono_apis "
                            "to confirm PASS. Do NOT shotgun more edits "
                            "hoping the next guess works.\n\n"
                            f"{_post_edit_validator_out}"
                        )

                # Spindle-removal anti-fix guard (fsi/sph HR-5 anti-fix warning). When the
                # prior failure was an SPH NaN / SIGABRT crash, the most
                # tempting "fix" is to delete the spindle FSI registration
                # — it silences the symptom because the spindle BCE markers
                # no longer exist to fall outside cMin/cMax. But it silently
                # disables wheel-fluid coupling: the vehicle drives on the
                # plate without sinking it, and the VLM review has no way
                # to detect the missing coupling from the rendered mp4.
                # The right fix is HR-11: expand the computational domain
                # to envelope the full vehicle trajectory.
                if (
                    _edit_succeeded_now
                    and prior_failure_was_sph_nan
                    and not pending_post_batch_nudge
                ):
                    pending_post_batch_nudge = (
                        "Your previous run crashed with NaN / SIGABRT in the "
                        "SPH solver (or 'out of min/max boundary' from "
                        "calcHashD). If your edit silenced this by REMOVING "
                        "any `sysFSI.AddFsiBody(spindle, ...)` call on a "
                        "vehicle wheel — REVERT that immediately. Per "
                        "fsi/sph HR-5 anti-fix warning, deleting spindle FSI registration "
                        "silently defeats wheel-fluid coupling: the vehicle "
                        "drives on the floating plate without pushing it "
                        "down, the plate never sinks, and the VLM review "
                        "passes a physically broken sim because the visual "
                        "still shows water + vehicle + plate. The CORRECT "
                        "fix (per HR-11) is to EXPAND the computational "
                        "domain: compute cMin/cMax as the union of "
                        "{fluid region, vehicle path across the full "
                        "sim_duration, all FSI body footprints} plus "
                        "5*initial_spacing of margin on every face. Then "
                        "keep the spindle registration loop intact."
                    )

                # Skill compliance self-critique. Fires after EVERY successful
                # edit (independent of fix-vs-generate mode). Coexists with
                # the invented-API nudge above — when both would set a nudge,
                # the invented-API one takes priority because invalid symbols
                # block compilation entirely. Once that run validates clean,
                # subsequent successful edits will pick up this self-critique
                # nudge naturally.
                if _edit_succeeded_now and not pending_post_batch_nudge:
                    pending_post_batch_nudge = SKILL_SELFCHECK_NUDGE

                if str(tool_name) == "rebut_review" and "Rebuttal submitted" in str(result):
                    rebuttal_submitted = True
                    self.logger.info(
                        f"[ToolLoop] rebuttal submitted at iter {loop_idx + 1}; "
                        f"terminating loop -- review will be re-run."
                    )
                    tool_results_batch.append(
                        self._build_tool_result_message(tool_call_id, str(result))
                    )
                    handoff_dict = handoff.model_dump()
                    handoff_dict.setdefault("metadata", {})["codegen_rebuttal"] = (
                        context.get("codegen_rebuttal") or ""
                    )
                    # Flush tool results
                    if tool_results_batch:
                        self._append_tool_results_to_messages(messages, tool_results_batch)
                    return (
                        str(context.get("current_code") or ""),
                        total_tool_calls,
                        edit_tool_call_count,
                        handoff_dict,
                        None,
                    )

                # Append tool result
                # read_file / grep_code / read_skill / read_skill_section /
                # query_skill: no truncation (model-requested reference content).
                _is_full_content = str(tool_name) in {
                    "read_file", "grep_code", "read_skill",
                    "read_skill_section", "query_skill",
                }
                # edit_file returns a diff-shaped summary that can be
                # multi-kilobyte but still high-signal; give it a larger head
                # budget and no tail (the tail of a diff is usually empty).
                settings = self.settings
                trunc_enabled = bool(getattr(settings, "tool_output_truncate_enabled", True))
                if _is_full_content or not trunc_enabled:
                    content = str(result or "")
                elif str(tool_name) == "edit_file":
                    content = compact_tool_output(result, max_chars=8000)
                else:
                    content = elide_middle(
                        result,
                        head_chars=int(getattr(settings, "tool_output_head_chars", 1500)),
                        tail_chars=int(getattr(settings, "tool_output_tail_chars", 500)),
                    )
                tool_results_batch.append(
                    self._build_tool_result_message(tool_call_id, content)
                )

            # Flush any remaining tool results for this iteration, THEN apply
            # a deferred user-nudge (see pending_post_batch_nudge note above).
            # Doing both in this order preserves the "tool_result must follow
            # tool_call" invariant that MiniMax enforces as error 2013.
            if tool_results_batch:
                self._append_tool_results_to_messages(messages, tool_results_batch)
            if pending_post_batch_nudge:
                self._append_user_nudge(messages, pending_post_batch_nudge)
                pending_post_batch_nudge = None

            # Anti-burn evaluation. The streak counts batches without forward
            # progress; reset on any of:
            #   (a) a successful edit landed (cumulative counter advanced);
            #   (b) an edit was attempted (write_file / edit_file called) --
            #       attempt counts as forward motion even if the tool itself
            #       rejected the input, since the next iteration
            #       will see the rejection feedback and adapt;
            #   (c) the LLM called validate_chrono_apis.
            # The thresholds bracket sane fix-loop length: 8 read-only turns
            # is normal investigation; 16 is "the model is stuck reading and
            # never going to write."
            if (
                edit_tool_call_count > edit_count_before_batch
                or edit_attempt_in_batch
                or validator_call_in_batch
            ):
                consecutive_turns_without_edit = 0
            else:
                consecutive_turns_without_edit += 1
                if consecutive_turns_without_edit >= NO_EDIT_ABORT_THRESHOLD:
                    self.logger.error(
                        "[ToolLoop] ABORTING after %d consecutive read-only "
                        "turns. The model has not produced any edit and is "
                        "burning budget on exploration. Returning current "
                        "code so the workflow can route the failure cleanly "
                        "instead of timing out at max_steps.",
                        consecutive_turns_without_edit,
                    )
                    break
                if (
                    consecutive_turns_without_edit == NO_EDIT_NUDGE_THRESHOLD
                    and not pending_post_batch_nudge
                ):
                    _edit_names = ", ".join(sorted(edit_tool_names))
                    pending_post_batch_nudge = (
                        f"STOP READING. You have made {consecutive_turns_without_edit} "
                        f"consecutive turns of tool calls WITHOUT calling any edit "
                        f"tool ({_edit_names}). Further reads will be wasted budget. "
                        f"On the next turn you MUST call {_edit_names} with the "
                        f"changes implied by the review feedback. If you genuinely "
                        f"do not know what to change, write a single-line file "
                        f"comment '# review feedback unparseable: <quote>' and "
                        f"call rebut_review() — do not keep exploring."
                    )
                    # Inject the nudge for the NEXT turn. The block above
                    # already consumed pending_post_batch_nudge for this turn,
                    # so we set it again here so the next iteration picks it
                    # up at the top of the loop.
                    self._append_user_nudge(messages, pending_post_batch_nudge)
                    pending_post_batch_nudge = None

        if total_tool_calls == 0:
            self.logger.warning("[ToolLoop] finished with zero tool calls")
        structured_error = None
        if edit_tool_call_count == 0 and not rebuttal_submitted:
            structured_failure_metadata = {
                "mode": effective_mode,
                "root_cause": ((failure_ctx.structured_error.summary if failure_ctx and failure_ctx.structured_error else failure_ctx.summary if failure_ctx else "") or ""),
                "current_question": (((handoff.metadata or {}).get("diagnostic_memory") or {}).get("current_question") or ""),
                "query_fingerprints": diagnostic_query_fingerprints,
                "diagnostic_queries_used": diagnostic_total_queries,
                "safe_fallback": "Prefer a validated local fallback such as removing the uncertain optional API call and preserving known-good visualization setup.",
            }
            structured_error = self._make_structured_codegen_error(
                summary="Fix mode completed without any edit tool call",
                error_type="tool_loop_no_edit",
                retryable=True,
                recommended_action="Apply a safe validated fallback or surface a structured failure instead of ending silently.",
                metadata=structured_failure_metadata,
            )
        handoff.metadata.setdefault("diagnostic_memory", {})
        handoff.metadata["diagnostic_memory"]["query_fingerprints"] = diagnostic_query_fingerprints
        handoff_dict = handoff.model_dump()
        if rebuttal_submitted:
            handoff_dict.setdefault("metadata", {})["codegen_rebuttal"] = (
                context.get("codegen_rebuttal") or ""
            )
        return (
            str(context.get("current_code") or ""),
            total_tool_calls,
            edit_tool_call_count,
            handoff_dict,
            structured_error,
        )

    async def _execute_with_tools(
        self,
        plan: SimulationPlan,
        compilation_feedback: Optional[Any],
        previous_code: Optional[str],
        state: Optional[dict],
        fix_mode: bool = False,
    ) -> Tuple[GeneratedCode, Dict[str, Any]]:
        """Execute generation/fix flow using tool-calling loop."""
        updated_state = dict(state or {})
        patch_text: Optional[str] = None
        patch_hunks: List[Dict[str, Any]] = []
        patch_apply_status = "not_attempted"
        patch_apply_error: Optional[str] = None
        applied_code: Optional[str] = None
        base_code: Optional[str] = None

        initial_code = previous_code or ""

        # -- Pre-create iteration directory so skill_read_log.json can be persisted --
        from chrono_code.agents.execution_agent import prepare_next_iteration_dir
        from chrono_code.tools.code_agent_tools import set_iteration_dir

        # Reuse existing iteration_dir from state (fix retries), or create new
        _existing_iter_dir = (state or {}).get("iteration_dir")
        if _existing_iter_dir:
            iteration_dir = Path(_existing_iter_dir).resolve()
            iteration_dir.mkdir(parents=True, exist_ok=True)
        else:
            history_root = self.settings.history_output_path()
            iteration_dir, _ = prepare_next_iteration_dir(history_root)
        set_iteration_dir(iteration_dir)
        updated_state["iteration_dir"] = str(iteration_dir)
        self.logger.info(f"[CodeGen] iteration_dir = {iteration_dir}")

        # Materialize step_context.json next to simulation.py so the generated
        # code's `Path(__file__).parent / "step_context.json"` open succeeds.
        # The file is otherwise only available as a codegen virtual file (via
        # read_file()); subprocess execution and manual `python simulation.py`
        # from any cwd both rely on the on-disk copy being in the iteration dir.
        _step_loop_for_ctx = (state or {}).get("step_loop") or (state or {}).get("_step_loop") or {}
        _step_ctx_for_disk = None
        if isinstance(_step_loop_for_ctx, dict) and _step_loop_for_ctx:
            _step_ctx_for_disk = _step_loop_for_ctx.get("step_context")
            if not _step_ctx_for_disk:
                try:
                    _step_ctx_for_disk = plan.build_step_context(
                        _step_loop_for_ctx.get("current_step_index", 0),
                        _step_loop_for_ctx.get("completed_steps", []),
                    ).model_dump()
                except Exception as exc:
                    self.logger.warning(
                        f"[CodeGen] step_context rebuild for iteration_dir dump failed: {exc}"
                    )
                    _step_ctx_for_disk = None
        if _step_ctx_for_disk:
            try:
                (iteration_dir / "step_context.json").write_text(
                    json.dumps(_step_ctx_for_disk, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.warning(f"[CodeGen] failed to write step_context.json: {exc}")

        attempted_fixes = (state or {}).get("attempted_fixes")
        if not isinstance(attempted_fixes, list):
            attempted_fixes = []

        _build_state = (state or {}).get("build") or {}
        tool_loop_result = await self._run_tool_loop(
            plan=plan,
            compilation_feedback=compilation_feedback,
            initial_code=initial_code,
            attempted_fixes=attempted_fixes if attempted_fixes else None,
            fix_mode=fix_mode,
            prior_handoff=(state or {}).get("llm_handoff") or (state or {}).get("_llm_handoff"),
            feedback_source=str((state or {}).get("feedback_source") or (state or {}).get("_feedback_source") or ""),
            error_history=(state or {}).get("error_history") or (state or {}).get("_error_history"),
            step_loop=(state or {}).get("step_loop") or (state or {}).get("_step_loop"),
            force_full_rewrite=bool(_build_state.get("force_full_rewrite", False)),
            repair_mode=bool(_build_state.get("repair_mode", False)) and fix_mode and bool(initial_code),
        )
        if isinstance(tool_loop_result, tuple) and len(tool_loop_result) == 5:
            final_code, tool_calls_count, edit_tool_calls_count, llm_handoff_dict, structured_error_dict = tool_loop_result
        elif isinstance(tool_loop_result, tuple) and len(tool_loop_result) == 3:
            final_code, tool_calls_count, edit_tool_calls_count = tool_loop_result
            llm_handoff_dict, structured_error_dict = None, None
        else:
            raise ValueError("_run_tool_loop() returned an unexpected result shape")

        updated_state["llm_handoff"] = llm_handoff_dict
        updated_state["structured_error"] = structured_error_dict

        # If the LLM submitted a rebuttal against the review-agent decision,
        # surface it on the top-level state so the outer workflow (step_review
        # node) can re-run the review with the rebuttal as additional context.
        _rebuttal_text = None
        if isinstance(llm_handoff_dict, dict):
            _rebuttal_text = ((llm_handoff_dict.get("metadata") or {}) or {}).get("codegen_rebuttal")
        if _rebuttal_text:
            updated_state["codegen_rebuttal"] = _rebuttal_text

        if previous_code is not None:
            base_code = previous_code
            if final_code != previous_code:
                patch_text = compute_unified_diff(previous_code, final_code)
                try:
                    patch_hunks = [h.to_dict() for h in parse_hunks(patch_text)]
                except Exception as exc:
                    self.logger.warning(f"Failed to parse patch hunks in tool mode: {exc}")
                    patch_hunks = []
                patch_apply_status = "applied"
                applied_code = final_code
            else:
                patch_apply_status = "not_attempted"
                if compilation_feedback is not None:
                    patch_apply_error = "tool_loop_completed_without_code_change"

        updated_state["tool_calls_count"] = tool_calls_count
        updated_state["edit_tool_calls_count"] = edit_tool_calls_count
        # Rebuttal takes priority over all other post-loop classifications.
        # If the LLM submitted a rebuttal, the code is intentionally unchanged
        # and we proceed as a successful codegen round; the review node will
        # re-run with the rebuttal as additional context.
        if updated_state.get("codegen_rebuttal"):
            self.logger.info(
                "[CodeGen] Rebuttal submitted -- skipping no-edit classification."
            )
            updated_state["latest_patch"] = None
            updated_state["latest_patch_status"] = "rebuttal"
            updated_state["latest_patch_error"] = None
            updated_state["latest_applied_code"] = None
            updated_state["fix_state"] = FIX_STATE_SUCCESS
            updated_state["fix_attempt"] = 1
            updated_state["fix_reason"] = "codegen_rebuttal_submitted"
            self._iteration += 1
            return GeneratedCode(
                code=final_code,
                retry_count=0,
                validation_status="not_validated",
                base_code=base_code,
                patch=None,
                hunks=[],
                applied_code=None,
                patch_apply_status="rebuttal",
                patch_apply_error=None,
            ), updated_state
        if previous_code is not None:
            no_edit_fix_loop = (
                compilation_feedback is not None
                and final_code == previous_code
                and edit_tool_calls_count == 0
            )
            if no_edit_fix_loop:
                reason_parts = ["no_tools_called"]
                patch_apply_error = (
                    f"tool_loop_completed_without_edit_tool_calls; {'; '.join(reason_parts)}"
                )
                patch_apply_status = "not_attempted"
            elif compilation_feedback is not None and final_code == previous_code:
                patch_apply_status = "failed"
                if not patch_apply_error:
                    patch_apply_error = "tool_loop_completed_without_code_change"
            updated_state["latest_patch"] = patch_text
            updated_state["latest_patch_status"] = patch_apply_status
            updated_state["latest_patch_error"] = patch_apply_error
            updated_state["latest_applied_code"] = applied_code
            if no_edit_fix_loop:
                updated_state["fix_state"] = FIX_STATE_TOOL_LOOP_NO_EDIT
                updated_state["structured_error"] = self._make_structured_codegen_error(summary=patch_apply_error or "tool_loop_completed_without_edit_tool_calls", error_type="tool_loop_no_edit", retryable=True)
                updated_state["fix_attempt"] = 1
                updated_state["fix_reason"] = patch_apply_error or "tool_loop_no_edit"
                _reason = patch_apply_error or "tool_loop_no_edit"
                updated_state["attempted_fixes"] = list(attempted_fixes) + [f"Attempt {len(attempted_fixes) + 1}: {_reason}"]
            elif patch_apply_status == "failed":
                updated_state["fix_state"] = FIX_STATE_PATCH_APPLY_FAILED
                updated_state["structured_error"] = self._make_structured_codegen_error(summary=patch_apply_error or "tool_loop_completed_without_code_change", error_type="patch_apply_failed", retryable=True)
                updated_state["fix_attempt"] = 1
                updated_state["fix_reason"] = patch_apply_error or "tool_loop_failed"
                _reason = patch_apply_error or "tool_loop_failed"
                updated_state["attempted_fixes"] = list(attempted_fixes) + [f"Attempt {len(attempted_fixes) + 1}: {_reason}"]
            else:
                updated_state["fix_state"] = FIX_STATE_SUCCESS
                updated_state["fix_attempt"] = 1
                updated_state["fix_reason"] = "tool_loop_completed"

        # Generation mode (no previous_code): if no edit tool was called, flag as error
        if previous_code is None and edit_tool_calls_count == 0:
            self.logger.error("[CodeGen] Generation mode completed without any edit tool call")
            updated_state["fix_state"] = FIX_STATE_TOOL_LOOP_NO_EDIT
            updated_state["structured_error"] = self._make_structured_codegen_error(
                summary="tool_loop_completed_without_edit_tool_calls; no_tools_called",
                error_type="tool_loop_no_edit",
                retryable=True,
            )
            updated_state["fix_attempt"] = 1
            updated_state["fix_reason"] = "tool_loop_no_edit"

        self._iteration += 1

        return GeneratedCode(
            code=final_code,
            retry_count=0,
            validation_status="not_validated",
            base_code=base_code,
            patch=patch_text,
            hunks=patch_hunks,
            applied_code=applied_code,
            patch_apply_status=patch_apply_status,
            patch_apply_error=patch_apply_error,
        ), updated_state

    async def execute(
        self,
        plan: SimulationPlan,
        compilation_feedback: Optional[Any] = None,
        previous_code: Optional[str] = None,
        messages: Optional[list] = None,
        state: Optional[dict] = None,
        fix_mode: bool = False,
    ) -> Tuple[GeneratedCode, Dict[str, Any]]:
        """
        Generate or fix PyChrono simulation code.

        Args:
            plan: Simulation plan
            compilation_feedback: Feedback from ReviewAgent or direct from compilation
            previous_code: Previous code if fixing errors
            messages: Conversation history for tracking previous errors/fixes
            state: Workflow state dict for error tracking and strategy escalation
            fix_mode: If True, simulation failed and we allow edit_file for fixes.
                     If False (default), we are in generation mode with one-shot write_file.

        Returns:
            Tuple of (GeneratedCode with validation status, updated state dict)
        """
        step_count = len(getattr(plan, "implementation_steps", []) or getattr(plan, "steps", []) or [])
        self.logger.info(f"Generating code for plan with {step_count} steps")
        self._emit_progress("build_codegen", "execute_start", progress_pct=5, step_count=step_count)
        self._emit_progress("build_codegen", "tool_mode_start", progress_pct=10)
        # Pipeline-stats lifecycle wrap: codegen does NOT go through
        # ``BaseAgent.invoke_llm`` / ``run_tool_loop``, so the engine's
        # ``_PipelineStatsCollector`` never sees a finished event for this
        # agent unless we emit one here. Per-turn ``_log_llm_usage`` calls
        # in ``_call_llm_with_tools`` populate ``_cumulative_usage``; we
        # diff before/after to attribute that delta to this execute() call.
        import time as _time
        _session_start = _time.time()
        _usage_before = dict(self._cumulative_usage)
        _calls_before = self._cumulative_calls
        emit_agent_lifecycle_event(
            agent=self.agent_name,
            state="started",
            model=self.model or "",
            provider=self.provider or "",
            session_kind="tool_loop",
        )
        max_internal_retries = get_settings().max_compilation_retries
        current_feedback = compilation_feedback
        current_previous = previous_code
        current_state = state
        generated_code = None
        updated_state = {}

        try:
            for internal_attempt in range(max_internal_retries):
                generated_code, updated_state = await self._execute_with_tools(
                    plan=plan,
                    compilation_feedback=current_feedback,
                    previous_code=current_previous,
                    state=current_state,
                    fix_mode=fix_mode,
                )
                if updated_state.get("fix_state") == FIX_STATE_PATCH_APPLY_FAILED:
                    patch_err = updated_state.get("latest_patch_error") or updated_state.get("fix_reason") or "patch apply failed"
                    if internal_attempt + 1 < max_internal_retries:
                        self.logger.info(
                            f"[InternalRetry] PATCH_APPLY_FAILED (attempt {internal_attempt + 1}/{max_internal_retries}), re-locating and generating new diff"
                        )
                        current_feedback = (
                            "PATCH APPLY FAILED - your unified diff could not be applied.\n"
                            f"Error: {patch_err}\n\n"
                            "Re-read the file with read_file to get the exact current content, "
                            "re-locate the target lines, then generate a NEW unified diff that correctly applies."
                        )
                        current_previous = generated_code.base_code or generated_code.code
                        current_state = updated_state
                        continue
                    self._emit_progress("build_codegen", "tool_mode_done", progress_pct=100)
                    return generated_code, updated_state

                utils_feedback = self._check_utils_calls(generated_code)
                if utils_feedback and internal_attempt + 1 < max_internal_retries:
                    self.logger.info(
                        f"[InternalRetry] UTILS_CALL_MISMATCH (attempt {internal_attempt + 1}/{max_internal_retries}), "
                        f"feeding {utils_feedback.count('[UtilsCallError]')} issue(s) back to codegen"
                    )
                    current_feedback = utils_feedback
                    current_previous = generated_code.code
                    current_state = updated_state
                    continue

                self._emit_progress("build_codegen", "tool_mode_done", progress_pct=100)
                return generated_code, updated_state

            self._emit_progress("build_codegen", "tool_mode_done", progress_pct=100)
            return generated_code, updated_state
        finally:
            _elapsed = _time.time() - _session_start
            _session_usage = _diff_usage(self._cumulative_usage, _usage_before)
            _session_calls = self._cumulative_calls - _calls_before
            emit_agent_lifecycle_event(
                agent=self.agent_name,
                state="finished",
                model=self.model or "",
                provider=self.provider or "",
                elapsed=_elapsed,
                usage=_session_usage,
                calls=_session_calls,
                session_kind="tool_loop",
            )
            self._persist_session_stats_to_dialog(
                session_kind="tool_loop",
                elapsed=_elapsed,
                usage=_session_usage,
                calls=_session_calls,
                turns=0,
            )

    def _format_compilation_feedback(self, feedback) -> str:
        """Format feedback for prompts as raw factual text only."""
        if feedback is None:
            return ""
        if isinstance(feedback, str):
            out = feedback
        elif isinstance(feedback, dict):
            parts = []
            for key in ("feedback_text", "error_text", "message"):
                val = feedback.get(key)
                if val:
                    parts.append(str(val))
            for issue in feedback.get("issues", []) or []:
                if isinstance(issue, dict):
                    desc = issue.get("description") or issue.get("message") or ""
                    if desc:
                        parts.append(str(desc))
                elif issue:
                    parts.append(str(issue))
            backtrace = feedback.get("backtrace") or feedback.get("traceback")
            if backtrace:
                parts.append(f"\nTraceback:\n{backtrace}")
            if parts:
                out = "\n".join(parts)
            else:
                try:
                    out = json.dumps(feedback, ensure_ascii=True, indent=2)
                except TypeError:
                    out = str(feedback)
        else:
            out = str(feedback or "")

        settings = getattr(self, "settings", None) or get_settings()
        if bool(getattr(settings, "tool_output_truncate_enabled", True)):
            head = int(getattr(settings, "tool_output_head_chars", 1500))
            tail = int(getattr(settings, "tool_output_tail_chars", 500))
            out = elide_middle(out, head_chars=head * 4, tail_chars=tail * 2)
        return out

    def _feedback_to_text(self, feedback: Any) -> str:
        """Normalize feedback into compact text for routing and extraction."""
        return self._format_compilation_feedback(feedback)


    OFFICE_REFERENCE_HEIGHTS: Dict[str, float] = {
        "computer_table": 0.75,
        "office_chair": 1.05,
        "paper_coffee_cup": 0.10,
        "macbook_pro_m3_16_inch_2024": 0.22,
        "ultrawide_monitor": 0.34,
        "airpods_max": 0.20,
    }

    # Sole remaining hard-coded routing decision: pure-scene plans bypass the
    # core/<plan_type> machinery and pre-inject only the dedicated scene skill.
    # All other plan types use core/<plan_type>.get_required_skills() (declared
    # in markdown) for pre-injection, plus the model-routed skill directory.
    DEDICATED_SCENE_SKILL = "scene/custom_assets_scene_convex_decomp"

    def _is_mbs_in_scene_plan(self, plan: SimulationPlan) -> bool:
        """Return True when the plan is an mbs_in_scene hybrid."""
        plan_type = getattr(plan, "plan_type", None) or ""
        return plan_type == "mbs_in_scene"

    def _is_fsi_in_scene_plan(self, plan: SimulationPlan) -> bool:
        """Return True when the plan is an FSI-coupled hybrid scene
        (plan_type=fsi_in_scene). Mirrors ``_is_mbs_in_scene_plan`` but
        triggers a different core skill (``core/fsi_in_scene``) and a
        different required-skills bundle (fsi/sph + veh/wheeled_vehicle
        FSI Coupling instead of veh/terrain + scene/custom_assets...).
        """
        plan_type = getattr(plan, "plan_type", None) or ""
        return plan_type == "fsi_in_scene"

    def _is_scene_plan(self, plan: SimulationPlan) -> bool:
        """Return True when the plan involves scene asset placement.

        Note: ``mbs_in_scene`` and ``fsi_in_scene`` plans are **not**
        treated as pure scene plans so that mbs / fsi / robot skills are
        still selected alongside the dedicated scene skill.
        """
        if self._is_mbs_in_scene_plan(plan) or self._is_fsi_in_scene_plan(plan):
            return False
        plan_type = getattr(plan, "plan_type", None) or ""
        if plan_type == "scene":
            return True
        assets = getattr(plan, "assets", None) or []
        if assets:
            return True
        return False

    def _has_known_office_assets(self, plan: SimulationPlan) -> bool:
        """Return True if the plan contains any assets from the known office set."""
        assets = getattr(plan, "assets", None) or []
        known = set(self.OFFICE_REFERENCE_HEIGHTS.keys())
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = (asset.get("name") or "").lower()
            filename = (asset.get("filename") or "").lower()
            if any(k in name or k in filename for k in known):
                return True
        return False

    def _build_office_height_override(self, plan: SimulationPlan, feedback: Optional[Any]) -> str:
        """Return a hard prompt override for known office asset heights."""
        if not self._has_known_office_assets(plan):
            return ""

        lines = [
            "OFFICE ASSET HEIGHT OVERRIDE (HIGHEST PRIORITY):",
            "For this dedicated office custom-assets scene, use the following authoritative target heights exactly.",
            "Do NOT infer, estimate, or replace these values from plan ideal_height when the asset name matches.",
            "The generated simulation.py MUST define a runtime TARGET_HEIGHTS dictionary with these exact values.",
        ]
        for name, height in self.OFFICE_REFERENCE_HEIGHTS.items():
            lines.append(f"- {name}: {height}")
        lines.extend([
            "For known assets above, always compute scale with scale_factor = TARGET_HEIGHTS[name] / raw_size[height_axis].",
            "If plan.json disagrees for one of these known assets, IGNORE the plan value and use this override.",
            "Only use plan ideal_height for assets not listed above.",
        ])
        return "\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _compute_plan_hash(plan: SimulationPlan) -> str:
        """Stable hash of plan content. Used to gate the LLM skill router and
        the skill-bundle build so identical plans across retries produce a
        byte-identical system prompt (Anthropic prompt-cache prerequisite)."""
        try:
            payload = plan.dump_all() if hasattr(plan, "dump_all") else plan.model_dump()
        except Exception:
            try:
                payload = plan.model_dump()
            except Exception:
                payload = {"_repr": repr(plan)}
        blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def _route_skills_via_llm(
        self,
        plan: SimulationPlan,
    ) -> Optional[Any]:
        """Call SkillRouterAgent (MiniMax) to pick skills + sections.

        Returns a ``RouterDecision`` (skills + per-skill section keys), or
        ``None`` when routing is disabled / fails. Caller stores result on
        ``self._llm_routed_skills`` (the .skills list) and
        ``self._llm_routed_sections`` (the .sections dict) for
        ``_resolve_skill_selection`` and the injection block to consume.
        """
        try:
            settings = get_settings()
            if not bool(getattr(settings, "skill_router_enabled", True)):
                return None
            # Scene plans short-circuit to the dedicated scene skill — no
            # routing call needed.
            if self._is_scene_plan(plan):
                return None
        except Exception:
            return None

        try:
            from chrono_code.agents.skill_router_agent import (
                SkillRouterAgent,
                build_plan_summary,
            )
        except Exception as exc:
            self.logger.warning("[SkillRouter] import failed: %s", exc)
            return None

        all_names = SkillRegistry.get_all_skill_names()
        if not all_names:
            return None
        # Section-aware directory: lists every skill's section keys so the
        # router can pick per-skill section subsets.
        directory = SkillRegistry.format_skill_directory(
            exclude=set(), with_sections=True,
        )
        plan_summary = build_plan_summary(plan)

        # Build per-skill section index for validating router output.
        valid_sections_per_skill: dict[str, set[str]] = {}
        for n in all_names:
            secs = SkillRegistry.list_sections(n) or []
            valid_sections_per_skill[n] = set(secs)

        try:
            router = SkillRouterAgent()
        except Exception as exc:
            self.logger.warning("[SkillRouter] init failed: %s", exc)
            return None

        try:
            decision = await router.route(
                plan_summary=plan_summary,
                skill_directory=directory,
                valid_skill_names=set(all_names),
                valid_sections_per_skill=valid_sections_per_skill,
            )
        except Exception as exc:
            self.logger.warning("[SkillRouter] route() raised: %s", exc)
            return None

        if not decision or not decision.skills:
            return None

        # Always ensure the core skill is present, even if the router missed
        # it. Pre-injection of the core skill is a hard contract for the
        # downstream skill_constraints / gate logic.
        plan_type = getattr(plan, "plan_type", None) or "mbs"
        core_name = f"core/{plan_type}"
        if core_name in all_names and core_name not in decision.skills:
            decision.skills = [core_name] + decision.skills
        return decision

    def _resolve_skill_selection(
        self,
        plan: SimulationPlan,
        feedback: Optional[Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Build the skill bundle.

        Selection of the pre-injected (``primary``) set, in priority order:
          1. ``self._llm_routed_skills`` if populated by ``_route_skills_via_llm``
             — Haiku reads plan + skill descriptions and picks per-plan.
          2. Legacy fallback: parse ``core/<plan_type>``'s "Required Skills"
             markdown table. Used when the router is disabled or fails.

        ``ordered`` (returned for the directory listing) always contains every
        registered skill so the model can on-demand ``read_skill(name)`` for
        anything outside the routed set.

        Scene plans (``plan_type='scene'``) short-circuit to the dedicated
        scene skill, matching the historical carve-out.

        ``feedback`` is no longer consulted here; the model sees the same
        feedback in its prompt and routes accordingly.
        """
        if self._is_scene_plan(plan):
            ordered = [self.DEDICATED_SCENE_SKILL]
            self.logger.info(
                f"Selected dedicated scene skill for scene plan: {ordered[0]}"
            )
            return ordered, ordered, []

        all_names = SkillRegistry.get_all_skill_names()
        plan_type = getattr(plan, "plan_type", None) or "mbs"
        core_name = f"core/{plan_type}"

        routed: Optional[List[str]] = getattr(self, "_llm_routed_skills", None)
        if routed:
            valid = set(all_names)
            primary = [n for n in routed if n in valid]
            self.logger.info(
                f"Skill bundle (llm-router): pre-injected={primary} "
                f"directory_size={len(all_names) - len(primary)}"
            )
            return all_names, primary, []

        # Fallback: legacy static-table path.
        core_skill = SkillRegistry._skills.get(core_name)
        primary: List[str] = []
        if core_skill is not None:
            declared = list(core_skill.get_required_skills())
            primary = [core_name] + [d for d in declared if d != core_name]
            primary = [n for n in primary if n in set(all_names)]
        self.logger.info(
            f"Skill bundle (static-table fallback): pre-injected={primary} "
            f"directory_size={len(all_names) - len(primary)}"
        )
        return all_names, primary, []

    def _select_skills_for_plan(
        self,
        plan: SimulationPlan,
        feedback: Optional[Any],
    ) -> List[str]:
        ordered, _, _ = self._resolve_skill_selection(plan, feedback)
        return ordered

    def _build_skill_bundle(self, plan: SimulationPlan, feedback: Optional[Any]) -> SkillBundle:
        ordered, primary, secondary = self._resolve_skill_selection(plan, feedback)
        bundle = SkillRegistry.build_bundle(
            ordered,
            bundle_name="selected",
            primary_skills=primary,
        )
        bundle.metadata.setdefault("secondary_skills", secondary)
        return bundle

    def _build_plan_summary(self, plan: SimulationPlan) -> Dict[str, Any]:
        simulation_parameters = dict(getattr(plan, "simulation_parameters", {}) or {})
        visualization = dict(getattr(plan, "visualization", {}) or {})
        camera = getattr(plan, "camera", None)
        vis_mode = visualization.get("mode", "vsg_with_sensor_camera")
        camera_layout = getattr(camera, "layout", None) if camera else None
        return {
            "time_step": simulation_parameters.get("time_step") or simulation_parameters.get("dt"),
            "visualization_mode": visualization.get("mode"),
            "camera_enabled": bool(getattr(camera, "enabled", False)) if camera is not None else False,
            "camera_layout": camera_layout,
            "camera_update_rule": "update_rate = 1 / dt",
            "must_preserve_vsg": vis_mode in ("vsg", "vsg_with_sensor_camera"),
            "must_preserve_sensor_camera": vis_mode in ("sensor_camera", "vsg_with_sensor_camera"),
        }

    def _normalize_query_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        token_parts = re.split(r"[\|\s,()]+", normalized)
        tokens = sorted(part for part in token_parts if part)
        return "|".join(tokens) if tokens else normalized

    def _tool_query_payload(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "grep_code":
            return str(tool_args.get("pattern") or "")
        if tool_name == "search_skills":
            return str(tool_args.get("query") or "")
        if tool_name == "read_skill":
            return str(tool_args.get("skill_name") or "")
        return json.dumps(tool_args, ensure_ascii=True, sort_keys=True)

    def _build_query_fingerprint(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        return f"{tool_name}:{self._normalize_query_text(self._tool_query_payload(tool_name, tool_args))}"

    def _build_query_family(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        failure_context: Optional[FailureContext],
    ) -> str:
        payload = self._normalize_query_text(self._tool_query_payload(tool_name, tool_args))
        failing_symbol = ""
        object_type = ""
        if failure_context is not None:
            failing_symbol = str((failure_context.metadata or {}).get("failing_symbol") or "").lower()
            object_type = str((failure_context.metadata or {}).get("object_type") or "").lower()
        if failing_symbol and failing_symbol in payload:
            return f"{tool_name}:symbol:{failing_symbol}"
        if object_type and object_type in payload:
            return f"{tool_name}:object:{object_type}"
        tokens = [part for part in payload.split("|") if part]
        return f"{tool_name}:generic:{'|'.join(tokens[:3])}"

    def _result_has_new_evidence(self, result: Any) -> bool:
        text = str(result or "").strip().lower()
        if not text:
            return False
        negative_markers = (
            "no matches found",
            "no matching skills found",
            "unknown skill",
            "section '",
            "not found in skill",
            "no api contract found",
            "unsupported path",
            "no code available",
        )
        return not any(marker in text for marker in negative_markers)

    def _build_current_structured_error(self, feedback: Optional[Any], feedback_source: str) -> Optional[StructuredError]:
        if feedback is None:
            return None
        details = extract_error_details(feedback)
        exception_type = str(details.get("exception_type") or "UnknownError")
        classified_error = classify_error(feedback)
        error_type = classified_error or re.sub(r"(?<!^)(?=[A-Z])", "_", exception_type).lower()
        signature = None
        if isinstance(feedback, dict):
            try:
                signature = ErrorSignatureExtractor.extract(feedback, self._iteration).get_hash()
            except Exception:
                signature = None
        metadata = {
            "exception_type": exception_type,
            "failing_symbol": details.get("missing_attr") or details.get("missing_name") or "",
            "object_type": details.get("object_type") or "",
            "user_frame": details.get("user_frame") or "",
            "core_error": details.get("core_error") or "",
            "feedback_source": feedback_source or "codegen",
            "classified_error": classified_error,
        }
        return StructuredError.from_message(
            error_type=error_type or "unknown_error",
            phase="build_codegen",
            summary=str(details.get("core_error") or str(feedback)[:240]),
            raw_message=self._feedback_to_text(feedback)[:4000],
            context_snippet=(details.get("user_frame") or ""),
            signature=signature,
            metadata=metadata,
        )

    def _build_failure_context(
        self,
        *,
        feedback: Optional[Any],
        feedback_source: str,
        previous_code: str,
    ) -> Optional[FailureContext]:
        if feedback is None:
            return None
        feedback_text = self._feedback_to_text(feedback) if feedback is not None else ""
        structured_error = self._build_current_structured_error(feedback, feedback_source)
        details = extract_error_details(feedback) if feedback is not None else {}
        user_frame = details.get("user_frame") or ""
        classified_error = ""
        if structured_error is not None:
            classified_error = str((structured_error.metadata or {}).get("classified_error") or structured_error.error_type or "")
        metadata = {
            "exception_type": details.get("exception_type") or "UnknownError",
            "failing_symbol": details.get("missing_attr") or details.get("missing_name") or "",
            "failing_function": user_frame.split(" in ", 1)[1] if " in " in user_frame else "",
            "user_frame": user_frame,
            "object_type": details.get("object_type") or "",
            "classified_error": classified_error,
            "diagnostic_budget": {
                "max_total_diagnostic_queries": 3,
                "max_same_family_queries": 1,
                "max_same_hypothesis_checks": 1,
            },
        }
        return FailureContext(
            source=feedback_source or "codegen",
            summary=(structured_error.summary if structured_error else (feedback_text[:500] or "retrying after previous failure")),
            structured_error=structured_error,
            recent_feedback=feedback_text[:4000],
            metadata=metadata,
        )

    def _build_handoff_packet(
        self,
        *,
        plan: SimulationPlan,
        bundle: SkillBundle,
        effective_mode: str,
        feedback: Optional[Any],
        feedback_source: str,
        previous_code: str,
        prior_handoff: Optional[dict],
        error_history: Optional[dict] = None,
        is_step_continuation: bool = False,
    ) -> LLMHandoff:
        prior = prior_handoff or {}
        failure_ctx = self._build_failure_context(
            feedback=feedback,
            feedback_source=feedback_source,
            previous_code=previous_code,
        )

        # Build history_context from error history
        history_ctx = None
        if error_history:
            prior_errors = []
            fingerprints = error_history.get("error_fingerprints", {})
            exec_retry_count = error_history.get("execution_retry_count", 0)
            fingerprint_groups: Dict[str, List[str]] = {}

            for fp, msg in fingerprints.items():
                raw_msg = re.sub(r"^\[RELATED to prior error\]\s*", "", str(msg or "")).strip()
                fp_group = fp.rsplit("|", 1)[0] if "|" in fp else fp
                prior_errors.append(PriorError(
                    fingerprint=fp,
                    summary=raw_msg[:200] if raw_msg else "",
                    phase="execution",
                    timestamp=datetime.now().isoformat(),
                    fingerprint_group=fp_group,
                    raw_message=raw_msg,
                ))
                fingerprint_groups.setdefault(fp_group, []).append(raw_msg[:200] if raw_msg else fp)

            repeated_error_summaries = [
                f"{group} repeated {len(items)} time(s)"
                for group, items in sorted(fingerprint_groups.items())
                if len(items) > 1
            ]
            do_not_repeat = [
                "Do not spend time re-fixing historical errors unless they appear in failure_context.",
            ]
            if repeated_error_summaries:
                do_not_repeat.append("If the same fingerprint group repeats, change strategy instead of re-reading the same stale region.")

            history_ctx = HistoryContext(
                summary=(f"{exec_retry_count} prior execution attempts" if exec_retry_count > 0 else "No prior attempts"),
                cycle_detected=exec_retry_count >= get_settings().max_retries,
                recent_attempts_available=bool(prior_errors),
                repeated_error_summaries=repeated_error_summaries,
                do_not_repeat=do_not_repeat,
                prior_errors=prior_errors,
                attempted_fixes=[],
            )

        return LLMHandoff(
            task_intent="fix_simulation_code" if effective_mode == "fix" else "generate_simulation_code",
            input_artifacts={
                "plan_available": True,
                "previous_code_available": bool(previous_code),
                "feedback_source": feedback_source or "none",
                "mode": effective_mode,
            },
            plan_summary=self._build_plan_summary(plan),
            decisions={
                **dict(prior.get("decisions") or {}),
                "selected_skills": bundle.skills,
                "video_generation_required": True,
                "preserve_vsg": plan.plan_type in ("scene", "mbs_in_scene", "fsi_in_scene"),
                "require_sensor_camera": plan.plan_type in ("scene", "mbs_in_scene", "fsi_in_scene"),
            },
            constraints=[
                "In generate mode, read plan.json as the primary specification; handoff.json is secondary metadata.",
                "In fix mode, read handoff.json first for failure_context and history_context.",
                "failure_context contains only the current error.",
                "In fix mode, default to handoff metadata -> grep_code -> local read_file window -> edit_file.",
                "Skill tools are minimal: use search_skills(...) to find relevant skills and read_skill(...) to inspect one.",
                "Use VSG + sensor cameras for all plans. Do NOT use Irrlicht.",
                "Use history_context only to avoid repeating failed fixes; do not treat it as the current traceback.",
                "For one unresolved current question, do not exceed the diagnostic query budget in failure_context.metadata.",
                "Do not repeat the same semantic grep/search/skill query for the same unresolved question.",
                "If the root cause is clear but binding evidence is still missing after budget is exhausted, stop searching and either apply a safe validated fallback or surface a structured failure.",
            ],
            failure_context=failure_ctx,
            history_context=history_ctx,
            next_expected_action="edit_file" if (effective_mode == "fix" or is_step_continuation) else "write_file",
            skill_bundle=bundle,
            metadata={
                "prior_handoff_present": bool(prior_handoff),
                "plan_type": getattr(plan, "plan_type", None),
                "diagnostic_memory": {
                    "current_question": (
                        failure_ctx.summary[:240]
                        if failure_ctx is not None else ""
                    ),
                    "query_fingerprints": dict((((prior.get("metadata") or {}).get("diagnostic_memory") or {}).get("query_fingerprints") or {})),
                },
            },
        )

    def _make_structured_codegen_error(
        self,
        *,
        summary: str,
        error_type: str,
        retryable: bool = True,
        operation: Optional[str] = None,
        recommended_action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return StructuredError.from_message(
            error_type=error_type,
            phase="build_codegen",
            summary=summary,
            raw_message=summary,
            operation=operation,
            retryable=retryable,
            recommended_action=recommended_action,
            metadata=metadata or {},
        ).model_dump()

    def _clean_code(self, code: str) -> str:
        """Remove markdown formatting from code."""
        # Remove markdown code blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 3:
                code = parts[1]

        return code.strip()
