"""
Base agent class for all Chrono-Agent agents.

Uses Anthropic and OpenAI SDKs directly instead of LangChain.
"""

import json
import asyncio
import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod

import anthropic
import openai

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore[assignment]

from chrono_agent.agents.exceptions import AgentLLMError
from chrono_agent.config import get_settings
from chrono_agent.utils.logger import AgentLogger
from chrono_agent.utils.dialog_manager import DialogManager
from chrono_agent.workflow.events import (
    emit_agent_lifecycle_event,
    emit_agent_thinking_event,
    emit_llm_stream_end_event,
    emit_llm_stream_start_event,
    emit_llm_text_delta_event,
    emit_tool_call_event,
)


_USAGE_KEYS: Tuple[str, ...] = ("input", "output", "cache_read", "cache_creation")


def _zero_usage() -> Dict[str, int]:
    return {k: 0 for k in _USAGE_KEYS}


def _diff_usage(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in _USAGE_KEYS}

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    """

    # Shared dialog manager across all agents in a session
    _shared_dialog_manager: Optional[DialogManager] = None

    def __init__(
        self,
        agent_name: str,
        agent_number: int,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize base agent.

        Args:
            agent_name: Name of the agent
            agent_number: Agent number (1, 2, 3, or 4)
            llm_provider: LLM provider ("anthropic", "openai", etc.), overrides config
            model: Specific model to use, overrides config
            temperature: Temperature for generation, overrides config
        """
        self.agent_name = agent_name
        self.agent_number = agent_number
        self.settings = get_settings()
        self.logger = AgentLogger(agent_name)

        # Get agent-specific config from settings (.env file)
        agent_config = self.settings.get_agent_config(agent_number)

        # Override with provided parameters (CLI args take precedence)
        self.provider = llm_provider or agent_config.get("provider")
        self.model = model or agent_config.get("model")
        self.temperature = temperature if temperature is not None else agent_config.get("temperature", 0.5)
        self.max_tokens = agent_config.get("max_tokens", 8192)
        self._api_base = agent_config.get("api_base")
        self._api_key = agent_config.get("api_key")

        # Log configuration
        self.logger.info(
            f"Agent {agent_number} ({agent_name}): {self.provider}/{self.model} (T={self.temperature})"
        )
        self.logger.debug(
            f"Agent {agent_number} config: provider={self.provider}, model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, api_base={self._api_base}"
        )

        # Initialize SDK client
        self.client = self._init_client(agent_config)

        # Feedback storage for inter-agent communication
        self.feedback_history = []

        # Access shared dialog manager or create if first agent
        if BaseAgent._shared_dialog_manager is None:
            BaseAgent._shared_dialog_manager = DialogManager(base_dir=str(get_settings().dialog_output_path()))
        self.dialog_manager = BaseAgent._shared_dialog_manager

        # Per-agent cumulative LLM-cost counters. Updated inside
        # ``_log_llm_usage`` every time a provider returns a response with
        # usage metadata, regardless of whether the call originated in
        # ``invoke_llm`` or one of the ``_tool_loop_*`` paths. Snapshotting
        # before/after a session yields the per-session diff that ends up
        # on the ``agent_lifecycle`` finished event.
        self._cumulative_usage: Dict[str, int] = _zero_usage()
        self._cumulative_calls: int = 0

    def _init_client(self, agent_config: Dict[str, Any]) -> Any:
        """
        Initialize the SDK client based on provider.

        Args:
            agent_config: Agent configuration dict from settings

        Returns:
            SDK client instance
        """
        provider = self.provider
        llm_config = self.settings.get_llm_config(provider)

        if provider == "anthropic":
            api_key = self._api_key or llm_config["api_key"]
            kwargs: Dict[str, Any] = {"api_key": api_key}
            final_base = self._api_base or llm_config.get("api_base")
            if final_base:
                kwargs["base_url"] = final_base
            self.logger.debug(
                f"Initializing Anthropic client: {self.model} (max_tokens={self.max_tokens}, base_url={final_base!r})"
            )
            return anthropic.AsyncAnthropic(**kwargs)

        elif provider in ("openai", "deepseek"):
            raw_key = self._api_key or llm_config.get("api_key")
            api_key = (raw_key or "").strip() if isinstance(raw_key, str) else str(raw_key or "").strip()
            if not api_key:
                raise ValueError(
                    "OpenAI-compatible provider selected but api_key is empty after strip. "
                    "Check OPENAI_API_KEY / OPENROUTER_API_KEY / DEEPSEEK_API_KEY in .env."
                )
            final_base = self._api_base or llm_config.get("api_base")
            self.logger.debug(
                f"Initializing OpenAI-compatible client: {self.model} (T={self.temperature}, base_url={final_base!r})"
            )
            oai_kwargs: Dict[str, Any] = {"api_key": api_key}
            if final_base:
                oai_kwargs["base_url"] = final_base
            # OpenRouter recommends optional headers
            if final_base and "openrouter.ai" in str(final_base).lower():
                oai_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/projectchrono/chrono-agent",
                    "X-Title": "Chrono-Agent",
                }
            return openai.AsyncOpenAI(**oai_kwargs)

        elif provider == "google":
            if genai is None:
                raise ValueError(
                    "Google provider requires google-generativeai. "
                    "Run `uv sync` to install project dependencies."
                )
            api_key = llm_config["api_key"]
            genai.configure(api_key=api_key)
            self.logger.debug(
                f"Initializing Google Gemini: {self.model} (T={self.temperature}, max_tokens={self.max_tokens})"
            )
            # Return the genai module itself; invoke_llm will use genai.GenerativeModel
            return genai

        elif provider == "ollama":
            ollama_model = agent_config.get("ollama_model") or agent_config.get("model")
            ollama_base_url = agent_config.get("ollama_base_url", "http://localhost:11434")
            if not ollama_model:
                raise ValueError("Ollama model not specified.")
            self.model = ollama_model
            self.logger.debug(
                f"Initializing Ollama (OpenAI-compat) client: {self.model} at {ollama_base_url}"
            )
            return openai.AsyncOpenAI(
                api_key="ollama",
                base_url=f"{ollama_base_url.rstrip('/')}/v1",
            )

        elif provider == "minimax":
            minimax_key = self._api_key or llm_config.get("api_key")
            minimax_base = self._api_base or llm_config.get("api_base", "https://api.minimaxi.com/v1")
            self.logger.debug(
                f"Initializing MiniMax (OpenAI-compat) client: {self.model} at {minimax_base}"
            )
            return openai.AsyncOpenAI(
                api_key=minimax_key,
                base_url=minimax_base,
            )

        elif provider == "ngc":
            ngc_key = llm_config["api_key"]
            ngc_base = self._api_base or llm_config.get("api_base", "https://integrate.api.nvidia.com")
            # Ensure base_url ends with /v1 for OpenAI-compatible endpoints
            if not ngc_base.rstrip("/").endswith("/v1"):
                ngc_base = ngc_base.rstrip("/") + "/v1"
            self.logger.debug(
                f"Initializing NGC (OpenAI-compat) client: {self.model} at {ngc_base}"
            )
            return openai.AsyncOpenAI(
                api_key=ngc_key,
                base_url=ngc_base,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def start_new_session(cls, user_prompt: str):
        """
        Start a new dialog session for all agents.

        Args:
            user_prompt: Initial user prompt for the session
        """
        cls._shared_dialog_manager = DialogManager(base_dir=str(get_settings().dialog_output_path()))
        cls._shared_dialog_manager.create_session(user_prompt)
        logger.info(f"Started new dialog session for prompt: {user_prompt[:100]}...")

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image_as_base64(image_path: Union[str, Path]) -> tuple:
        """Load an image file and return (base64_data, media_type)."""
        path = Path(image_path)
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return data, mime

    def _build_anthropic_image_blocks(self, images: List[Union[str, Path]]) -> list:
        """Build Anthropic-format image content blocks."""
        blocks = []
        for img in images:
            b64_data, media_type = self._load_image_as_base64(img)
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            })
        return blocks

    def _build_openai_image_blocks(self, images: List[Union[str, Path]]) -> list:
        """Build OpenAI vision-format image content blocks (Chat Completions)."""
        blocks = []
        for img in images:
            b64_data, media_type = self._load_image_as_base64(img)
            blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_data}",
                },
            })
        return blocks

    def _build_openai_responses_image_blocks(self, images: List[Union[str, Path]]) -> list:
        """Build OpenAI Responses-API image content blocks.

        The Responses API uses ``input_image`` with ``image_url`` as a flat
        string (not a nested object like Chat Completions).
        """
        blocks = []
        for img in images:
            b64_data, media_type = self._load_image_as_base64(img)
            blocks.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{b64_data}",
            })
        return blocks

    def _is_openai_reasoning_model(self) -> bool:
        """True iff provider is OpenAI proper and the model is a reasoning
        family (gpt-5*, o1*, o3*, o4*) whose reasoning is only surfaced
        through the Responses API.

        We require the client's ``base_url`` to point at ``api.openai.com``
        because OpenAI-compat endpoints reached via the openai provider
        (OpenRouter, Azure-variant proxies) often do NOT implement the
        Responses API and would 404 on ``responses.create``. When a
        non-OpenAI base_url is detected, the call falls back to the
        Chat Completions thinking branch, which still extracts reasoning
        from providers that expose ``delta.reasoning_content``.
        """
        if (self.provider or "").lower() != "openai":
            return False
        base_url = str(getattr(self.client, "base_url", "") or "")
        if base_url and "api.openai.com" not in base_url:
            return False
        m = (self.model or "").lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    async def invoke_llm(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        parse_json: bool = False,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        skip_dialog_logging: bool = False,
        images: Optional[List[Union[str, Path]]] = None,
        enable_thinking: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: User prompt
            system_message: Optional system message
            parse_json: Whether to parse response as JSON
            max_retries: Maximum retry attempts for failed responses
            temperature: Optional temperature override for this specific call
            skip_dialog_logging: Skip logging to dialog manager
            images: Optional list of image paths for vision models
            enable_thinking: Request extended-thinking / reasoning content
                from the provider when supported. Routing rules:
                  * anthropic → streaming + ``thinking={"type":"enabled"}``
                  * openai (api.openai.com) + gpt-5/o-series → Responses API
                  * other openai-compat (DeepSeek-R1, Qwen-Reasoning) →
                    Chat Completions stream, captures ``reasoning_content``
                  * google → no-op (empty thinking returned)

        Returns:
            LLM response (string or parsed JSON)
        """
        effective_temp = temperature if temperature is not None else self.temperature
        last_error = None
        content = ""

        # Bracket the whole retry loop with one ``agent_lifecycle`` session
        # so the dialog UI gets a single "✓ done · tokens · time" line per
        # invoke_llm call (rather than one per retry). Use cumulative-usage
        # snapshots to compute the per-session diff: ``_log_llm_usage``
        # accumulates into ``self._cumulative_usage`` on every provider
        # response, so ``after - before`` is the cost of this session even
        # when retries push usage up.
        import time as _time
        _session_start = _time.time()
        _usage_before = dict(self._cumulative_usage)
        _calls_before = self._cumulative_calls
        emit_agent_lifecycle_event(
            agent=self.agent_name,
            state="started",
            model=self.model or "",
            provider=self.provider or "",
            session_kind="invoke_llm",
        )

        try:
            for attempt in range(max_retries + 1):
                try:
                    self.logger.debug(f"Invoking LLM with prompt length: {len(prompt)} (attempt {attempt + 1})")

                    # Log prompt to dialog (only on first attempt)
                    if attempt == 0 and not skip_dialog_logging and self.dialog_manager:
                        await asyncio.to_thread(
                            self.dialog_manager.log_prompt,
                            agent_name=self.agent_name,
                            prompt=prompt,
                            metadata={
                                "system_message": bool(system_message),
                                "temperature": effective_temp,
                                "parse_json": parse_json,
                                "has_images": bool(images),
                                "attempt": attempt + 1,
                            },
                        )

                    content, thinking_text = await self._call_provider(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=effective_temp,
                        images=images,
                        enable_thinking=enable_thinking,
                    )

                    self.logger.debug(f"Received response length: {len(content)}")

                    # Log response to dialog. When the provider produced extended-
                    # thinking / reasoning content (only present when enable_thinking
                    # was on AND the provider supports it — Anthropic thinking,
                    # DeepSeek-R1 / o1 reasoning_content), prepend a `# THINKING`
                    # block so the dialog artifact mirrors the tool-loop format.
                    if not skip_dialog_logging and self.dialog_manager and content:
                        if thinking_text:
                            body = (
                                "# THINKING\n" + thinking_text[:20000]
                                + "\n\n# TEXT\n" + content[:10000]
                            )
                        else:
                            body = content
                        await asyncio.to_thread(
                            self.dialog_manager.log_response,
                            agent_name=self.agent_name,
                            response=body,
                            metadata={
                                "response_length": len(content),
                                "thinking_chars": len(thinking_text or ""),
                                "attempt": attempt + 1,
                                "parse_json": parse_json,
                            },
                        )

                    # Check for empty response
                    if not content or not content.strip():
                        raise AgentLLMError(
                            agent_name=self.agent_name,
                            operation="invoke_llm",
                            message="LLM returned empty response",
                        )

                    if parse_json:
                        json_content = self._extract_json(content)
                        if not json_content or not json_content.strip():
                            self.logger.debug(f"Raw response: {content[:500]}...")
                            raise AgentLLMError(
                                agent_name=self.agent_name,
                                operation="invoke_llm",
                                message="No JSON found in LLM response",
                            )
                        try:
                            return json.loads(json_content)
                        except json.JSONDecodeError as exc:
                            self.logger.debug(f"Initial JSON parse failed, attempting fix: {exc}")
                            fixed_content = self._fix_json(json_content)
                            try:
                                return json.loads(fixed_content)
                            except json.JSONDecodeError as exc2:
                                self.logger.warning(
                                    f"JSON parse failed after fix. First error: {exc}; after _fix_json: {exc2}. "
                                    f"Preview: {content[:800] if content else '(empty)'}..."
                                )
                                raise AgentLLMError(
                                    agent_name=self.agent_name,
                                    operation="invoke_llm",
                                    message=(
                                        f"Failed to parse JSON response: {exc2} "
                                        f"(raw parse error: {exc})"
                                    ),
                                    original_exception=exc2,
                                ) from exc2

                    return content

                except AgentLLMError as e:
                    last_error = e
                    self.logger.error(f"Error invoking LLM (attempt {attempt + 1}): {e}")
                    if attempt < max_retries:
                        continue
                    raise

            # If we get here, all retries failed
            if isinstance(last_error, AgentLLMError):
                raise last_error
            raise AgentLLMError(
                agent_name=self.agent_name,
                operation="invoke_llm",
                message="LLM invocation failed after all retries",
                original_exception=last_error,
            )
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
                session_kind="invoke_llm",
            )
            self._persist_session_stats_to_dialog(
                session_kind="invoke_llm",
                elapsed=_elapsed,
                usage=_session_usage,
                calls=_session_calls,
                turns=0,
            )

    async def _call_provider(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        images: Optional[List[Union[str, Path]]] = None,
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        """
        Make the actual SDK call to the provider and return text content.

        Args:
            prompt: User prompt text
            system_message: Optional system message
            temperature: Temperature for this call
            images: Optional image paths for multimodal
            enable_thinking: When True and provider supports it (Anthropic
                extended thinking, OpenAI-compat ``reasoning_content``),
                request thinking and capture it for dialog logging. Silent
                no-op on providers without thinking support (Google).

        Returns:
            ``(text, thinking_text)``. ``thinking_text`` is "" when thinking
            was not requested or the provider didn't supply it.
        """
        temp = temperature if temperature is not None else self.temperature

        try:
            if self.provider == "anthropic":
                return await self._call_anthropic(prompt, system_message, temp, images, enable_thinking)
            elif self.provider == "google":
                # Google path doesn't surface thinking through the SDK shape we use;
                # return empty thinking text.
                text = await self._call_google(prompt, system_message, temp, images)
                return text, ""
            elif (
                enable_thinking
                and self._is_openai_reasoning_model()
            ):
                # gpt-5 / o-series via Chat Completions return no reasoning text
                # (it's opaque server-side). The Responses API exposes
                # reasoning summaries via ``output[].type=='reasoning'``.
                return await self._call_openai_responses(prompt, system_message, images)
            else:
                # openai (non-reasoning), deepseek, ollama, minimax, ngc, openrouter —
                # all OpenAI-compatible. The Chat Completions thinking branch in
                # _call_openai_compat still serves DeepSeek-R1 / qwen-reasoning,
                # which expose ``delta.reasoning_content`` natively.
                return await self._call_openai_compat(prompt, system_message, temp, images, enable_thinking)
        except AgentLLMError:
            raise
        except BaseException as exc:
            raise AgentLLMError(
                agent_name=self.agent_name,
                operation="invoke_llm",
                message=str(exc),
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Anthropic prompt-cache helpers
    # ------------------------------------------------------------------

    # Anthropic requires a per-block minimum for cache_control to take
    # effect (1024 tokens for Sonnet / Haiku 4.x). Below that, the
    # cache-creation cost outweighs the read savings. We approximate
    # tokens as ``len(text) / 3.5`` — conservative so we don't stamp
    # cache_control on short system prompts that won't meet the floor.
    _ANTHROPIC_CACHE_MIN_CHARS = 4000

    def _should_cache_anthropic_prompt(self) -> bool:
        """Whether to mark Anthropic system / tools blocks as cacheable."""
        if str(self.provider or "").lower() != "anthropic":
            return False
        try:
            return bool(getattr(self.settings, "use_prompt_cache", True))
        except Exception:
            return True

    def _build_cached_system_blocks(self, system_message: str) -> list:
        """Wrap a plain system string in a list-of-blocks with cache_control
        on the final block. The Anthropic API caches the longest common
        prefix up to and including the last block tagged with
        ``cache_control``, so a single tagged block is sufficient here.
        """
        blocks: list = [{"type": "text", "text": system_message}]
        if len(system_message) >= self._ANTHROPIC_CACHE_MIN_CHARS:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def _tag_tools_with_cache(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a copy of ``tools`` with cache_control on the last entry.

        Tool-loop ``tools=`` arrays are stable across all turns of a single
        tool loop, so caching them alongside the system prompt saves
        tokens on every retry / tool result turn.
        """
        if not tools:
            return tools
        tagged = list(tools)
        # Estimate combined tool-definition size; skip if under the threshold.
        approx_chars = sum(len(json.dumps(t, ensure_ascii=False)) for t in tagged)
        if approx_chars < self._ANTHROPIC_CACHE_MIN_CHARS:
            return tagged
        tagged[-1] = {**tagged[-1], "cache_control": {"type": "ephemeral"}}
        return tagged

    def _log_anthropic_cache_usage(
        self,
        usage: Any,
        *,
        where: str,
    ) -> None:
        """Emit cache-hit stats so operators can tell whether the cache
        is actually firing. Called after every Anthropic call that went
        through the cache-enabled path.
        """
        if usage is None:
            return
        try:
            read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            input_toks = int(getattr(usage, "input_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if read or created:
            self.logger.info(
                f"[PromptCache:{where}] read={read} created={created} "
                f"uncached_input={input_toks}"
            )

    # ------------------------------------------------------------------
    # Generic token usage observability
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: Any, provider: Optional[str]) -> Dict[str, int]:
        """Normalize per-provider usage fields to a common dict.

        Returns {"input": int, "output": int, "cache_read": int,
        "cache_creation": int}. Missing fields default to 0. Unknown
        providers or response shapes return all zeros — logging callers
        should no-op on all-zero results.
        """
        out = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        if response is None:
            return out
        prov = (provider or "").lower()

        # Anthropic: response.usage with input_tokens / output_tokens /
        # cache_{read,creation}_input_tokens
        usage = getattr(response, "usage", None)
        if usage is not None and prov == "anthropic":
            try:
                out["input"] = int(getattr(usage, "input_tokens", 0) or 0)
                out["output"] = int(getattr(usage, "output_tokens", 0) or 0)
                out["cache_read"] = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
                out["cache_creation"] = int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                )
            except (TypeError, ValueError):
                pass
            return out

        # OpenAI-compatible (openai, deepseek, ollama, minimax, ngc): usage.prompt_tokens /
        # completion_tokens; some providers expose cached input via
        # prompt_tokens_details.cached_tokens
        if usage is not None and prov in ("openai", "deepseek", "ollama", "minimax", "ngc"):
            try:
                out["input"] = int(getattr(usage, "prompt_tokens", 0) or 0)
                out["output"] = int(getattr(usage, "completion_tokens", 0) or 0)
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    out["cache_read"] = int(getattr(details, "cached_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            return out

        # Google Gemini: response.usage_metadata with *_token_count fields
        if prov == "google":
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                try:
                    out["input"] = int(getattr(meta, "prompt_token_count", 0) or 0)
                    out["output"] = int(getattr(meta, "candidates_token_count", 0) or 0)
                    out["cache_read"] = int(
                        getattr(meta, "cached_content_token_count", 0) or 0
                    )
                except (TypeError, ValueError):
                    pass
            return out

        return out

    def _log_llm_usage(
        self,
        response: Any,
        *,
        where: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Log normalized token usage for a single LLM call; return the usage dict.

        Gated by settings.log_token_usage. All-zero usage (unknown provider
        or missing fields) still returns the zero dict so the caller can
        aggregate across turns, but nothing is logged.
        """
        usage = self._extract_usage(response, self.provider)
        # Accumulate into the per-agent cumulative counter regardless of
        # the log gate — pipeline-level token totals must stay correct
        # even when ``log_token_usage`` is off. Only the textual log line
        # below is gated.
        total = usage["input"] + usage["output"] + usage["cache_read"] + usage["cache_creation"]
        if total > 0:
            for k in _USAGE_KEYS:
                self._cumulative_usage[k] += int(usage.get(k, 0))
            self._cumulative_calls += 1
        try:
            enabled = bool(getattr(self.settings, "log_token_usage", True))
        except Exception:
            enabled = True
        if not enabled:
            return usage
        if total == 0:
            return usage
        extra_str = ""
        if extra:
            extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())
        self.logger.info(
            f"[{self.agent_name}:{where}] usage: input={usage['input']} "
            f"output={usage['output']} cache_read={usage['cache_read']} "
            f"cache_creation={usage['cache_creation']}{extra_str}"
        )
        return usage

    def _log_tool_loop_totals(
        self,
        *,
        totals: Dict[str, int],
        turns: int,
    ) -> None:
        """Log a one-line summary of accumulated usage for a finished tool loop."""
        try:
            enabled = bool(getattr(self.settings, "log_token_usage", True))
        except Exception:
            enabled = True
        if not enabled:
            return
        total = sum(totals.values())
        if total == 0:
            return
        self.logger.info(
            f"[{self.agent_name}] tool_loop done: {turns} turns "
            f"total_input={totals['input']} total_output={totals['output']} "
            f"total_cache_read={totals['cache_read']} "
            f"total_cache_creation={totals['cache_creation']}"
        )

    # ------------------------------------------------------------------
    # Per-session stats persistence
    # ------------------------------------------------------------------

    def _persist_session_stats_to_dialog(
        self,
        *,
        session_kind: str,
        elapsed: float,
        usage: Dict[str, int],
        calls: int,
        turns: int,
    ) -> None:
        """Append a session-end stats record under the agent's dialog dir.

        Writes a numbered ``NNN_stats.json`` next to the prompts/responses
        already stored under ``dialog/sessions/<session>/<agent>/`` and
        registers the entry in ``summary.json`` (key ``"sessions"``) so a
        post-mortem can reconstruct the per-session token spend.

        Errors are swallowed: dialog persistence is best-effort and must
        never abort an agent run.
        """
        dm = self.dialog_manager
        if dm is None:
            return
        try:
            dm.log_session_stats(
                agent_name=self.agent_name,
                session_kind=session_kind,
                elapsed=float(elapsed),
                usage=dict(usage),
                calls=int(calls),
                turns=int(turns),
                provider=str(self.provider or ""),
                model=str(self.model or ""),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug(f"Failed to persist session stats: {exc}")

    async def _call_anthropic(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        images: Optional[List[Union[str, Path]]],
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        """Call Anthropic Messages API. Returns ``(text, thinking_text)``.

        With ``enable_thinking=True``, switches to streaming + the
        ``thinking={"type":"enabled", ...}`` parameter so the model performs
        extended thinking and we accumulate ``thinking_delta`` events into
        ``thinking_text``. Anthropic requires ``temperature=1.0`` when
        thinking is enabled — we force it and warn if the caller asked for
        a different value. Without thinking, the original non-streaming
        ``messages.create`` path is used unchanged.
        """
        content_blocks: list = []
        if images:
            content_blocks.extend(self._build_anthropic_image_blocks(images))
        content_blocks.append({"type": "text", "text": prompt})

        use_cache = self._should_cache_anthropic_prompt()
        if not enable_thinking:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content_blocks}],
            }
            if system_message:
                if use_cache:
                    kwargs["system"] = self._build_cached_system_blocks(system_message)
                else:
                    kwargs["system"] = system_message

            response = await self.client.messages.create(**kwargs)

            if use_cache:
                self._log_anthropic_cache_usage(
                    getattr(response, "usage", None), where="invoke_llm"
                )
            self._log_llm_usage(response, where="invoke_llm")

            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "\n".join(text_parts), ""

        # enable_thinking=True path: streaming + thinking budget. Anthropic's
        # API rejects the request unless temperature == 1.0, so override.
        if abs(temperature - 1.0) > 1e-6:
            self.logger.warning(
                f"[invoke_llm] enable_thinking=True forces temperature=1.0 "
                f"(was {temperature}); Anthropic constraint."
            )
        thinking_budget = min(4000, max(1024, self.max_tokens // 4))
        stream_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 1.0,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
            "messages": [{"role": "user", "content": content_blocks}],
        }
        if system_message:
            if use_cache:
                stream_kwargs["system"] = self._build_cached_system_blocks(system_message)
            else:
                stream_kwargs["system"] = system_message

        thinking_chunks: List[str] = []
        async with self.client.messages.stream(**stream_kwargs) as stream:
            async for sevent in stream:
                if getattr(sevent, "type", "") == "content_block_delta":
                    delta = getattr(sevent, "delta", None)
                    if delta and getattr(delta, "type", "") == "thinking_delta":
                        chunk = getattr(delta, "thinking", "") or ""
                        if chunk:
                            thinking_chunks.append(chunk)
            final_msg = await stream.get_final_message()

        if use_cache:
            self._log_anthropic_cache_usage(
                getattr(final_msg, "usage", None), where="invoke_llm"
            )
        self._log_llm_usage(final_msg, where="invoke_llm")

        text_parts = []
        for block in final_msg.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts), "".join(thinking_chunks)

    async def _call_openai_compat(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        images: Optional[List[Union[str, Path]]],
        enable_thinking: bool = False,
    ) -> Tuple[str, str]:
        """Call OpenAI-compatible API. Returns ``(text, thinking_text)``.

        With ``enable_thinking=True``, switches to ``stream=True`` and
        accumulates ``delta.reasoning_content`` chunks (DeepSeek-R1, OpenAI
        o1+, Qwen-Reasoning expose this; vanilla gpt-4 / minimax leave it
        ``None`` and ``thinking_text`` ends up "" — that's fine, we just
        log nothing).
        """
        messages: list = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        if images:
            user_content: list = self._build_openai_image_blocks(images)
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        if not enable_thinking:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=self.max_tokens,
            )
            self._log_llm_usage(response, where="invoke_llm")
            choice = response.choices[0]
            return choice.message.content or "", ""

        thinking_chunks: List[str] = []
        text_chunks: List[str] = []
        stream_iter = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream_iter:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                thinking_chunks.append(rc)
            c = getattr(delta, "content", None)
            if c:
                text_chunks.append(c)
        # Streaming responses don't carry final usage on every provider; skip.
        return "".join(text_chunks), "".join(thinking_chunks)

    async def _call_openai_responses(
        self,
        prompt: str,
        system_message: Optional[str],
        images: Optional[List[Union[str, Path]]],
    ) -> Tuple[str, str]:
        """Call OpenAI Responses API with reasoning enabled.

        Used for gpt-5 / o-series whose Chat Completions endpoint hides the
        reasoning trace. Returns ``(text, thinking_text)`` where
        ``thinking_text`` aggregates the model's reasoning summaries.

        Notes:
            - ``temperature`` is intentionally omitted: reasoning models on
              the Responses API ignore it (or require 1.0).
            - ``system_message`` maps to the ``instructions`` field —
              Responses API does not accept a ``system`` role inside
              ``input``.
            - Image content uses the Responses-API-specific shape
              (``input_image`` with flat ``image_url`` string).
        """
        user_content: list = []
        if images:
            user_content.extend(self._build_openai_responses_image_blocks(images))
        user_content.append({"type": "input_text", "text": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": user_content}],
            "reasoning": {"effort": "medium", "summary": "auto"},
            "max_output_tokens": self.max_tokens,
        }
        if system_message:
            kwargs["instructions"] = system_message

        response = await self.client.responses.create(**kwargs)
        self._log_llm_usage(response, where="invoke_llm")

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for c in getattr(item, "content", None) or []:
                    t = getattr(c, "text", None)
                    if t:
                        text_parts.append(t)
            elif item_type == "reasoning":
                for s in getattr(item, "summary", None) or []:
                    t = getattr(s, "text", None)
                    if t:
                        thinking_parts.append(t)

        # Fallback: ``output_text`` is a SDK convenience that concatenates
        # all message text. Use it only when our manual walk found nothing
        # (e.g. shape changes in future SDK versions).
        text = "\n".join(text_parts)
        if not text:
            text = getattr(response, "output_text", "") or ""
        thinking = "\n\n".join(thinking_parts)
        return text, thinking

    async def _call_google(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        images: Optional[List[Union[str, Path]]],
    ) -> str:
        """Call Google Generative AI (Gemini) API."""
        if genai is None:
            raise ValueError("google-generativeai is not installed.")

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_message if system_message else None,
            generation_config=generation_config,
        )

        parts = []
        if images:
            import PIL.Image
            for img_path in images:
                img = PIL.Image.open(str(img_path))
                parts.append(img)
        parts.append(prompt)

        response = await asyncio.to_thread(model.generate_content, parts)
        self._log_llm_usage(response, where="invoke_llm")
        return response.text

    # ------------------------------------------------------------------
    # Tool-loop artifact logging helpers
    # ------------------------------------------------------------------

    async def _log_tool_turn_prompt(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        step: int,
        terminal_tool_name: Optional[str],
    ) -> None:
        """Persist the prompt sent on this tool-loop turn to dialog_manager.

        Writes one ``NNN_prompt.txt`` per turn under the agent's session
        subdir so operators can inspect what the LLM saw. Without this,
        the tool loop is a black box on disk — only the terminal
        ``submit_plan`` result (via the single invoke_llm log) was visible.
        """
        if not self.dialog_manager:
            return
        try:
            lines = [
                "# SYSTEM PROMPT",
                system_prompt[:20000]
                + ("\n... [truncated]" if len(system_prompt) > 20000 else ""),
                "",
                "# MESSAGES (oldest → newest, last turn is what this request posts)",
            ]
            for idx, msg in enumerate(messages[-6:]):  # last 6 turns is plenty
                role = msg.get("role", "?") if isinstance(msg, dict) else "?"
                content = self._flatten_message_content(
                    msg.get("content") if isinstance(msg, dict) else msg
                )
                lines.append(f"\n--- message[{idx}] role={role} ---")
                lines.append(content[:10000])
            await asyncio.to_thread(
                self.dialog_manager.log_prompt,
                agent_name=self.agent_name,
                prompt="\n".join(lines),
                metadata={
                    "tool_loop_step": step + 1,
                    "terminal_tool_name": terminal_tool_name,
                    "message_count": len(messages),
                },
            )
        except Exception as exc:  # noqa: BLE001 — logging must never fail the run
            self.logger.warning(f"[ToolLoopLog] prompt log failed: {exc}")

    async def _log_tool_turn_response(
        self,
        text: str,
        tool_calls: List[Dict[str, Any]],
        step: int,
        thinking: str = "",
    ) -> None:
        """Persist the response from this tool-loop turn to dialog_manager.

        ``tool_calls`` is a list of ``{"name": str, "args": Any}`` dicts —
        caller formats them so this helper stays provider-agnostic.
        ``thinking`` carries the model's extended-thinking / reasoning
        content when the provider exposes it (Anthropic ``thinking_delta``,
        OpenAI ``reasoning_content`` on o1+/reasoning models). Empty
        otherwise. Persisting it here turns the tool loop from a black box
        into a reviewable trail of the model's intermediate reasoning.
        """
        if not self.dialog_manager:
            return
        try:
            # Some providers (minimax, qwen-reasoning, DeepSeek-R1 in legacy
            # text mode) embed chain-of-thought inline as ``<think>...</think>``
            # tags inside the visible content rather than via a separate
            # ``reasoning_content`` API field. Lift those tags out so they
            # land in ``# THINKING`` instead of polluting ``# TEXT``. After
            # this normalization the on-disk artifact looks the same
            # regardless of whether the provider used the API field or the
            # inline-tag convention.
            inline_blocks = re.findall(r"<think>(.*?)</think>", text or "", flags=re.DOTALL)
            if inline_blocks:
                inline_text = "\n".join(b.strip() for b in inline_blocks if b.strip())
                if inline_text:
                    thinking = (thinking + "\n" + inline_text) if thinking else inline_text
                text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()

            parts: List[str] = []
            if thinking:
                parts.append("# THINKING\n" + thinking[:20000])
            if text:
                parts.append("# TEXT\n" + text[:10000])
            if tool_calls:
                parts.append("# TOOL CALLS")
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    try:
                        args_str = json.dumps(args, indent=2, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_str = str(args)
                    parts.append(f"\n--- {name} ---")
                    parts.append(args_str[:8000])
            body = "\n".join(parts) if parts else "(empty response)"
            await asyncio.to_thread(
                self.dialog_manager.log_response,
                agent_name=self.agent_name,
                response=body,
                metadata={
                    "tool_loop_step": step + 1,
                    "tool_call_count": len(tool_calls),
                    "text_chars": len(text or ""),
                    "thinking_chars": len(thinking or ""),
                },
            )
        except Exception as exc:  # noqa: BLE001 — logging must never fail the run
            self.logger.warning(f"[ToolLoopLog] response log failed: {exc}")

    # ------------------------------------------------------------------
    # Tool-calling loop (Claude native tool_use API)
    # ------------------------------------------------------------------

    # Tools whose executors mutate shared ``context`` state (primarily
    # ``context['current_code']``) or have arbitrary shell side effects.
    # When any tool in a batch is in this set, the batch is serialized;
    # otherwise the batch runs in parallel via ``asyncio.gather``.
    _MUTATING_TOOLS = frozenset({
        "write_file",
        "edit_file",
        "rebut_review",
        "bash",
    })

    async def _execute_tool_batch(
        self,
        tool_use_blocks,
        tool_executors: Dict[str, Callable],
        *,
        step: int,
    ) -> List[Dict[str, Any]]:
        """Execute all tool_use blocks from one LLM turn.

        Parallel when safe, serial when any tool mutates shared state. The
        returned list preserves the original input order so it zips 1:1 with
        ``tool_use_blocks`` for the tool_use_id pairing.
        """
        any_mutating = any(b.name in self._MUTATING_TOOLS for b in tool_use_blocks)

        async def run_one(call_idx: int, tool_block) -> Dict[str, Any]:
            tool_name = tool_block.name
            tool_input = tool_block.input
            tool_id = tool_block.id

            self.logger.debug(f"Executing tool: {tool_name}")

            executor = tool_executors.get(tool_name)
            if executor is None:
                result_content = f"Error: Unknown tool '{tool_name}'"
                is_error = True
            else:
                try:
                    if asyncio.iscoroutinefunction(executor):
                        result = await executor(**tool_input)
                    else:
                        result = await asyncio.to_thread(executor, **tool_input)
                    result_content = str(result) if result is not None else ""
                    is_error = False
                except Exception as e:
                    result_content = f"Error executing {tool_name}: {e}"
                    is_error = True
                    self.logger.warning(f"Tool {tool_name} failed: {e}")

            emit_tool_call_event(
                tool_name=tool_name,
                tool_args=dict(tool_input) if isinstance(tool_input, dict) else {},
                status="error" if is_error else "complete",
                result_preview=result_content,
                call_index=call_idx,
                loop_iter=step + 1,
            )

            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_content,
                "is_error": is_error,
            }

        if any_mutating or len(tool_use_blocks) == 1:
            results: List[Dict[str, Any]] = []
            for i, tb in enumerate(tool_use_blocks):
                results.append(await run_one(i, tb))
            return results

        return list(await asyncio.gather(
            *(run_one(i, tb) for i, tb in enumerate(tool_use_blocks))
        ))

    async def run_tool_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executors: Dict[str, Any],
        max_steps: int = 30,
        images: Optional[List[Union[str, Path]]] = None,
        terminal_tool_name: Optional[str] = None,
    ) -> str:
        """
        Generic tool-calling loop using Claude's native tool_use API.

        Emits lifecycle / tool_call / llm_text_delta events to the workflow
        event bus so a live CLI renderer can surface per-agent progress
        without having to poll log output.

        Args:
            system_prompt: System prompt string
            user_prompt: User prompt string
            tools: List of Claude tool definitions (dicts with name, description, input_schema)
            tool_executors: Dict mapping tool name -> async/sync callable
            max_steps: Max tool loop iterations
            images: Optional image paths for multimodal
            terminal_tool_name: If set, the loop exits immediately after a
                tool batch that includes this tool name (without sending
                its ``tool_result`` back to the model). Callers typically
                stash the terminal tool's payload in a closure container
                and read it after ``run_tool_loop`` returns.

        Returns:
            Final text response from the model (or the terminal tool's
            result string when ``terminal_tool_name`` fired).
        """
        import time as _time
        emit_agent_lifecycle_event(
            agent=self.agent_name,
            state="started",
            model=self.model or "",
            provider=self.provider or "",
            session_kind="tool_loop",
        )
        started_at = _time.time()
        usage_before = dict(self._cumulative_usage)
        calls_before = self._cumulative_calls
        # Inner tool_loop methods stash their per-loop turn count here so
        # the lifecycle finished event can carry it without having to
        # change their return signatures.
        self._last_tool_loop_turns: int = 0
        try:
            if self.provider == "anthropic":
                return await self._tool_loop_anthropic(
                    system_prompt, user_prompt, tools, tool_executors, max_steps, images,
                    terminal_tool_name,
                )
            else:
                return await self._tool_loop_openai_compat(
                    system_prompt, user_prompt, tools, tool_executors, max_steps, images,
                    terminal_tool_name,
                )
        finally:
            elapsed = _time.time() - started_at
            session_usage = _diff_usage(self._cumulative_usage, usage_before)
            session_calls = self._cumulative_calls - calls_before
            session_turns = int(getattr(self, "_last_tool_loop_turns", 0) or 0)
            emit_agent_lifecycle_event(
                agent=self.agent_name,
                state="finished",
                model=self.model or "",
                provider=self.provider or "",
                elapsed=elapsed,
                usage=session_usage,
                calls=session_calls,
                turns=session_turns,
                session_kind="tool_loop",
            )
            self._persist_session_stats_to_dialog(
                session_kind="tool_loop",
                elapsed=elapsed,
                usage=session_usage,
                calls=session_calls,
                turns=session_turns,
            )

    async def _tool_loop_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executors: Dict[str, Any],
        max_steps: int,
        images: Optional[List[Union[str, Path]]],
        terminal_tool_name: Optional[str] = None,
    ) -> str:
        """Tool loop using Anthropic's native tool_use blocks with streaming.

        Each LLM turn is streamed via ``client.messages.stream(...)``; text
        and thinking deltas plus tool_use starts fan out as workflow events
        so the CLI renderer can show tokens landing in real time.
        """
        # Build initial user content
        user_content: list = []
        if images:
            user_content.extend(self._build_anthropic_image_blocks(images))
        user_content.append({"type": "text", "text": user_prompt})

        messages: list = [{"role": "user", "content": user_content}]

        # Prepare cacheable system + tools once per loop. Both are stable
        # across every turn — only ``messages`` changes each iteration — so
        # the cache window covers the whole loop and pays back the creation
        # cost in 1-2 turns.
        use_cache = self._should_cache_anthropic_prompt()
        if use_cache:
            system_arg: Any = self._build_cached_system_blocks(system_prompt)
            tools_arg = self._tag_tools_with_cache(tools)
        else:
            system_arg = system_prompt
            tools_arg = tools

        usage_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        completed_turns = 0

        for step in range(max_steps):
            self.logger.debug(f"Tool loop step {step + 1}/{max_steps}")

            await self._log_tool_turn_prompt(
                system_prompt=system_prompt,
                messages=messages,
                step=step,
                terminal_tool_name=terminal_tool_name,
            )

            emit_llm_stream_start_event(self.agent_name, loop_iter=step + 1)
            final_text_accum: list[str] = []
            thinking_accum: list[str] = []
            try:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_arg,
                    messages=messages,
                    tools=tools_arg,
                ) as stream:
                    async for sevent in stream:
                        et = getattr(sevent, "type", "")
                        if et == "content_block_delta":
                            delta = getattr(sevent, "delta", None)
                            dt = getattr(delta, "type", "") if delta else ""
                            if dt == "text_delta":
                                text = getattr(delta, "text", "") or ""
                                if text:
                                    final_text_accum.append(text)
                                    emit_llm_text_delta_event(
                                        self.agent_name, text, loop_iter=step + 1,
                                    )
                            elif dt == "thinking_delta":
                                thinking = getattr(delta, "thinking", "") or ""
                                if thinking:
                                    thinking_accum.append(thinking)
                                    emit_agent_thinking_event(
                                        self.agent_name, thinking, loop_iter=step + 1,
                                    )
                        elif et == "content_block_start":
                            block = getattr(sevent, "content_block", None)
                            if block and getattr(block, "type", "") == "tool_use":
                                emit_tool_call_event(
                                    tool_name=getattr(block, "name", "?"),
                                    tool_args={},  # input arrives via input_json_delta
                                    status="start",
                                    call_index=getattr(sevent, "index", 0),
                                    loop_iter=step + 1,
                                )
                    response = await stream.get_final_message()
                if use_cache:
                    self._log_anthropic_cache_usage(
                        getattr(response, "usage", None),
                        where=f"tool_loop_step{step + 1}",
                    )
                step_usage = self._log_llm_usage(
                    response, where=f"tool_loop_step{step + 1}"
                )
                for k, v in step_usage.items():
                    usage_totals[k] += v
                completed_turns = step + 1
            finally:
                emit_llm_stream_end_event(
                    self.agent_name,
                    final_text="".join(final_text_accum),
                    loop_iter=step + 1,
                )

            # Check for tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # Log the response artifact for this turn (text + tool calls + thinking).
            await self._log_tool_turn_response(
                text="".join(final_text_accum),
                tool_calls=[
                    {"name": b.name, "args": dict(b.input) if isinstance(b.input, dict) else b.input}
                    for b in tool_use_blocks
                ],
                step=step,
                thinking="".join(thinking_accum),
            )

            if not tool_use_blocks:
                # No tool calls - extract final text
                text_parts = [b.text for b in response.content if hasattr(b, "text")]
                self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
                self._last_tool_loop_turns = completed_turns
                return "\n".join(text_parts)

            # Append assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and build tool_result blocks. Multiple
            # tool_use blocks in one turn are executed in parallel when safe
            # (all are side-effect-free); serialized when any of them mutates
            # shared state (write_file / edit_file touch
            # ``context['current_code']`` that validators read; ``bash`` can
            # have arbitrary side effects). Parallel execution cuts codegen
            # wall-clock ~50% on the common case of read_skill × 3 +
            # read_file × 2 per turn.
            tool_results = await self._execute_tool_batch(
                tool_use_blocks, tool_executors, step=step
            )

            # Terminal-tool short-circuit: caller has stashed the payload via
            # the executor's closure; no need to feed results back to the LLM.
            if terminal_tool_name and any(
                tb.name == terminal_tool_name for tb in tool_use_blocks
            ):
                terminal_result = next(
                    (tr["content"] for tr, tb in zip(tool_results, tool_use_blocks)
                     if tb.name == terminal_tool_name),
                    "",
                )
                self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
                self._last_tool_loop_turns = completed_turns
                return str(terminal_result)

            messages.append({"role": "user", "content": tool_results})

        # Exhausted max_steps
        self.logger.warning(f"Tool loop exhausted {max_steps} steps")
        self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
        self._last_tool_loop_turns = completed_turns
        return f"Tool loop completed after {max_steps} steps without final response."

    async def _tool_loop_openai_compat(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executors: Dict[str, Any],
        max_steps: int,
        images: Optional[List[Union[str, Path]]],
        terminal_tool_name: Optional[str] = None,
    ) -> str:
        """Tool loop using OpenAI function calling format, with streaming.

        Each turn uses ``stream=True`` and aggregates ``choice.delta`` chunks
        so text tokens and tool_call arg fragments are exposed as workflow
        events as they arrive.
        """
        # Convert Claude tool format to OpenAI function format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

        messages: list = [{"role": "system", "content": system_prompt}]

        if images:
            user_content: list = self._build_openai_image_blocks(images)
            user_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        # Request usage stats in the final stream chunk when enabled. Some
        # OpenAI-compat backends may reject this kwarg; set log_token_usage=False
        # to skip it.
        try:
            usage_logging_enabled = bool(getattr(self.settings, "log_token_usage", True))
        except Exception:
            usage_logging_enabled = True

        usage_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        completed_turns = 0

        for step in range(max_steps):
            self.logger.debug(f"Tool loop step {step + 1}/{max_steps}")

            # Persist this turn's prompt to dialog BEFORE firing the LLM —
            # mirrors the Anthropic branch and is what makes ``planning/`` /
            # ``codegen/`` (any agent on an OpenAI-compat provider) record
            # per-turn artifacts. Without it the OpenAI path was a black
            # box on disk: only the upstream ``invoke_llm`` triage call was
            # logged; every submit_plan / write_file tool-loop turn vanished.
            await self._log_tool_turn_prompt(
                system_prompt=system_prompt,
                messages=messages,
                step=step,
                terminal_tool_name=terminal_tool_name,
            )

            emit_llm_stream_start_event(self.agent_name, loop_iter=step + 1)

            # Accumulators rebuilt per-step
            text_accum: list[str] = []
            # Capture provider-side chain-of-thought streamed alongside the
            # visible answer. DeepSeek-R1 / Moonshot k1.5 / OpenAI o1+ emit
            # ``delta.reasoning_content`` chunks; vanilla gpt-4 / minimax
            # leave it None and we just store an empty thinking buffer.
            thinking_accum: list[str] = []
            # Map index -> partial tool call record so fragmented chunks
            # can be stitched back together.
            tool_call_accum: Dict[int, Dict[str, Any]] = {}
            tool_call_started_emitted: Dict[int, bool] = {}
            finish_reason: Optional[str] = None
            # Holds the final chunk's ``usage`` object when
            # ``stream_options={"include_usage": True}`` is in effect.
            final_usage_chunk: Any = None

            try:
                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_completion_tokens": self.max_tokens,
                    "tools": openai_tools if openai_tools else None,
                    "stream": True,
                }
                if usage_logging_enabled:
                    create_kwargs["stream_options"] = {"include_usage": True}
                stream = await self.client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    # The final chunk with ``include_usage`` carries only
                    # ``usage`` and has an empty ``choices`` array.
                    if not chunk.choices:
                        if getattr(chunk, "usage", None) is not None:
                            final_usage_chunk = chunk
                        continue
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    # Text delta
                    if getattr(delta, "content", None):
                        piece = delta.content
                        text_accum.append(piece)
                        emit_llm_text_delta_event(
                            self.agent_name, piece, loop_iter=step + 1,
                        )

                    # Reasoning-content delta (DeepSeek-R1 / Moonshot / o1+).
                    # Captured separately so it lands in the dialog file's
                    # ``# THINKING`` section instead of being mixed into the
                    # visible answer text.
                    rc_piece = getattr(delta, "reasoning_content", None)
                    if rc_piece:
                        thinking_accum.append(rc_piece)
                        emit_agent_thinking_event(
                            self.agent_name, rc_piece, loop_iter=step + 1,
                        )

                    # Tool-call deltas: arrive as fragments with an index
                    for tcd in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tcd, "index", 0) or 0
                        rec = tool_call_accum.setdefault(idx, {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        })
                        if getattr(tcd, "id", None):
                            rec["id"] = tcd.id
                        fn = getattr(tcd, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                rec["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                rec["arguments"] += fn.arguments
                        # Emit "start" event as soon as we know the name
                        if rec["name"] and not tool_call_started_emitted.get(idx):
                            emit_tool_call_event(
                                tool_name=rec["name"],
                                tool_args={},  # args arrive as a streamed JSON fragment
                                status="start",
                                call_index=idx,
                                loop_iter=step + 1,
                            )
                            tool_call_started_emitted[idx] = True

                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
            finally:
                emit_llm_stream_end_event(
                    self.agent_name,
                    final_text="".join(text_accum),
                    loop_iter=step + 1,
                )

            if final_usage_chunk is not None:
                step_usage = self._log_llm_usage(
                    final_usage_chunk, where=f"tool_loop_step{step + 1}"
                )
                for k, v in step_usage.items():
                    usage_totals[k] += v
            completed_turns = step + 1

            # Persist this turn's response (text + tool calls + thinking).
            # Build a provider-neutral tool-call list — name + parsed JSON
            # args — matching the format ``_log_tool_turn_response`` accepts.
            _logged_tool_calls: List[Dict[str, Any]] = []
            for _idx, _rec in sorted(tool_call_accum.items()):
                try:
                    _parsed_args = json.loads(_rec["arguments"]) if _rec["arguments"] else {}
                except json.JSONDecodeError:
                    _parsed_args = {"_raw": _rec["arguments"]}
                _logged_tool_calls.append({
                    "name": _rec["name"] or "?",
                    "args": _parsed_args,
                })
            await self._log_tool_turn_response(
                text="".join(text_accum),
                tool_calls=_logged_tool_calls,
                step=step,
                thinking="".join(thinking_accum),
            )

            if finish_reason != "tool_calls" or not tool_call_accum:
                self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
                self._last_tool_loop_turns = completed_turns
                return "".join(text_accum)

            # Re-assemble OpenAI assistant message (needed in history for the
            # follow-up tool messages to be accepted).
            assistant_tool_calls = [
                {
                    "id": rec["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": rec["name"],
                        "arguments": rec["arguments"],
                    },
                }
                for idx, rec in sorted(tool_call_accum.items())
            ]
            messages.append({
                "role": "assistant",
                "content": "".join(text_accum) or None,
                "tool_calls": assistant_tool_calls,
            })

            # Execute each tool call
            terminal_result: Optional[str] = None
            for idx, rec in sorted(tool_call_accum.items()):
                tool_name = rec["name"] or "?"
                try:
                    tool_args = json.loads(rec["arguments"]) if rec["arguments"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                self.logger.debug(f"Executing tool: {tool_name}")

                executor = tool_executors.get(tool_name)
                is_error = False
                if executor is None:
                    result_content = f"Error: Unknown tool '{tool_name}'"
                    is_error = True
                else:
                    try:
                        if asyncio.iscoroutinefunction(executor):
                            result = await executor(**tool_args)
                        else:
                            result = await asyncio.to_thread(executor, **tool_args)
                        result_content = str(result) if result is not None else ""
                    except Exception as e:
                        result_content = f"Error executing {tool_name}: {e}"
                        is_error = True
                        self.logger.warning(f"Tool {tool_name} failed: {e}")

                emit_tool_call_event(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    status="error" if is_error else "complete",
                    result_preview=result_content,
                    call_index=idx,
                    loop_iter=step + 1,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": rec["id"] or f"call_{idx}",
                    "content": result_content,
                })

                if terminal_tool_name and tool_name == terminal_tool_name:
                    terminal_result = result_content

            # Terminal-tool short-circuit: caller reads the payload from the
            # executor's closure, no need to feed results back to the LLM.
            if terminal_result is not None:
                self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
                self._last_tool_loop_turns = completed_turns
                return str(terminal_result)

        # Exhausted max_steps
        self.logger.warning(f"Tool loop exhausted {max_steps} steps")
        self._log_tool_loop_totals(totals=usage_totals, turns=completed_turns)
        self._last_tool_loop_turns = completed_turns
        return f"Tool loop completed after {max_steps} steps without final response."

    # ------------------------------------------------------------------
    # JSON extraction and fixing (provider-independent)
    # ------------------------------------------------------------------

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that might contain markdown formatting.

        Args:
            text: Text potentially containing JSON

        Returns:
            Extracted JSON string
        """
        if not text or not text.strip():
            self.logger.warning("Empty text passed to _extract_json")
            return ""

        original_text = text

        # Strategy 1: Look for ```json code block specifically
        if "```json" in text:
            try:
                text = text.split("```json")[1].split("```")[0].strip()
                self.logger.debug("Extracted from ```json code block")
            except IndexError:
                self.logger.warning("Failed to extract JSON from ```json code block")
                text = original_text

        # Strategy 2: If no ```json block, try to find any code block containing JSON
        elif "```" in text:
            try:
                parts = text.split("```")
                json_found = False
                for i, part in enumerate(parts):
                    stripped = part.strip()
                    if i % 2 == 1:  # This is a code block content
                        if stripped.startswith(("json", "JSON")):
                            text = stripped[4:].strip()
                            json_found = True
                            self.logger.debug(f"Found JSON in code block {i}")
                            break
                        elif stripped.startswith("{"):
                            text = stripped
                            json_found = True
                            self.logger.debug(f"Found JSON object in code block {i}")
                            break

                if not json_found:
                    self.logger.debug("No JSON code block found, searching in original text")
                    text = original_text
            except IndexError:
                self.logger.warning("Failed to extract from generic code block")
                text = original_text

        # Strategy 3: Find JSON object boundaries in the text
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            extracted = text[start:end]
            self.logger.debug(f"Extracted JSON of length {len(extracted)}")
            return extracted

        # Strategy 4: Try to find array JSON
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            extracted = text[start:end]
            self.logger.debug(f"Extracted JSON array of length {len(extracted)}")
            return extracted

        # Strategy 5: Search in original text if we modified it
        if text != original_text:
            start = original_text.find("{")
            end = original_text.rfind("}") + 1
            if start != -1 and end > start:
                extracted = original_text[start:end]
                self.logger.debug(f"Found JSON in original text, length {len(extracted)}")
                return extracted

        self.logger.warning(f"Could not find JSON boundaries in response. First 200 chars: {text[:200]}")
        return text.strip()

    def _fix_json(self, text: str) -> str:
        """
        Fix common JSON syntax errors in LLM output.

        Args:
            text: Raw JSON string that failed to parse

        Returns:
            Fixed JSON string
        """
        original = text

        # Remove markdown code block markers
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'^```\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text.strip())

        # BOM / smart quotes (common in LLM output)
        if text.startswith("\ufeff"):
            text = text[1:]
        text = (
            text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )

        # Fix trailing commas before } or ]
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Fix missing commas between newlines with key-value pairs
        text = re.sub(r'(\n\s*})([ \t]*\n\s*[{["\'])', r'\1,\2', text)

        # Fix single quotes to double quotes (unescaped ones)
        text = re.sub(r"(?<!\\)(?<!')'(?!:)(?!s)(?!')", '"', text)

        # Fix unquoted keys (simple case: key without quotes)
        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

        self.logger.debug(f"JSON fix applied, length: {len(text)}")
        return text

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        """Convert message content blocks into plain text for logging."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(part for part in parts if part)
        return str(content or "")

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Rough token estimate for diagnostics only."""
        if not text:
            return 0
        chunks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, int(len(chunks) * 0.75))

    def add_feedback(self, feedback: str, from_agent: str):
        """
        Add feedback from another agent.

        Args:
            feedback: Feedback message
            from_agent: Name of the agent providing feedback
        """
        self.feedback_history.append({
            "from": from_agent,
            "message": feedback,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
        self.logger.info(f"Received feedback from {from_agent}: {feedback[:100]}...")

    def get_feedback(self) -> str:
        """
        Get all feedback as a formatted string.

        Returns:
            Formatted feedback string
        """
        if not self.feedback_history:
            return ""

        feedback_lines = ["Previous feedback from other agents:"]
        for fb in self.feedback_history:
            feedback_lines.append(f"- From {fb['from']}: {fb['message']}")

        return "\n".join(feedback_lines)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the agent's main task.

        This method must be implemented by all concrete agent classes.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name})"
