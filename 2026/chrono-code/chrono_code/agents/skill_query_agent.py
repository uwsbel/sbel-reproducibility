"""Lightweight Haiku-backed skill Q&A sub-agent.

Instead of returning 3-5K tokens of full SKILL.md content to the main
codegen model, this sub-agent reads the document (on a cheaper Haiku
model) and returns a focused 300-800 token answer to a specific
question. The SKILL.md content is cached via Anthropic's prompt cache
so that repeated queries on the same skill land in the ~90%-discount
cache path.

Cost model (per query, rough):
  - Main-model ``read_skill``: ~5K Sonnet input tokens (full doc returned).
  - ``query_skill`` via this sub-agent:
      * Haiku input:  ~5K (document + system + question)
      * Haiku output: ~500 (answer)
      * Main-model input: ~500 (short answer)
    Haiku is ~4× cheaper than Sonnet, so the net is ~30-40% of the
    original Sonnet cost for skills that would have been read in full.

This class deliberately does NOT extend ``BaseAgent`` — it has no tool
loop, no dialog manager, and doesn't consume one of the agent1..4
config slots. It's a one-shot Q&A utility owned by the codegen tool
executor.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import anthropic

from chrono_code.config import get_settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You answer questions about a single PyChrono SKILL.md document. "
    "Ground every statement in the document content — if the answer "
    "isn't present, say so explicitly (\"not covered in this skill\") "
    "rather than guessing or mixing in general PyChrono knowledge. "
    "Keep answers tight (≤300 words). When the question asks about an "
    "API, function signature, or specific code pattern, quote the "
    "relevant snippet verbatim from the document. When the question is "
    "conceptual, give a short prose answer plus a pointer to the "
    "section of the document that contains the authoritative version."
)


class SkillQueryAgent:
    """One-shot Anthropic Q&A agent for SKILL.md retrieval.

    Not async-factory; callers await :meth:`query` directly.
    """

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self.provider = (
            provider
            or getattr(settings, "skill_query_subagent_provider", "anthropic")
        ).lower()
        if self.provider != "anthropic":
            raise ValueError(
                f"SkillQueryAgent only supports 'anthropic' provider; got {self.provider}"
            )
        self.model = model or getattr(
            settings, "skill_query_subagent_model", "claude-haiku-4-5-20251001"
        )
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else getattr(settings, "skill_query_subagent_max_tokens", 1024)
        )
        # Factual retrieval — deterministic output helps cache hit rate on
        # repeated identical questions.
        self.temperature = 0.0

        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError(
                "SkillQueryAgent: anthropic_api_key is not configured. "
                "Set ANTHROPIC_API_KEY in .env or disable "
                "skill_query_subagent_enabled."
            )
        client_kwargs: dict[str, Any] = {"api_key": key}
        base = api_base or getattr(settings, "anthropic_api_base", None)
        if base:
            client_kwargs["base_url"] = base
        self.client = anthropic.AsyncAnthropic(**client_kwargs)

    async def query(
        self,
        skill_name: str,
        skill_content: str,
        question: str,
    ) -> str:
        """Ask ``question`` about ``skill_content`` (the SKILL.md text).

        Returns the Haiku-generated short answer as a plain string. The
        caller should surface this answer as a tool_result to the
        primary codegen loop.
        """
        if not skill_content:
            return f"(no content for skill '{skill_name}')"
        if not question or not question.strip():
            return "query_skill requires a non-empty question."

        user_text = (
            f"SKILL document: {skill_name}\n"
            f"<document>\n{skill_content}\n</document>\n\n"
            f"Question: {question.strip()}"
        )
        # Cache both the system prompt and the document block so
        # repeated queries on the same skill during a codegen session
        # land in Anthropic's prompt cache (5-min ephemeral TTL).
        system_blocks = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        user_blocks = [
            {
                "type": "text",
                "text": user_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_blocks,
            messages=[{"role": "user", "content": user_blocks}],
        )

        # Usage logging — gated by the shared log_token_usage flag.
        try:
            settings = get_settings()
            if bool(getattr(settings, "log_token_usage", True)):
                usage = getattr(response, "usage", None)
                if usage is not None:
                    logger.info(
                        "[SkillQueryAgent:%s] usage: input=%s output=%s "
                        "cache_read=%s cache_creation=%s",
                        skill_name,
                        getattr(usage, "input_tokens", 0),
                        getattr(usage, "output_tokens", 0),
                        getattr(usage, "cache_read_input_tokens", 0),
                        getattr(usage, "cache_creation_input_tokens", 0),
                    )
        except Exception:
            # Never let observability break the tool.
            pass

        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_parts).strip() or "(empty response)"
