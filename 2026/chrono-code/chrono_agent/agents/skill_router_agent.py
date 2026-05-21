"""LLM-based skill router for codegen pre-injection.

Replaces the static "Required Skills" markdown table in core skills (and the
keyword-based gate in tools/code_agent_tools.py) with a single LLM call
that reads:

  * a compact plan summary (plan_type + scene_objects + assets +
    visualization mode)
  * the skill directory (name + description for every registered skill)

…and returns the minimal set of skill names the codegen LLM must consult
before writing code.

Provider: defaults to MiniMax via the OpenAI-compatible API — the same
endpoint the codegen agent uses, with the same MINIMAX_API_KEY. Cheap and
low-latency. Override via skill_router_* settings if needed.

Design:
  * One-shot tool_use call with strict schema → reliable list output.
  * Module-level cache keyed on plan-content + skill-set hash so a single
    CLI run (multiple codegen iterations) pays exactly one router call.
  * On any error returns ``None`` so the caller keeps the legacy
    static-table behaviour as a safety net.

Owned by the codegen agent, not BaseAgent — same lightweight pattern as
``SkillQueryAgent``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, List, Optional

import openai

from chrono_agent.config import get_settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a skill router for a PyChrono code-generation pipeline. Given a "
    "simulation plan and a directory of available skills (each with a one-line "
    "description AND a per-skill section index), pick TWO things:\n\n"
    "  1. ``selected_skills`` — the MINIMAL set of skill NAMES the codegen "
    "model must consult to write correct code.\n"
    "  2. ``selected_sections`` — for each big skill (more than ~5 sections), "
    "the MINIMAL list of SECTION KEYS within that skill the codegen model "
    "must read. This lets us inject only the relevant Patterns / Hard Rules "
    "instead of the whole multi-thousand-line skill file.\n\n"
    "Skill-selection rules:\n"
    "1. ALWAYS include the core skill matching plan_type "
    "(core/<plan_type>: core/scene, core/mbs, or core/mbs_in_scene).\n"
    "2. Include a skill ONLY if its description matches a feature actually "
    "present in the plan. Match against scene_objects fields (domain_type, "
    "fsi_registration, primitive, role), assets[*].type, and visualization "
    "mode.\n"
    "3. Be aggressive about EXCLUDING skills whose features are absent. For "
    "example, do NOT include veh/terrain when no terrain object exists in "
    "scene_objects, even if the plan has a vehicle. Do NOT include "
    "scene/custom_assets_* when all scene_objects use procedural primitives.\n"
    "4. Cross-domain hints: SPH/FSI/fluid/water/floating fields → fsi/sph; "
    "vehicle assets → veh/wheeled_vehicle + veh/driver; terrain object → "
    "veh/terrain; sensor camera in visualization → sens/camera + "
    "sens/sensor_manager; vsg in visualization → vsg.\n\n"
    "Section-selection rules:\n"
    "5. For each selected skill, look at its ``Sections:`` list. Use the "
    "section keys EXACTLY as written (lowercase, including the em-dash and "
    "any subtitle).\n"
    "6. ALWAYS include ``api contract`` and ``purpose`` (or ``when to use``) "
    "for selected skills that expose them.\n"
    "7. Pick Pattern* sections that match plan features — e.g. plan with "
    "``fsi_registration: CreatePointsBoxInterior`` needs Pattern D from "
    "fsi/sph; plan with vehicle + fsi needs ``fsi coupling — wheel spindle "
    "registration`` from veh/wheeled_vehicle.\n"
    "8. Always pick the ``hard rules`` section (or whatever the skill calls "
    "its invariant block) when present — those are non-negotiable.\n"
    "9. Skip sections that don't apply: do NOT pick ``pattern g — vsg-only "
    "mp4 recording`` if the plan doesn't render mp4; do NOT pick "
    "performance-tuning sections unless the plan tunes performance.\n"
    "10. For tiny skills (≤5 sections) you MAY omit the skill from "
    "``selected_sections`` entirely; the system will inject the whole file.\n"
    "11. For big skills, omitting them from ``selected_sections`` falls back "
    "to a budget-driven tier cascade — usually NOT what you want, since the "
    "cascade may drop the very Patterns you need. Always provide an explicit "
    "section list for big skills.\n\n"
    "Return ONLY via the submit_skills function call."
)


# OpenAI-style function tool. MiniMax follows the OpenAI Chat Completions
# tool_calls protocol verbatim.
_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_skills",
        "description": (
            "Submit the minimal set of skill names + per-skill section keys "
            "required for this plan. Use names and section keys exactly as "
            "they appear in the directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selected_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Skill names from the directory, ordered by importance. "
                        "Core skill first."
                    ),
                },
                "selected_sections": {
                    "type": "object",
                    "description": (
                        "Per-skill section selection. Keys are skill names "
                        "(must also appear in selected_skills); values are "
                        "lowercase section keys exactly as they appear under "
                        "the skill's ``Sections:`` line in the directory. "
                        "Omit a skill from this object to fall back to "
                        "whole-file injection (with budget tier cascade) — "
                        "use only for tiny skills."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "One short sentence per skill explaining why it's needed "
                        "for THIS plan. Helps debug router decisions."
                    ),
                },
            },
            "required": ["selected_skills"],
        },
    },
}


# Module-level cache: (plan_hash, skills_hash) -> (selected_skills, selected_sections).
# Survives across codegen iterations within one CLI run (same process).
_ROUTER_CACHE: dict[str, "RouterDecision"] = {}


class RouterDecision:
    """Pair of (skill_names, sections_per_skill) returned by SkillRouterAgent.

    ``sections_per_skill`` is a dict mapping skill name → list of lowercase
    section keys to inject. A skill not present in the dict falls back to
    whole-file injection (with the budget tier cascade) downstream.
    """

    __slots__ = ("skills", "sections")

    def __init__(self, skills: List[str], sections: dict[str, List[str]]):
        self.skills = list(skills)
        self.sections = {k: list(v) for k, v in sections.items()}

    def copy(self) -> "RouterDecision":
        return RouterDecision(self.skills, self.sections)


def _hash_inputs(plan_summary: str, skill_directory: str) -> str:
    h = hashlib.sha256()
    h.update(plan_summary.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(skill_directory.encode("utf-8", errors="replace"))
    return h.hexdigest()


def build_plan_summary(plan: Any) -> str:
    """Compact plan summary fed to the router. JSON keeps structure visible."""
    summary: dict[str, Any] = {}

    def _g(name: str, default: Any = None) -> Any:
        if hasattr(plan, name):
            return getattr(plan, name, default)
        if isinstance(plan, dict):
            return plan.get(name, default)
        return default

    summary["plan_type"] = _g("plan_type")

    vis = _g("visualization") or {}
    if hasattr(vis, "model_dump"):
        vis = vis.model_dump()
    summary["visualization_mode"] = (vis or {}).get("mode") if isinstance(vis, dict) else None

    assets = _g("assets") or []
    asset_brief = []
    for a in assets:
        if hasattr(a, "model_dump"):
            a = a.model_dump()
        if isinstance(a, dict):
            asset_brief.append({
                "name": a.get("name"),
                "type": a.get("type"),
            })
    summary["assets"] = asset_brief

    scene_objs = _g("scene_objects") or []
    so_brief = []
    for so in scene_objs:
        if hasattr(so, "model_dump"):
            so = so.model_dump()
        if isinstance(so, dict):
            so_brief.append({
                k: so.get(k) for k in (
                    "name", "role", "construction_source", "primitive",
                    "domain_type", "fsi_registration", "fixed",
                ) if so.get(k) is not None
            })
    summary["scene_objects"] = so_brief

    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


class SkillRouterAgent:
    """One-shot LLM call (MiniMax via OpenAI-compat) that picks skills from
    a description directory."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        # Default to the same MiniMax endpoint as the codegen agent.
        self.model = (
            model
            or getattr(settings, "skill_router_model", None)
            or getattr(settings, "minimax_model", None)
            or "MiniMax-M2.7-highspeed"
        )
        self.api_base = (
            api_base
            or getattr(settings, "skill_router_api_base", None)
            or getattr(settings, "minimax_api_base", None)
            or "https://api.minimaxi.com/v1"
        )
        self.api_key = (
            api_key
            or getattr(settings, "skill_router_api_key", None)
            or getattr(settings, "minimax_api_key", None)
        )
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else getattr(settings, "skill_router_max_tokens", 1024)
        )
        self.temperature = 0.0

        if not self.api_key:
            raise ValueError(
                "SkillRouterAgent: no API key. Set MINIMAX_API_KEY (or "
                "SKILL_ROUTER_API_KEY) in .env."
            )
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    async def route(
        self,
        plan_summary: str,
        skill_directory: str,
        valid_skill_names: set[str],
        valid_sections_per_skill: Optional[dict[str, set[str]]] = None,
    ) -> Optional["RouterDecision"]:
        """Return a RouterDecision (skill names + per-skill section keys),
        or ``None`` on failure.

        ``valid_skill_names`` filters hallucinated skill names.
        ``valid_sections_per_skill`` (optional) filters hallucinated section
        keys — anything the router emits that isn't a real section is
        dropped, leaving the skill in ``selected_skills`` so it falls back
        to whole-file injection.
        """
        cache_key = _hash_inputs(plan_summary, skill_directory)
        cached = _ROUTER_CACHE.get(cache_key)
        if cached is not None:
            logger.info(
                "[SkillRouter] cache hit (%d skills, %d skills with sections)",
                len(cached.skills), len(cached.sections),
            )
            return cached.copy()

        user_text = (
            f"Plan:\n<plan>\n{plan_summary}\n</plan>\n\n"
            f"Available skills:\n<directory>\n{skill_directory}\n</directory>\n\n"
            "Pick the minimal required skill set + per-skill section list "
            "via submit_skills."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[_TOOL_SCHEMA],
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_skills"},
                },
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("[SkillRouter] LLM call failed: %s", exc)
            return None

        # Usage logging.
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                logger.info(
                    "[SkillRouter] usage: prompt=%s completion=%s total=%s",
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                    getattr(usage, "total_tokens", 0),
                )
        except Exception:
            pass

        try:
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []
        except Exception:
            tool_calls = []
        if not tool_calls:
            logger.warning("[SkillRouter] no tool_calls in response")
            return None

        # Find the submit_skills call.
        raw_args: Optional[str] = None
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is not None and getattr(fn, "name", "") == "submit_skills":
                raw_args = getattr(fn, "arguments", "") or ""
                break
        if not raw_args:
            logger.warning("[SkillRouter] submit_skills call missing arguments")
            return None

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            logger.warning("[SkillRouter] tool args not valid JSON: %s", exc)
            return None

        selected = parsed.get("selected_skills") or []
        sections_raw = parsed.get("selected_sections") or {}
        reasoning = parsed.get("reasoning") or ""
        if not isinstance(selected, list):
            return None
        if not isinstance(sections_raw, dict):
            sections_raw = {}

        seen: set[str] = set()
        clean: List[str] = []
        dropped: List[str] = []
        for name in selected:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            if n not in valid_skill_names:
                dropped.append(n)
                continue
            seen.add(n)
            clean.append(n)

        if dropped:
            logger.warning(
                "[SkillRouter] dropped %d unknown skill names: %s",
                len(dropped), dropped,
            )
        if not clean:
            logger.warning("[SkillRouter] router returned no valid skills")
            return None

        # Validate per-skill section keys; drop unknowns silently so a typo
        # in one section name degrades to whole-file injection rather than
        # failing the whole route.
        clean_sections: dict[str, List[str]] = {}
        for skill_name, secs in sections_raw.items():
            sk = str(skill_name or "").strip()
            if sk not in seen:
                continue
            if not isinstance(secs, list):
                continue
            valid = (valid_sections_per_skill or {}).get(sk)
            picked: List[str] = []
            for s in secs:
                key = str(s or "").strip().lower()
                if not key:
                    continue
                if valid is not None and key not in valid:
                    matched = next((v for v in valid if key in v), None)
                    if matched is None:
                        continue
                    key = matched
                if key in picked:
                    continue
                picked.append(key)
            if picked:
                clean_sections[sk] = picked

        decision = RouterDecision(clean, clean_sections)
        logger.info(
            "[SkillRouter] selected %d skills via %s: %s "
            "(sectioned: %s)%s",
            len(decision.skills), self.model, decision.skills,
            {k: len(v) for k, v in decision.sections.items()},
            f" — reasoning: {reasoning}" if reasoning else "",
        )
        _ROUTER_CACHE[cache_key] = decision.copy()
        return decision
