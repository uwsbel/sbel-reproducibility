"""6-phase PlanningAgent pipeline. See plan_agent.md."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from chrono_agent.agents.prompts import planning_prompts
from chrono_agent.models.clarification import StructuredClarification
from chrono_agent.models.plan import SimulationPlan, set_user_spec_context
from chrono_agent.models.plan_markdown_parser import (
    TOKEN_RE,
    collect_clarification_tokens,
    has_unresolved_tokens,
    parse_plan_markdown,
)
from chrono_agent.models.user_spec import extract_user_spec_regex

logger = logging.getLogger(__name__)


AnswerMap = Dict[str, Any]
BatchCallback = Callable[[List[StructuredClarification]], Awaitable[AnswerMap]]


# ---- Phase 1: Extract -----------------------------------------------------


async def phase1_extract(
    agent,
    user_prompt: str,
    images: Optional[List[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """Single LLM call: prompt + catalog → list of {name, kind, ...}."""
    catalog = agent._scan_asset_catalog() or {}
    catalog_block = _format_catalog(catalog)

    prompt = planning_prompts.PHASE1_EXTRACT_PROMPT.substitute(
        user_prompt=user_prompt,
        catalog_block=catalog_block,
    )
    raw = await agent.invoke_llm(prompt=prompt, parse_json=False, images=images)
    objects = _parse_phase1_output(raw, catalog)
    listing = _render_phase1_listing(objects, catalog)

    return {
        "objects": objects,
        "catalog_block": catalog_block,
        "phase1_listing": listing,
        "requested": [o["name"] for o in objects],
    }


def _format_catalog(catalog: Dict[str, Dict[str, Any]]) -> str:
    """One catalog row per line: name + type + filename or factory. No prose."""
    if not catalog:
        return "(no catalog assets — pure mbs plan)"
    lines: List[str] = []
    for entry in catalog.values():
        atype = entry.get("type", "?")
        if atype == "wrapper_vehicle":
            loader = f"factory={entry.get('factory', '')}"
        else:
            loader = f"filename={entry.get('filename', '')}"
        line = f"- {entry['name']} (type={atype}, {loader})"
        aliases = entry.get("aliases") or []
        if aliases:
            line += f" aliases={aliases}"
        lines.append(line)
    return "\n".join(lines)


def _parse_phase1_output(raw: Any, catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse the Phase 1 JSON array. Drop entries that fail validation."""
    text = raw if isinstance(raw, str) else json.dumps(raw)
    text = _strip_markdown_fences(text)
    # Pull out the first [...] block in case the model added prose.
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        logger.warning("[Phase1] no JSON array in output")
        return []
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("[Phase1] JSON parse failed: %s", exc)
        return []
    if not isinstance(parsed, list):
        return []

    catalog_names = {e["name"] for e in catalog.values()}
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        kind = str(item.get("kind") or "").strip().lower()
        if not name or kind not in {"asset", "procedural"} or name in seen:
            continue
        if kind == "asset":
            cat = str(item.get("catalog") or "").strip()
            atype = str(item.get("asset_type") or "").strip()
            if cat not in catalog_names:
                logger.warning("[Phase1] asset %r names unknown catalog %r — skipping", name, cat)
                continue
            out.append({
                "name": name, "kind": "asset", "catalog": cat,
                "asset_type": atype, "rationale": item.get("rationale", ""),
            })
        else:
            prim = str(item.get("primitive") or "").strip()
            if prim not in {"box", "sphere", "cylinder", "grid", "fluid_domain", "generated_boundary"}:
                logger.warning("[Phase1] procedural %r has bad primitive %r — skipping", name, prim)
                continue
            out.append({
                "name": name, "kind": "procedural", "primitive": prim,
                "rationale": item.get("rationale", ""),
            })
        seen.add(name)
    return out


def _render_phase1_listing(objects: List[Dict[str, Any]], catalog: Dict[str, Dict[str, Any]]) -> str:
    """Per-object decisions for Phase 2: name → kind + loader info.

    Trailing annotation lists same-noun groups (chair_1..chair_7 →
    chairs[×7]) so Phase 2 can emit a SINGLE
    ``scene_layout_strategy[chairs]`` clarification rather than 49
    individual position tokens.
    """
    if not objects:
        return "(no objects extracted)"
    by_name = {e["name"]: e for e in catalog.values()}
    lines: List[str] = []
    for o in objects:
        if o["kind"] == "asset":
            entry = by_name.get(o["catalog"]) or {}
            atype = o.get("asset_type") or entry.get("type", "?")
            if atype == "wrapper_vehicle":
                loader = f"factory={entry.get('factory', '')}"
            else:
                loader = f"filename={entry.get('filename', '')}"
            lines.append(
                f"- {o['name']} → ASSET: catalog={o['catalog']}, asset_type={atype}, {loader}"
            )
        else:
            lines.append(f"- {o['name']} → PROCEDURAL: primitive={o['primitive']}")

    groups = _detect_object_groups(objects)
    if groups:
        lines.append("")
        lines.append("SAME-NOUN GROUPS (≥2 instances; emit ONE scene_layout_strategy[group] token, NOT per-instance position tokens):")
        for group_name, members in groups.items():
            lines.append(f"  - {group_name}[×{len(members)}] = {{{', '.join(members)}}}")
    return "\n".join(lines)


_GROUP_SUFFIX_RE = re.compile(r"^(?P<noun>[a-z][a-z_]*[a-z])_(?P<idx>\d+)$")


def _detect_object_groups(objects: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return {plural_noun: [member_name_1, member_name_2, ...]} for groups ≥ 2.

    Recognises ``chair_1..chair_N``, ``tree_3``, ``rock_07`` etc. The
    plural is computed naively (append ``s`` unless already ending in
    ``s``) so codegen / Phase 5b can key on it. Singletons are ignored.
    """
    by_noun: Dict[str, List[str]] = {}
    for o in objects:
        m = _GROUP_SUFFIX_RE.match(str(o.get("name") or ""))
        if not m:
            continue
        noun = m.group("noun")
        by_noun.setdefault(noun, []).append(o["name"])
    out: Dict[str, List[str]] = {}
    for noun, members in by_noun.items():
        if len(members) < 2:
            continue
        plural = noun if noun.endswith("s") else f"{noun}s"
        out[plural] = sorted(members)
    return out


# ---- Phase 2: Draft -------------------------------------------------------


async def phase2_draft(
    agent,
    user_prompt: str,
    user_spec,
    phase1: Dict[str, Any],
    images: Optional[List[Union[str, Path]]] = None,
) -> str:
    """One LLM call. Output: plan markdown with optional <<ASK_*>> tokens."""
    image_grounding_block = ""
    if images:
        skill_body = _load_skill_text("planning/image_grounding")
        image_grounding_block = (
            "## IMAGE GROUNDING (REQUIRED — images are attached)\n\n"
            + skill_body
            + "\n\nThe plan markdown MUST include a top-level "
            "`## image_observation` section (see OUTPUT FORMAT below) carrying "
            "the 5-step procedure literally — enumerate / positions / "
            "orientations / viewpoint / cross-check. The `viewpoint:` line "
            "feeds the camera-rules protocol's image-grounded override.\n"
        )
    system_prompt = planning_prompts.PHASE2_DRAFT_SYSTEM_PROMPT.substitute(
        user_prompt=user_prompt,
        user_spec_block=user_spec.render_for_prompt(),
        catalog_block=phase1["catalog_block"],
        phase1_listing=phase1["phase1_listing"],
        image_grounding_block=image_grounding_block,
        token_rules=planning_prompts.TOKEN_PROTOCOL_RULES,
    )
    raw = await agent.invoke_llm(
        prompt=user_prompt,
        system_message=system_prompt,
        parse_json=False,
        images=images,
        enable_thinking=True,
    )
    return _strip_markdown_fences(str(raw))


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing ``` fences if the model wrapped output."""
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


# ---- Phase 3: Collect (pure Python) ---------------------------------------


def phase3_collect(draft_markdown: str) -> List[StructuredClarification]:
    """Regex over the draft → list of choice/number clarifications.

    Filters orphan tokens whose ``target_field`` references an
    ``objects[<name>]`` not declared in the same draft's ``## objects``
    block — Phase 2 occasionally hallucinates extra entities (e.g.
    ``chair_5`` when only chair_1..chair_4 are listed). Surfacing those
    questions confuses the user because the resulting answer has nowhere
    to land in the final plan.
    """
    items = collect_clarification_tokens(draft_markdown)
    known_objects = _extract_object_names(draft_markdown)
    out: List[StructuredClarification] = []
    dropped = 0
    for item in items:
        if _is_orphan_target(item["target_field"], known_objects):
            dropped += 1
            logger.warning(
                "[Phase3] dropping orphan token target=%r (no matching "
                "object in draft; known=%s)",
                item["target_field"], sorted(known_objects),
            )
            continue
        if item["kind"] == "choice":
            out.append(StructuredClarification(
                question=item["question"],
                kind="choice",
                target_field=item["target_field"],
                options=item["options"],
                allow_other=False,
            ))
        else:
            out.append(StructuredClarification(
                question=item["question"],
                kind="number",
                target_field=item["target_field"],
                unit=item["unit"],
            ))
    if dropped:
        logger.info("[Phase3] dropped %d orphan token(s); %d kept", dropped, len(out))
    return out


_OBJECT_REF_RE = re.compile(r"^objects\[([^\]]+)\]")


def _is_orphan_target(target_field: str, known_objects: set) -> bool:
    """Return True iff ``target_field`` names an object not in the draft.

    Targets that don't reference an object (``simulation_parameters.*``,
    ``scene_layout_strategy[<group>]``) are never orphans — only
    ``objects[<name>].*`` paths are validated.
    """
    m = _OBJECT_REF_RE.match(target_field or "")
    if not m:
        return False
    return m.group(1) not in known_objects


def _extract_object_names(draft_markdown: str) -> set:
    """Pull object names from the ``## objects`` YAML block.

    Returns a set of names; empty when the draft has no ``## objects``
    block (e.g. malformed Phase 2 output) — in that case nothing is
    classified as orphan, so the legacy behaviour is preserved.
    """
    try:
        plan_dict, _ = parse_plan_markdown(draft_markdown)
    except Exception as exc:
        logger.warning("[Phase3] could not parse draft for orphan check: %s", exc)
        return set()
    objs = plan_dict.get("objects") or []
    names: set = set()
    if isinstance(objs, list):
        for o in objs:
            if isinstance(o, dict):
                name = str(o.get("name") or "").strip()
                if name:
                    names.add(name)
    return names


# ---- Phase 4: Ask UI ------------------------------------------------------


async def phase4_ask(
    items: List[StructuredClarification],
    callback: Optional[BatchCallback],
) -> AnswerMap:
    """Hand the full list to the UI; expect {target_field: answer} back."""
    if not items:
        return {}
    if callback is None:
        logger.warning("[Pipeline] no clarification callback; skipping %d question(s)", len(items))
        return {}
    answers = await callback(items)
    if not isinstance(answers, dict):
        raise TypeError(f"callback must return Dict[target_field, answer]; got {type(answers).__name__}")
    return answers


# ---- Phase 5: Finalize ----------------------------------------------------


async def phase5_finalize(agent, draft_markdown: str, answers: AnswerMap) -> str:
    """One LLM call: substitute answers + resolve relations using the skill."""
    if not answers and not has_unresolved_tokens(draft_markdown):
        return draft_markdown

    system_prompt = planning_prompts.PHASE5_FINALIZE_SYSTEM_PROMPT.substitute(
        draft_markdown=draft_markdown,
        answers_block=_render_answers_block(answers),
        scene_coord_skill=_load_skill_text("planning/scene_coordinate_system"),
    )
    raw = await agent.invoke_llm(
        prompt="Output the finalized plan markdown now.",
        system_message=system_prompt,
        parse_json=False,
    )
    return _strip_markdown_fences(str(raw))


def _render_answers_block(answers: AnswerMap) -> str:
    if not answers:
        return "(no answers — user did not provide clarifications)"
    return "\n".join(
        f"- {tf} = {json.dumps(answers[tf], ensure_ascii=False)}"
        for tf in sorted(answers)
    )


# ---- Phase 5b: Backfill defaults for unresolved tokens --------------------


async def phase5b_backfill_defaults(agent, final_markdown: str) -> str:
    """One LLM call: replace any remaining <<ASK_*>> tokens with safe defaults.

    Phase 5 leaves tokens untouched when the user provided no answers.
    This phase rewrites the draft so codegen never sees literal
    ``<<ASK_*>>`` strings, and tags every backfilled field via an
    ``(inferred default: ...)`` annotation in the owning object's
    description (or a top-of-plan ``## inferred_defaults`` section for
    plan-level fields).
    """
    if not has_unresolved_tokens(final_markdown):
        return final_markdown

    system_prompt = planning_prompts.PHASE5B_BACKFILL_SYSTEM_PROMPT.substitute(
        final_markdown=final_markdown,
    )
    raw = await agent.invoke_llm(
        prompt="Replace every remaining <<ASK_*>> token with the inferred default and emit the full markdown.",
        system_message=system_prompt,
        parse_json=False,
    )
    out = _strip_markdown_fences(str(raw))
    if has_unresolved_tokens(out):
        # LLM didn't fully clean up — surface this so the user notices
        # the plan is still incomplete rather than silently shipping a
        # broken plan to codegen. (We can't safely raise here because
        # the user explicitly chose not to answer; just warn.)
        leftover_count = sum(1 for _ in TOKEN_RE.finditer(out))
        logger.warning(
            "[Phase5b] backfill returned %d remaining token(s); plan may break codegen",
            leftover_count,
        )
    return out


def _load_skill_text(skill_name: str) -> str:
    """Load a skill's markdown body (front-matter stripped)."""
    try:
        from chrono_agent.skills import SkillRegistry
        text = SkillRegistry.get_skill_fragment(skill_name)
        if text:
            return text
    except Exception as exc:
        logger.warning("[Pipeline] skill load failed for %r: %s", skill_name, exc)
    candidate = Path(__file__).resolve().parent.parent / "skills" / skill_name / "SKILL.md"
    if not candidate.exists():
        return f"(skill '{skill_name}' not found)"
    body = candidate.read_text(encoding="utf-8")
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].lstrip()
    return body


# ---- Phase 6: Modify ------------------------------------------------------


async def phase6_modify(agent, current_plan_markdown: str, modification_text: str) -> str:
    """One LLM call: apply the user's modification text to the current plan."""
    system_prompt = planning_prompts.PHASE6_MODIFY_SYSTEM_PROMPT.substitute(
        current_plan_markdown=current_plan_markdown,
        modification_text=modification_text,
        token_rules=planning_prompts.TOKEN_PROTOCOL_RULES,
    )
    raw = await agent.invoke_llm(
        prompt=modification_text,
        system_message=system_prompt,
        parse_json=False,
        enable_thinking=True,
    )
    return _strip_markdown_fences(str(raw))


# ---- Drivers --------------------------------------------------------------


async def run_pipeline(
    agent,
    user_prompt: str,
    images: Optional[List[Union[str, Path]]] = None,
    clarification_batch_callback: Optional[BatchCallback] = None,
) -> SimulationPlan:
    """Phases 1→2→3→4→5. Phase 6 lives in run_modify."""
    user_spec = extract_user_spec_regex(user_prompt)

    phase1 = await phase1_extract(agent, user_prompt, images=images)
    if phase1["requested"] and not user_spec.required_assets:
        user_spec.required_assets = phase1["requested"]
    logger.info("[Pipeline] phase1: %d objects", len(phase1["requested"]))

    draft = await phase2_draft(agent, user_prompt, user_spec, phase1, images=images)
    logger.info("[Pipeline] phase2: draft %d chars", len(draft))

    items = phase3_collect(draft)
    logger.info("[Pipeline] phase3: %d clarifications", len(items))

    answers = await phase4_ask(items, clarification_batch_callback)
    logger.info("[Pipeline] phase4: %d answers", len(answers))

    final_md = await phase5_finalize(agent, draft, answers)
    logger.info("[Pipeline] phase5: final %d chars", len(final_md))

    if has_unresolved_tokens(final_md):
        final_md = await phase5b_backfill_defaults(agent, final_md)
        logger.info("[Pipeline] phase5b: backfilled %d chars", len(final_md))

    return _markdown_to_plan(final_md, user_spec)


async def run_modify(
    agent,
    current_plan: SimulationPlan,
    modification_text: str,
    images: Optional[List[Union[str, Path]]] = None,
    clarification_batch_callback: Optional[BatchCallback] = None,
) -> SimulationPlan:
    """Phase 6 → re-run Phase 3-5 on the modified draft."""
    user_spec = extract_user_spec_regex(current_plan.plan_markdown or "")

    current_md = current_plan.plan_markdown or ""
    if not current_md:
        raise ValueError("current_plan must carry plan_markdown to modify")

    modified = await phase6_modify(agent, current_md, modification_text)
    logger.info("[Pipeline] phase6: modified %d chars", len(modified))

    items = phase3_collect(modified)
    answers = await phase4_ask(items, clarification_batch_callback)
    final_md = await phase5_finalize(agent, modified, answers)

    if has_unresolved_tokens(final_md):
        final_md = await phase5b_backfill_defaults(agent, final_md)
        logger.info("[Pipeline] phase5b: backfilled %d chars", len(final_md))

    return _markdown_to_plan(final_md, user_spec)


def _markdown_to_plan(final_md: str, user_spec) -> SimulationPlan:
    """Parse final markdown → SimulationPlan."""
    plan_dict, _ = parse_plan_markdown(final_md)
    plan_dict.setdefault("plan_markdown", final_md)

    token = set_user_spec_context(user_spec)
    try:
        return SimulationPlan(**plan_dict)
    finally:
        set_user_spec_context(None, token=token)
