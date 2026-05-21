"""
Runtime registry for Agent Skills in Chrono-Agent.
"""

from __future__ import annotations

import importlib
import logging
import re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.sax.saxutils import escape

from chrono_agent.models.handoff import SkillBundle
from chrono_agent.skills.base import ChronoSkill, SkillIssue
from chrono_agent.skills.loader import parse_skill_md

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Singleton-style registry for loaded runtime skills."""

    _skills: Dict[str, ChronoSkill] = {}
    _discovered: bool = False
    _class_to_skills: Dict[str, List[str]] = {}   # "ChAABB" -> ["mbs/system_create", ...]
    _method_to_skills: Dict[str, List[str]] = {}   # "GetBoundingBox" -> ["mbs/system_create", ...]

    @classmethod
    def register(cls, skill: ChronoSkill) -> None:
        key = skill.canonical_name
        if key in cls._skills:
            raise ValueError(f"Duplicate skill registration for '{key}'")
        cls._skills[key] = skill

    @classmethod
    def discover(cls, skills_dir: Path) -> None:
        if cls._discovered:
            return
        if not skills_dir.exists():
            logger.debug("Skills directory does not exist: %s", skills_dir)
            cls._discovered = True
            return

        def _load_one(skill_dir: Path) -> None:
            try:
                props, instructions = parse_skill_md(skill_dir)
                rel_parts = skill_dir.relative_to(skills_dir).parts
                canonical_name = "/".join(rel_parts)
                skill_py = skill_dir / "skill.py"
                if skill_py.exists():
                    module_path = "chrono_agent.skills." + ".".join(rel_parts) + ".skill"
                    module = importlib.import_module(module_path)
                    if not hasattr(module, "create_skill"):
                        raise ValueError(f"Skill module {module_path} must expose create_skill()")
                    skill = module.create_skill(props, instructions, skill_dir)
                    if not isinstance(skill, ChronoSkill):
                        raise ValueError(f"create_skill() for '{props.name}' did not return ChronoSkill")
                    skill.canonical_name = canonical_name
                else:
                    skill = ChronoSkill(props, instructions, skill_dir, canonical_name=canonical_name)
                cls.register(skill)
                logger.info("Registered runtime skill: %s", canonical_name)
            except Exception as exc:
                logger.warning("Failed to load skill from %s: %s", skill_dir, exc)

        for skill_md in sorted(skills_dir.rglob("SKILL.md")):
            skill_dir = skill_md.parent
            rel_parts = skill_dir.relative_to(skills_dir).parts
            if any(part.startswith("_") for part in rel_parts):
                continue
            _load_one(skill_dir)
        cls._build_reverse_index()
        cls._discovered = True

    @classmethod
    def get_available_skills_xml(cls) -> str:
        if not cls._skills:
            return "<available_skills></available_skills>"
        lines = ["<available_skills>"]
        for skill in cls._skills.values():
            skill_md_path = skill.skill_dir / "SKILL.md"
            lines.append("  <skill>")
            lines.append(f"    <name>{escape(skill.canonical_name)}</name>")
            lines.append(f"    <description>{escape(skill.properties.description)}</description>")
            lines.append(f"    <location>{escape(str(skill_md_path.resolve()))}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    @classmethod
    def get_all_skill_names(cls) -> List[str]:
        return sorted(cls._skills.keys())

    @classmethod
    def format_skill_directory(
        cls,
        exclude: Optional[set] = None,
        with_sections: bool = False,
    ) -> str:
        """Render every available skill as ``name — description`` lines.

        Goes into the codegen system prompt so the model can pick what to
        ``read_skill(name)`` itself, rather than relying on a keyword router
        upstream. ``exclude`` lets callers drop skills that are already
        pre-injected (e.g. core + declared required children) so the
        directory only advertises the on-demand tail.

        One stable-ordered line per skill keeps the cached system-prompt
        prefix stable across runs (important for prompt-cache hits on the
        Anthropic path).

        When ``with_sections`` is True, each skill entry is followed by a
        ``Sections:`` line listing the lowercase section keys in document
        order. This is the format the LLM skill router consumes — it lets
        the router pick per-skill section subsets, so the codegen prompt
        only carries the Patterns / Hard Rules relevant to THIS plan
        instead of every skill's full multi-thousand-line contents.
        """
        excl = exclude or set()
        lines: List[str] = []
        for name in sorted(cls._skills.keys()):
            if name in excl:
                continue
            skill = cls._skills[name]
            desc = (skill.properties.description or "").strip()
            desc = " ".join(desc.split())
            lines.append(f"- {name} — {desc}")
            if with_sections:
                section_keys = [
                    k for k in skill.get_sections().keys() if k != "__root__"
                ]
                if section_keys:
                    lines.append(f"  Sections: {', '.join(section_keys)}")
        return "\n".join(lines)

    @classmethod
    def get_skill_fragment(cls, name: str) -> Optional[str]:
        skill = cls._skills.get(name)
        if skill is None:
            return None
        return (skill.get_prompt_fragment() or "").strip() or None

    @classmethod
    def get_skill_summary(cls, name: str, max_chars: int = 800) -> Optional[str]:
        skill = cls._skills.get(name)
        if skill is None:
            return None
        return skill.get_summary(max_chars=max_chars)

    @classmethod
    def get_skill_section(cls, name: str, section_name: str) -> Optional[str]:
        skill = cls._skills.get(name)
        if skill is None:
            return None
        try:
            return skill.get_section(section_name)
        except KeyError:
            return None

    @classmethod
    def list_sections(cls, name: str) -> Optional[List[str]]:
        """Return section heading names for a skill (or None if skill unknown).

        Returned in document order (NOT sorted) so the LLM router sees the
        same sequence the human-authored skill flows in. Stable order also
        keeps router prompt cache hits consistent.
        """
        skill = cls._skills.get(name)
        if skill is None:
            return None
        keys = [k for k in skill.get_sections().keys() if k != "__root__"]
        return keys

    @classmethod
    def render_sections(
        cls,
        name: str,
        section_keys: List[str],
    ) -> Optional[str]:
        """Render selected sections of a skill as a single markdown block.

        Used by the codegen agent to inject only the LLM-router-chosen
        sections (instead of FULL/MEDIUM/COMPACT whole-file tiers).
        Skips section keys that don't resolve, so a typo in the router
        output degrades the section injection rather than failing it.
        """
        skill = cls._skills.get(name)
        if skill is None:
            return None
        sections = skill.get_sections()
        seen: set = set()
        chunks: List[str] = [f"## Skill: {skill.canonical_name} [SECTIONED]"]
        if skill.properties.description:
            chunks.append(f"Description: {skill.properties.description}")
        for raw_key in section_keys:
            key = (raw_key or "").strip().lower()
            if not key or key in seen:
                continue
            content = sections.get(key)
            if content is None:
                # Try a substring fallback so router can use shortened keys
                # (e.g. "pattern d" vs "pattern d — floating body fsi registration").
                for full_key, body in sections.items():
                    if full_key == "__root__":
                        continue
                    if key in full_key:
                        content = body
                        key = full_key
                        break
            if content is None:
                continue
            seen.add(key)
            chunks.append(content)
        if len(chunks) <= 2:  # nothing matched
            return None
        return "\n\n".join(chunks).strip()

    @classmethod
    def get_skill_api_contract(cls, name: str) -> Optional[Dict[str, List[str]]]:
        skill = cls._skills.get(name)
        if skill is None:
            return None
        contract = skill.get_api_contract()
        return contract or None

    # ── Reverse index: error → skill mapping ───────────────────────────────

    @classmethod
    def _build_reverse_index(cls) -> None:
        """Build reverse indexes from API contracts: class/method -> skill names."""
        cls._class_to_skills.clear()
        cls._method_to_skills.clear()

        for name, skill in cls._skills.items():
            contract = skill.get_api_contract() or {}

            # Index allowed_classes: extract bare class name from "chrono.ChFoo"
            for class_entry in (contract.get("allowed_classes") or []):
                bare = class_entry.split(".")[-1].strip()
                if bare:
                    cls._class_to_skills.setdefault(bare, []).append(name)

            for method_entry in (contract.get("allowed_methods") or []):
                # Extract method name from patterns like "body.SetMass(mass)"
                m = _re.match(r'(?:\w+\.)?(\w+)\s*\(', method_entry)
                if m:
                    cls._method_to_skills.setdefault(m.group(1), []).append(name)
                # Also extract class references like "ChCollisionSystem"
                for class_ref in _re.findall(r'Ch\w+', method_entry):
                    cls._class_to_skills.setdefault(class_ref, []).append(name)

            # Index canonical_examples for class references
            for example in (contract.get("canonical_examples") or []):
                for class_ref in _re.findall(r'Ch\w+', example):
                    cls._class_to_skills.setdefault(class_ref, []).append(name)

        # Deduplicate
        cls._class_to_skills = {k: list(dict.fromkeys(v)) for k, v in cls._class_to_skills.items()}
        cls._method_to_skills = {k: list(dict.fromkeys(v)) for k, v in cls._method_to_skills.items()}

        logger.info(
            "Skill reverse index: %d class entries, %d method entries",
            len(cls._class_to_skills), len(cls._method_to_skills),
        )

    @classmethod
    def find_skills_for_error(
        cls,
        class_name: Optional[str] = None,
        method_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find skills relevant to a validator error involving class_name.method_name.

        Returns list of dicts: ``{skill_name, api_contract, match_source}``.
        Deduplicated and limited to 3 most relevant.
        """
        candidate_skills: Dict[str, str] = {}  # skill_name -> match_source

        if class_name:
            bare = class_name.split(".")[-1]
            for skill_name in cls._class_to_skills.get(bare, []):
                candidate_skills[skill_name] = f"class:{bare}"

        if method_name:
            for skill_name in cls._method_to_skills.get(method_name, []):
                source = candidate_skills.get(skill_name, "")
                candidate_skills[skill_name] = (
                    f"{source}+method:{method_name}" if source else f"method:{method_name}"
                )

        results: List[Dict[str, Any]] = []
        for skill_name, source in candidate_skills.items():
            contract = cls.get_skill_api_contract(skill_name)
            if contract is None:
                continue
            score = source.count("+") + 1
            results.append({
                "skill_name": skill_name,
                "api_contract": contract,
                "match_source": source,
                "_score": score,
            })

        results.sort(key=lambda r: -r["_score"])
        for r in results:
            r.pop("_score", None)
        return results[:3]

    @classmethod
    def search(cls, query: str, domain: Optional[str] = None, limit: int = 8) -> List[Dict[str, str]]:
        q = (query or "").strip().lower()
        normalized_domain = (domain or "").strip().lower().rstrip("/")
        hits: List[tuple[int, ChronoSkill]] = []
        for name, skill in cls._skills.items():
            if normalized_domain and not name.startswith(normalized_domain + "/") and name != normalized_domain:
                continue
            contract = skill.get_api_contract() or {}
            contract_values = [
                str(item).lower()
                for key in ("allowed_classes", "allowed_methods", "canonical_examples")
                for item in (contract.get(key) or [])
            ]
            haystacks = [
                name.lower(),
                skill.properties.description.lower(),
                skill.get_summary(max_chars=1200).lower(),
                *contract_values,
            ]
            score = 0
            if q:
                for haystack in haystacks:
                    if q in haystack:
                        score += 5 if haystack in contract_values else 3
                for token in q.split():
                    if any(token in haystack for haystack in haystacks):
                        score += 2 if any(token in haystack for haystack in contract_values) else 1
            else:
                score = 1
            if score > 0:
                hits.append((score, skill))
        hits.sort(key=lambda item: (-item[0], item[1].canonical_name))
        results: List[Dict[str, str]] = []
        for score, skill in hits[: max(1, int(limit))]:
            results.append(
                {
                    "name": skill.canonical_name,
                    "description": skill.properties.description,
                    "summary": skill.get_summary(max_chars=400),
                    "score": str(score),
                }
            )
        return results

    @classmethod
    def _render_compact_skill_fragment(cls, skill: ChronoSkill) -> str:
        summary = skill.get_summary(max_chars=280)
        contract = skill.get_api_contract() or {}
        lines = [f"## Skill: {skill.canonical_name} [COMPACT]", f"Summary: {summary}"]
        for key in ("allowed_classes", "allowed_methods", "canonical_examples"):
            values = contract.get(key) or []
            if not values:
                continue
            lines.append(f"{key}: {'; '.join(values[:6])}")
        return "\n".join(lines)

    @classmethod
    def _render_medium_skill_fragment(cls, skill: ChronoSkill, max_chars: int = 4000) -> str:
        """Render skill with full API contract and key rules/pitfalls (no full prose)."""
        lines = [f"## Skill: {skill.canonical_name} [MEDIUM]"]
        lines.append(f"Description: {skill.properties.description}")

        # Full API contract (no truncation)
        contract = skill.get_api_contract() or {}
        for key in ("allowed_classes", "allowed_methods", "canonical_examples"):
            values = contract.get(key) or []
            if values:
                lines.append(f"\n{key}:")
                for v in values:
                    lines.append(f"  - {v}")

        # Extract pitfall/rule/warning sections
        _PITFALL_KEYWORDS = ("pitfall", "warning", "avoid", "do not", "common mistake", "gotcha", "rule")
        sections = skill.get_sections()
        pitfall_keys = [
            k for k in sections
            if k != "__root__" and any(w in k for w in _PITFALL_KEYWORDS)
        ]
        for key in pitfall_keys[:3]:
            section_text = sections[key][:800]
            lines.append(f"\n### {key}")
            lines.append(section_text)

        result = "\n".join(lines)
        return result[:max_chars]

    @classmethod
    def _render_full_skill_fragment(cls, skill: ChronoSkill) -> str:
        """Render full skill content for tier-1 injection."""
        return f"## Skill: {skill.canonical_name} [FULL]\n\n{skill.get_prompt_fragment().strip()}"

    @classmethod
    def build_bundle(
        cls,
        names: List[str],
        bundle_name: str = "selected",
        *,
        include_full_text: bool = False,
        primary_skills: Optional[List[str]] = None,
    ) -> SkillBundle:
        ordered: List[str] = []
        seen: Set[str] = set()
        primary_summaries: List[str] = []
        secondary_summaries: List[str] = []
        full_text_parts: List[str] = []
        section_index: Dict[str, List[str]] = {}
        api_contracts: Dict[str, Dict[str, Any]] = {}
        primary = [name for name in (primary_skills or []) if name in names]
        primary_set = set(primary)
        for name in names:
            skill = cls._skills.get(name)
            if skill is None or name in seen:
                continue
            seen.add(name)
            ordered.append(name)
            compact = cls._render_compact_skill_fragment(skill)
            if name in primary_set:
                primary_summaries.append(compact)
            else:
                secondary_summaries.append(f"- {name}: {skill.get_summary(max_chars=220)}")
            if include_full_text:
                full_text_parts.append(f"## Skill: {name}\n\n{skill.get_prompt_fragment().strip()}")
            section_index[name] = sorted(key for key in skill.get_sections().keys() if key != "__root__")
            contract = skill.get_api_contract() or {}
            if contract:
                api_contracts[name] = contract
        summary_parts: List[str] = []
        if primary_summaries:
            summary_parts.append("## Primary Skills\n\n" + "\n\n".join(primary_summaries))
        if secondary_summaries:
            summary_parts.append("## Secondary Skills\n" + "\n".join(secondary_summaries))
        return SkillBundle(
            name=bundle_name,
            skills=ordered,
            summary="\n\n".join(summary_parts).strip(),
            full_text="\n\n".join(full_text_parts),
            section_index=section_index,
            metadata={
                "count": len(ordered),
                "primary_skills": primary,
                "secondary_skills": [name for name in ordered if name not in primary_set],
                "api_contracts": api_contracts,
            },
        )

    @classmethod
    def get_prompt_fragments_for(
        cls,
        names: List[str],
        primary_skills: Optional[List[str]] = None,
        max_full_text: int = 3,
    ) -> str:
        """Render skill fragments with tiered injection.

        - Tier 1 (FULL): first ``max_full_text`` primary skills get complete content.
        - Tier 2 (MEDIUM): remaining primary skills get API contract + pitfalls.
        - Tier 3 (COMPACT): secondary skills get one-line summary + tool hint.
        """
        bundle = cls.build_bundle(names, bundle_name="prompt", primary_skills=primary_skills)
        if bundle.skills:
            logger.info("Skills selectively injected: %s", ", ".join(bundle.skills))
        primary = bundle.metadata.get("primary_skills") or []
        secondary = bundle.metadata.get("secondary_skills") or []

        parts: List[str] = []

        # Tier 1: full text for top primary skills
        full_names = primary[:max_full_text]
        for name in full_names:
            skill = cls._skills.get(name)
            if skill is not None:
                parts.append(cls._render_full_skill_fragment(skill))

        # Tier 2: medium rendering for remaining primary skills
        for name in primary[max_full_text:]:
            skill = cls._skills.get(name)
            if skill is not None:
                parts.append(cls._render_medium_skill_fragment(skill))

        # Tier 3: compact + tool hint for secondary skills
        if secondary:
            parts.append("## Secondary Skills (call read_skill(name) before using)")
            for name in secondary:
                skill = cls._skills.get(name)
                if skill is not None:
                    parts.append(f"- {name}: {skill.get_summary(max_chars=220)}")

        tier_info = f"full={len(full_names)} medium={len(primary) - len(full_names)} compact={len(secondary)}"
        logger.info("Tiered skill injection: %s", tier_info)
        return "\n\n".join(parts).strip()

    @classmethod
    def get_prompt_fragments(cls) -> str:
        bundle = cls.build_bundle(cls.get_all_skill_names(), bundle_name="all")
        if bundle.skills:
            logger.info("Skills used in code generation prompt: %s", ", ".join(bundle.skills))
        else:
            logger.info("Skills used in code generation prompt: (none contributed)")
        parts = [cls._render_compact_skill_fragment(cls._skills[name]) for name in bundle.skills if name in cls._skills]
        return "\n\n".join(parts).strip()

    @classmethod
    def validate_all(cls, code: str) -> Tuple[bool, List[SkillIssue]]:
        all_issues: List[SkillIssue] = []
        for skill in cls._skills.values():
            try:
                _, issues = skill.validate(code)
                all_issues.extend(issues)
            except Exception as exc:
                all_issues.append(
                    SkillIssue(
                        skill_name=skill.canonical_name,
                        category="skill_validate_exception",
                        severity="warning",
                        message=f"Skill validate failed: {exc}",
                        auto_fixable=False,
                    )
                )
        is_valid = not any(issue.severity == "critical" for issue in all_issues)
        return is_valid, all_issues

    @classmethod
    def inject_all(cls, code: str) -> Tuple[str, List[str]]:
        patched = code
        logs: List[str] = []
        invoked: List[str] = []
        for skill in cls._skills.values():
            invoked.append(skill.canonical_name)
            try:
                patched, count = skill.inject(patched)
                if count > 0:
                    logs.append(f"{skill.canonical_name}: injected {count} section(s)")
            except Exception as exc:
                logs.append(f"{skill.canonical_name}: injection failed ({exc})")
        logger.info("Skills invoked during code injection: %s", ", ".join(invoked))
        return patched, logs
