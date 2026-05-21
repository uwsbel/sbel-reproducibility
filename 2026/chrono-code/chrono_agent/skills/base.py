"""
Chrono-Agent runtime extensions for Agent Skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chrono_agent.skills.loader import SkillProperties


@dataclass
class SkillIssue:
    """Issue raised by a runtime skill validation."""

    skill_name: str
    category: str
    severity: str
    message: str
    auto_fixable: bool = False


class ChronoSkill:
    """
    Base class for Chrono-Agent runtime skills.

    A skill is loaded from a standards-compliant SKILL.md file and can optionally
    provide deterministic validate/inject hooks for generated code.
    """

    def __init__(self, properties: SkillProperties, instructions: str, skill_dir: Path, canonical_name: str | None = None):
        self.properties = properties
        self.instructions = instructions
        self.skill_dir = skill_dir
        self.canonical_name = canonical_name or properties.name
        self._section_cache: Dict[str, str] | None = None
        self._summary_cache: str | None = None

    def get_prompt_fragment(self) -> str:
        """Return full instructions for prompt injection."""
        return self.instructions

    def get_summary(self, max_chars: int = 800) -> str:
        """Return a compact summary for bundle/index use."""
        if self._summary_cache is None:
            summary = self.properties.description.strip()
            paragraphs = [p.strip() for p in self.instructions.split("\n\n") if p.strip()]
            for para in paragraphs:
                if para.startswith("#"):
                    continue
                summary = f"{summary} {para}".strip()
                if len(summary) >= max_chars:
                    break
            self._summary_cache = " ".join(summary.split())[:max_chars]
        return self._summary_cache[:max_chars]

    def get_sections(self) -> Dict[str, str]:
        """Parse markdown headings into section text blocks."""
        if self._section_cache is not None:
            return self._section_cache

        sections: Dict[str, List[str]] = {}
        current = "__root__"
        sections[current] = []
        for line in self.instructions.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("### "):
                current = stripped.lstrip("#").strip().lower()
                sections.setdefault(current, [])
            sections.setdefault(current, []).append(line)

        self._section_cache = {
            name: "\n".join(lines).strip()
            for name, lines in sections.items()
            if "\n".join(lines).strip()
        }
        return self._section_cache

    def get_section(self, section_name: str) -> str:
        """Return a section by normalized heading name."""
        key = (section_name or "").strip().lower()
        if not key:
            return self.instructions
        sections = self.get_sections()
        if key in sections:
            return sections[key]
        for name, content in sections.items():
            if key in name:
                return content
        raise KeyError(f"Section '{section_name}' not found in {self.canonical_name}")

    def get_required_skills(self) -> List[str]:
        """Parse required child skill names from 'Required Skills' table(s).

        Scans all markdown tables under headings containing 'required skills'.
        Extracts backtick-quoted skill names from the first column.
        Returns a deduplicated list preserving first-occurrence order.
        """
        import re

        sections = self.get_sections()
        required: List[str] = []
        seen: set = set()
        for heading, content in sections.items():
            if "required skill" not in heading:
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                # Skip table header/separator rows
                cells = [c.strip() for c in stripped.split("|")]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                if cells[0].startswith("---") or cells[0].lower() == "skill":
                    continue
                # Extract backtick-quoted skill name
                match = re.search(r"`([^`]+)`", cells[0])
                if match:
                    name = match.group(1)
                    # Skip dynamic/placeholder entries like "owning scene/* skill"
                    if name.startswith("owning") or "*" in name:
                        continue
                    if name not in seen:
                        seen.add(name)
                        required.append(name)
        return required

    def get_api_contract(self) -> Dict[str, Any]:
        """Return a lightweight API contract parsed from an optional skill section."""
        try:
            raw = self.get_section("api contract")
        except KeyError:
            return {}

        contract: Dict[str, Any] = {}
        current_key: str | None = None
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                current_key = stripped[:-1].strip().lower().replace(" ", "_")
                contract.setdefault(current_key, [])
                continue
            if stripped.startswith("- ") and current_key:
                contract.setdefault(current_key, []).append(stripped[2:].strip())

        return {k: v for k, v in contract.items() if v}

    def validate(self, code: str) -> Tuple[bool, List[SkillIssue]]:
        return True, []

    def inject(self, code: str) -> Tuple[str, int]:
        return code, 0

    def load_asset(self, relative_path: str) -> str:
        path = self.skill_dir / "assets" / relative_path
        return path.read_text(encoding="utf-8")

    def load_reference(self, relative_path: str) -> str:
        path = self.skill_dir / "references" / relative_path
        return path.read_text(encoding="utf-8")
