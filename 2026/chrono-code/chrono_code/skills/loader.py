"""
Agent Skills metadata and SKILL.md loader.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import strictyaml


@dataclass
class SkillProperties:
    """Metadata parsed from a SKILL.md frontmatter block."""

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


def find_skill_md(skill_dir: Path) -> Optional[Path]:
    """Find SKILL.md in a skill directory."""
    for candidate in ("SKILL.md", "skill.md"):
        path = skill_dir / candidate
        if path.exists():
            return path
    return None


def _parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Parse YAML frontmatter and markdown body from SKILL.md content."""
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter delimiter '---'")

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter delimiter '---'")

    closing_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break
    if closing_idx is None:
        raise ValueError("SKILL.md frontmatter must be closed with '---'")

    frontmatter_str = "\n".join(lines[1:closing_idx])
    body = "\n".join(lines[closing_idx + 1:]).strip()
    try:
        parsed = strictyaml.load(frontmatter_str)
    except strictyaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter: {exc}") from exc

    metadata = parsed.data
    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")

    return metadata, body


def parse_skill_md(skill_dir: Path) -> Tuple[SkillProperties, str]:
    """
    Parse SKILL.md and return (properties, markdown_instructions).
    """
    skill_md = find_skill_md(skill_dir)
    if skill_md is None:
        raise ValueError(f"Missing SKILL.md in: {skill_dir}")

    metadata, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))

    name = metadata.get("name", "")
    description = metadata.get("description", "")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Field 'name' must be a non-empty string")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Field 'description' must be a non-empty string")

    # Agent Skills naming expectation: directory name matches skill name.
    if skill_dir.name != name.strip():
        raise ValueError(
            f"Skill directory '{skill_dir.name}' must match skill name '{name.strip()}'"
        )

    raw_meta = metadata.get("metadata", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, dict):
        raise ValueError("Field 'metadata' must be a mapping when present")

    normalized_meta = {str(k): str(v) for k, v in raw_meta.items()}
    props = SkillProperties(
        name=name.strip(),
        description=description.strip(),
        license=metadata.get("license"),
        compatibility=metadata.get("compatibility"),
        metadata=normalized_meta,
    )
    return props, body
