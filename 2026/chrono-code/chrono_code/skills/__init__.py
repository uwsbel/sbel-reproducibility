"""
Chrono-Code runtime skills package.
"""

from pathlib import Path

from chrono_code.skills.registry import SkillRegistry

_skills_dir = Path(__file__).parent
SkillRegistry.discover(_skills_dir)

__all__ = ["SkillRegistry"]
