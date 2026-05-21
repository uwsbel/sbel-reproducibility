"""
Body name classification for stability exemption.

The scene-placement stability check flags every body whose end-of-sim
linear/angular velocity exceeds a tolerance. For intentionally-moving
assets (robots, rovers, manipulators), that produces noisy failures on
navigation steps. This module centralizes the logic that decides which
body names should be excluded from the stability check.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

ROBOT_LINK_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^(FL|FR|RL|RR)_"),
    re.compile(r"_(hip|thigh|calf|foot|rotor)$"),
    re.compile(r"^calflower\d*$", re.IGNORECASE),
    re.compile(r"_calflower\d*$", re.IGNORECASE),
    re.compile(r"^Head_", re.IGNORECASE),
    re.compile(r"^base(_link)?$", re.IGNORECASE),
    re.compile(r"^trunk$", re.IGNORECASE),
    re.compile(r"^imu", re.IGNORECASE),
    re.compile(r"^radar", re.IGNORECASE),
    re.compile(r"^front_camera", re.IGNORECASE),
    re.compile(r"_wheel(_link)?$", re.IGNORECASE),
    re.compile(r"^(left|right)_(arm|leg|gripper)", re.IGNORECASE),
)


def _normalize(name: str) -> str:
    return name.strip().lower()


def is_dynamic_link(body_name: str, exclusion_roots: Iterable[str] = ()) -> bool:
    """Return True if ``body_name`` should be skipped by the stability check.

    A body is considered dynamic when any of the following holds:

    * its name starts with one of ``exclusion_roots`` (URDF parsers
      typically namespace links as ``<root>/<link>`` or ``<root>_<link>``,
      so we match both separators plus bare-root equality), OR
    * its name matches any of ``ROBOT_LINK_PATTERNS``.
    """
    if not body_name:
        return False
    name = _normalize(body_name)
    for root in exclusion_roots:
        if not root:
            continue
        r = _normalize(root)
        if name == r or name.startswith(f"{r}/") or name.startswith(f"{r}_") or name.startswith(f"{r}."):
            return True
    for pat in ROBOT_LINK_PATTERNS:
        if pat.search(body_name):
            return True
    return False


def filter_stable_bodies(
    bodies: Dict[str, Dict[str, float]],
    exclusion_roots: Iterable[str] = (),
) -> Dict[str, Dict[str, float]]:
    """Return a copy of ``bodies`` with dynamic links removed."""
    roots = tuple(exclusion_roots or ())
    return {n: row for n, row in bodies.items() if not is_dynamic_link(n, roots)}


def split_dynamic_static(
    bodies: Dict[str, Dict[str, float]],
    exclusion_roots: Iterable[str] = (),
) -> tuple[Dict[str, Dict[str, float]], Set[str]]:
    """Return ``(static_bodies_dict, dynamic_names_set)``."""
    roots = tuple(exclusion_roots or ())
    static: Dict[str, Dict[str, float]] = {}
    dynamic: Set[str] = set()
    for name, row in bodies.items():
        if is_dynamic_link(name, roots):
            dynamic.add(name)
        else:
            static[name] = row
    return static, dynamic
