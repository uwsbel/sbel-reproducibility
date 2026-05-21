"""AST-driven signature injector (Tier 3b).

Given a piece of generated PyChrono code, extract the set of
chrono/vsg/sens/veh API symbols and project-util function names it
references, look each up in the skill+utils index built by
:mod:`chrono_agent.skills.index_builder`, and format a compact
"Referenced APIs" block suitable for appending to a codegen tool
result.

This is the retrieval-at-generation-time piece of the mature SWE-agent
pattern: don't make the LLM remember which skill describes an API it
just called — tell it, on the next turn, exactly what the authoritative
reference looks like (signature + first doc line + shortest skill
snippet). The LLM self-corrects on the following iteration.

Keep this cheap: pure AST walk + index lookup, no LLM calls.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set

from chrono_agent.skills.index_builder import (
    CHRONO_MODULE_PREFIXES,
    get_or_build_index,
)


# Maximum number of symbol entries we emit per tool result — past this
# the LLM tunes out. We prioritize utils first (highest-value,
# project-specific), then skill-documented APIs.
MAX_INJECTED_ENTRIES = 12


def extract_api_references(code: str) -> Dict[str, Set[str]]:
    """Return ``{"chrono_apis": set[...], "utils": set[...]}`` — the set
    of API symbols and util function names referenced in ``code``.

    ``chrono_apis`` covers dotted calls like ``chrono.ChBody`` or
    ``sens.ChCameraSensor`` at AST ``Attribute``-on-``Name`` sites.
    ``utils`` covers bare ``Name`` call sites that match a known util
    function name (so we pick up both ``setup_preview_camera(...)`` and
    ``scene_assets.add_visual_assets(...)``-style imports).
    """
    chrono_apis: Set[str] = set()
    utils_refs: Set[str] = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"chrono_apis": chrono_apis, "utils": utils_refs}

    # Known util names come from the index; we match call sites against
    # that set so "foo()" from some unrelated module doesn't get pulled
    # in as if it were a project util.
    try:
        idx = get_or_build_index()
        known_utils = set(idx.get("utils", {}).keys())
    except Exception:
        known_utils = set()

    for node in ast.walk(tree):
        # chrono.ChBody / sens.ChCameraSensor style: Attribute.value is Name
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            prefix = node.value.id
            if prefix in CHRONO_MODULE_PREFIXES:
                chrono_apis.add(f"{prefix}.{node.attr}")
        # setup_preview_camera(...), add_visual_assets(...), etc.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in known_utils:
                utils_refs.add(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in known_utils:
                utils_refs.add(func.attr)

    return {"chrono_apis": chrono_apis, "utils": utils_refs}


def format_reference_block(
    code: str,
    max_entries: int = MAX_INJECTED_ENTRIES,
) -> Optional[str]:
    """Format a compact "Referenced APIs" block for ``code``.

    Returns None when nothing relevant was found (e.g. empty file,
    syntax-error bailout, or code uses no indexed symbols) — callers
    should skip appending in that case.
    """
    refs = extract_api_references(code)
    if not refs["chrono_apis"] and not refs["utils"]:
        return None

    try:
        idx = get_or_build_index()
    except Exception:
        return None

    skill_idx = idx.get("skills", {}) or {}
    utils_idx = idx.get("utils", {}) or {}

    lines: List[str] = []
    budget = max_entries

    # Utils first — highest-value, project-specific; signature mistakes
    # here (e.g. setup_preview_camera's attach_body vs cam_body) are the
    # most common class of bugs.
    for name in sorted(refs["utils"]):
        if budget <= 0:
            break
        entry = utils_idx.get(name)
        if not entry:
            continue
        sig = entry.get("signature", "")
        doc = entry.get("docline", "")
        mod = entry.get("module", "")
        lines.append(f"  • {name}{sig}")
        if mod:
            lines.append(f"      from: {mod}")
        if doc:
            lines.append(f"      doc: {doc[:160]}")
        budget -= 1

    # Then chrono APIs — skill-documented symbols get their shortest
    # snippet. Sorted for determinism.
    for symbol in sorted(refs["chrono_apis"]):
        if budget <= 0:
            break
        entries = skill_idx.get(symbol)
        if not entries:
            continue
        # Take the shortest snippet across sources
        best = min(entries, key=lambda e: len(e.get("snippet", "")))
        source = best.get("source", "")
        snippet = best.get("snippet", "")
        if not snippet:
            continue
        # Trim snippet to ~140 chars to keep the block skimmable
        trimmed = snippet if len(snippet) <= 140 else snippet[:137] + "..."
        lines.append(f"  • {symbol}  [{source}]")
        lines.append(f"      {trimmed}")
        budget -= 1

    if not lines:
        return None

    truncated_note = (
        f"\n  (...truncated — {max_entries} references shown; "
        "more may apply)"
        if (len(refs["chrono_apis"]) + len(refs["utils"])) > max_entries
        else ""
    )
    return (
        "=== Referenced APIs (harness-injected signatures) ===\n"
        + "\n".join(lines)
        + truncated_note
    )
