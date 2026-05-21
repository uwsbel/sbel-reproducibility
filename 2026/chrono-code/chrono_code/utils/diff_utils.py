"""Utilities for unified diff generation and parsing.

The apply-side helpers (``apply_unified_patch`` / ``apply_unified_patch_soft``
/ ``PatchApplyError`` / ``PatchApplyResult``) were removed when
``apply_patch`` was dropped from the codegen tool harness in favor of
``edit_file`` (substring replace) + ``write_file`` (whole-file rewrite).
The motivation is documented in
``dialog-sessions-session-20260429-112754-glittery-pixel.md``: multi-hunk
application produced too many failure modes (line-number drift, partial-
apply rollback, agent-fallback hunk drops) for too little gain when
each edit can be expressed as an independent substring replacement.

What this module still provides — and why each survives:

* ``compute_unified_diff`` — used to LOG diffs between successive
  ``current_code`` snapshots in agent transcripts and step handoffs;
  pure rendering, no application semantics.
* ``parse_hunks`` / ``DiffHunk`` — the codegen agent records hunk-level
  metadata in handoff state for the planning agent's "what changed
  between steps" view; consumers don't apply, they inspect.
* ``filter_diff_by_hunk_decisions`` — the plan-diff approval UI keeps
  only the hunks the human approved; output goes back to ``compute_*``
  rendering, never to an applier.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


HUNK_HEADER_RE = re.compile(
    r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@(?:\s*(.*))?$"
)


@dataclass
class DiffHunk:
    """A parsed unified diff hunk."""

    id: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "old_start": self.old_start,
            "old_count": self.old_count,
            "new_start": self.new_start,
            "new_count": self.new_count,
            "header": self.header,
            "lines": self.lines,
        }


def compute_unified_diff(
    old_code: str,
    new_code: str,
    fromfile: str = "a/simulation.py",
    tofile: str = "b/simulation.py",
    context_lines: int = 3,
) -> str:
    """Compute unified diff text between two code strings."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=context_lines,
        lineterm="",
    )
    return "\n".join(diff_lines)


def parse_hunks(diff_text: str) -> List[DiffHunk]:
    """Parse hunks from unified diff text."""
    if not diff_text.strip():
        return []

    lines = diff_text.splitlines()
    hunks: List[DiffHunk] = []
    i = 0
    hunk_index = 1
    while i < len(lines):
        line = lines[i]
        match = HUNK_HEADER_RE.match(line)
        if not match:
            i += 1
            continue

        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        header = match.group(5) or ""
        hunk_lines: List[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            if lines[i].startswith("@@"):
                break
            if lines[i] == r"\ No newline at end of file":
                i += 1
                continue
            hunk_lines.append(lines[i])
            i += 1

        hunks.append(
            DiffHunk(
                id=f"hunk-{hunk_index}",
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                header=header,
                lines=hunk_lines,
            )
        )
        hunk_index += 1

    return hunks


def filter_diff_by_hunk_decisions(
    diff_text: str,
    decisions: Dict[str, str],
    include_statuses: Tuple[str, ...] = ("approved",),
) -> str:
    """Keep only hunks whose decision status is in ``include_statuses``.

    ``decisions`` maps hunk_id → status (e.g. approved/rejected/pending).
    """
    if not diff_text.strip():
        return diff_text

    all_lines = diff_text.splitlines()
    header_lines: List[str] = []
    i = 0
    while i < len(all_lines) and not all_lines[i].startswith("@@"):
        header_lines.append(all_lines[i])
        i += 1

    hunks = parse_hunks(diff_text)
    selected_hunks: List[DiffHunk] = [
        h for h in hunks if decisions.get(h.id, "pending") in include_statuses
    ]

    if not selected_hunks:
        return ""

    out_lines = list(header_lines)
    for h in selected_hunks:
        header_suffix = f" {h.header}" if h.header else ""
        old_count = f",{h.old_count}" if h.old_count != 1 else ""
        new_count = f",{h.new_count}" if h.new_count != 1 else ""
        out_lines.append(
            f"@@ -{h.old_start}{old_count} +{h.new_start}{new_count} @@{header_suffix}"
        )
        out_lines.extend(h.lines)
    return "\n".join(out_lines)
