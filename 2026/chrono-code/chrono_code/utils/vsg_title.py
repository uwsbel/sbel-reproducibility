"""Window-title rewriting for the generated PyChrono ``simulation.py``.

Two responsibilities, run on the script just before subprocess execution:

1. Sync ``vis.SetWindowTitle("Step N ...")`` to the workflow's *actual* current
   step. Codegen may bake a stale ``Step 1`` literal that does not match the
   ``current_step_index`` the step-loop has reached this iteration.
2. ASCII-sanitize the title. The VSG default font font ships only basic ASCII
   glyphs; em-dashes (``\\u2014``), en-dashes, smart quotes, CJK characters,
   etc. render as boxes / mojibake in the title bar.

Allowed characters are ``[A-Za-z0-9 ]`` only — every other byte is replaced
with a single space and consecutive spaces are collapsed. We do not attempt
unicode-to-ascii transliteration; that would require an extra dependency and
the title is a label, not a translation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Strip every char that is not an ASCII letter, digit, or space.
_TITLE_ALLOWED = re.compile(r"[^A-Za-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")

# Match any ``Step <num>`` or ``Step <num> of <num>`` token (case-insensitive)
# anywhere in the title — codegen may write ``Step 1: ...`` or ``... — Step 1``.
# Stripped before re-prepending the canonical ``Step N of M`` so we never end
# up with two competing step counters in the same title.
_STALE_STEP_RE = re.compile(
    r"\b[Ss]tep\s+\d+(?:\s+of\s+\d+)?\s*[:\-]?",
)

_MAX_DESC_LEN = 80

# Match ``<anything>.SetWindowTitle("..." or '...')`` allowing internal escapes.
_SET_WINDOW_TITLE_RE = re.compile(
    r"""(\.SetWindowTitle\s*\(\s*)
        (['"])
        ((?:\\.|(?!\2).)*)
        \2
        (\s*\))
    """,
    re.VERBOSE | re.DOTALL,
)


def sanitize_title_text(text: str) -> str:
    """Reduce *text* to ``[A-Za-z0-9 ]+`` with collapsed runs of whitespace."""
    if not text:
        return ""
    cleaned = _TITLE_ALLOWED.sub(" ", text)
    return _MULTISPACE.sub(" ", cleaned).strip()


def build_window_title(
    step_number: Optional[int],
    total_steps: Optional[int],
    description: str = "",
) -> str:
    """Canonical title used for VSG windows during step-by-step execution.

    Form: ``Step N of M <description>`` when both numbers are given, else just
    the sanitized description. Description is stripped of any embedded
    ``Step <n>`` tokens so the canonical ``Step N of M`` is the only counter
    in the result.
    """
    desc = description or ""
    desc = _STALE_STEP_RE.sub(" ", desc)
    desc = sanitize_title_text(desc)
    if len(desc) > _MAX_DESC_LEN:
        desc = desc[:_MAX_DESC_LEN].rstrip()

    if step_number is not None and total_steps is not None:
        prefix = f"Step {int(step_number)} of {int(total_steps)}"
        return f"{prefix} {desc}".rstrip() if desc else prefix
    return desc


def _decode_python_string_literal(raw: str) -> str:
    """Best-effort decode of escape sequences in a Python string literal body."""
    try:
        return bytes(raw, "utf-8").decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return raw


def rewrite_window_titles(
    code: str,
    step_info: Optional[Dict[str, object]] = None,
) -> Tuple[str, int]:
    """Rewrite every ``vis.SetWindowTitle(...)`` literal in *code*.

    The rewrite always sanitizes the title to ``[A-Za-z0-9 ]``. When
    ``step_info`` is provided (``step_number`` + ``total_steps``), the
    canonical ``Step N of M`` prefix is force-synced — any stale step
    counter codegen baked in is stripped first.

    Returns ``(rewritten_code, replacement_count)``. Replacement count is
    zero when the script has no ``SetWindowTitle`` call (then the original
    code is returned unchanged).
    """
    n_replacements = 0

    step_number: Optional[int] = None
    total_steps: Optional[int] = None
    if step_info:
        sn = step_info.get("step_number")
        ts = step_info.get("total_steps")
        if isinstance(sn, int) and isinstance(ts, int) and sn >= 1 and ts >= 1:
            step_number = sn
            total_steps = ts

    def repl(match: re.Match) -> str:
        nonlocal n_replacements
        n_replacements += 1
        original_literal = match.group(3)
        decoded = _decode_python_string_literal(original_literal)
        new_title = build_window_title(step_number, total_steps, decoded)
        # ``new_title`` is by construction ASCII letters/digits/spaces — safe
        # to embed inside a double-quoted Python string with no escaping.
        return f'{match.group(1)}"{new_title}"{match.group(4)}'

    new_code = _SET_WINDOW_TITLE_RE.sub(repl, code)
    return new_code, n_replacements
