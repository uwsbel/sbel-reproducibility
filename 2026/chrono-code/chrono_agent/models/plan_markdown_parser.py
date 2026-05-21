"""
Parse the PlanningAgent's fill-in-the-blank markdown output into a dict
suitable for ``SimulationPlan(**dict)``.

Why a markdown parser instead of a strict JSON-schema tool?

The previous design jammed the full ``SimulationPlan.model_json_schema()``
(19 KB of nested $defs) into ``submit_plan``'s ``input_schema``. LLMs'
token distribution is hostile to deeply-nested discriminated unions — they
consistently emit ``vlm: ["hint text"]`` instead of
``vlm: [{"hint": "hint text"}]``, ``constraints: "string"`` instead of
``constraints: ["string"]``, and so on. Every such "shape slip" costs a
repair round.

Mature SWE-agents (Aider, Cursor, Devin, Claude Code's TodoWrite) keep
their tool surfaces narrow and let the LLM emit structured DOCUMENTS that
a deterministic post-processor extracts. This module is that
post-processor for planning.

## The template

Plans are markdown with H2 sections. Each section is either:

1. A YAML code fence (for anything non-trivial — dicts, nested lists),
2. A dash-prefixed list (for flat string lists like ``objectives``), or
3. A single scalar line (for ``plan_type``).

See ``PHASE2_DRAFT_SYSTEM_PROMPT`` in planning_prompts.py for the template.

## Parse errors

``PlanMarkdownParseError`` carries a per-section breakdown so the repair
loop can give the LLM a surgical correction list rather than a generic
"bad YAML".
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# Sections the LLM is allowed to emit. Any unknown `## X` is logged but not
# a hard error — newer LLMs sometimes invent narrative sections, and we'd
# rather preserve them as a debug hint than reject the whole plan.
KNOWN_SECTIONS: Tuple[str, ...] = (
    "plan_type",
    "simulation_parameters",
    "objectives",
    "implementation_steps",
    "scene_objects",
    "assets",
    "objects",  # new unified schema (plan_agent.md §4)
    "topology",
    "visualization",
    "recording_mode",
    "clarifications_needed",
    "geometry_relations",
    "camera",
)

# Sections whose body is expected to be a YAML block.
# ``implementation_steps`` is a YAML list-of-dicts matching the
# ``SimulationStep`` schema (description / assets / camera / constraints).
# ``geometry_relations`` is a YAML list-of-dicts matching ``GeometryRelation``.
YAML_SECTIONS: Tuple[str, ...] = (
    "simulation_parameters",
    "implementation_steps",
    "scene_objects",
    "assets",
    "objects",  # new unified schema (plan_agent.md §4)
    "topology",
    "visualization",
    "geometry_relations",
    "camera",
)

# Sections whose body is expected to be a flat dash-prefixed list of strings.
LIST_SECTIONS: Tuple[str, ...] = (
    "objectives",
)

# Sections that accept EITHER a YAML fence (list of dicts / mixed entries)
# OR a flat dash-prefixed bullet list. Used for ``clarifications_needed``
# so the planner can mix legacy free-text questions with structured
# StructuredClarification (question / options / allow_other) entries
# without forcing the simpler form into a dict shape.
HYBRID_LIST_SECTIONS: Tuple[str, ...] = (
    "clarifications_needed",
)

# Sections whose body is a single scalar value.
SCALAR_SECTIONS: Tuple[str, ...] = ("plan_type", "recording_mode")


class PlanMarkdownParseError(ValueError):
    """Raised when the plan markdown cannot be parsed into a dict.

    The ``field_errors`` attribute follows the same shape as Pydantic's
    ``ValidationError.errors()`` so the repair-loop error formatter can
    reuse its existing layout.
    """

    def __init__(self, message: str, field_errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.field_errors = field_errors or []


_SECTION_HEADER_RE = re.compile(r"^##[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*$", re.MULTILINE)
_YAML_FENCE_RE = re.compile(r"^```(?:yaml|yml)?\s*\n(.*?)\n```", re.DOTALL | re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^[ \t]*-[ \t]+(.*)$")
_YAML_NATURAL_LANGUAGE_KEYS = frozenset({
    "description",
    "role",
    "purpose",
    "notes",
    "note",
    "summary",
})
_YAML_STRING_SCALAR_KEYS = frozenset({
    "predicate",
})
_YAML_IMPLICIT_NON_STRING_VALUES = frozenset({
    "y",
    "yes",
    "n",
    "no",
    "true",
    "false",
    "on",
    "off",
    "null",
    "~",
})
_YAML_KEY_VALUE_RE = re.compile(r"^([ \t]*)([A-Za-z_][A-Za-z0-9_-]*)([ \t]*:[ \t]*)(.*)$")


def _extract_yaml_block(body: str) -> Optional[str]:
    """Return the contents of the first fenced code block in ``body``.

    Handles ``yaml``, ``yml``, and bare triple-backtick fences. Returns
    ``None`` when there is no fenced block — the caller treats the whole
    body as raw YAML in that case (LLMs sometimes skip the fence).
    """
    m = _YAML_FENCE_RE.search(body)
    if m:
        return m.group(1)
    return None


def _parse_list_body(body: str) -> List[str]:
    """Extract ``- item`` lines into a list of stripped strings.

    Lines that are not list items are ignored (preamble / trailing notes
    don't block parsing). An empty result is allowed — the caller decides
    whether that's legitimate for the section.
    """
    items: List[str] = []
    for raw_line in body.splitlines():
        m = _LIST_ITEM_RE.match(raw_line)
        if m:
            text = m.group(1).strip()
            # Drop placeholder markers like "(none)" / "(empty)".
            if text.lower() in ("(none)", "(empty)", "n/a", "none"):
                continue
            if text:
                items.append(text)
    return items


def _parse_scalar_body(body: str) -> str:
    """Take the first non-empty line of ``body`` as the scalar value.

    Tolerates surrounding blank lines and trailing commentary.
    """
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def _quote_yaml_scalar(value: str) -> str:
    return yaml.safe_dump(value, default_style='"').strip()


def _sanitize_yaml_scalars(yaml_text: str) -> str:
    """Quote one-line YAML scalars that LLMs commonly emit unsafely.

    LLMs often emit fields like ``description: Water: density 1000``.
    In YAML, the second ``: `` is syntax unless the value is quoted or a
    block scalar. They also emit values like ``predicate: on``; YAML 1.1
    resolves ``on`` as boolean ``True``, but the plan schema expects a
    string. This sanitizer is intentionally narrow: it only touches known
    free-text/string keys, only on single-line values, and only when the
    value is not already quoted or block-style.
    """
    out: List[str] = []
    for line in yaml_text.splitlines():
        m = _YAML_KEY_VALUE_RE.match(line)
        if not m:
            out.append(line)
            continue

        indent, key, sep, value = m.groups()
        stripped = value.strip()
        if not stripped or stripped[0] in ("'", '"', "|", ">") or stripped.startswith("[") or stripped.startswith("{"):
            out.append(line)
            continue

        if key in _YAML_NATURAL_LANGUAGE_KEYS and ": " in stripped:
            out.append(f"{indent}{key}{sep}{_quote_yaml_scalar(stripped)}")
            continue

        if key in _YAML_STRING_SCALAR_KEYS and stripped.lower() in _YAML_IMPLICIT_NON_STRING_VALUES:
            out.append(f"{indent}{key}{sep}{_quote_yaml_scalar(stripped)}")
            continue

        out.append(line)

    return "\n".join(out)


def _parse_yaml_body(body: str, section: str, errors: List[Dict[str, Any]]) -> Any:
    """Parse a YAML section body. Records parse failures into ``errors``.

    Returns ``None`` on failure so the section is absent from the final
    dict (downstream Pydantic will complain about any required missing
    field — at which point the repair loop fires with the YAML error in
    context).
    """
    yaml_text = _extract_yaml_block(body) or body
    # LLMs routinely mirror the template's visual indent and emit every YAML
    # line with a 4-space prefix. ``.strip()`` only removes leading whitespace
    # of the whole block, so line 1 lands at col 0 while line 2+ stay at col 4
    # — PyYAML then sees an inconsistent mapping and blows up on the second
    # colon ("mapping values are not allowed here"). Dedent per-line before
    # parsing so the parser accepts any consistent indent the LLM chose.
    yaml_text = textwrap.dedent(yaml_text).strip()
    yaml_text = _sanitize_yaml_scalars(yaml_text)
    if not yaml_text:
        return None
    try:
        return yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        # PyYAML's mark-based line numbers are relative to the YAML text,
        # not the whole markdown. That's fine — the LLM sees the section
        # name and a fragment to locate the problem.
        errors.append({
            "loc": (section,),
            "msg": f"YAML parse error: {exc}",
            "type": "yaml_parse_error",
        })
        logger.warning(f"[plan_markdown_parser] YAML parse failed in '{section}': {exc}")
        return None


def parse_plan_markdown(md: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse a plan markdown document into a dict for ``SimulationPlan(**)``.

    Returns ``(parsed_dict, field_errors)`` — non-fatal errors (e.g. a
    single YAML section that failed) are reported without aborting; the
    caller decides whether to raise or attempt construction with the
    partial dict. Missing required fields are surfaced by Pydantic's
    validation, not here — this parser's job is only faithful extraction.
    """
    if not md or not md.strip():
        raise PlanMarkdownParseError(
            "Plan markdown is empty.",
            field_errors=[{
                "loc": ("content",),
                "msg": "Empty markdown body submitted to submit_plan_markdown",
                "type": "empty_body",
            }],
        )

    # Split by ## headers. re.split with a capturing group gives us
    # [preamble, section_name_1, body_1, section_name_2, body_2, ...].
    parts = _SECTION_HEADER_RE.split(md)
    if len(parts) < 3:
        raise PlanMarkdownParseError(
            "No '## section' headers found in plan markdown. Every plan "
            "MUST use the filled-in template with ## headers.",
            field_errors=[{
                "loc": ("structure",),
                "msg": "no ## headers detected",
                "type": "structure_missing_headers",
            }],
        )

    result: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    seen_sections: List[str] = []
    for name, body in zip(parts[1::2], parts[2::2]):
        name = name.strip().lower()
        seen_sections.append(name)
        body_stripped = (body or "").strip()

        if name in YAML_SECTIONS:
            value = _parse_yaml_body(body_stripped, name, errors)
            if value is not None:
                result[name] = value
        elif name in HYBRID_LIST_SECTIONS:
            # Prefer YAML when a fenced block is present (structured
            # entries with question / options keys); fall back to a
            # flat bullet list when it is not (legacy free-text
            # questions). Mixing both shapes inside the same list is
            # allowed because the field type is List[Union[str, dict]].
            value: Any = None
            if _YAML_FENCE_RE.search(body_stripped):
                value = _parse_yaml_body(body_stripped, name, errors)
            if isinstance(value, list):
                result[name] = value
            else:
                result[name] = _parse_list_body(body_stripped)
        elif name in LIST_SECTIONS:
            result[name] = _parse_list_body(body_stripped)
        elif name in SCALAR_SECTIONS:
            result[name] = _parse_scalar_body(body_stripped)
        else:
            # Unknown section — record in errors as a warning-level entry
            # but still try to route it:
            #   * YAML fence present → treat as YAML
            #   * Dash-prefix lines → treat as list
            #   * Otherwise → scalar
            logger.info(
                f"[plan_markdown_parser] unknown section '{name}' — "
                "attempting heuristic parse"
            )
            if _YAML_FENCE_RE.search(body_stripped):
                v = _parse_yaml_body(body_stripped, name, errors)
                if v is not None:
                    result[name] = v
            elif _LIST_ITEM_RE.search(body_stripped):
                result[name] = _parse_list_body(body_stripped)
            else:
                scalar = _parse_scalar_body(body_stripped)
                if scalar:
                    result[name] = scalar

    if not result:
        raise PlanMarkdownParseError(
            "Plan markdown parsed to an empty dict. Sections seen: "
            f"{seen_sections or '(none)'}. Check the template.",
            field_errors=errors + [{
                "loc": ("structure",),
                "msg": f"all sections empty; seen={seen_sections}",
                "type": "empty_result",
            }],
        )

    return result, errors


# ============================================================================
# NEW pipeline: token grammar (Phase 3 collector — see plan_agent.md §3)
# ============================================================================

# Match either:
#   <<ASK_CHOICE: target | question | label1 | label2 | ...>>
#   <<ASK_NUMBER: target | question | unit>>
# Greedy match across pipes; the closing `>>` anchors the end.
TOKEN_RE = re.compile(
    r"<<ASK_(?P<kind>CHOICE|NUMBER):\s*"
    r"(?P<target>[^|>]+?)\s*\|\s*"
    r"(?P<question>[^|>]+?)\s*"
    r"(?:\|\s*(?P<rest>.+?)\s*)?>>",
    re.DOTALL,
)

def collect_clarification_tokens(markdown: str) -> List[Dict[str, Any]]:
    """Extract all <<ASK_*>> tokens from a Phase 2 draft markdown.

    Returns a deduplicated list (first occurrence per ``target_field`` wins),
    each entry shaped as::

        {
            "kind": "choice" | "number",
            "target_field": "objects[plate].pose.position.z",
            "question": "What is the plate's z-position?",
            # choice only:
            "options": ["bottom_flush_water", "center_at_water"],
            # number only:
            "unit": "m",
        }

    Raises ``PlanMarkdownParseError`` when a token violates the protocol:
    - ASK_CHOICE option label contains digits
    - ASK_CHOICE has fewer than 2 labels
    - ASK_NUMBER missing unit
    """
    items: List[Dict[str, Any]] = []
    seen_targets: set[str] = set()
    errors: List[Dict[str, Any]] = []

    for match in TOKEN_RE.finditer(markdown or ""):
        kind_upper = match.group("kind")
        target = (match.group("target") or "").strip()
        question = (match.group("question") or "").strip()
        rest = (match.group("rest") or "").strip()

        if not target or not question:
            errors.append({
                "loc": ("token",),
                "msg": f"malformed token: target={target!r} question={question!r}",
                "type": "token_malformed",
            })
            continue

        if target in seen_targets:
            continue
        seen_targets.add(target)

        if kind_upper == "CHOICE":
            labels = [seg.strip() for seg in rest.split("|") if seg.strip()]
            if len(labels) < 2:
                errors.append({
                    "loc": ("token", target),
                    "msg": (
                        f"ASK_CHOICE for {target!r} has fewer than 2 labels "
                        f"({labels!r}); choice tokens must offer at least "
                        "two distinct options."
                    ),
                    "type": "ask_choice_too_few_labels",
                })
                continue
            # Reject numeric digits inside labels — that is the whole reason
            # the new pipeline exists.
            for label in labels:
                if re.search(r"\d", label):
                    errors.append({
                        "loc": ("token", target),
                        "msg": (
                            f"ASK_CHOICE label {label!r} for {target!r} "
                            "contains a digit. Labels must be pure category "
                            "names (e.g. 'bottom_flush_water_surface'). Use "
                            "ASK_NUMBER instead if you need a numeric "
                            "answer."
                        ),
                        "type": "ask_choice_label_has_digit",
                    })
                    break
            else:
                items.append({
                    "kind": "choice",
                    "target_field": target,
                    "question": question,
                    "options": labels,
                })
        elif kind_upper == "NUMBER":
            unit = rest
            if not unit:
                errors.append({
                    "loc": ("token", target),
                    "msg": (
                        f"ASK_NUMBER for {target!r} is missing a unit. Use "
                        "'1' for dimensionless or e.g. 'm', 'kg/m^3'."
                    ),
                    "type": "ask_number_missing_unit",
                })
                continue
            items.append({
                "kind": "number",
                "target_field": target,
                "question": question,
                "unit": unit,
            })

    if errors:
        raise PlanMarkdownParseError(
            f"Plan draft contains {len(errors)} invalid <<ASK_*>> token(s).",
            field_errors=errors,
        )

    return items


def has_unresolved_tokens(markdown: str) -> bool:
    """True if the markdown still contains any <<ASK_*>> token."""
    return bool(TOKEN_RE.search(markdown or ""))
