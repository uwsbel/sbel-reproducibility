"""Structured clarification request used by PlanningAgent.

This module models the data shape for clarifications surfaced to the user.
The new pipeline (see plan_agent.md §3-4) splits clarifications into two
**mutually exclusive** kinds — categorical and numeric — to eliminate the
"options-with-numeric-value" failure mode where the agent fabricated
specific numbers (e.g. z=1.0 / z=1.15) and forced the user to pick one.

Two kinds:
  - ``kind="choice"``: pure multi-choice. ``options`` is a list of label
    strings (no numeric values). Used for relations / sides / orientations
    (e.g. ``bottom_flush_water_surface``, ``side_minus_y``).
  - ``kind="number"``: pure free-form numeric input. ``unit`` carries the
    expected physical unit (``m``, ``kg/m^3``, ``Pa·s``, ``s``, ``1`` for
    dimensionless). No options.

The legacy plain-string clarification (a free-text question with no
options) and the legacy ``StructuredClarification`` (with the embedded
per-option ``value`` field) remain accepted by Pydantic for backward
compatibility while the workflow migrates. New code should emit
``kind="choice"`` or ``kind="number"`` exclusively.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ClarificationOption(BaseModel):
    """One labelled choice the user can pick (CHOICE kind only).

    For the new pipeline, the ``label`` is the only user-facing text and
    must NOT contain numeric values — the agent is forbidden from
    inventing numbers and packaging them as options. ``relation_pattern``
    optionally points at a pattern in the
    ``planning/scene_coordinate_system`` skill so Phase 5 can resolve the
    pattern to numeric coordinates.
    """

    label: str = Field(description="Short text shown to the user (1-5 words).")
    description: str = Field(
        default="",
        description=(
            "Optional one-sentence explanation of what choosing this option "
            "means. Mention the downstream consequence (e.g. 'platform "
            "extends outside the tank') so the user can compare without "
            "reading code."
        ),
    )
    relation_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Optional pattern name from chrono_agent/skills/planning/"
            "scene_coordinate_system/SKILL.md (e.g. 'platform_flush_wall_outer'). "
            "Set when the option resolves a geometry-relation clarification "
            "so Phase 5 can read the matching skill subsection."
        ),
    )
    preview: Optional[str] = Field(
        default=None,
        description="Optional multi-line ASCII / text preview rendered with the option.",
    )


class StructuredClarification(BaseModel):
    """A clarification question — categorical or numeric.

    New shape (preferred):
      - ``kind="choice"`` + ``options: List[str]`` — multi-choice, no numbers
      - ``kind="number"`` + ``unit: str`` — free-form numeric input

    Legacy shape (still accepted):
      - ``options: List[ClarificationOption]`` with embedded values; the
        validator will infer ``kind="choice"`` and convert option records
        to label strings so downstream code only sees the new shape.
    """

    question: str = Field(
        description="One-line question. Should reference involved bodies by name."
    )
    kind: Literal["choice", "number"] = Field(
        default="choice",
        description=(
            "Question kind. 'choice' = multi-choice (use ``options``); "
            "'number' = free-form numeric input (use ``unit``). The two are "
            "mutually exclusive — never mix options with a unit."
        ),
    )
    target_field: Optional[str] = Field(
        default=None,
        description=(
            "Plan-internal dot path the answer fills, e.g. "
            "``objects[plate].pose.position.z`` or "
            "``objects[plate].topology.relation``. Used by Phase 5 to "
            "substitute the answer back into the draft."
        ),
    )
    target_name: Optional[str] = Field(
        default=None,
        description="Entity name when ``target_field`` references a body.",
    )

    # --- choice-only fields ---
    options: List[str] = Field(
        default_factory=list,
        description=(
            "Choice options — label strings only, no numeric values. "
            "Required when kind='choice'; ignored when kind='number'."
        ),
    )
    option_details: List[ClarificationOption] = Field(
        default_factory=list,
        description=(
            "Optional rich option records (with descriptions / "
            "relation_patterns / previews). Aligned 1:1 with ``options`` "
            "when present. Phase 5 reads ``relation_pattern`` here when "
            "resolving categorical answers to numeric coordinates."
        ),
    )

    # --- number-only fields ---
    unit: Optional[str] = Field(
        default=None,
        description=(
            "Physical unit for kind='number' answers (e.g. 'm', 'kg/m^3', "
            "'Pa·s', 's', '1' for dimensionless). Required when "
            "kind='number'; must be None when kind='choice'."
        ),
    )

    # --- legacy/UI fields ---
    allow_other: bool = Field(
        default=False,
        description=(
            "Legacy: when True, the UI appends an 'Other (text input)' "
            "option to a choice question. The new pipeline keeps choice "
            "and number strictly separated, so this defaults to False."
        ),
    )
    body_context: Optional[List[str]] = Field(
        default=None,
        description="Optional body names involved in the question.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_options(cls, data):
        """Migrate the legacy ``List[ClarificationOption]`` shape.

        Old plans / repair flows still emit options as a list of dicts with
        ``label`` / ``description`` / ``value`` / ``relation_pattern``.
        Convert them into the new flat string list (``options``) plus the
        rich record list (``option_details``) so downstream code sees the
        new shape uniformly. The embedded ``value`` field is dropped — the
        new pipeline forbids numeric values inside choice options.
        """
        if not isinstance(data, dict):
            return data
        opts = data.get("options")
        if isinstance(opts, list) and opts and isinstance(opts[0], dict):
            details = []
            labels: List[str] = []
            for entry in opts:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label") or "").strip()
                if not label:
                    continue
                labels.append(label)
                details.append(
                    {
                        "label": label,
                        "description": str(entry.get("description") or ""),
                        "relation_pattern": entry.get("relation_pattern"),
                        "preview": entry.get("preview"),
                    }
                )
            data["options"] = labels
            data["option_details"] = details
            data.setdefault("kind", "choice")
        return data

    @model_validator(mode="after")
    def _validate_kind_consistency(self) -> "StructuredClarification":
        """Enforce kind ↔ (options | unit) consistency.

        choice → options must be non-empty, unit must be None.
        number → unit must be set, options must be empty.
        """
        if self.kind == "choice":
            if not self.options:
                # Allow legacy plain-string clarifications that lacked
                # options entirely; downstream code treats them as text.
                # Only enforce when somebody explicitly set kind='choice'.
                pass
            if self.unit is not None:
                raise ValueError(
                    "kind='choice' must not carry a 'unit'; "
                    "use kind='number' for numeric input."
                )
        elif self.kind == "number":
            if self.options:
                raise ValueError(
                    "kind='number' must not carry 'options'; "
                    "use kind='choice' for multi-choice questions."
                )
            if not self.unit:
                raise ValueError(
                    "kind='number' requires a non-empty 'unit' "
                    "(e.g. 'm', 'kg/m^3', '1' for dimensionless)."
                )
        return self
