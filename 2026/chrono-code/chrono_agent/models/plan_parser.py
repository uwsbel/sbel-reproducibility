"""
Markdown plan parser — extracts structured fields from plan markdown.

Used by:
- PlanningAgent._parse_markdown_plan (at plan creation time)
- SimulationPlan.coerce_all_fields (at deserialization time, re-hydration)
"""

import ast
import re
from typing import Any, Dict, List, Optional


def _coerce_scalar(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if text[0] in "[{(\"'" or re.fullmatch(r"-?\d+(\.\d+)?", text):
            return ast.literal_eval(text)
    except Exception:
        pass
    return text


def _normalize_heading(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _split_sections(markdown: str, heading_prefix: str = "## ") -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in (markdown or "").splitlines():
        if line.startswith(heading_prefix):
            current = _normalize_heading(line[len(heading_prefix):])
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _parse_simple_bullets(section_text: str) -> Dict[str, str]:
    items: Dict[str, str] = {}
    for raw_line in (section_text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        items[key] = match.group(2).strip()
    return items


def _parse_list_section(section_text: str) -> List[str]:
    results: List[str] = []
    for raw_line in (section_text or "").splitlines():
        line = raw_line.strip()
        if re.match(r"^[-*]\s+", line):
            results.append(re.sub(r"^[-*]\s+", "", line).strip())
        elif re.match(r"^\d+\.\s+", line):
            results.append(re.sub(r"^\d+\.\s+", "", line).strip())
    return [item for item in results if item]


# ---------------------------------------------------------------------------
# Format-agnostic field extraction from broken / mixed LLM output
# ---------------------------------------------------------------------------

def _find_matching_bracket(text: str, start: int) -> int:
    """Return the index of the bracket that closes the one at *start*, or -1."""
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
        elif ch == "\\" and in_str:
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
    return -1


def _extract_json_value(text: str, field_name: str) -> Any:
    """Extract a single JSON value for *field_name* from potentially broken JSON.

    Returns the parsed Python object, or ``None`` if the field cannot be found
    or its value cannot be decoded.  Each field is extracted independently so
    one broken neighbour does not affect others.
    """
    import json as _json

    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*')
    m = pattern.search(text)
    if not m:
        return None

    rest = text[m.end():]
    if not rest:
        return None

    first = rest.lstrip()[0] if rest.lstrip() else ""
    offset = m.end() + (len(rest) - len(rest.lstrip()))

    if first in "[{":
        end = _find_matching_bracket(text, offset)
        if end == -1:
            return None
        fragment = text[offset : end + 1]
        try:
            return _json.loads(fragment)
        except (ValueError, TypeError):
            return None
    elif first == '"':
        str_m = re.match(r'"((?:[^"\\]|\\.)*)"', rest.lstrip())
        if str_m:
            return str_m.group(1)
    elif first in "tfnTFN":
        for literal, val in [("true", True), ("false", False), ("null", None)]:
            if rest.lstrip().lower().startswith(literal):
                return val
    else:
        num_m = re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", rest.lstrip())
        if num_m:
            try:
                return _json.loads(num_m.group(0))
            except (ValueError, TypeError):
                pass
    return None


def extract_plan_fields_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract plan fields individually from any text that looks like a plan.

    Unlike ``json.loads`` this does **not** require the entire blob to be valid
    JSON.  Each known field is located and decoded independently, so one broken
    field (e.g. ``simulation_parameters``) does not prevent recovery of the rest.

    Returns ``None`` when the text does not appear to contain plan data at all.
    """
    text = (raw_text or "").strip()
    # Strip markdown code fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()

    if '"plan_type"' not in text and '"implementation_steps"' not in text:
        return None

    result: Dict[str, Any] = {}

    # --- scalar / string fields ---
    for field in ("plan_type",):
        val = _extract_json_value(text, field)
        if val is not None:
            result[field] = val

    # --- object fields (skip silently when broken) ---
    for field in (
        "simulation_parameters",
        "visualization",
        "topology",
        "camera",
    ):
        val = _extract_json_value(text, field)
        if isinstance(val, dict):
            result[field] = val

    # --- list-of-string fields ---
    for field in (
        "objectives",
        "implementation_steps",
        "scene_objects",
        "clarifications_needed",
    ):
        val = _extract_json_value(text, field)
        if isinstance(val, list):
            if field == "scene_objects":
                result[field] = [s for s in val if isinstance(s, dict)]
            else:
                result[field] = [str(s) for s in val if s]

    # --- assets (list of objects) ---
    val = _extract_json_value(text, "assets")
    if isinstance(val, list):
        result["assets"] = val

    val = _extract_json_value(text, "scene_objects")
    if isinstance(val, list):
        result["scene_objects"] = [s for s in val if isinstance(s, dict)]

    return result if result else None


def extract_implementation_steps(raw_text: str) -> List[str]:
    """Format-agnostic extraction of implementation steps from any LLM output.

    Tries three strategies and returns the first non-empty result:

    1. JSON field extraction via :func:`_extract_json_value`.
    2. Bulleted / numbered lines after a recognisable heading.
    3. Quoted strings inside a broken JSON array fragment.
    """
    text = raw_text or ""

    # --- Strategy 1: structured JSON field ---
    val = _extract_json_value(text, "implementation_steps")
    if isinstance(val, list):
        steps = [str(s).strip() for s in val if s]
        if steps:
            return steps

    # --- Strategy 2: heading + bullet / numbered lines ---
    heading_pat = re.compile(
        r"(?:^|\n)\s*(?:#{1,4}\s+|[*_]{1,2})?"
        r"[Ii]mplementation\s+[Ss]teps[*_]{0,2}\s*:?\s*\n",
    )
    hm = heading_pat.search(text)
    if hm:
        block = text[hm.end() :]
        steps = _parse_list_section(block.split("\n\n")[0])
        if steps:
            return steps

    # --- Strategy 3: quoted strings near the field key ---
    m = re.search(r'"implementation_steps"\s*:\s*\[', text, re.IGNORECASE)
    if m:
        # Grab everything from the opening '[' to the next ']' (or end)
        start = m.end() - 1
        end = _find_matching_bracket(text, start)
        fragment = text[start : end + 1] if end != -1 else text[start : start + 2000]
        found = re.findall(r'"((?:[^"\\]|\\.)+)"', fragment)
        steps = [s.strip() for s in found if len(s.strip()) > 15]
        if steps:
            return steps

    return []


def _parse_named_subsections(section_text: str) -> Dict[str, str]:
    return _split_sections(section_text, heading_prefix="### ")


def _parse_kv_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for part in [p.strip() for p in str(text or "").split(";") if p.strip()]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        pairs[key.strip().lower().replace(" ", "_")] = value.strip()
    return pairs


def _parse_assets_section(section_text: str) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for name, content in _parse_named_subsections(section_text).items():
        bullet_map = _parse_simple_bullets(content)
        asset: Dict[str, Any] = {
            "name": name,
            "type": bullet_map.get("type", "mesh"),
            "filename": bullet_map.get("filename", ""),
            "description": bullet_map.get("description", name),
            "apply_to": bullet_map.get("apply_to", name),
        }
        ideal_height_raw = bullet_map.get("ideal_height")
        if ideal_height_raw:
            parsed = _coerce_scalar(ideal_height_raw)
            if isinstance(parsed, (int, float)):
                asset["ideal_height"] = float(parsed)
        assets.append(asset)
    return assets


def _parse_topology_section(section_text: str) -> Dict[str, Any]:
    bullet_map = _parse_simple_bullets(section_text)
    subsections = _parse_named_subsections(section_text)
    topology: Dict[str, Any] = {
        "gravity_axis": bullet_map.get("gravity_axis", "-z"),
        "working_plane": bullet_map.get("working_plane", "xz"),
        # Markdown plan path is legacy; sensor cameras are now expressed via the
        # JSON `cameras` dict (camera+x/-x/+y/-y). Markdown parsers leave them
        # unset so the validator's coerce_all_fields keeps them None.
        "cameras": None,
        "camera_target": _coerce_scalar(bullet_map.get("camera_target", "[]")) or None,
        "camera_up": _coerce_scalar(bullet_map.get("camera_up", "[]")) or None,
        "body_positions": None,
        "body_geometry": None,
        "joints": None,
        "physical_predicates": None,
        "scene_predicates": None,
    }

    # ---- Scene Predicates subsection ----
    scene_pred_text = subsections.get("scene predicates", "")
    if scene_pred_text:
        sp_bullets = _parse_simple_bullets(scene_pred_text)
        topology["orientation_convention"] = sp_bullets.get("orientation_convention", "y_up_to_z_up")

        physical_predicates: List[Dict[str, str]] = []
        scene_predicates: List[Dict[str, Any]] = []
        in_physical = False
        in_scene = False
        for raw_line in scene_pred_text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("- physical_predicates:"):
                in_physical = True
                in_scene = False
                continue
            if stripped.startswith("- scene_predicates:"):
                in_scene = True
                in_physical = False
                continue
            if stripped.startswith("- ") and not stripped.startswith("  - "):
                in_physical = False
                in_scene = False
                continue
            if not stripped.startswith("- ") or "|" not in stripped:
                continue

            parts = [part.strip() for part in stripped.lstrip("- ").split("|")]
            if in_physical and len(parts) >= 3:
                physical_predicates.append({
                    "subject": parts[0],
                    "predicate": parts[1],
                    "object": parts[2],
                })
            elif in_scene and len(parts) >= 3:
                pred: Dict[str, Any] = {
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2],
                }
                for part in parts[3:]:
                    if ":" not in part:
                        continue
                    key, value = part.split(":", 1)
                    key = key.strip().lower()
                    if key == "position":
                        pred["position"] = _parse_kv_pairs(value)
                    elif key == "orientation":
                        pred["orientation"] = value.strip()
                scene_predicates.append(pred)

        topology["physical_predicates"] = physical_predicates or None
        topology["scene_predicates"] = scene_predicates or None

    # ---- Body Positions subsection ----
    body_positions_text = subsections.get("body positions", "")
    if body_positions_text:
        body_positions: Dict[str, Dict[str, str]] = {}
        for raw_line in body_positions_text.splitlines():
            line = raw_line.strip()
            match = re.match(r"^- ([^:]+):\s*x=(.*?);\s*y=(.*?);\s*z=(.*)$", line)
            if not match:
                continue
            body_positions[match.group(1).strip()] = {
                "x": match.group(2).strip(),
                "y": match.group(3).strip(),
                "z": match.group(4).strip(),
            }
        topology["body_positions"] = body_positions or None

    # ---- Body Geometry subsection ----
    body_geometry_text = subsections.get("body geometry", "")
    if body_geometry_text:
        body_geometry: Dict[str, Dict[str, float]] = {}
        for raw_line in body_geometry_text.splitlines():
            line = raw_line.strip()
            match = re.match(r"^- ([^:]+):\s*length=(.*?);\s*radius=(.*)$", line)
            if not match:
                continue
            body_geometry[match.group(1).strip()] = {
                "length": float(_coerce_scalar(match.group(2).strip()) or 0.0),
                "radius": float(_coerce_scalar(match.group(3).strip()) or 0.0),
            }
        topology["body_geometry"] = body_geometry or None

    # ---- Joints subsection ----
    joints_text = subsections.get("joints", "")
    if joints_text:
        joints: List[Dict[str, Any]] = []
        for raw_line in joints_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            parts = [part.strip() for part in line[2:].split("|")]
            if len(parts) < 4:
                continue
            joints.append(
                {
                    "body1": parts[0],
                    "body2": parts[1],
                    "type": parts[2],
                    "location": parts[3],
                }
            )
        topology["joints"] = joints or None

    return topology


def infer_plan_type(markdown: str) -> str:
    """Infer plan_type from markdown content when the Plan Meta section is missing or unparseable.

    Checks for scene-indicative signals (assets section, scene predicates,
    physical predicates) vs MBS signals (joints, body geometry). Returns
    the most likely type; defaults to "mbs" only when no signals are found.
    """
    sections = _split_sections(markdown)

    # First try: explicit Plan Meta
    meta = _parse_simple_bullets(sections.get("plan meta", ""))
    explicit = meta.get("plan_type", "").strip().lower()
    if explicit in ("scene", "mbs", "mbs_in_scene"):
        return explicit

    # Heuristic: look at which sections are present
    has_assets = bool(sections.get("assets", "").strip())
    topo_text = sections.get("topology", "")
    has_scene_predicates = "scene_predicates:" in topo_text or "physical_predicates:" in topo_text
    has_mbs_fields = bool(
        sections.get("joints", "").strip()
        or "### Body Positions" in topo_text
        or "### Body Geometry" in topo_text
        or "### Joints" in topo_text
    )

    if has_assets and has_mbs_fields:
        return "mbs_in_scene"
    if has_assets or has_scene_predicates:
        return "scene"
    return "mbs"


def parse_markdown_to_fields(markdown: str, fallback_plan_type: str = "mbs") -> Dict[str, Any]:
    """Parse planner Markdown into a SimulationPlan-compatible dict.

    This is the single source of truth for markdown → structured fields conversion.
    """
    sections = _split_sections(markdown)
    meta = _parse_simple_bullets(sections.get("plan meta", ""))
    simulation_parameters = {
        key: _coerce_scalar(value)
        for key, value in _parse_simple_bullets(sections.get("simulation parameters", "")).items()
    }
    visualization = {
        key: _coerce_scalar(value)
        for key, value in _parse_simple_bullets(sections.get("visualization", "")).items()
    }
    topology = _parse_topology_section(sections.get("topology", "")) if sections.get("topology") else None
    assets = _parse_assets_section(sections.get("assets", "")) if sections.get("assets") else None
    scene_objects = (
        _parse_assets_section(sections.get("scene objects", ""))
        if sections.get("scene objects")
        else None
    )

    # For mbs_in_scene plans, auto-populate body_positions from scene_predicates
    plan_type = str(meta.get("plan_type") or fallback_plan_type)
    if (
        plan_type == "mbs_in_scene"
        and topology is not None
        and not topology.get("body_positions")
        and topology.get("scene_predicates")
    ):
        body_positions_from_scene: Dict[str, Dict[str, str]] = {}
        for sp in topology["scene_predicates"]:
            if isinstance(sp, dict) and sp.get("position"):
                body_positions_from_scene[sp["subject"]] = dict(sp["position"])
        if body_positions_from_scene:
            topology["body_positions"] = body_positions_from_scene

    impl_steps = _parse_list_section(sections.get("implementation steps", ""))
    if not impl_steps:
        impl_steps = extract_implementation_steps(markdown)

    return {
        "plan_type": plan_type,
        "simulation_parameters": simulation_parameters,
        "objectives": _parse_list_section(sections.get("objectives", "")),
        "implementation_steps": impl_steps,
        "clarifications_needed": _parse_list_section(sections.get("clarifications needed", "")),
        "visualization": visualization or {"mode": "vsg_with_sensor_camera", "apis": []},
        "topology": topology,
        "assets": assets,
        "scene_objects": scene_objects or [],
        "plan_markdown": markdown.strip(),
    }
