"""
Code Agent tools for unified tool-calling flow.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chrono_code.config import get_settings
from chrono_code.skills import SkillRegistry
from chrono_code.tools.file_explorer_tools import make_file_explorer_tools
from chrono_code.tools.signature_injector import format_reference_block
# diff_utils still hosts compute_unified_diff / parse_hunks for logging
# and handoff metadata (e.g. hunk summaries in the codegen agent's
# transcript), but the apply-side helpers are no longer reachable from
# the tool harness — apply_patch was removed when edit_file became the
# only mutation primitive.

logger = logging.getLogger(__name__)

DEFAULT_LOGICAL_FILE = "simulation.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_chrono_apis.py"

# In-process Chrono-API validator: load the script as a module exactly once
# per agent process so the heavy `pychrono` import (and its caches) is paid
# once. Set to ``False`` after a failed import so we stop retrying.
_VALIDATE_CHRONO_APIS_MODULE: Any = None  # module | None | False (sentinel: tried and failed)

# Memoize validation results by SHA-256 of the code text. The codegen tool
# loop frequently calls _run_chrono_api_validation on the same content
# (no-op edits, validate-after-write hooks on unchanged regions); a tiny
# FIFO cache turns the repeat calls into instant returns.
_VALIDATION_CACHE: Dict[str, str] = {}
_VALIDATION_CACHE_MAX = 16


def _load_validate_chrono_apis_module():
    """Import ``scripts/validate_chrono_apis.py`` as a Python module.

    The script lives outside the package, so we use ``spec_from_file_location``
    rather than the regular import machinery. Module-level code (imports +
    cache dicts) runs once per agent process — subsequent calls reuse the
    already-imported pychrono and the validator's own ``_cache`` /
    ``_return_type_cache``. Returns the module on success, ``None`` on
    failure (caller falls back to subprocess).
    """
    global _VALIDATE_CHRONO_APIS_MODULE
    if _VALIDATE_CHRONO_APIS_MODULE is False:
        return None
    if _VALIDATE_CHRONO_APIS_MODULE is not None:
        return _VALIDATE_CHRONO_APIS_MODULE
    try:
        spec = importlib.util.spec_from_file_location(
            "_chrono_code_validate_chrono_apis_inproc",
            str(VALIDATOR_SCRIPT_PATH),
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _VALIDATE_CHRONO_APIS_MODULE = module
        logger.info(
            "[chrono_api_validator] loaded in-process; subsequent validations "
            "skip subprocess startup + pychrono re-import"
        )
        return module
    except Exception as exc:
        logger.warning(
            "[chrono_api_validator] in-process load failed (%s); "
            "falling back to subprocess for every validation",
            exc,
        )
        _VALIDATE_CHRONO_APIS_MODULE = False
        return None


def _format_validate_chrono_apis_output(path: str, valid: list, issues: list) -> str:
    """Mirror ``scripts/validate_chrono_apis.py`` ``main()``'s stdout format.

    The downstream ``_run_chrono_api_validation`` parses ``INVALID: <n>`` to
    set the PASS/FAIL banner; in-process and subprocess paths must produce
    byte-identical output here so callers can't tell which path ran.
    """
    parts = [
        f"Validating: {path}",
        "",
        "=" * 70,
        f"VALID: {len(valid)}  |  INVALID: {len(issues)}",
        "=" * 70,
    ]
    if issues:
        parts.append("")
        parts.append(f"{'LINE':<6} {'CHAIN':<45} ERROR")
        parts.append("-" * 70)
        for line, chain, err in issues:
            parts.append(f"{line:<6} {chain:<45}")
            parts.append(f"       → {err}")
            parts.append("")
    return "\n".join(parts)


def _validation_cache_put(code_sha: str, result_text: str) -> None:
    """Insert into the FIFO cache, evicting the oldest entry on overflow."""
    if code_sha in _VALIDATION_CACHE:
        # Refresh insertion order so MRU stays warm
        _VALIDATION_CACHE.pop(code_sha)
    elif len(_VALIDATION_CACHE) >= _VALIDATION_CACHE_MAX:
        _VALIDATION_CACHE.pop(next(iter(_VALIDATION_CACHE)))
    _VALIDATION_CACHE[code_sha] = result_text


def _resolve_validator_python() -> str:
    """Choose a Python interpreter that can import the project's Chrono bindings."""
    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")

    # Common local env path for this project.
    candidates.append(Path.home() / "miniconda3" / "envs" / "chrono-code" / "bin" / "python")

    # Fallback to the current interpreter.
    candidates.append(Path(sys.executable))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def _build_validator_env(python_executable: str) -> Dict[str, str]:
    """Build an env that can import pychrono.parsers reliably from subprocesses."""
    env = os.environ.copy()

    python_path = Path(python_executable).resolve()
    env_prefix = python_path.parents[1] if python_path.parent.name == "bin" else Path(sys.prefix)
    lib_dir = env_prefix / "lib"

    ld_parts: List[str] = []
    if lib_dir.exists():
        ld_parts.append(str(lib_dir))

    # Some local installs keep tinyxml2 only in the conda package cache.
    miniconda_root = Path.home() / "miniconda3" / "pkgs"
    if miniconda_root.exists():
        tinyxml_candidates = sorted(miniconda_root.glob("tinyxml2-*/lib"), reverse=True)
        for candidate in tinyxml_candidates:
            if (candidate / "libtinyxml2.so.11").exists():
                ld_parts.append(str(candidate))
                break

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_parts.append(existing_ld)
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    chrono_python_dir = env_prefix / "share" / "chrono" / "python"
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts: List[str] = []
    if chrono_python_dir.exists():
        pythonpath_parts.append(str(chrono_python_dir))
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    if pythonpath_parts:
        env["PYTHONPATH"] = ":".join(pythonpath_parts)

    return env


def _current_simulation_path() -> Path:
    """Return the canonical live simulation path for the current codegen turn.

    Resolution order:
      1. ``_iteration_dir`` set by ``set_iteration_dir(...)`` from
         ``CodeGenerationAgent._execute_with_tools``. This is the
         iteration the CURRENT codegen invocation is writing into.
      2. Fallback: ``settings.visualization_output_path`` for back-compat
         with paths that don't go through codegen (rare).

    Why this order matters (the iter_005/iter_006 collision bug):
    ``settings.visualization_output_path`` is a workflow-level mutable
    field that ``ExecutionAgent.execute`` overwrites at the START of
    each simulation run. After a fix-mode retry, the next codegen
    invocation creates a NEW ``iteration_dir`` (e.g., iter_006), but
    ``settings.visualization_output_path`` still points at the previous
    iter (iter_005, last touched by execution). If we read it here, the
    tool's ``_sync_code`` writes the freshly-edited code into the OLD
    iter dir; the new iter dir's ``simulation.py`` then comes from
    ``ExecutionAgent``'s independent copy of ``generated_code.code``,
    which is the same string but a different path — leaving two
    iteration dirs with byte-identical files and no audit trail of
    what each codegen turn actually changed. Reading
    ``_iteration_dir`` instead pins the tool to the dir THIS codegen
    is logically operating on, which matches ``set_iteration_dir`` and
    the ``skill_read_log.json`` write target.
    """
    if _iteration_dir is not None:
        return Path(_iteration_dir).resolve() / DEFAULT_LOGICAL_FILE
    settings = get_settings()
    return Path(settings.visualization_output_path).resolve() / DEFAULT_LOGICAL_FILE


def _is_supported_path(path: Optional[str], logical_file: str) -> bool:
    if not path:
        return True
    normalized = path.strip().replace("\\", "/")
    return normalized == logical_file or normalized.endswith(f"/{logical_file}")


def _sync_code(code: str) -> str:
    """Persist current code to the active runtime simulation path."""
    simulation_path = _current_simulation_path()
    simulation_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_path.write_text(code or "", encoding="utf-8")

    return f"Saved current code to {simulation_path}."


# Module-level ref set by code_generation_agent before tool loop starts.
_iteration_dir: Optional[Path] = None


def set_iteration_dir(path: Path) -> None:
    """Called by code_generation_agent to tell tools where to persist SKILL.md."""
    global _iteration_dir
    _iteration_dir = path


def _append_skill_read_log(**entry: Any) -> None:
    """Append one record to ``iteration_dir/skill_read_log.json``.

    SWE-agent observability: every skill read (full doc, section, search) leaves
    an audit trail on disk so a run can be replayed cold — which skills the
    model consulted for which plan step. No-op when ``_iteration_dir`` is unset
    (tests / bare tool-executor usage).
    """
    if _iteration_dir is None:
        return
    entry.setdefault("ts", time.time())
    path = _iteration_dir / "skill_read_log.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
    except (OSError, ValueError):
        existing = []
    existing.append(entry)
    try:
        path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def _enrich_validation_with_skills(validator_output: str) -> str:
    """Parse validator error lines and append relevant skill API contracts.

    Scans for error patterns like ``'MethodName' not found in class hierarchy of ClassName``
    and uses :meth:`SkillRegistry.find_skills_for_error` to find matching skills.
    """
    if "INVALID: 0" in validator_output or "CHRONO_API_VALIDATION: PASS" in validator_output:
        return validator_output

    error_pairs: set[tuple[str, str]] = set()

    # Pattern: "'method' not found in class hierarchy of ClassName"
    for match in re.finditer(
        r"'(\w+)'\s+not found in class hierarchy of (\w+)", validator_output
    ):
        method, cls = match.group(1), match.group(2)
        error_pairs.add((cls, method))

    # Pattern: "Attribute 'X' not found in ..."
    for match in re.finditer(
        r"Attribute '(\w+)' not found in", validator_output
    ):
        error_pairs.add(("", match.group(1)))

    # Pattern: class names appearing in error lines (ChXxx)
    for match in re.finditer(r"\b(Ch\w+)\b", validator_output):
        cls_name = match.group(1)
        # Only add if not already covered
        if not any(c == cls_name for c, _ in error_pairs):
            error_pairs.add((cls_name, ""))

    if not error_pairs:
        return validator_output

    seen_skills: set[str] = set()
    snippets: list[str] = []
    for cls_name, method_name in error_pairs:
        results = SkillRegistry.find_skills_for_error(
            class_name=cls_name or None,
            method_name=method_name or None,
        )
        for r in results:
            skill_name = r["skill_name"]
            if skill_name in seen_skills:
                continue
            seen_skills.add(skill_name)
            contract = r["api_contract"]
            lines = [f"  Skill: {skill_name} (matched via {r['match_source']})"]
            for key in ("allowed_classes", "allowed_methods", "canonical_examples"):
                items = contract.get(key, [])
                if items:
                    lines.append(f"    {key}: {'; '.join(items[:8])}")
            snippets.append("\n".join(lines))

    if not snippets:
        return validator_output

    enrichment = (
        "\n\n--- AUTO-RETRIEVED SKILL API CONTRACTS ---\n"
        "The following skill API contracts are relevant to the invalid APIs above.\n"
        "Use these exact method signatures to fix the errors:\n\n"
        + "\n\n".join(snippets)
    )
    return validator_output + enrichment


def _normalize_skill_name(name: str) -> str:
    """Canonicalize a skill name for set-membership comparison."""
    return (name or "").strip().strip("'\"").lower()


# Max number of times the skill-required gate will nudge the LLM per codegen
# tool-loop iteration. After this many nudges, the gate unlocks so the LLM
# can make progress even if it's stubbornly refusing to call read_skill —
# the auto-validator still catches bad output downstream.
MAX_SKILL_GATE_NUDGES = 2


def _derive_required_skills(
    plan_dict: Dict[str, Any],
    step_context: Optional[Dict[str, Any]] = None,
    llm_routed_skills: Optional[List[str]] = None,
) -> set:
    """Return the set of skill names the codegen LLM MUST read before it is
    allowed to ``write_file`` / ``edit_file`` for this plan.

    When ``llm_routed_skills`` is provided (populated by the Haiku skill
    router in CodeGenerationAgent._route_skills_via_llm), it is used
    verbatim as the required set — the LLM router has already picked the
    minimal skills based on plan content + skill descriptions. The
    keyword rules below run only as a fallback when the router is
    disabled or failed.

    Fallback rules (kept conservative — false-positive nudges are tolerable,
    missing a real requirement is not):
    * ``plan.visualization.mode`` containing ``sensor`` → require ``sens/camera``
    * ``plan_type == "mbs_in_scene"`` → require ``core/mbs_in_scene``
    * ``plan_type == "scene"`` → require ``core/scene``
    * ``plan_type == "mbs"`` → require ``core/mbs``
    * Vehicle keywords (hmmwv / wheeled_vehicle / chvehicle) anywhere in
      plan JSON → require ``veh/wheeled_vehicle`` and ``veh/driver``
    * ``step_context.step_assets`` non-empty → require
      ``scene/custom_assets_scene_convex_decomp`` (the project-private
      ``AssetDescriptor`` / ``add_visual_assets`` / ``add_collision_via_subbodies``
      utilities live there; this closes the "first asset step is a no-op"
      hole where plan-level rules never triggered the asset skill).
    """
    if llm_routed_skills:
        return {str(n).strip() for n in llm_routed_skills if str(n).strip()}

    required: set = set()
    if not isinstance(plan_dict, dict):
        plan_dict = {}

    plan_type = str(plan_dict.get("plan_type") or "").lower()
    vis_mode = ""
    vis = plan_dict.get("visualization")
    if isinstance(vis, dict):
        vis_mode = str(vis.get("mode") or "").lower()

    if "sensor" in vis_mode:
        required.add("sens/camera")
    if plan_type == "mbs_in_scene":
        required.add("core/mbs_in_scene")
    elif plan_type == "fsi_in_scene":
        required.add("core/fsi_in_scene")
    elif plan_type == "scene":
        required.add("core/scene")
    elif plan_type == "mbs":
        required.add("core/mbs")

    # Vehicle detection via coarse keyword scan over serialized plan.
    try:
        blob = json.dumps(plan_dict, default=str).lower()
    except (TypeError, ValueError):
        blob = ""
    if any(k in blob for k in ("hmmwv", "chwheeledvehicle", "wheeled_vehicle", "chvehicle")):
        required.add("veh/wheeled_vehicle")
        required.add("veh/driver")

    # Step-level asset detection. When the current step's context lists
    # any step_assets (trees/bushes/rocks/cottage, or any AssetDescriptor-
    # shaped item), the codegen MUST read the scene-assets skill — the
    # AssetDescriptor / add_visual_assets / add_collision_via_subbodies
    # utilities are project-private and not discoverable from Chrono docs.
    # Before this rule, the LLM used to voluntarily read_skill on its
    # first asset-step turn, burning the turn with no code written.
    if isinstance(step_context, dict):
        step_assets = step_context.get("step_assets")
        if isinstance(step_assets, (list, tuple)) and len(step_assets) > 0:
            required.add("scene/custom_assets_scene_convex_decomp")

    return required


def _build_skill_gate_nudge(missing: list, required: list) -> str:
    """Format the nudge string returned to the LLM when skills are missing."""
    joined_missing = ", ".join(f"'{s}'" for s in sorted(missing))
    joined_required = ", ".join(f"'{s}'" for s in sorted(required))
    return (
        "REFUSED: skill-gate.\n"
        f"For this plan you must call read_skill() on these skills before "
        f"write_file / edit_file is accepted: {joined_required}.\n"
        f"Missing reads this turn: {joined_missing}.\n"
        "BATCHING TIP: the harness serialises tool_use blocks when any of them "
        "is a mutating tool (write_file/edit_file), so you can emit "
        "all the read_skill calls AND the write_file call in a SINGLE assistant "
        "turn — the skill reads execute first, the gate sees them as satisfied, "
        "and write_file proceeds on the same turn with no extra round-trip.\n"
        "Example batch: [read_skill(missing_skill_1), read_skill(missing_skill_2), "
        "write_file(content=...)].\n"
        "This is enforced by the tool harness — it won't unblock until you call "
        "read_skill (or you exhaust the nudge budget)."
    )


def _compact_validator_block(full_output: str, header: str) -> str:
    """Squeeze a validator's verbose PASS/FAIL block into a single status line
    plus just the violation lines on FAIL.

    Each validator's script prints:
        SOMETHING_VALIDATION: PASS / FAIL
        (maybe a summary)
        VIOLATIONS: N   (only on FAIL)
          - violation 1
          - violation 2

    For the auto-hook use case we want a short, skimmable summary — not the
    full multi-paragraph block of each validator, since three of them run
    together and the LLM would drown.
    """
    if not full_output:
        return f"{header}: (no output)"
    # Pull the status line and any violation bullets.
    lines = full_output.splitlines()
    status_line = next((l for l in lines if "VALIDATION: PASS" in l or "VALIDATION: FAIL" in l
                        or "INVARIANTS: PASS" in l or "INVARIANTS: FAIL" in l), "")
    bullets = [l for l in lines if l.lstrip().startswith("- ") or l.lstrip().startswith("* ")]
    if "PASS" in status_line:
        return f"{header}: PASS"
    # Use the consistent "header: FAIL" format regardless of which script
    # produced the FAIL (scripts print different banner names like
    # SIM_DURATION_VALIDATION vs VISUAL_INVARIANTS).
    body = f"{header}: FAIL"
    if bullets:
        body += "\n" + "\n".join(bullets[:8])  # cap noise
        if len(bullets) > 8:
            body += f"\n  ...and {len(bullets) - 8} more violations"
    return body


def _classify_validator_status(full_output: str) -> str:
    """Classify a raw validator output as 'PASS' / 'FAIL' / 'UNKNOWN'."""
    if not full_output:
        return "UNKNOWN"
    for line in full_output.splitlines():
        if ("VALIDATION: PASS" in line) or ("INVARIANTS: PASS" in line):
            return "PASS"
        if ("VALIDATION: FAIL" in line) or ("INVARIANTS: FAIL" in line):
            return "FAIL"
    return "UNKNOWN"


def _extract_validator_valid_count(full_output: str) -> Optional[int]:
    """Read ``VALID: <n>  |  INVALID: <m>`` from validator output, return n.

    Used by the PASS path of ``_run_all_post_edit_validators`` so the agent
    sees how many chains the validator actually inspected — a PASS with
    0 chains means the validator coverage missed the new code (e.g. all
    calls used unbound aliases that bypassed the visitor before fix 1.A,
    or the file was empty / syntax-erroneous). Returns None on parse miss.
    """
    if not full_output:
        return None
    match = re.search(r"VALID:\s*(\d+)\s*\|", full_output)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_chrono_api_violations(full_output: str) -> List[str]:
    """Parse ``validate_chrono_apis.py`` tabular output into one-line violations.

    The validator script emits issues as paired rows::

        LINE   CHAIN                                         ERROR
        ----------------------------------------------------------------------
        159    sysMBS.Get_bodylist
               → 'Get_bodylist' not found in class hierarchy of ChSystemSMC...

    Each pair becomes ``"line 159: sysMBS.Get_bodylist — 'Get_bodylist' not found..."``.
    Returns ``[]`` when the output has no violation rows (the upstream caller
    should still report FAIL but with an honest "no detail parseable" hint).
    """
    if not full_output:
        return []
    in_issues = False
    pending_chain: Optional[str] = None
    pending_line: Optional[str] = None
    issues: List[str] = []
    for raw in full_output.splitlines():
        stripped = raw.strip()
        if not in_issues:
            if stripped.startswith("LINE") and "CHAIN" in stripped and "ERROR" in stripped:
                in_issues = True
            continue
        if not stripped:
            continue
        # Skip the dashed separator under the LINE header.
        if set(stripped) <= {"-"}:
            continue
        if stripped.startswith("→") or stripped.startswith("→"):
            err_text = stripped.lstrip("→→").strip()
            if pending_chain is not None:
                tag = f"line {pending_line}: " if pending_line else ""
                issues.append(f"{tag}{pending_chain} — {err_text}")
                pending_chain = None
                pending_line = None
            continue
        match = re.match(r"^(\d+)\s+(\S.*?)\s*$", stripped)
        if match:
            pending_line = match.group(1)
            pending_chain = match.group(2)
    return issues


def _run_all_post_edit_validators(code: str, plan: Dict[str, Any]) -> str:
    """Run post-edit validators and return a compact summary block.

    Called automatically by ``write_file`` / ``edit_file``.
    Currently only the Chrono-API validator is wired in.

    Output shape:
    - PASS → ``validators: chrono_apis PASS``
    - FAIL → ``chrono_apis FAIL (<n> violations):`` + parsed rows (up to 8)
      with line numbers from the validator script.

    Why this parser: the legacy implementation looked for ``- `` / ``* ``
    bullets, which match the visual_invariants/sim_duration validators but
    NOT ``validate_chrono_apis.py`` (it emits a tabular ``LINE  CHAIN  →
    ERROR`` block). On every chrono-api FAIL the bullet matcher returned
    ``[]`` so the block printed ``chrono_apis FAIL:`` with empty body —
    agents read the empty FAIL as "nothing actionable" and proceeded
    (this is the iter_004 ``Get_bodylist`` regression in
    session_20260429_112754).
    """
    out = _run_chrono_api_validation(code)
    status = _classify_validator_status(out)

    if status != "FAIL":
        # Surface coverage so agents can tell whether validator actually saw
        # the new code (PASS with 0 chains validated == validator did nothing
        # useful, e.g. on a syntax-error-only file).
        valid_count = _extract_validator_valid_count(out)
        suffix = f" ({valid_count} chains validated)" if valid_count is not None else ""
        return f"validators: chrono_apis {status}{suffix}"

    violations = _extract_chrono_api_violations(out)
    if not violations:
        # Legacy bullet shape — keep as fallback in case other validators
        # ever join this auto-hook with a different output format.
        violations = [
            line.strip()
            for line in out.splitlines()
            if line.lstrip().startswith("- ") or line.lstrip().startswith("* ")
        ]

    block = f"chrono_apis FAIL ({len(violations)} violations):"
    if violations:
        block += "\n" + "\n".join(f"  - {v}" for v in violations[:8])
        if len(violations) > 8:
            block += f"\n  ...and {len(violations) - 8} more violations"
    else:
        block += "\n  (validator reported FAIL but no detail rows were parseable — see chrono_apis raw output above)"
    return block


def _run_chrono_api_validation(code: str) -> str:
    """Validate Chrono API usage for the current code and return a trace-friendly status block.

    Three-layer perf path:
      1. SHA cache hit → instant return (no AST walk, no subprocess)
      2. In-process validator import → ~300-700 ms saved per call vs subprocess
         (skips Python interpreter startup AND the heavy pychrono re-import,
         since the agent process already has pychrono loaded)
      3. Subprocess fallback → for environments where the in-process import
         can't reach pychrono (rare; e.g. running the agent under a Python
         that lacks the chrono bindings)

    Behaviour and output format are byte-identical across all three paths.
    """
    saved_message = _sync_code(code)
    simulation_path = _current_simulation_path()

    # Layer 1: SHA cache. Identical code text → identical validator output.
    code_sha = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
    cached_block = _VALIDATION_CACHE.get(code_sha)
    if cached_block is not None:
        return f"{saved_message}\n\n{cached_block}"

    # Layer 2: in-process import + direct call.
    module = _load_validate_chrono_apis_module()
    if module is not None and hasattr(module, "validate_file"):
        try:
            valid, issues = module.validate_file(str(simulation_path))
        except Exception as exc:
            logger.warning(
                "[chrono_api_validator] in-process call failed (%s); "
                "falling back to subprocess for this run",
                exc,
            )
        else:
            output = _format_validate_chrono_apis_output(
                str(simulation_path), valid, issues
            )
            status = "PASS" if not issues else "FAIL"
            block_parts = [
                f"CHRONO_API_VALIDATION: {status}",
                "Chrono API validation:\n" + output,
            ]
            result_block = "\n\n".join(block_parts)
            if status == "FAIL":
                result_block = _enrich_validation_with_skills(result_block)
            _validation_cache_put(code_sha, result_block)
            return f"{saved_message}\n\n{result_block}"

    # Layer 3: subprocess fallback (legacy path).
    python_executable = _resolve_validator_python()
    env = _build_validator_env(python_executable)
    try:
        result = subprocess.run(
            [python_executable, str(VALIDATOR_SCRIPT_PATH), str(simulation_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except Exception as exc:
        return (
            f"{saved_message}\n\n"
            "CHRONO_API_VALIDATION: ERROR\n\n"
            f"Chrono API validation could not run: {type(exc).__name__}: {exc}"
        )

    output = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    invalid_count = None
    match = re.search(r"INVALID:\s*(\d+)", output)
    if match:
        invalid_count = int(match.group(1))
    status = "PASS" if invalid_count == 0 else "FAIL"

    block_parts = [f"CHRONO_API_VALIDATION: {status}"]
    if output:
        block_parts.append("Chrono API validation:\n" + output)
    if err:
        block_parts.append("Validator stderr:\n" + err)
    if not output and not err:
        block_parts.append(
            f"Chrono API validation finished with exit code {result.returncode}."
        )

    result_block = "\n\n".join(block_parts)
    if status == "FAIL":
        result_block = _enrich_validation_with_skills(result_block)
    _validation_cache_put(code_sha, result_block)
    return f"{saved_message}\n\n{result_block}"


def make_code_agent_tools(
    context: Dict[str, Any],
    logical_file: str = DEFAULT_LOGICAL_FILE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """Create code agent tools for the unified tool-calling flow.

    Returns:
        A tuple of (tool_definitions, tool_executors) where tool_definitions
        is a list of Claude tool schema dicts and tool_executors is a dict
        mapping tool name to callable function.
    """

    # Initialize skill-gate tracking: which skills are required for this
    # plan + step, which ones the LLM has read this session, how many times
    # we've already nudged. See _derive_required_skills for the policy.
    # Both the plan dict and the per-step context are consulted: plan-level
    # rules (plan_type, vehicle keywords) come from ``context['plan']``;
    # step-level rules (step_assets → scene asset skill) come from
    # ``context['step_context']`` if present.
    plan_for_gate = context.get("plan") if isinstance(context.get("plan"), dict) else {}
    step_ctx_for_gate = context.get("step_context") if isinstance(context.get("step_context"), dict) else None
    routed_for_gate = context.get("llm_routed_skills")
    if not isinstance(routed_for_gate, list):
        routed_for_gate = None
    context.setdefault(
        "required_skills",
        _derive_required_skills(
            plan_for_gate or {},
            step_context=step_ctx_for_gate,
            llm_routed_skills=routed_for_gate,
        ),
    )
    context.setdefault("skills_read", set())
    context.setdefault("skill_nudges_fired", 0)

    def _skill_gate_check() -> Optional[str]:
        """Return a nudge string to block the edit, or None to allow it.

        Gate satisfaction sources (any one is enough per skill):
          1. The LLM explicitly called read_skill / read_skill_section /
             query_skill on the skill this session (``skills_read``).
          2. The skill was pre-injected into the system prompt by the
             codegen agent (``preinjected_skill_names``) — the content is
             already in every turn's input, so demanding a second
             tool-call surfacing is one turn of pure ceremony. Gated by
             settings.skill_gate_treat_preinjected_as_read.
        """
        required = context.get("required_skills") or set()
        read = context.get("skills_read") or set()
        satisfied_by_read = {_normalize_skill_name(r) for r in read}
        # Merge in pre-injected skills when enabled.
        try:
            _relaxed = bool(
                getattr(
                    get_settings(),
                    "skill_gate_treat_preinjected_as_read",
                    True,
                )
            )
        except Exception:
            _relaxed = True
        if _relaxed:
            preinjected = context.get("preinjected_skill_names") or set()
            satisfied_by_read = satisfied_by_read | {
                _normalize_skill_name(p) for p in preinjected
            }
        missing = {
            s for s in required
            if _normalize_skill_name(s) not in satisfied_by_read
        }
        if not missing:
            return None
        nudges = int(context.get("skill_nudges_fired") or 0)
        if nudges >= MAX_SKILL_GATE_NUDGES:
            # Exhausted — let the LLM through; auto-validators will catch
            # any resulting broken code on the next turn.
            return None
        context["skill_nudges_fired"] = nudges + 1
        return _build_skill_gate_nudge(sorted(missing), sorted(required))

    def read_file(
        path: str = logical_file,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read a file by name (optional line window).

        Supported files:
        - simulation.py
        - plan.json
        - previous_attempts.txt
        - handoff.json
        """
        normalized = (path or "").strip().replace("\\", "/")
        extra_files: Dict[str, str] = context.get("extra_files") or {}
        if normalized in extra_files:
            content = extra_files[normalized]
            if start_line is None and end_line is None:
                return content
            lines = content.splitlines()
            total = len(lines)
            start = 1 if start_line is None else int(start_line)
            end = total if end_line is None else int(end_line)
            clipped_end = min(end, total)
            return "\n".join(lines[start - 1 : clipped_end])

        if not _is_supported_path(path, logical_file):
            available = [logical_file] + list(extra_files.keys())
            return f"Unsupported path: {path}. Available files: {', '.join(available)}"

        source = str(context.get("current_code") or "")
        if not source:
            return f"{logical_file} is empty. Use write_file to create the initial code."
        if start_line is None and end_line is None:
            return source

        lines = source.splitlines()
        total = len(lines)
        if total == 0:
            return f"{logical_file} is empty. Use write_file to create the initial code."

        start = 1 if start_line is None else int(start_line)
        end = total if end_line is None else int(end_line)
        if start < 1 or end < 1:
            return "Invalid line range: start_line/end_line must be >= 1."
        if start > end:
            return "Invalid line range: start_line must be <= end_line."
        if start > total:
            return f"Invalid line range: start_line={start} exceeds file length {total}."

        clipped_end = min(end, total)
        return "\n".join(lines[start - 1 : clipped_end])

    def grep_code(pattern: str, path: Optional[str] = None) -> str:
        """Search code using regex pattern and return matching lines."""
        if not _is_supported_path(path, logical_file):
            return f"Unsupported path: {path}. Only {logical_file} is available."

        source = str(context.get("current_code") or "")
        if not source:
            return "No code available."

        broad_patterns = {".", ".*", "^.*$", ".+", ".*?"}
        if (pattern or "").strip() in broad_patterns:
            return (
                "Pattern too broad. Use a specific symbol or function name, then call "
                "read_file(start_line=..., end_line=...) for a local window."
            )

        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            return f"Invalid regex: {exc}"

        hits: List[str] = []
        for line_no, line in enumerate(source.splitlines(), start=1):
            if compiled.search(line):
                hits.append(f"{line_no}: {line}")
            if len(hits) >= 50:
                break

        if not hits:
            return "No matches found."
        first_line = int(hits[0].split(":", 1)[0])
        start = max(1, first_line - 40)
        end = first_line + 40
        hint = f"\n\nNext step: read_file(path='{logical_file}', start_line={start}, end_line={end})"
        return "\n".join(hits) + hint

    # apply_patch was removed: the harness now uses edit_file (substring
    # replace) + write_file (whole-file rewrite) as the only mutation
    # primitives. Multi-hunk diff application carried failure modes
    # (line-number drift, partial-apply rollback, agent fallback to
    # edit_file silently dropping unrelated hunks — the iter_002
    # NameError(veh) chain in session_20260429_112754) that don't exist
    # when each edit is a self-contained substring replacement. See dialog
    # plan dialog-sessions-session-20260429-112754-glittery-pixel.md.

    def write_file(content: str, path: str = logical_file) -> str:
        """Replace current logical file with full content."""
        if not _is_supported_path(path, logical_file):
            return f"Unsupported path: {path}. Only {logical_file} is available."
        # Skill-required gate (Tier 2.2): block the edit until the LLM has
        # called read_skill() for every required skill for this plan.
        _gate = _skill_gate_check()
        if _gate is not None:
            return _gate
        context["current_code"] = content or ""
        code_str = str(context["current_code"])
        plan_dict = context.get("plan") if isinstance(context.get("plan"), dict) else {}
        auto_report = _run_all_post_edit_validators(code_str, plan_dict or {})
        ref_block = format_reference_block(code_str) or ""
        return (
            f"Wrote {len(code_str)} chars to {logical_file}.\n\n"
            f"{_sync_code(code_str)}\n\n"
            f"{auto_report}\n\n"
            + (f"{ref_block}\n\n" if ref_block else "")
            + "If any validator above shows FAIL, fix the listed violations before moving on. "
            "If all show PASS, you may proceed."
        )

    def edit_file(
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace an exact substring in the current logical file."""
        _gate = _skill_gate_check()
        if _gate is not None:
            return _gate

        base = str(context.get("current_code") or "")
        if not base.strip():
            return (
                f"{logical_file} is empty. Use write_file to create the initial code."
            )

        if not isinstance(old_string, str) or old_string == "":
            return "edit_file: old_string is required and must be non-empty."
        if not isinstance(new_string, str):
            return "edit_file: new_string must be a string."
        if old_string == new_string:
            return "edit_file: old_string and new_string are identical; no change."

        count = base.count(old_string)
        if count == 0:
            return (
                "edit_file: old_string not found. Re-read the file and quote an exact "
                "substring, including surrounding whitespace and punctuation."
            )
        if count > 1 and not replace_all:
            return (
                f"edit_file: old_string is ambiguous — appears {count} times. "
                "Either extend old_string with more unique surrounding context, "
                "or pass replace_all=true to rewrite every occurrence."
            )

        if replace_all:
            patched = base.replace(old_string, new_string)
            changed = count
        else:
            patched = base.replace(old_string, new_string, 1)
            changed = 1

        context["current_code"] = patched
        plan_dict = context.get("plan") if isinstance(context.get("plan"), dict) else {}
        auto_report = _run_all_post_edit_validators(patched, plan_dict or {})
        ref_block = format_reference_block(patched) or ""
        occurrence_note = (
            f"Replaced {changed} occurrence(s) in {logical_file}. "
            f"New code length: {len(patched)} chars."
        )

        return (
            f"{occurrence_note}\n\n"
            f"{_sync_code(patched)}\n\n"
            f"{auto_report}\n\n"
            + (f"{ref_block}\n\n" if ref_block else "")
            + "If any validator above shows FAIL, fix the listed violations before moving on. "
            "If all show PASS, you may proceed."
        )

    def validate_chrono_apis(path: str = logical_file) -> str:
        """Validate current code with scripts/validate_chrono_apis.py."""
        if not _is_supported_path(path, logical_file):
            return f"Unsupported path: {path}. Only {logical_file} is available."
        return _run_chrono_api_validation(str(context.get("current_code") or ""))

    def read_skill(skill_name: str) -> str:
        """Read a full PyChrono skill by canonical path name."""
        name = (skill_name or "").strip()
        fragment = SkillRegistry.get_skill_fragment(name)
        if fragment is None:
            available = SkillRegistry.get_all_skill_names()
            return f"Unknown skill '{name}'. Available: {', '.join(available)}"
        # Record the read for the skill-required gate. Normalize for
        # comparison (case / quotes / whitespace).
        reads = context.setdefault("skills_read", set())
        reads.add(_normalize_skill_name(name))
        _append_skill_read_log(
            event="read_skill", skill_name=name, chars=len(fragment)
        )
        return fragment

    def _normalize_query_question(q: str) -> str:
        """Stable key for the query_skill answer cache.

        Lowercase + collapse whitespace so trivially-different phrasings
        of the same question still hit the cache. Intentionally does NOT
        do semantic normalization — only near-identical strings should
        collide.
        """
        return " ".join((q or "").strip().lower().split())

    async def query_skill(skill_name: str, question: str) -> str:
        """Ask a focused question about a SKILL.md — Haiku answers, returns a
        short extract instead of the full 3-5K-token doc. See
        :class:`chrono_code.agents.skill_query_agent.SkillQueryAgent`.

        Two layers of caching, from cheapest to most expensive on a miss:
          1. In-process LRU on (skill_name, normalized_question) — keyed
             by strings so repeat identical asks return instantly without
             any API call.
          2. Anthropic prompt cache (``cache_control: ephemeral``) on the
             skill document block inside SkillQueryAgent — same skill
             asked different questions still hits the ~90%-discount read
             path on the document body.
        """
        name = (skill_name or "").strip()
        q = (question or "").strip()
        if not name:
            return "query_skill requires skill_name."
        if not q:
            return "query_skill requires a non-empty question."
        # Fetch full skill text (reuses the same registry path as read_skill).
        fragment = SkillRegistry.get_skill_fragment(name)
        if fragment is None:
            available = SkillRegistry.get_all_skill_names()
            return f"Unknown skill '{name}'. Available: {', '.join(available)}"

        # --- Layer 1: in-process answer cache ---------------------------
        _settings = get_settings()
        cache_enabled = bool(getattr(_settings, "skill_query_cache_enabled", True))
        cache_max = int(getattr(_settings, "skill_query_cache_max_entries", 32))
        cache = context.get("_skill_query_cache")
        if cache_enabled and cache is None:
            from collections import OrderedDict
            cache = OrderedDict()
            context["_skill_query_cache"] = cache
        cache_key = None
        if cache_enabled:
            cache_key = (_normalize_skill_name(name), _normalize_query_question(q))
            hit = cache.get(cache_key)
            if hit is not None:
                # LRU bump: move to end.
                cache.move_to_end(cache_key)
                # Still satisfy the skill-gate on cache hits — the model
                # did engage with the authoritative doc (just earlier in
                # this session).
                reads = context.setdefault("skills_read", set())
                reads.add(_normalize_skill_name(name))
                _append_skill_read_log(
                    event="query_skill_cache_hit",
                    skill_name=name,
                    chars=len(hit),
                    query=q,
                )
                return f"[query_skill answer for {name} (cached)]\n{hit}"

        # --- Layer 2: Haiku sub-agent call -----------------------------
        # Lazy-cache the sub-agent instance on ``context`` so repeat calls
        # within one codegen session reuse the same Haiku client (and,
        # importantly, hit its 5-min prompt cache window on the skill text).
        subagent = context.get("_skill_query_subagent")
        if subagent is None:
            # Local import keeps ``code_agent_tools`` importable even when
            # ``anthropic`` is missing (e.g. unit tests running without the
            # SDK installed).
            from chrono_code.agents.skill_query_agent import SkillQueryAgent
            subagent = SkillQueryAgent()
            context["_skill_query_subagent"] = subagent
        try:
            answer = await subagent.query(name, fragment, q)
        except Exception as exc:
            return (
                f"query_skill failed: {exc}. "
                f"Fall back to read_skill('{name}') for the full document."
            )

        # Write-through into the LRU cache. Evict oldest when over budget.
        if cache_enabled and cache_key is not None:
            cache[cache_key] = answer
            cache.move_to_end(cache_key)
            while len(cache) > cache_max:
                cache.popitem(last=False)

        # Satisfy the skill-gate just like read_skill / read_skill_section —
        # the model has engaged with the authoritative doc via the proxy.
        reads = context.setdefault("skills_read", set())
        reads.add(_normalize_skill_name(name))
        _append_skill_read_log(
            event="query_skill", skill_name=name, chars=len(answer), query=q
        )
        return f"[query_skill answer for {name}]\n{answer}"

    def read_skill_section(skill_name: str, heading: str) -> str:
        """Return one section of a skill, addressed by its markdown heading.

        Narrow-beam counterpart to ``read_skill``: useful when you know the
        skill but only need e.g. 'api contract' or 'common mistakes'.
        Satisfies the skill-gate just like ``read_skill`` (reading any section
        proves you've engaged with the authoritative doc).
        """
        name = (skill_name or "").strip()
        head = (heading or "").strip()
        if not head:
            return "heading is required; pass a markdown section name (e.g. 'api contract')."
        # Validate skill exists first, so a bogus name can't trivially satisfy the gate.
        if SkillRegistry.get_skill_fragment(name) is None:
            available = SkillRegistry.get_all_skill_names()
            return f"Unknown skill '{name}'. Available: {', '.join(available)}"
        section = SkillRegistry.get_skill_section(name, head)
        if not section:
            sections = SkillRegistry.list_sections(name) or []
            return (
                f"No section '{head}' in '{name}'. "
                f"Available sections: {', '.join(sections) if sections else '(none)'}."
            )
        reads = context.setdefault("skills_read", set())
        reads.add(_normalize_skill_name(name))
        _append_skill_read_log(
            event="read_skill_section",
            skill_name=name,
            heading=head,
            chars=len(section),
        )
        return section

    def rebut_review(reasoning: str) -> str:
        """Push back on a review-agent decision when you believe the code is ALREADY correct.

        Use this ONLY in these situations:
        - The step_feedback complains about an orientation, position, predicate, or visual
          detail that the current simulation.py already implements correctly.
        - The review's complaint is based on a misreading of the image, a misinterpretation
          of the asset's canonical frame, or a false-positive from the visual description.
        - Making any edit would actively make the simulation worse, and the only correct
          action is to keep the code as-is and explain why.

        DO NOT use rebut_review to avoid doing work. If even part of the review feedback
        is valid, make the edit instead -- rebuttals are for full disagreement only.
        """
        text = (reasoning or "").strip()
        if not text:
            return "rebut_review rejected: reasoning is empty."
        if context.get("review_feedback") is None:
            return (
                "rebut_review rejected: there is no review feedback to rebut in this "
                "run. This tool is only usable after a step_review failure."
            )
        context["codegen_rebuttal"] = text
        return (
            "Rebuttal submitted to the review agent. The review will be re-run with "
            "your explanation as additional context. You do NOT need to make any code "
            "edits in this run unless the review is accepted and still flags issues."
        )

    def search_skills(query: str, domain: str = "", limit: int = 4) -> str:
        """Search the skill index by keyword and optional domain prefix."""
        results = SkillRegistry.search(query=query, domain=domain or None, limit=limit)
        _append_skill_read_log(
            event="search_skills",
            query=query,
            domain=domain or "",
            hits=len(results),
        )
        if not results:
            return "No matching skills found."
        return json.dumps(results, ensure_ascii=True, indent=2)

    # --- API RAG ---
    _api_query_agent: Any = None  # lazy-init on first call

    async def query_api(question: str) -> str:
        """Search the PyChrono API documentation and get a Haiku-generated answer.

        Use this when you need the exact constructor signature, method parameters,
        or description for any PyChrono class or function (e.g. ChBody, ChSystem,
        ChLinkMotorRotationSpeed, SetContactMethod, etc.).
        """
        nonlocal _api_query_agent
        if not question or not question.strip():
            return "query_api requires a non-empty question."
        try:
            from chrono_code.tools.api_rag import ApiQueryAgent
            if _api_query_agent is None:
                _api_query_agent = ApiQueryAgent()
            answer = await _api_query_agent.query(question)
            _append_skill_read_log(
                event="query_api",
                query=question,
                chars=len(answer),
            )
            return f"[PyChrono API answer]\n{answer}"
        except Exception as exc:
            logger.warning("[query_api] failed: %s", exc)
            return f"query_api failed: {exc}. Use bash('python -c \"import pychrono; help(...)\"') as fallback."

    BASH_MAX_CALLS_PER_TURN = 5
    BASH_DEFAULT_TIMEOUT = 20
    BASH_MAX_TIMEOUT = 60
    BASH_OUTPUT_BYTES = 8 * 1024
    BASH_ALLOWED_COMMANDS = frozenset({
        "python", "python3", "pip",
        "ls", "cat", "head", "tail", "grep", "find", "which",
    })
    _bash_call_counter = [0]

    def bash(command: str, timeout: int = BASH_DEFAULT_TIMEOUT) -> str:
        """Run a read-only introspection command (allowlisted) and return EXIT/STDOUT/STDERR.

        Use this to check the real PyChrono API at runtime, e.g.
            bash('python -c "import pychrono as c; help(c.ChAABB)"')
            bash('pip show pychrono')
            bash('find /path/to/data -name "*.obj" -maxdepth 3')
        """
        if _bash_call_counter[0] >= BASH_MAX_CALLS_PER_TURN:
            return (
                f"bash: call budget exceeded ({BASH_MAX_CALLS_PER_TURN} calls per turn). "
                "Act on what you have learned; no further bash allowed this turn."
            )

        if not isinstance(command, str) or not command.strip():
            return "bash: command is empty."

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return f"bash: cannot parse command ({exc}). Use plain arguments, no shell operators."

        if not argv:
            return "bash: command is empty after parsing."

        head_cmd = argv[0]
        base_name = Path(head_cmd).name
        if base_name not in BASH_ALLOWED_COMMANDS:
            return (
                f"bash: command '{base_name}' is not allowed. "
                f"Allowed: {', '.join(sorted(BASH_ALLOWED_COMMANDS))}. "
                "Compose with plain flags/arguments (no pipes, redirects, or shell operators)."
            )

        # Reject shell metacharacters that shlex would keep as literal tokens —
        # these have no meaning with shell=False and usually signal the agent
        # expected shell features we don't support.
        forbidden_tokens = {"|", "||", "&", "&&", ";", ">", ">>", "<", "`", "$("}
        for tok in argv[1:]:
            if tok in forbidden_tokens or tok.startswith("$(") or tok.startswith("`"):
                return (
                    f"bash: token {tok!r} looks like a shell operator, which is not supported. "
                    "Use a single simple command without pipes/redirects."
                )

        try:
            timeout_int = int(timeout)
        except (TypeError, ValueError):
            timeout_int = BASH_DEFAULT_TIMEOUT
        timeout_int = max(1, min(timeout_int, BASH_MAX_TIMEOUT))

        python_executable = _resolve_validator_python()
        env = _build_validator_env(python_executable)

        # Substitute python/python3/pip with the chrono-aware interpreter so
        # imports (pychrono, etc.) resolve in the subprocess.
        if base_name in ("python", "python3"):
            argv[0] = python_executable
        elif base_name == "pip":
            pip_path = Path(python_executable).with_name("pip")
            argv[0] = str(pip_path) if pip_path.exists() else head_cmd

        _bash_call_counter[0] += 1

        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_int,
                env=env,
                cwd=str(REPO_ROOT),
            )
        except subprocess.TimeoutExpired:
            return (
                f"bash: command timed out after {timeout_int}s. "
                "Narrow the scope (e.g. a smaller `find` root) or raise timeout up to "
                f"{BASH_MAX_TIMEOUT}s."
            )
        except FileNotFoundError:
            return f"bash: executable '{argv[0]}' not found on PATH."
        except OSError as exc:
            return f"bash: OS error: {exc}"

        def _clip(text: str) -> str:
            if len(text) <= BASH_OUTPUT_BYTES:
                return text
            return text[:BASH_OUTPUT_BYTES] + f"\n...[truncated, {len(text) - BASH_OUTPUT_BYTES} more bytes]"

        return (
            f"EXIT: {proc.returncode}\n"
            f"STDOUT:\n{_clip(proc.stdout or '')}\n"
            f"STDERR:\n{_clip(proc.stderr or '')}"
        )

    # ── Build tool definitions ────────────────────────────────────────────

    tool_definitions: List[Dict[str, Any]] = [
        {
            "name": "read_file",
            "description": (
                "Read a file by name (optional line window). "
                "Supported files: simulation.py, plan.json, previous_attempts.txt, handoff.json."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read. Defaults to simulation.py.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line number to return (1-based).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line number to return (1-based).",
                    },
                },
            },
        },
        {
            "name": "grep_code",
            "description": "Search code using regex pattern and return matching lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to search. Defaults to simulation.py.",
                    },
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "edit_file",
            "description": (
                "Replace an exact substring of the current code with new text. "
                "Quote enough surrounding context in old_string to make it "
                "unique, then give the replacement as new_string. Fails if "
                "old_string appears zero times, or more than once without "
                "replace_all=true. For multiple independent edits, call "
                "edit_file multiple times; for full rewrites use write_file."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact substring to locate in the current code. "
                            "Must include enough surrounding context to be unique."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text (may be empty to delete).",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "If true, replace every occurrence of old_string. "
                            "Defaults to false (unique match required)."
                        ),
                    },
                },
                "required": ["old_string", "new_string"],
            },
        },
        {
            "name": "write_file",
            "description": (
                "Replace the current logical file with full content. Use only "
                "for full rewrites or initial code; prefer edit_file for "
                "incremental changes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to write. Defaults to simulation.py.",
                    },
                },
                "required": ["content"],
            },
        },
        {
            "name": "validate_chrono_apis",
            "description": "Validate current code with scripts/validate_chrono_apis.py.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to validate. Defaults to simulation.py.",
                    },
                },
            },
        },
        {
            "name": "read_skill",
            "description": (
                "Read a full PyChrono skill by canonical path name. Returns "
                "the entire SKILL.md (typically 3-5K tokens). Prefer "
                "query_skill(name, question) when you have a specific "
                "question — it returns a focused answer on a cheaper model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Canonical skill name (e.g. 'mbs/system_create').",
                    },
                },
                "required": ["skill_name"],
            },
        },
        {
            "name": "read_skill_section",
            "description": (
                "Return a single markdown section of a skill, addressed by its "
                "heading (case-insensitive, substring match). Use when you only "
                "need a narrow slice of a skill (e.g. 'api contract', 'common "
                "mistakes', 'minimal example') and don't want to re-read the "
                "whole doc. Satisfies the skill-gate just like read_skill."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Canonical skill name (e.g. 'mbs/system_create').",
                    },
                    "heading": {
                        "type": "string",
                        "description": (
                            "Section heading to fetch (case-insensitive; matches "
                            "any heading containing this text as a substring)."
                        ),
                    },
                },
                "required": ["skill_name", "heading"],
            },
        },
        {
            "name": "search_skills",
            "description": "Search the skill index by keyword and optional domain prefix.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional domain prefix to filter by (e.g. 'mbs', 'veh').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results. Defaults to 4.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "bash",
            "description": (
                "Run a read-only introspection command (allowlisted) and get EXIT/STDOUT/STDERR. "
                "Use this to verify PyChrono APIs at runtime before editing code — e.g. "
                "bash('python -c \"import pychrono as c; help(c.ChAABB)\"'), "
                "bash('pip show pychrono'), "
                "bash('find data/robot -name \"*.urdf\" -maxdepth 3'). "
                "Allowed commands: python, python3, pip, ls, cat, head, tail, grep, find, which. "
                "No pipes, redirects, or shell operators. "
                f"Budget: up to 5 calls per turn, timeout up to 60s (default 20s)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Single command line with arguments, parsed by shlex.split. "
                            "Example: 'python -c \"import pychrono as c; print(dir(c.ChAABB))\"'."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before the command is killed. Default 20, max 60.",
                    },
                },
                "required": ["command"],
            },
        },
    ]

    tool_executors: Dict[str, Callable] = {
        "read_file": read_file,
        "grep_code": grep_code,
        "edit_file": edit_file,
        "write_file": write_file,
        "validate_chrono_apis": validate_chrono_apis,
        "read_skill": read_skill,
        "read_skill_section": read_skill_section,
        "search_skills": search_skills,
        "bash": bash,
    }

    # Conditionally register the query_skill (Haiku sub-agent) tool. It's a
    # pure superset — callers that don't enable it still have read_skill /
    # read_skill_section available.
    _settings = get_settings()
    if bool(getattr(_settings, "skill_query_subagent_enabled", True)):
        tool_definitions.append({
            "name": "query_skill",
            "description": (
                "Ask a focused question about a SKILL.md and get a short "
                "answer (≤300 words) instead of the full document. The "
                "answer is grounded strictly in the skill's text — if the "
                "skill doesn't cover your question it will say so. "
                "PREFERRED over read_skill when you have a specific "
                "question (API shape, parameter name, example pattern). "
                "Satisfies the skill-gate just like read_skill."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Canonical skill name (e.g. 'mbs/system_create').",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Specific question about the skill. Examples: "
                            "'What is the exact signature of "
                            "setup_preview_camera?', 'How do I attach a "
                            "sensor camera to a body?', 'Does this skill "
                            "cover TMEASY tires?'."
                        ),
                    },
                },
                "required": ["skill_name", "question"],
            },
        })
        tool_executors["query_skill"] = query_skill

    # query_api: Haiku-backed PyChrono API lookup from pre-built chunk index.
    # Always registered when ANTHROPIC_API_KEY is available (no separate toggle).
    if getattr(_settings, "anthropic_api_key", None):
        tool_definitions.append({
            "name": "query_api",
            "description": (
                "Look up PyChrono API documentation: constructor signatures, method "
                "parameters, and class descriptions. Answers are grounded strictly in "
                "the official API index — no hallucinated signatures. "
                "Use this when you need the exact signature of a class constructor or "
                "method (e.g. 'What parameters does ChLinkMotorRotationSpeed take?', "
                "'What is the signature of ChBody.SetContactMethod?'). "
                "Returns a short focused answer (≤ 300 words)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "Natural language question about a PyChrono class or method. "
                            "Examples: 'What are the constructor arguments of ChBodyEasyBox?', "
                            "'How do I set the tire type on a WheeledVehicle?', "
                            "'What methods does ChCameraSensor have for setting resolution?'."
                        ),
                    },
                },
                "required": ["question"],
            },
        })
        tool_executors["query_api"] = query_api

    # rebut_review is only meaningful after a step_review failure.
    if context.get("review_feedback"):
        tool_definitions.append({
            "name": "rebut_review",
            "description": (
                "Push back on a review-agent decision when you believe the code is ALREADY correct. "
                "Use this ONLY when the step_feedback complains about something the code already "
                "implements correctly, the review is based on a misreading, or making any edit "
                "would actively make the simulation worse. DO NOT use to avoid doing work."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "A concise explanation addressed to the review agent. Cite the "
                            "specific line or predicate you checked, the asset's canonical axis, "
                            "and why the review's claim is wrong."
                        ),
                    },
                },
                "required": ["reasoning"],
            },
        })
        tool_executors["rebut_review"] = rebut_review

    # Append file explorer tools for asset discovery fallback
    explorer_defs, explorer_executors = make_file_explorer_tools()
    tool_definitions.extend(explorer_defs)
    tool_executors.update(explorer_executors)

    # Append Claude-Code-style tools (glob, todo_write/read, web_fetch,
    # bash_background/output/kill, spawn_subagent). State for todos and
    # background processes lives in the same ``context`` dict so everything
    # shares one session.
    from chrono_code.tools.claude_code_tools import make_claude_code_tools
    cc_defs, cc_executors = make_claude_code_tools(
        state=context,
        default_tools=tool_definitions + explorer_defs,
        default_executors={**tool_executors, **explorer_executors},
    )
    tool_definitions.extend(cc_defs)
    tool_executors.update(cc_executors)

    return tool_definitions, tool_executors
