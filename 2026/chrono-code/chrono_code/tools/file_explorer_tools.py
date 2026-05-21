"""
File exploration tools for asset discovery.

Provides list_directory, read_file_content, find_files, find_assets, and
list_chrono_assets tools that can be registered with any tool-calling agent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 200
MAX_FIND_RESULTS = 100

# Binary file extensions that should not be read as text
_BINARY_EXTENSIONS = frozenset({
    ".obj", ".stl", ".step", ".stp", ".iges", ".igs",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif",
    ".mp4", ".avi", ".mov", ".mkv",
    ".bin", ".dat", ".npy", ".npz", ".pkl", ".pickle",
    ".zip", ".gz", ".tar", ".bz2", ".7z",
    ".so", ".dylib", ".dll", ".exe",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".sqlite", ".db",
})


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def _describe_mesh_bounds(obj_path: "os.PathLike | str") -> str:
    """Load ``obj_path`` via pychrono and return a one-line bbox summary.

    Returns an empty string on any load failure or when the path isn't an
    OBJ — caller folds this straight into the tool response, so a silent
    degrade just means "no bbox info available".

    Output shape:
        "bbox (native): 34.0 x 20.7 x 14.7 m — WARNING: native X-span > 15 m
        may overwhelm a 30 m scene; consider scale ~0.4 via
        AssetDescriptor(scale=...)."
    """
    try:
        p = str(obj_path).lower()
        if not p.endswith(".obj"):
            return ""
        # Defer pychrono import so this module stays importable without it.
        import pychrono.core as chrono  # type: ignore

        mesh = chrono.ChTriangleMeshConnected()
        ok = mesh.LoadWavefrontMesh(str(obj_path), True, False)  # load_uv=False, cheap
        if not ok:
            return ""
        verts = mesh.GetCoordsVertices()
        if not verts or len(verts) < 3:
            return ""
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]
        zs = [v.z for v in verts]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        span_z = max(zs) - min(zs)
        summary = (
            f"bbox (native): {span_x:.1f} x {span_y:.1f} x {span_z:.1f} m"
        )
        # Heuristic warning: anything > 10 m in any horizontal dimension is
        # likely to blow out scenes under ~30 m. The project's canonical
        # scaling knob is the plan's ``assets[].ideal_height`` field — a
        # target physical height in meters after uniform mesh scaling. That
        # flows through the plan → codegen handoff and wins the reviewer's
        # "ideal_height provided but never used" check. Map the native span
        # to a plausible ideal_height so the LLM can drop it straight into
        # the plan's assets block.
        max_horiz = max(span_x, span_y)
        if max_horiz > 10.0:
            # Scale target: largest horizontal dimension ~7.5 m (about a
            # quarter of a 30 m scene). Propagate that scale to the height
            # so we recommend an ideal_height instead of a raw scale factor.
            target_horiz = 7.5
            suggested_ideal_height = round(span_z * target_horiz / max_horiz, 2)
            summary += (
                f"  --  WARNING: this mesh is {max_horiz:.1f} m in its "
                f"largest horizontal dimension and {span_z:.1f} m tall — too "
                f"big for a typical 30 m scene at native scale. Set "
                f"`ideal_height: {suggested_ideal_height}` on this asset's "
                f"plan entry (target physical height in meters; codegen will "
                f"pick a uniform scale preserving aspect ratio). Or, if the "
                f"scene really is ≥{max_horiz * 3:.0f} m across, leave "
                f"ideal_height off and use the native size."
            )
        return "  " + summary
    except Exception:  # noqa: BLE001 — bbox probing is a nicety, never fatal
        return ""


def _asset_stem_tokens(stem: str) -> List[str]:
    """Tokenize an asset filename stem for fuzzy suggestion matching.

    Splits on underscore / dash / digits so ``tree_small`` → ['tree', 'small']
    and ``rock05`` → ['rock', '05']. Keeps 2+-char alpha tokens plus digit
    groups so stem-based substring checks like ``'tree' in stem`` become
    ``token in tokens(stem)`` — more robust against pluralization and
    compound names than a naked substring match.
    """
    import re as _re

    return [tok for tok in _re.findall(r"[a-z]+|[0-9]+", stem.lower()) if tok]


def make_file_explorer_tools() -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """Create file exploration tools for asset discovery.

    Returns:
        A tuple of (tool_definitions, tool_executors) where tool_definitions
        is a list of Claude tool schema dicts and tool_executors is a dict
        mapping tool name to callable function.
    """

    def list_directory(path: str = ".") -> str:
        """List files and directories at the given path.

        Returns a formatted listing showing directories and files with sizes.
        Use this to explore project structure and discover asset files.
        """
        target = Path(path).resolve()
        if not target.exists():
            return f"Error: path does not exist: {path}"
        if not target.is_dir():
            return f"Error: not a directory: {path}"

        entries: List[str] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return f"Error: permission denied: {path}"

        for child in children:
            if child.is_dir():
                entries.append(f"[DIR]  {child.name}/")
            else:
                try:
                    size = _human_size(child.stat().st_size)
                except OSError:
                    size = "?"
                entries.append(f"[FILE] {child.name}  ({size})")

            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"... truncated ({len(list(target.iterdir()))} total entries)")
                break

        if not entries:
            return f"Directory is empty: {path}"
        return f"Contents of {path} ({len(entries)} entries):\n" + "\n".join(entries)

    def read_file_content(path: str, max_lines: int = MAX_READ_LINES) -> str:
        """Read the content of a file.

        Useful for inspecting asset configs, JSON metadata, YAML files,
        Python scripts, CSV headers, etc. Binary files return a size summary.

        Path resolution order for relative paths:
          1. As given (resolved against ``os.getcwd()``).
          2. Falls back to ``<iteration_dir>/<path>`` when the codegen has
             registered an iteration directory via ``set_iteration_dir``.
             This lets the model write ``read_file_content('step_context.json')``
             or ``read_file_content('plan.md')`` in step-mode retries even
             though the cwd is the project root, NOT the iteration directory.
             Without this fallback, retry-mode codegen burns its tool budget
             on list_directory / glob trying to locate files it was told
             were "available" by the system prompt.
        """
        original_path = path
        target = Path(path)
        if not target.is_absolute():
            cwd_resolved = (Path.cwd() / path).resolve()
            if cwd_resolved.exists():
                target = cwd_resolved
            else:
                # Fallback: try the codegen's current iteration directory.
                # Lazy import keeps file_explorer_tools standalone-importable.
                try:
                    from chrono_code.tools.code_agent_tools import _iteration_dir as _iter_dir
                except Exception:
                    _iter_dir = None
                if _iter_dir is not None:
                    iter_resolved = (Path(_iter_dir) / path).resolve()
                    if iter_resolved.exists():
                        target = iter_resolved
                    else:
                        target = cwd_resolved   # keep original error message anchored at cwd
                else:
                    target = cwd_resolved
        else:
            target = target.resolve()

        if not target.exists():
            return f"Error: file does not exist: {original_path}"
        if not target.is_file():
            return f"Error: not a file: {original_path}"

        try:
            size = target.stat().st_size
        except OSError:
            size = 0

        # Binary detection
        if target.suffix.lower() in _BINARY_EXTENSIONS:
            return f"Binary file: {path} ({_human_size(size)})"

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return f"Error: permission denied: {path}"
        except Exception as exc:
            return f"Error reading file: {type(exc).__name__}: {exc}"

        lines = text.splitlines()
        total = len(lines)
        cap = min(max_lines, MAX_READ_LINES)
        if total > cap:
            return (
                f"File: {path} ({total} lines, showing first {cap}):\n"
                + "\n".join(lines[:cap])
                + f"\n... ({total - cap} more lines)"
            )
        return f"File: {path} ({total} lines):\n" + "\n".join(lines)

    def find_files(pattern: str, search_path: str = ".") -> str:
        """Find files matching a glob pattern recursively.

        Examples:
            find_files("*.obj", "data/")  - find all OBJ mesh files
            find_files("*.json", "data/vehicle/") - find JSON configs
            find_files("robot*", "data/") - find robot-related files
        """
        root = Path(search_path).resolve()
        if not root.exists():
            return f"Error: search path does not exist: {search_path}"
        if not root.is_dir():
            return f"Error: not a directory: {search_path}"

        matches: List[str] = []
        try:
            for p in root.rglob(pattern):
                if p.is_file():
                    try:
                        rel = p.relative_to(root)
                    except ValueError:
                        rel = p
                    try:
                        size = _human_size(p.stat().st_size)
                    except OSError:
                        size = "?"
                    matches.append(f"{rel}  ({size})")
                    if len(matches) >= MAX_FIND_RESULTS:
                        break
        except PermissionError:
            return f"Error: permission denied while searching: {search_path}"

        if not matches:
            return f"No files matching '{pattern}' found in {search_path}"

        header = f"Found {len(matches)} file(s) matching '{pattern}' in {search_path}"
        if len(matches) >= MAX_FIND_RESULTS:
            header += f" (showing first {MAX_FIND_RESULTS})"
        return header + ":\n" + "\n".join(matches)

    # ── Chrono data path resolution ────────────────────────────────────────

    def _get_chrono_data_path() -> Optional[Path]:
        """Return the Chrono built-in data root, or None if unavailable.

        Resolution order:
        1. pychrono.core.GetChronoDataPath()
        2. CHRONO_DATA_DIR environment variable
        3. sys.prefix / "share/chrono/data" (conda fallback)
        """
        # 1. Try pychrono import
        try:
            import pychrono.core as chrono
            p = Path(chrono.GetChronoDataPath())
            if p.is_dir():
                return p
        except Exception:
            pass

        # 2. Environment variable
        env = os.environ.get("CHRONO_DATA_DIR")
        if env:
            p = Path(env)
            if p.is_dir():
                return p

        # 3. Conda fallback
        p = Path(sys.prefix) / "share" / "chrono" / "data"
        if p.is_dir():
            return p

        return None

    def find_assets(pattern: str = "*.obj", source: str = "all") -> str:
        """Find 3D asset files from project data/ and/or Chrono built-in data.

        Results are grouped by source. Project assets show project-relative
        paths (e.g. data/scene/office_chair/office_chair.obj). Chrono built-in
        assets show paths relative to the chrono data root
        (e.g. sensor/offroad/tree1.obj), matching chrono.GetChronoDataFile()
        convention.
        """
        sections: List[str] = []

        # ── Project assets ───────────────────────────────────────────────
        if source in ("project", "all"):
            project_root = Path("data").resolve()
            if project_root.is_dir():
                matches: List[str] = []
                for p in sorted(project_root.rglob(pattern)):
                    if p.is_file():
                        try:
                            rel = p.relative_to(project_root.parent)  # keep "data/..." prefix
                        except ValueError:
                            rel = p
                        try:
                            size = _human_size(p.stat().st_size)
                        except OSError:
                            size = "?"
                        matches.append(f"  {rel}  ({size})")
                        if len(matches) >= MAX_FIND_RESULTS:
                            break
                if matches:
                    sections.append(f"[Project] {len(matches)} file(s):\n" + "\n".join(matches))
                elif source == "project":
                    sections.append(f"[Project] No files matching '{pattern}' in data/")

        # ── Chrono built-in assets ───────────────────────────────────────
        if source in ("chrono", "all"):
            chrono_root = _get_chrono_data_path()
            if chrono_root is not None:
                matches = []
                for p in sorted(chrono_root.rglob(pattern)):
                    if p.is_file():
                        try:
                            rel = p.relative_to(chrono_root)
                        except ValueError:
                            rel = p
                        try:
                            size = _human_size(p.stat().st_size)
                        except OSError:
                            size = "?"
                        matches.append(f"  {rel}  ({size})")
                        if len(matches) >= MAX_FIND_RESULTS:
                            break
                if matches:
                    sections.append(f"[Chrono built-in] {len(matches)} file(s):\n" + "\n".join(matches))
                elif source == "chrono":
                    sections.append(f"[Chrono built-in] No files matching '{pattern}'")
            elif source == "chrono":
                sections.append("[Chrono built-in] Chrono data directory not found")

        if not sections:
            return f"No files matching '{pattern}' found (source={source})"
        return "\n\n".join(sections)

    def check_asset_path(path: str) -> str:
        """Verify that an asset file path resolves to a real file on disk
        AND report its mesh bounds when it's an OBJ.

        Tries three resolution strategies in order:

        1. Project-relative: the path is interpreted relative to the current
           working directory (typical for ``data/scene/...`` / ``data/robot/...``).
        2. Chrono data root: the path is passed to ``GetChronoDataFile()`` to
           see if it resolves under the bundled Chrono assets
           (``sensor/offroad/tree1.obj`` etc.).
        3. Absolute: if the path is already absolute, just check it directly.

        On a hit for a ``.obj`` file, the mesh is loaded via
        ``ChTriangleMeshConnected.LoadWavefrontMesh`` and its axis-aligned
        bounding-box span (x, y, z meters) is appended to the response.
        This catches the ``sensor/offroad/cottage.obj`` class of surprise —
        native dimensions 34 × 21 × 15 m, far too large for a 30×30 m scene
        without scaling. The LLM can then thread an appropriate
        ``scale=`` argument through ``AssetDescriptor`` or equivalent.

        On miss, suggests the closest catalog entries via token-overlap
        fuzzy match so the LLM can self-correct instead of writing a path
        that will SIGSEGV at ``ChSensorManager.Update()`` time.

        Designed for use by BOTH the code agent (preflight when writing a
        path into generated Python) and the code reviewer (static check on
        every OBJ / URDF literal in the emitted simulation.py).
        """
        if not path or not isinstance(path, str):
            return "Error: check_asset_path requires a non-empty string path."
        path = path.strip()

        # Strategy 1: project-relative
        project_candidate = Path(path)
        if project_candidate.is_file():
            try:
                size = _human_size(project_candidate.stat().st_size)
            except OSError:
                size = "?"
            msg = (
                f"OK (project): {path}  ({size}). "
                "Load directly via open() / mesh loaders; do NOT wrap in "
                "chrono.GetChronoDataFile() for project paths."
            )
            bbox = _describe_mesh_bounds(project_candidate)
            return msg + bbox

        # Strategy 2: Chrono data root
        chrono_root = _get_chrono_data_path()
        if chrono_root is not None:
            chrono_candidate = chrono_root / path
            if chrono_candidate.is_file():
                try:
                    size = _human_size(chrono_candidate.stat().st_size)
                except OSError:
                    size = "?"
                msg = (
                    f"OK (chrono): {path}  ({size}). "
                    "Load via chrono.GetChronoDataFile(\"" + path + "\")."
                )
                bbox = _describe_mesh_bounds(chrono_candidate)
                return msg + bbox

        # Strategy 3: absolute path already
        abs_candidate = Path(path).expanduser()
        if abs_candidate.is_absolute() and abs_candidate.is_file():
            msg = (
                f"OK (absolute): {abs_candidate}. "
                "Prefer project-relative or chrono-data-relative references "
                "in committed code."
            )
            bbox = _describe_mesh_bounds(abs_candidate)
            return msg + bbox

        # --- miss path: gather suggestions -------------------------------
        # Token-based match so singular / plural / compound stems still
        # surface useful candidates: 'trees_large' → 'trees' / 'large',
        # 'fixedterrain' → 'fixedterrain', 'tree1' → 'tree' / '1'.
        stem = Path(path).stem.lower()
        query_tokens = set(_asset_stem_tokens(stem))
        suggestions: List[str] = []

        def _is_candidate_match(candidate_stem: str) -> bool:
            cand_tokens = set(_asset_stem_tokens(candidate_stem))
            # Direct substring still matches (covers 'fixedterrain' vs 'fixedterrain.obj').
            if stem in candidate_stem or candidate_stem in stem:
                return True
            # Token overlap with prefix-tolerant matching — catches
            # singular/plural (tree/trees, rock/rocks) and compound-word
            # variants. Require both tokens ≥3 chars and one to be a prefix
            # of the other.
            for qt in query_tokens:
                if len(qt) < 3:
                    continue
                for ct in cand_tokens:
                    if len(ct) < 3:
                        continue
                    if qt == ct or qt.startswith(ct) or ct.startswith(qt):
                        return True
            return False

        # Scan project data/
        project_root = Path("data").resolve()
        if project_root.is_dir():
            for p in project_root.rglob("*.obj"):
                if p.is_file() and _is_candidate_match(p.stem.lower()):
                    try:
                        rel = p.relative_to(project_root.parent)
                    except ValueError:
                        rel = p
                    suggestions.append(f"(project) {rel}")
                    if len(suggestions) >= 6:
                        break

        # Scan chrono data root if not already saturated
        if chrono_root is not None and len(suggestions) < 6:
            for p in chrono_root.rglob("*.obj"):
                if p.is_file() and _is_candidate_match(p.stem.lower()):
                    try:
                        rel = p.relative_to(chrono_root)
                    except ValueError:
                        rel = p
                    suggestions.append(f"(chrono) {rel}")
                    if len(suggestions) >= 6:
                        break

        if suggestions:
            return (
                f"NOT FOUND: {path}\n"
                "This path does not exist as a project-relative file, a "
                "Chrono-data-relative file, or an absolute file. The "
                "simulation will fail to load this asset (often SIGSEGV at "
                "ChSensorManager.Update()).\n"
                "Closest catalog matches:\n  - "
                + "\n  - ".join(suggestions)
                + "\nPick one of these, or use find_assets() / "
                "list_chrono_assets() to browse further."
            )
        return (
            f"NOT FOUND: {path}\n"
            "No similar files found in data/ or the Chrono data root. "
            "Use find_assets(pattern=...) or list_chrono_assets() to "
            "discover what actually exists before referencing this path."
        )

    def list_chrono_assets(category: str = "") -> str:
        """List Chrono built-in asset directories and files.

        Without arguments, shows top-level categories with OBJ file counts.
        With a category path, shows the OBJ files within that category.

        Paths shown are relative to the chrono data root, matching the format
        used by chrono.GetChronoDataFile().
        """
        chrono_root = _get_chrono_data_path()
        if chrono_root is None:
            return "Error: Chrono data directory not found"

        if category:
            target = chrono_root / category
            if not target.is_dir():
                return f"Error: category not found: {category}"
            entries: List[str] = []
            for p in sorted(target.rglob("*.obj")):
                if p.is_file():
                    rel = p.relative_to(chrono_root)
                    try:
                        size = _human_size(p.stat().st_size)
                    except OSError:
                        size = "?"
                    entries.append(f"  {rel}  ({size})")
                    if len(entries) >= MAX_FIND_RESULTS:
                        break
            if not entries:
                return f"No OBJ files in {category}/"
            return f"OBJ files in {category}/ ({len(entries)}):\n" + "\n".join(entries)

        # Top-level summary: group by first path component
        counts: dict[str, int] = {}
        for p in chrono_root.rglob("*.obj"):
            if p.is_file():
                try:
                    rel = p.relative_to(chrono_root)
                except ValueError:
                    continue
                top = rel.parts[0] if rel.parts else "."
                counts[top] = counts.get(top, 0) + 1

        if not counts:
            return "No OBJ files found in Chrono data directory"

        lines = [f"Chrono built-in asset categories ({sum(counts.values())} OBJ files total):"]
        for cat in sorted(counts):
            lines.append(f"  {cat}/  ({counts[cat]} OBJ files)")
        lines.append("\nUse list_chrono_assets(category) to drill into a specific category.")
        return "\n".join(lines)

    # ── Build tool definitions and executors ──────────────────────────────

    tool_definitions: List[Dict[str, Any]] = [
        {
            "name": "list_directory",
            "description": (
                "List files and directories at the given path. "
                "Returns a formatted listing showing directories and files with sizes. "
                "Use this to explore project structure and discover asset files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory.",
                    },
                },
            },
        },
        {
            "name": "read_file_content",
            "description": (
                "Read the content of a file. "
                "Useful for inspecting asset configs, JSON metadata, YAML files, "
                "Python scripts, CSV headers, etc. Binary files return a size summary."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to return. Defaults to 200.",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "find_files",
            "description": (
                "Find files matching a glob pattern recursively. "
                "Examples: find_files('*.obj', 'data/'), find_files('*.json', 'data/vehicle/'), "
                "find_files('robot*', 'data/')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g., '*.obj', '*.json').",
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Root directory to search from. Defaults to '.'.",
                    },
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find_assets",
            "description": (
                "Find 3D asset files from project data/ and/or Chrono built-in data. "
                "Results are grouped by source. Project assets show project-relative "
                "paths (e.g. data/scene/office_chair/office_chair.obj). Chrono built-in "
                "assets show paths relative to the chrono data root "
                "(e.g. sensor/offroad/tree1.obj), matching chrono.GetChronoDataFile() convention."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g. '*.obj', 'tree*', '*.stl').",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "'project' = repo data/ only, "
                            "'chrono' = Chrono built-in only, "
                            "'all' = both (default)."
                        ),
                    },
                },
            },
        },
        {
            "name": "check_asset_path",
            "description": (
                "Verify a single asset file path (e.g. 'data/scene/cottage/cottage.obj' "
                "or 'sensor/offroad/tree1.obj') resolves to a real file under the "
                "project data/ tree, the Chrono built-in data root, or an absolute "
                "location. "
                "ON A HIT for an OBJ, the mesh is loaded and its native axis-aligned "
                "bounding-box (x × y × z meters) is reported, along with an oversize "
                "warning + suggested scale factor when any horizontal dimension "
                "exceeds 10 m. Example: `sensor/offroad/cottage.obj` reports bbox "
                "34 × 21 × 15 m — far too big for a 30 m scene without scaling. "
                "ON A MISS, returns the closest matching catalog entries by filename "
                "stem so you can self-correct. "
                "USE THIS BEFORE writing any OBJ/URDF path into generated Python: "
                "(a) hallucinated paths cause tiny_obj to fail silently, then SIGSEGV "
                "inside ChSensorManager.Update() or ChVisualSystemVSG; (b) loading "
                "an oversized mesh at scale=1.0 in a small scene produces visually "
                "broken scenes and unreviewable VLM frames."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Asset path to verify, e.g. 'data/scene/cottage/cottage.obj' "
                            "or 'sensor/offroad/tree1.obj'. Project-relative, "
                            "chrono-data-relative, or absolute."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "list_chrono_assets",
            "description": (
                "List Chrono built-in asset directories and files. "
                "Without arguments, shows top-level categories with OBJ file counts. "
                "With a category path, shows the OBJ files within that category. "
                "Paths shown are relative to the chrono data root, matching the format "
                "used by chrono.GetChronoDataFile()."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Sub-path like 'sensor/offroad', 'models', 'vehicle/hmmwv'. "
                            "Empty string shows top-level summary."
                        ),
                    },
                },
            },
        },
    ]

    tool_executors: Dict[str, Callable] = {
        "list_directory": list_directory,
        "read_file_content": read_file_content,
        "find_files": find_files,
        "find_assets": find_assets,
        "check_asset_path": check_asset_path,
        "list_chrono_assets": list_chrono_assets,
    }

    return tool_definitions, tool_executors


