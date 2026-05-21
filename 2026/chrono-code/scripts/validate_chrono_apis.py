#!/usr/bin/env python3
"""
Validate PyChrono API calls in a Python script by introspecting pychrono at runtime.

Tracks variable types from constructor calls and method-return assignments so
instance methods on saved temporaries can also be validated (e.g.
``ground_vis = ground.GetVisualShape(0); ground_vis.SetMaterial(...)``).

Usage:
    python scripts/validate_chrono_apis.py outputs/simulation.py
"""

import ast
import difflib
import re
import sys
import importlib

# Cache for validated lookups
_cache = {}
_return_type_cache = {}


def get_pychrono_attr(path: str, module_name: str = "pychrono.core"):
    """Dynamically look up an attribute from a pychrono module.

    *path* is the dotted attribute path (e.g. ``'ChSystemNSC'`` or
    ``'ChCollisionSystem.Type_BULLET'``).  *module_name* is the full
    Python module to import (e.g. ``'pychrono.core'``, ``'pychrono.vsg3d'``).
    """
    cache_key = (module_name, path)
    if cache_key in _cache:
        return _cache[cache_key], None

    parts = path.split(".")
    attr_name = parts[-1]

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None, f"Cannot import {module_name}"

    obj = mod
    for part in parts[:-1]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            _cache[cache_key] = None
            return None, f"Module/Class not found: {'.'.join(parts[:-1])}"

    try:
        result = getattr(obj, attr_name)
        _cache[cache_key] = result
        return result, None
    except AttributeError:
        _cache[cache_key] = None
        candidates = [a for a in dir(obj) if not a.startswith('__')]
        suggestions = difflib.get_close_matches(attr_name, candidates, n=3, cutoff=0.5)
        msg = f"Attribute '{attr_name}' not found in {obj}"
        if suggestions:
            msg += f". SUGGESTION: Did you mean {' or '.join(repr(s) for s in suggestions)}?"
        return None, msg


def has_attr(obj, attr: str) -> bool:
    """Check if obj has the given attribute."""
    try:
        getattr(obj, attr)
        return True
    except AttributeError:
        return False


def get_class_methods(cls) -> set:
    """Get all methods/attributes of a class including its MRO."""
    methods = set()
    for c in cls.__mro__:
        if hasattr(c, "__dict__"):
            methods.update(c.__dict__.keys())
    return methods


def suggest_similar_methods(cls, method_name: str, n: int = 3, cutoff: float = 0.5) -> list[str]:
    """Find method names similar to *method_name* using fuzzy matching."""
    methods = get_class_methods(cls)
    if not method_name.startswith('_'):
        candidates = [m for m in methods if not m.startswith('_')]
    else:
        candidates = list(methods)
    return difflib.get_close_matches(method_name, candidates, n=n, cutoff=cutoff)


def is_valid_method(cls, method_name: str) -> tuple[bool, str]:
    """Check if method_name is a valid method on cls or any class in its MRO."""
    methods = get_class_methods(cls)
    if method_name in methods:
        return True, ""
    suggestions = suggest_similar_methods(cls, method_name)
    msg = f"'{method_name}' not found in class hierarchy of {cls.__name__}"
    if suggestions:
        msg += f". SUGGESTION: Did you mean {' or '.join(repr(s) for s in suggestions)}?"
    return False, msg


# ── Argument count validation from SWIG docstrings ────────────────────────────

_sig_cache: dict[tuple, list[int]] = {}


def _split_params(params_str: str) -> list[str]:
    """Split parameter string on commas, respecting nested parentheses.

    SWIG default values can contain commas inside parentheses, e.g.
    ``ChQuaterniond q=chrono::ChQuaternion(1, 0, 0, 0)``.  A naive
    ``split(',')`` would break that into 4 spurious parameters.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())
    return parts


def _find_matching_close_paren(line: str, open_pos: int) -> int:
    """Find the closing ')' that matches the '(' at *open_pos*, respecting nesting."""
    depth = 0
    for i in range(open_pos, len(line)):
        if line[i] == '(':
            depth += 1
        elif line[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _parse_swig_arg_counts(obj) -> list[int]:
    """Parse SWIG docstring to extract the allowed argument counts (excluding self).

    SWIG docstrings follow a regular format::

        MethodName(ClassName self, double x, double y)
        MethodName(ClassName self, double x, double y, double z)

    Each line is an overload.  We count commas after ``self`` to get the
    number of required positional parameters for each overload, and return
    a sorted, deduplicated list of allowed counts.
    """
    doc = getattr(obj, '__doc__', '') or ''
    if not doc:
        return []

    counts: set[int] = set()
    for line in doc.splitlines():
        line = line.strip()
        # Match "Name(..." pattern
        paren = line.find('(')
        if paren < 0:
            continue
        close = _find_matching_close_paren(line, paren)
        if close <= paren:
            continue
        params_str = line[paren + 1:close].strip()
        if not params_str:
            counts.add(0)
            continue
        params = _split_params(params_str)
        # Remove 'self' parameter (always first for instance methods)
        if params and 'self' in params[0]:
            params = params[1:]
        # Count required (no default) vs optional (has '=') parameters.
        # Allow any count from n_required to n_total.
        n_required = sum(1 for p in params if '=' not in p)
        n_total = len(params)
        for n in range(n_required, n_total + 1):
            counts.add(n)
    return sorted(counts)


def get_allowed_arg_counts(cls_or_func, method_name: str | None = None) -> list[int]:
    """Return the list of allowed positional argument counts for a callable.

    Looks up the SWIG docstring and parses overload signatures.  Results are
    cached for performance.
    """
    key = (id(cls_or_func), method_name)
    if key in _sig_cache:
        return _sig_cache[key]

    if method_name is not None:
        # Instance method: find in MRO
        obj = None
        if isinstance(cls_or_func, type):
            for c in cls_or_func.__mro__:
                if method_name in getattr(c, '__dict__', {}):
                    obj = c.__dict__[method_name]
                    break
        if obj is None:
            obj = getattr(cls_or_func, method_name, None)
    else:
        obj = cls_or_func
        # For class constructor calls (e.g. chrono.ChBodyEasyBox(...)),
        # the signatures are on __init__, not the class __doc__.
        if isinstance(obj, type) and hasattr(obj, '__init__'):
            obj = obj.__init__

    if obj is None:
        _sig_cache[key] = []
        return []

    counts = _parse_swig_arg_counts(obj)
    _sig_cache[key] = counts
    return counts


def check_arg_count(cls_or_func, method_name: str | None, n_args: int) -> tuple[bool, str]:
    """Validate the number of positional arguments against SWIG overloads.

    Returns (True, "") if valid or unknown, (False, message) if definitely wrong.
    """
    allowed = get_allowed_arg_counts(cls_or_func, method_name)
    if not allowed:
        # Cannot determine — don't flag
        return True, ""
    if n_args in allowed:
        return True, ""
    target = method_name or getattr(cls_or_func, '__name__', str(cls_or_func))
    return False, (
        f"'{target}' expects {' or '.join(str(a) for a in allowed)} args, got {n_args}"
    )


def resolve_method_return_type(cls, method_name: str, modules=None):
    """Try to determine the return type of cls.method_name by parsing SWIG docstrings.

    *modules* is an iterable of pychrono module paths to search for the return
    type (e.g. ``["pychrono.core", "pychrono.vehicle"]``).  When ``None``,
    only ``pychrono.core`` is searched — which misses types defined in
    sub-modules like ``pychrono.vehicle.ChWheeledVehicle``.
    """
    cache_key = (cls, method_name)
    if cache_key in _return_type_cache:
        return _return_type_cache[cache_key]

    # Find the method in the MRO
    method = None
    for c in cls.__mro__:
        if method_name in getattr(c, '__dict__', {}):
            method = c.__dict__[method_name]
            break

    if method is None:
        _return_type_cache[cache_key] = None
        return None

    doc = getattr(method, '__doc__', '') or ''
    # Match SWIG-style return types: "-> ClassName" or "-> std::shared_ptr< chrono::ClassName >"
    match = re.search(r'->\s*(?:std::shared_ptr<\s*chrono::(\w+)\s*>|(\w+))', doc)
    if match:
        type_name = match.group(1) or match.group(2)
        # Search all known pychrono modules, not just pychrono.core.
        search_modules = list(modules or []) + ["pychrono.core"]
        # Deduplicate while preserving order.
        seen = set()
        for mod in search_modules:
            if mod in seen:
                continue
            seen.add(mod)
            ret_cls, _ = get_pychrono_attr(type_name, mod)
            if ret_cls is not None and isinstance(ret_cls, type):
                _return_type_cache[cache_key] = ret_cls
                return ret_cls

    _return_type_cache[cache_key] = None
    return None


# ── AST Visitor ────────────────────────────────────────────────────────────────

# Module alias literals that are conventional pychrono import targets in
# this codebase (mirrors the system prompt's import rule and the helper
# imports in ``chrono_agent.utils``). When code uses ``veh.WheeledVehicle``
# or ``fsi.ChFsiSystemSPH`` etc. without binding the alias via
# ``import pychrono.vehicle as veh``, the runtime raises ``NameError`` —
# the visitor catches that statically here so codegen sees the bug on the
# same turn it wrote the code, not on the next execution iteration.
_KNOWN_PYCHRONO_ALIASES: dict[str, str] = {
    "chrono": "pychrono.core",
    "chronovsg": "pychrono.vsg3d",
    "veh": "pychrono.vehicle",
    "sens": "pychrono.sensor",
    "fsi": "pychrono.fsi",
    "fea": "pychrono.fea",
    "robot": "pychrono.robot",
    "irr": "pychrono.irrlicht",
}


def _undefined_alias_error(alias: str, attr: str) -> str:
    """Format the canonical 'unbound module alias' error message."""
    expected = _KNOWN_PYCHRONO_ALIASES.get(alias, "pychrono.<module>")
    return (
        f"Module alias '{alias}' is not imported (used as '{alias}.{attr}'). "
        f"Add `import {expected} as {alias}` to the imports section."
    )


class ModuleAliasTable:
    """Tracks import aliases → full pychrono module names.

    Always contains the implicit ``chrono`` → ``pychrono.core`` mapping.
    Additional entries are discovered from ``import pychrono.xxx as alias``
    statements in the analysed file.
    """

    # Known pychrono import patterns: ``import pychrono as chrono`` and
    # ``import pychrono.core as chrono`` both map to ``pychrono.core``.
    _CORE_ALIASES = {"chrono"}

    def __init__(self):
        self._map: dict[str, str] = {"chrono": "pychrono.core"}

    def add(self, alias: str, module_path: str):
        self._map[alias] = module_path

    def get(self, alias: str) -> str | None:
        return self._map.get(alias)

    def is_known(self, alias: str) -> bool:
        return alias in self._map

    def all_modules(self) -> list[str]:
        """Return all known pychrono module paths (deduplicated)."""
        return list(dict.fromkeys(self._map.values()))


class SymbolTable:
    """Tracks variable name → Chrono class/type mappings."""

    def __init__(self):
        # var_name -> class (the Chrono class, e.g. ChBody)
        self._vars = {}

    def add(self, name: str, chrono_class):
        self._vars[name] = chrono_class

    def get(self, name: str):
        return self._vars.get(name)


class ValidateVisitor(ast.NodeVisitor):
    def __init__(self, code):
        self.code = code
        self.lines = code.splitlines()
        self.modules = ModuleAliasTable()
        self.symbols = SymbolTable()
        self.issues = []  # list of (line, chain, msg)
        self.valid_chains = set()
        # ``ast.Attribute`` ids already validated by ``visit_Call`` so the
        # generic_visit recursion into the call's func subtree doesn't
        # re-validate the same chain via ``visit_Attribute``.
        self._covered_attr_nodes: set[int] = set()

    # ── Import discovery ──────────────────────────────────────────────────────

    def visit_Import(self, node):
        """Handle ``import pychrono.xxx as alias``."""
        for alias_node in node.names:
            name = alias_node.name  # e.g. "pychrono.vsg3d"
            asname = alias_node.asname  # e.g. "chronovsg"
            if name.startswith("pychrono") and asname:
                # Normalise bare "pychrono" to "pychrono.core"
                mod = name if name != "pychrono" else "pychrono.core"
                self.modules.add(asname, mod)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle ``from pychrono import xxx``  (less common, but possible)."""
        self.generic_visit(node)

    # ── Assignment tracking ───────────────────────────────────────────────────

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                rhs_type = self._resolve_expr_type(node.value)
                if rhs_type is not None:
                    self.symbols.add(target.id, rhs_type)
        self.generic_visit(node)

    def _resolve_rhs_class(self, node):
        """Resolve an ``alias.XXX()`` call to its class."""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                mod = self.modules.get(func.value.id)
                if mod is not None:
                    cls, _ = get_pychrono_attr(func.attr, mod)
                    if cls is not None and isinstance(cls, type):
                        return cls
        return None

    def _resolve_expr_type(self, node):
        """Resolve the Chrono type of an arbitrary expression node."""
        if isinstance(node, ast.Name):
            return self.symbols.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Recursively resolve the type of the object, then get the return type
            obj_type = self._resolve_expr_type(node.func.value)
            if obj_type is not None:
                return resolve_method_return_type(
                    obj_type, node.func.attr, self.modules.all_modules()
                )
            # alias.XXX() — constructor call
            if isinstance(node.func.value, ast.Name):
                mod = self.modules.get(node.func.value.id)
                if mod is not None:
                    cls, _ = get_pychrono_attr(node.func.attr, mod)
                    if cls is not None and isinstance(cls, type):
                        return cls
        return None

    def _format_chain(self, node):
        """Build a human-readable chain string from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._format_chain(node.value)
            return f"{base}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{self._format_chain(node.func)}()"
        return "..."

    def _record_valid(self, chain):
        self.valid_chains.add(chain)

    def _get_module_for_alias(self, name: str) -> str | None:
        """Return the pychrono module path for *name* if it is a known alias."""
        return self.modules.get(name)

    def _mark_chain_covered(self, node: ast.AST) -> None:
        """Mark every ``ast.Attribute`` node along an attribute chain so that
        ``visit_Attribute`` skips them on the generic_visit recursion."""
        cur = node
        while isinstance(cur, ast.Attribute):
            self._covered_attr_nodes.add(id(cur))
            cur = cur.value

    @staticmethod
    def _count_call_args(node: ast.Call) -> int:
        """Count positional arguments in an AST Call node."""
        return len(node.args)

    def _check_args(self, node: ast.Call, cls_or_func, method_name: str | None, display_chain: str):
        """Validate argument count against SWIG docstring signatures."""
        n_args = self._count_call_args(node)
        ok, err = check_arg_count(cls_or_func, method_name, n_args)
        if not ok:
            self.issues.append((node.lineno, display_chain, err))

    def visit_Call(self, node):
        func = node.func

        if isinstance(func, ast.Attribute):
            # Case 1: alias.XXX() — direct class call like chrono.ChBody() / chronovsg.ChVisualSystemVSG()
            if isinstance(func.value, ast.Name) and (mod := self._get_module_for_alias(func.value.id)):
                chain = func.attr
                val, err = get_pychrono_attr(chain, mod)
                if val is not None:
                    self._record_valid(chain)
                    self._check_args(node, val, None, f"{func.value.id}.{chain}")
                else:
                    self.issues.append((node.lineno, f"{func.value.id}.{chain}", err))
                # Suppress duplicate validation by visit_Attribute on recursion.
                self._covered_attr_nodes.add(id(func))

            # Case 2: var.XXX() — instance method call like body.SetName()
            elif isinstance(func.value, ast.Name):
                var_name = func.value.id
                method = func.attr
                var_class = self.symbols.get(var_name)
                if var_class is not None:
                    ok, err = is_valid_method(var_class, method)
                    if ok:
                        self._record_valid(f"{var_class.__name__}.{method}")
                        self._check_args(node, var_class, method, f"{var_name}.{method}")
                    else:
                        self.issues.append((node.lineno, f"{var_name}.{method}", err))
                elif var_name in _KNOWN_PYCHRONO_ALIASES:
                    # Conventional pychrono alias used without ``import ... as <name>``.
                    # This was the iter_002 NameError(veh) regression in
                    # session_20260429_112754: validator silently passed the
                    # call because var_class was None.
                    self.issues.append(
                        (
                            node.lineno,
                            f"{var_name}.{method}",
                            _undefined_alias_error(var_name, method),
                        )
                    )
                    # Suppress duplicate from visit_Attribute on generic_visit recursion.
                    self._covered_attr_nodes.add(id(func))

            # Case 3: alias.XXX.YYY() — chained class call like chrono.ChCollisionSystem.Type_BULLET()
            elif isinstance(func.value, ast.Attribute):
                base = func.value
                if isinstance(base.value, ast.Name) and (mod := self._get_module_for_alias(base.value.id)):
                    full_chain = f"{base.attr}.{func.attr}"
                    cls, err = get_pychrono_attr(full_chain, mod)
                    if cls is not None:
                        self._record_valid(full_chain)
                    else:
                        # Maybe it's alias.ChSystemNSC().Something() — instance method on result
                        base_cls, _ = get_pychrono_attr(base.attr, mod)
                        if base_cls is not None and isinstance(base_cls, type):
                            ok, err = is_valid_method(base_cls, func.attr)
                            if ok:
                                self._record_valid(f"{base.attr}.{func.attr}")
                                self._check_args(node, base_cls, func.attr, f"{base.attr}.{func.attr}")
                            else:
                                self.issues.append((node.lineno, f"{base.attr}.{func.attr}", err))
                        else:
                            self.issues.append((node.lineno, full_chain, err))
                    # Cover both ``func`` (alias.X.Y) and ``func.value`` (alias.X)
                    # so ``visit_Attribute`` doesn't double-report on recursion.
                    self._mark_chain_covered(func)
                else:
                    # Case 4: expr.method() — chained calls like system.GetX().SetY()
                    obj_type = self._resolve_expr_type(func.value)
                    if obj_type is not None:
                        method = func.attr
                        chain = self._format_chain(node.func)
                        ok, err = is_valid_method(obj_type, method)
                        if ok:
                            self._record_valid(f"{obj_type.__name__}.{method}")
                            self._check_args(node, obj_type, method, chain)
                        else:
                            self.issues.append((node.lineno, chain, err))

            # Case 5: result_of_call.method() — e.g. system.GetX().SetY() when value is a Call
            elif isinstance(func.value, ast.Call):
                obj_type = self._resolve_expr_type(func.value)
                if obj_type is not None:
                    method = func.attr
                    chain = self._format_chain(node.func)
                    ok, err = is_valid_method(obj_type, method)
                    if ok:
                        self._record_valid(f"{obj_type.__name__}.{method}")
                        self._check_args(node, obj_type, method, chain)
                    else:
                        self.issues.append((node.lineno, chain, err))

        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Skip if a longer enclosing chain has already been validated (e.g.
        # ``visit_Call`` handled ``alias.X.Y()`` and marked the inner Attributes).
        if id(node) in self._covered_attr_nodes:
            self.generic_visit(node)
            return

        # Walk down the attribute chain to find the root. Handles arbitrary
        # depth: ``alias.X``, ``alias.X.Y``, ``alias.X.Y.Z`` — the last form
        # covers enum values used as arguments, e.g.
        # ``sysMBS.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)``.
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value

        if isinstance(cur, ast.Name) and (mod := self._get_module_for_alias(cur.id)):
            parts.reverse()
            chain = ".".join(parts)
            val, err = get_pychrono_attr(chain, mod)
            if val is not None:
                self._record_valid(chain)
            else:
                self.issues.append((node.lineno, f"{cur.id}.{chain}", err))
            # Mark every Attribute node in this chain so the generic_visit
            # recursion (which descends into ``node.value`` etc.) does not
            # re-validate any prefix and produce duplicate issues.
            self._mark_chain_covered(node)
        elif isinstance(cur, ast.Name) and cur.id in _KNOWN_PYCHRONO_ALIASES:
            # ``veh.QUNIT`` / ``fsi.IntegrationScheme_RK2`` style attribute
            # access on an unbound conventional alias — same NameError class
            # of bug as the visit_Call branch above, but for non-call sites
            # (enum values, constants).
            parts.reverse()
            chain = ".".join(parts)
            self.issues.append(
                (
                    node.lineno,
                    f"{cur.id}.{chain}",
                    _undefined_alias_error(cur.id, chain),
                )
            )
            self._mark_chain_covered(node)

        self.generic_visit(node)


def validate_file(path: str):
    with open(path) as f:
        code = f.read()

    tree = ast.parse(code)
    visitor = ValidateVisitor(code)
    visitor.visit(tree)

    issues = sorted(visitor.issues, key=lambda x: x[0])

    valid = sorted(visitor.valid_chains)
    return valid, issues


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_chrono_apis.py <file.py>")
        sys.exit(1)

    path = sys.argv[1]
    valid, issues = validate_file(path)

    print(f"Validating: {path}\n")
    print("=" * 70)
    print(f"VALID: {len(valid)}  |  INVALID: {len(issues)}")
    print("=" * 70)

    if issues:
        print(f"\n{'LINE':<6} {'CHAIN':<45} ERROR")
        print("-" * 70)
        for line, chain, err in issues:
            # Show the actual line for context
            print(f"{line:<6} {chain:<45}")
            print(f"       → {err}")
            print()
