"""Static AST validator for calls to chrono_code.utils.* functions.

Parses generated simulation code, resolves every name that refers to a
public `chrono_code.utils` symbol (across the supported import styles),
and binds each call site against the real `inspect.Signature`. A mismatch
(unknown keyword, too many positional args, missing required arg) produces
a `UsabilityIssue` that the codegen fix loop feeds back to the model as a
hard error.

This is the "post-generation type-check" layer in the
"signature exposure + on-demand lookup + post-check" standard triplet; it
catches the "misused argument" class of failures that plain signature
injection alone does not.
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from chrono_code.utils._signature_digest import get_signatures, get_fq_signatures

logger = logging.getLogger(__name__)


_UTILS_PKG = "chrono_code.utils"


@dataclass
class UtilsCallIssue:
    """A single mismatched call to a chrono_code.utils symbol."""

    lineno: int
    col: int
    callee_display: str  # e.g. "setup_preview_camera" or "utils.FootprintRegistry"
    canonical: str        # canonical short name we matched against (e.g. "setup_preview_camera")
    signature: str        # rendered expected signature, e.g. "(viewer, *, attach_body, ...)"
    reason: str           # human-readable error from Signature.bind

    def format(self) -> str:
        return (
            f"[UtilsCallError] line {self.lineno}: "
            f"`{self.callee_display}` — {self.reason}. "
            f"Expected: `{self.canonical}{self.signature}`."
        )


def validate_utils_calls(code: str) -> List[UtilsCallIssue]:
    """Return a list of mismatched util calls in `code` (empty if all OK).

    The function is tolerant of syntax errors (returns [] — the python
    compile check already catches those separately) and of unresolvable
    names (skips them rather than false-flagging).
    """
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    short_sigs = get_signatures()
    fq_sigs = get_fq_signatures()
    if not short_sigs and not fq_sigs:
        return []

    resolver = _ImportResolver()
    resolver.visit(tree)

    issues: List[UtilsCallIssue] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = resolver.resolve_call(node.func)
        if resolved is None:
            continue
        canonical_name, display = resolved
        sig = short_sigs.get(canonical_name) or fq_sigs.get(canonical_name)
        if sig is None:
            continue
        issue = _bind_and_diagnose(node, sig, canonical_name, display)
        if issue is not None:
            issues.append(issue)
    return issues


def format_issues_for_feedback(issues: List[UtilsCallIssue]) -> str:
    """Render issues as a compilation_feedback-ready text block."""
    if not issues:
        return ""
    lines = [
        "Static validation found incorrect calls to chrono_code.utils API. "
        "Fix these before any other change (do NOT reimplement the utility inline):",
    ]
    for issue in issues:
        lines.append(f"- {issue.format()}")
    lines.append(
        "Re-read the signatures in the UTILS API REFERENCE section of the "
        "system prompt and adjust the call sites."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _ImportResolver(ast.NodeVisitor):
    """Walk imports once, then resolve call targets on demand.

    Supports:
      * `from chrono_code.utils import X`                        -> X
      * `from chrono_code.utils import X as Y`                   -> Y -> X
      * `from chrono_code.utils.scene_assets import X`           -> X (fq via submodule)
      * `from chrono_code.utils.scene_assets import X as Y`      -> Y -> X
      * `import chrono_code.utils as u`                          -> u.X
      * `import chrono_code.utils.scene_assets as sa`            -> sa.X
      * `import chrono_code.utils`                               -> chrono_code.utils.X
      * `import chrono_code.utils.scene_assets`                  -> chrono_code.utils.scene_assets.X
    """

    def __init__(self) -> None:
        # local_name -> canonical_short_name (lookup in short_sigs)
        self._short_aliases: Dict[str, str] = {}
        # local_name -> fully_qualified_module_path (lookup via {module}.{attr} in fq_sigs)
        self._module_aliases: Dict[str, str] = {}

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802 (ast API)
        mod = node.module or ""
        if not (mod == _UTILS_PKG or mod.startswith(_UTILS_PKG + ".")):
            return
        for alias in node.names:
            local = alias.asname or alias.name
            # For `from chrono_code.utils import X` we trust the short name;
            # for `from chrono_code.utils.submod import X` we also key on the
            # short name (utils __all__ is flat).
            self._short_aliases[local] = alias.name
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            full = alias.name
            if full != _UTILS_PKG and not full.startswith(_UTILS_PKG + "."):
                continue
            if alias.asname:
                self._module_aliases[alias.asname] = full
            else:
                # `import a.b.c` binds the top-level `a` locally but `a.b.c.X`
                # is still reachable as an Attribute chain; we record the full
                # path so attribute-chain resolution below can match it.
                self._module_aliases[full] = full
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Call-site resolution
    # ------------------------------------------------------------------

    def resolve_call(self, func: ast.AST) -> Optional[Tuple[str, str]]:
        """Return (canonical_lookup_key, display_name) or None.

        The lookup key is tried first against `get_signatures()` (short
        keyed by __all__ name) and falls back to `get_fq_signatures()`
        (fully-qualified `module.attr`).
        """
        if isinstance(func, ast.Name):
            canonical = self._short_aliases.get(func.id)
            if canonical is not None:
                return canonical, func.id
            return None

        if isinstance(func, ast.Attribute):
            chain = _attr_chain(func)
            if chain is None:
                return None
            head, *rest = chain
            if not rest:
                return None
            attr = rest[-1]
            prefix_parts = [head] + rest[:-1]
            prefix = ".".join(prefix_parts)

            # Case 1: `u.X` where u is alias for chrono_code.utils(.sub)
            mod_path = self._module_aliases.get(head)
            if mod_path is not None:
                middle = ".".join(rest[:-1])
                full_module = mod_path if not middle else f"{mod_path}.{middle}"
                display = f"{prefix}.{attr}"
                if not _looks_like_utils(full_module):
                    return None
                # Package-level access (`u.X` where u == chrono_code.utils) goes
                # through the re-exported flat __all__, so we key on the short
                # name. Submodule access falls back to the fully-qualified key.
                if full_module == _UTILS_PKG:
                    return attr, display
                return f"{full_module}.{attr}", display

            # Case 2: attribute chain starting with a bare package name, e.g.
            # `chrono_code.utils.scene_assets.X` when user wrote
            # `import chrono_code.utils.scene_assets`. The head `chrono_code`
            # is recorded verbatim in _module_aliases (see visit_Import).
            if head == "chrono_code":
                full_module = ".".join([head] + rest[:-1])
                if _looks_like_utils(full_module):
                    display = f"{full_module}.{attr}"
                    # Same flat-vs-submodule split as Case 1.
                    if full_module == _UTILS_PKG:
                        return attr, display
                    return f"{full_module}.{attr}", display

        return None


def _attr_chain(node: ast.AST) -> Optional[List[str]]:
    """Flatten an Attribute/Name chain into a list of identifiers.

    `a.b.c` -> ["a", "b", "c"]. Returns None on expressions that aren't
    pure attribute access (e.g. `foo().bar`).
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    parts.reverse()
    return parts


def _looks_like_utils(module_path: str) -> bool:
    return module_path == _UTILS_PKG or module_path.startswith(_UTILS_PKG + ".")


def _bind_and_diagnose(
    call: ast.Call,
    sig: inspect.Signature,
    canonical: str,
    display: str,
) -> Optional[UtilsCallIssue]:
    """Try to bind AST args against `sig`; return issue on mismatch.

    We do an explicit unknown-kwarg pre-check before delegating to
    `Signature.bind`, because bind() raises "missing a required argument"
    as soon as it hits a gap — which hides the real problem when the
    agent typoed the kwarg name (the observed failure mode).
    """
    pos_args: List[object] = []
    kw_args: Dict[str, object] = {}
    for a in call.args:
        if isinstance(a, ast.Starred):
            return None  # can't statically validate
        pos_args.append(_SENTINEL)
    for kw in call.keywords:
        if kw.arg is None:
            return None  # **kwargs expansion -> bail out
        kw_args[kw.arg] = _SENTINEL

    # Pre-check: unknown keyword arguments, unless the signature accepts **kwargs.
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_keyword:
        allowed_kw = {
            name
            for name, p in sig.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        for kwname in kw_args:
            if kwname not in allowed_kw:
                return UtilsCallIssue(
                    lineno=getattr(call, "lineno", 0),
                    col=getattr(call, "col_offset", 0),
                    callee_display=display,
                    canonical=canonical,
                    signature=str(sig),
                    reason=f"got an unexpected keyword argument '{kwname}'",
                )

    try:
        sig.bind(*pos_args, **kw_args)
    except TypeError as exc:
        return UtilsCallIssue(
            lineno=getattr(call, "lineno", 0),
            col=getattr(call, "col_offset", 0),
            callee_display=display,
            canonical=canonical,
            signature=str(sig),
            reason=str(exc),
        )
    return None


_SENTINEL = object()
