"""
Utilities for deterministic error compaction and fingerprinting.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, List, Tuple, Optional

# Error classification patterns for PyChrono errors
ERROR_CLASSIFICATION_PATTERNS = {
    "binding_type_mismatch": [
        # Python binding type mismatch where C++/wrapped types leak into the error
        r"(?:TypeError|ArgumentError):.*(?:SColorf|ChColorf|ChColor|tuple|list)",
        r"(?:TypeError|ArgumentError):.*(?:incompatible function arguments|overload|no matching function)",
        r"(?:irr::video::SColorf|SColorf)",
    ],
    "simple_attribute_error": [
        # AttributeError where the object type and missing attribute are clear
        r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
    ],
    "name_error": [
        # NameError - variable not defined
        r"NameError: name '(\w+)' is not defined",
    ],
    "import_error": [
        # Module not found
        r"ModuleNotFoundError: No module named 'pychrono\.(\w+)'",
        r"ImportError: cannot import name '(\w+)' from 'pychrono",
    ],
    "type_error_args": [
        # Missing required arguments
        r"TypeError: .* missing required argument",
        r"TypeError: .* got an unexpected keyword argument",
    ],
    "syntax_error": [
        r"SyntaxError:",
    ],
}


def classify_error(feedback: Any) -> str:
    """
    Classify an error into a category that determines fix strategy.

    Returns one of:
    - "binding_type_mismatch": Binding-layer type/signature uncertainty
    - "simple_attribute_error": Clear AttributeError - 1 LLM call with direct patch
    - "name_error": NameError - often missing import or typo
    - "import_error": Import/Module error - direct fix possible
    - "type_error_args": TypeError with args - 1 LLM call
    - "complex": Other errors requiring full tool loop
    """
    primary_text, _ = _to_text_blocks(feedback)
    merged = primary_text or str(feedback) or ""

    for category, patterns in ERROR_CLASSIFICATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, merged, re.IGNORECASE):
                return category

    return "complex"


def extract_error_details(feedback: Any) -> dict:
    """
    Extract structured details from an error for fast patching.

    Returns dict with:
    - exception_type: e.g., "AttributeError"
    - object_type: e.g., "ChVector3d" (for AttributeError)
    - missing_attr: e.g., "Normalize" (for AttributeError)
    - missing_name: e.g., "chrono" (for NameError)
    - user_frame: file:line location
    """
    primary_text, backtrace_text = _to_text_blocks(feedback)
    merged = primary_text or str(feedback) or ""

    details = {
        "exception_type": _extract_exception_type(merged),
        "object_type": None,
        "missing_attr": None,
        "missing_name": None,
        "user_frame": _extract_user_frame(backtrace_text, primary_text),
        "core_error": _extract_core_line(primary_text or merged),
    }

    # Extract object type and missing attribute from AttributeError
    attr_match = re.search(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", merged)
    if attr_match:
        details["object_type"] = attr_match.group(1)
        details["missing_attr"] = attr_match.group(2)

    # Extract missing name from NameError
    name_match = re.search(r"NameError: name '(\w+)' is not defined", merged)
    if name_match:
        details["missing_name"] = name_match.group(1)

    return details


def _to_text_blocks(feedback: Any) -> Tuple[str, str]:
    """Normalize feedback into (primary_text, backtrace_text)."""
    if feedback is None:
        return "", ""

    if isinstance(feedback, dict):
        lines: List[str] = []
        primary = str(feedback.get("feedback_text") or "").strip()
        if primary:
            lines.append(primary)
        for issue in feedback.get("issues", []) or []:
            if isinstance(issue, dict):
                desc = str(issue.get("description") or "").strip()
            else:
                desc = str(issue).strip()
            if desc:
                lines.append(desc)
        backtrace = str(feedback.get("backtrace") or "")
        return "\n".join(lines).strip(), backtrace.strip()

    text = str(feedback).strip()
    return text, ""


def _extract_exception_type(text: str) -> str:
    patterns = [
        r"\b([A-Za-z_]\w*(?:Error|Exception))\b",
        r"\b(SIG[A-Z]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "UnknownError"


def _extract_core_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Unknown error"

    preferred_patterns = [
        r"\b[A-Za-z_]\w*(?:Error|Exception)\b",
        r"takes .* positional arguments? but .* were given",
        r"invalid null reference",
        r"Traceback \(most recent call last\):",
    ]
    for pattern in preferred_patterns:
        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                return line[:240]
    return lines[0][:240]


def _extract_user_frame(backtrace_text: str, primary_text: str) -> str:
    merged = f"{backtrace_text}\n{primary_text}".strip()
    if not merged:
        return "unknown"

    frame_pattern = re.compile(r'File "([^"]+)", line (\d+)(?:, in ([^\n]+))?')
    frames = frame_pattern.findall(merged)
    if not frames:
        return "unknown"

    # Prefer user code frames.
    for file_path, line_no, func in reversed(frames):
        normalized = file_path.replace("\\", "/")
        if normalized.endswith("simulation.py"):
            func_part = f" in {func.strip()}" if func else ""
            return f"{normalized}:{line_no}{func_part}"

    file_path, line_no, func = frames[-1]
    func_part = f" in {func.strip()}" if func else ""
    return f"{file_path}:{line_no}{func_part}"


def compact_error(feedback: Any, max_context_lines: int = 10) -> str:
    """
    Build a concise, structured error summary.

    Output keeps only key fields:
    - exception type
    - core error line
    - last user-code frame (if available)
    - a few deduplicated context lines
    """
    primary_text, backtrace_text = _to_text_blocks(feedback)
    merged = f"{primary_text}\n{backtrace_text}".strip()
    if not merged:
        return ""

    exception_type = _extract_exception_type(merged)
    core_line = _extract_core_line(primary_text or merged)
    user_frame = _extract_user_frame(backtrace_text, primary_text)

    context_candidates: List[str] = []
    for line in (primary_text + "\n" + backtrace_text).splitlines():
        s = line.strip()
        if not s:
            continue
        if (
            "Traceback" in s
            or "File " in s
            or "Error" in s
            or "Exception" in s
            or "invalid" in s.lower()
            or "failed" in s.lower()
        ):
            context_candidates.append(s)

    deduped: List[str] = []
    seen = set()
    for line in context_candidates:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
        if len(deduped) >= max(1, max_context_lines):
            break

    lines = [
        f"exception_type: {exception_type}",
        f"core_error: {core_line}",
        f"user_frame: {user_frame}",
    ]
    if deduped:
        lines.append("context:")
        lines.extend(f"- {line}" for line in deduped)
    return "\n".join(lines)


def get_user_frame_from_feedback(feedback: Any) -> str:
    """
    Extract the user-code frame (e.g. simulation.py:123) from feedback.
    Returns empty string if not found.
    """
    primary_text, backtrace_text = _to_text_blocks(feedback)
    return _extract_user_frame(backtrace_text, primary_text)


def extract_code_snippet_at_frame(
    code: str, user_frame: str, context_lines: int = 5
) -> str:
    """
    Extract code lines around the error location from user_frame.

    user_frame format: "path/simulation.py:123" or "path/simulation.py:123 in func"
    Returns formatted snippet with line numbers, or empty string if parsing fails.
    """
    if not code or not user_frame or user_frame == "unknown":
        return ""
    match = re.search(r":(\d+)(?:\s+in\s+|\s*$)", user_frame)
    if not match:
        return ""
    try:
        line_no = int(match.group(1))
    except (ValueError, IndexError):
        return ""
    lines = code.splitlines()
    if not lines:
        return ""
    start = max(0, line_no - 1 - context_lines)
    end = min(len(lines), line_no + context_lines)
    snippet_lines = []
    for i in range(start, end):
        marker = ">>> " if i == line_no - 1 else "    "
        snippet_lines.append(f"{marker}{i + 1:4d} | {lines[i]}")
    return "\n".join(snippet_lines)


def fingerprint_error(feedback: Any) -> str:
    """
    Generate a deterministic fingerprint key for a failure.

    Format:
      <ExceptionType>|<UserFrame>|<hash8>
    """
    primary_text, backtrace_text = _to_text_blocks(feedback)
    merged = f"{primary_text}\n{backtrace_text}".strip()
    if not merged:
        return "UnknownError|unknown|00000000"

    exception_type = _extract_exception_type(merged)
    core_line = _extract_core_line(primary_text or merged)
    user_frame = _extract_user_frame(backtrace_text, primary_text)
    base = f"{exception_type}|{user_frame}|{core_line.lower()}"
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"{exception_type}|{user_frame}|{digest}"


def compact_tool_output(result: Any, max_chars: int = 1500) -> str:
    """Simple truncation of tool output."""
    text = str(result or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n... [truncated]"


def elide_middle(
    result: Any,
    head_chars: int = 1500,
    tail_chars: int = 500,
) -> str:
    """Head + tail truncation for large tool outputs (SWE-agent-style).

    Short texts are returned as-is. For long inputs we keep ``head_chars``
    leading characters and ``tail_chars`` trailing characters with a
    single-line elision marker between them — preserving both the high-
    signal head (grep matches, first errors) and the tail (last stack
    frame where the exception actually surfaced).
    """
    text = "" if result is None else str(result)
    max_total = max(0, head_chars) + max(0, tail_chars)
    if not text or len(text) <= max_total:
        return text
    elided = len(text) - max_total
    head = text[: max(0, head_chars)].rstrip()
    tail = text[-max(0, tail_chars):].lstrip() if tail_chars > 0 else ""
    marker = f"\n... [{elided} chars elided] ...\n"
    return f"{head}{marker}{tail}"
