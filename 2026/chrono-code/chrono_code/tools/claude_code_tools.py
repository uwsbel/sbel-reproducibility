"""Claude-Code-style tools bolted onto the code agent.

Adds the high-value primitives that Claude Code exposes to its subagent but
that were previously missing here:

* ``glob``               — cross-directory file pattern matching, mtime-sorted
* ``todo_write``         — persistent task checklist for multi-step fixes
* ``todo_read``          — read the current task list
* ``web_fetch``          — pull a URL's text content (no search, just fetch)
* ``spawn_subagent``     — delegate a self-contained research/coding question
                          to a fresh sub-model, isolates main context
* ``bash_background``    — start a long-running shell command, return handle
* ``bash_output``        — tail stdout/stderr from a background handle
* ``bash_kill``          — send SIGTERM (and then SIGKILL) to a background handle

All share process-wide state in the returned ``context`` dict so the tools
compose cleanly with the existing ``code_agent_tools.make_code_agent_tools``
closure.
"""

from __future__ import annotations

import asyncio
import fnmatch
import glob as _glob
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chrono_code.utils.logger import get_logger

logger = get_logger(__name__)


# --- glob -------------------------------------------------------------------


def _make_glob_tool() -> Tuple[Dict[str, Any], Callable]:
    """Fast file pattern matching, mtime-sorted.

    Examples:
        glob("**/*.py")                         — all python files in tree
        glob("chrono_code/agents/*.py")        — top-level agent modules
        glob("history/iteration_*/cam/*.mp4")   — all cam videos across runs
    """

    def glob_impl(pattern: str, path: str = ".", head_limit: int = 100) -> str:
        root = Path(path).resolve()
        if not root.exists():
            return f"Error: path does not exist: {path}"
        if not root.is_dir():
            return f"Error: not a directory: {path}"

        try:
            matches_raw = list(root.glob(pattern))
        except (NotImplementedError, ValueError) as e:
            return f"Error: invalid glob pattern {pattern!r}: {e}"

        files_with_mtime: List[Tuple[float, str]] = []
        for p in matches_raw:
            if p.is_file():
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = 0.0
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                files_with_mtime.append((mtime, str(rel)))

        files_with_mtime.sort(key=lambda t: -t[0])

        if not files_with_mtime:
            return f"No files matching '{pattern}' in {path}"

        limit = max(1, int(head_limit))
        truncated = len(files_with_mtime) > limit
        selected = files_with_mtime[:limit]
        lines = [p for _mt, p in selected]
        header = f"Found {len(files_with_mtime)} file(s) matching '{pattern}' (mtime-sorted, newest first)"
        if truncated:
            header += f" — showing first {limit}"
        return header + "\n" + "\n".join(lines)

    tool_def = {
        "name": "glob",
        "description": (
            "Find files by glob pattern, sorted by modification time (newest first). "
            "Supports cross-directory patterns like 'src/**/*.py'. "
            "Prefer this over find_files when you want recency ordering or are "
            "writing a multi-level pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.py' or 'history/iteration_*/cam/*.mp4'.",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search from. Defaults to current dir.",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Max results to return. Default 100.",
                },
            },
            "required": ["pattern"],
        },
    }
    return tool_def, glob_impl


# --- todo_write / todo_read ------------------------------------------------


@dataclass
class _TodoEntry:
    id: str
    content: str
    status: str = "pending"   # pending | in_progress | completed | cancelled
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def _make_todo_tools(state: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Callable]]:
    """TodoWrite/TodoRead pair.

    The LLM uses this to track multi-step work within a single agent session.
    State lives in the caller-supplied ``state`` dict so the same list
    survives across tool calls.
    """

    state.setdefault("todos", [])

    def _render(todos: List[_TodoEntry]) -> str:
        if not todos:
            return "Todo list is empty."
        status_icon = {
            "pending":     "[ ]",
            "in_progress": "[~]",
            "completed":   "[x]",
            "cancelled":   "[-]",
        }
        lines = ["Todo list:"]
        for t in todos:
            icon = status_icon.get(t.status, "[?]")
            lines.append(f"  {icon} ({t.id[:8]}) {t.content}")
        pending = sum(1 for t in todos if t.status == "pending")
        in_prog = sum(1 for t in todos if t.status == "in_progress")
        done = sum(1 for t in todos if t.status == "completed")
        lines.append(
            f"-- {done} done, {in_prog} in progress, {pending} pending, "
            f"{len(todos) - done - in_prog - pending} cancelled"
        )
        return "\n".join(lines)

    def todo_write(todos: List[Dict[str, Any]]) -> str:
        """Replace the full todo list.

        Each entry is ``{"id"?: str, "content": str, "status"?: str}``.
        Missing ids are auto-generated. Missing status defaults to "pending".
        """
        if not isinstance(todos, list):
            return "Error: `todos` must be a list."
        new_entries: List[_TodoEntry] = []
        existing_by_id: Dict[str, _TodoEntry] = {t.id: t for t in state["todos"]}
        for raw in todos:
            if not isinstance(raw, dict):
                return f"Error: each todo must be a dict, got {type(raw).__name__}"
            content = str(raw.get("content") or "").strip()
            if not content:
                return "Error: each todo must have non-empty `content`"
            tid = str(raw.get("id") or "")
            status = str(raw.get("status") or "pending")
            if status not in {"pending", "in_progress", "completed", "cancelled"}:
                return f"Error: invalid status {status!r} (allowed: pending/in_progress/completed/cancelled)"
            if tid and tid in existing_by_id:
                old = existing_by_id[tid]
                new_entries.append(_TodoEntry(
                    id=tid,
                    content=content,
                    status=status,
                    created_at=old.created_at,
                    updated_at=time.time(),
                ))
            else:
                new_entries.append(_TodoEntry(
                    id=tid or uuid.uuid4().hex[:12],
                    content=content,
                    status=status,
                ))
        state["todos"] = new_entries
        return _render(new_entries)

    def todo_read() -> str:
        return _render(state["todos"])

    write_def = {
        "name": "todo_write",
        "description": (
            "Replace the current todo list. Use this to track multi-step work "
            "(e.g. 'fix import error, then add missing validator, then re-run'). "
            "Mark items 'in_progress' when starting and 'completed' immediately "
            "after finishing — don't batch status updates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Stable id to preserve history. Omit to auto-generate."},
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["todos"],
        },
    }
    read_def = {
        "name": "todo_read",
        "description": "Show the current todo list with statuses.",
        "input_schema": {"type": "object", "properties": {}},
    }
    return [(write_def, todo_write), (read_def, todo_read)]


# --- web_fetch --------------------------------------------------------------


_WEB_FETCH_TIMEOUT = 30.0
_WEB_FETCH_MAX_CHARS = 200_000
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _make_web_fetch_tool() -> Tuple[Dict[str, Any], Callable]:
    """Pull text content from a URL.

    Strict: only http/https, 30 s timeout, 200 KB cap, no redirects to
    file://. No search, no summarization — just raw content. If the page is
    HTML, a light text extraction is applied; otherwise the body is returned
    as-is.
    """

    def _strip_html(html: str) -> str:
        try:
            from html.parser import HTMLParser
        except ImportError:
            return html

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self._skip = 0
                self._parts: List[str] = []

            def handle_starttag(self, tag, attrs):
                if tag in {"script", "style", "noscript"}:
                    self._skip += 1

            def handle_endtag(self, tag):
                if tag in {"script", "style", "noscript"}:
                    self._skip = max(0, self._skip - 1)

            def handle_data(self, data):
                if self._skip:
                    return
                s = data.strip()
                if s:
                    self._parts.append(s)

        p = _Stripper()
        try:
            p.feed(html)
        except Exception:
            return html
        return "\n".join(p._parts)

    def web_fetch(url: str) -> str:
        try:
            from urllib.parse import urlparse
        except ImportError:
            return "Error: urllib unavailable"
        try:
            parsed = urlparse(url)
        except Exception as e:
            return f"Error: invalid URL {url!r}: {e}"
        if parsed.scheme not in _ALLOWED_SCHEMES:
            return f"Error: only http/https schemes allowed, got {parsed.scheme!r}"
        if not parsed.netloc:
            return f"Error: URL missing host: {url}"

        req = Request(url, headers={"User-Agent": "chrono-code/0.1 (+https://anthropic.com)"})
        try:
            with urlopen(req, timeout=_WEB_FETCH_TIMEOUT) as resp:
                ctype = resp.headers.get("Content-Type", "").lower()
                raw = resp.read(_WEB_FETCH_MAX_CHARS + 1)
        except HTTPError as e:
            return f"HTTP {e.code} {e.reason} for {url}"
        except URLError as e:
            return f"Network error for {url}: {e.reason}"
        except Exception as e:
            return f"Fetch failed for {url}: {e}"

        body_text: str
        try:
            body_text = raw.decode("utf-8", errors="replace")
        except Exception:
            body_text = str(raw)

        if "html" in ctype:
            body_text = _strip_html(body_text)

        truncated = len(body_text) > _WEB_FETCH_MAX_CHARS
        body_text = body_text[:_WEB_FETCH_MAX_CHARS]
        header = f"Fetched {url} ({ctype or 'unknown content-type'})"
        if truncated:
            header += f" — truncated at {_WEB_FETCH_MAX_CHARS} chars"
        return header + "\n" + body_text

    tool_def = {
        "name": "web_fetch",
        "description": (
            "Fetch a URL's text content (no search — supply the exact URL). "
            "Useful for reading official docs (e.g. api.projectchrono.org) "
            "when a skill doesn't cover the needed API. HTTP/HTTPS only, "
            "30s timeout, 200 KB cap. HTML pages are lightly stripped to text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full HTTP/HTTPS URL to fetch.",
                },
            },
            "required": ["url"],
        },
    }
    return tool_def, web_fetch


# --- background bash --------------------------------------------------------


@dataclass
class _BgProc:
    handle: str
    proc: subprocess.Popen
    started_at: float = field(default_factory=time.time)
    stdout_chunks: List[str] = field(default_factory=list)
    stderr_chunks: List[str] = field(default_factory=list)
    stdout_offset: int = 0
    stderr_offset: int = 0
    _readers_started: bool = False


def _make_background_bash_tools(
    state: Dict[str, Any],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[Tuple[Dict[str, Any], Callable]]:
    """bash_background / bash_output / bash_kill tool triplet.

    ``state['bg_procs']`` holds ``{handle: _BgProc}`` so tools can cross-ref.
    Output is drained by a small thread per stream into in-memory buffers,
    which is fine because we cap each at 2 MB and the LLM rarely asks for
    more than the last few KB.
    """
    import threading

    state.setdefault("bg_procs", {})
    MAX_BUFFER_CHARS = 2_000_000

    def _start_stream_reader(bg: _BgProc, stream, chunks: List[str]) -> None:
        def reader():
            try:
                for line in iter(stream.readline, b""):
                    try:
                        chunks.append(line.decode("utf-8", errors="replace"))
                    except Exception:
                        chunks.append(repr(line))
                    # Cap memory: drop oldest halves when we exceed MAX.
                    total = sum(len(c) for c in chunks)
                    if total > MAX_BUFFER_CHARS:
                        while chunks and sum(len(c) for c in chunks) > MAX_BUFFER_CHARS // 2:
                            chunks.pop(0)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
        t = threading.Thread(target=reader, daemon=True)
        t.start()

    def bash_background(command: str, cwd_override: Optional[str] = None) -> str:
        if not command or not command.strip():
            return "Error: command is empty"
        target_cwd = cwd_override or cwd or os.getcwd()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=target_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # new process group → bash_kill reaches children
            )
        except Exception as e:
            return f"Error starting background command: {e}"
        handle = f"bg_{uuid.uuid4().hex[:8]}"
        bg = _BgProc(handle=handle, proc=proc)
        state["bg_procs"][handle] = bg
        _start_stream_reader(bg, proc.stdout, bg.stdout_chunks)
        _start_stream_reader(bg, proc.stderr, bg.stderr_chunks)
        bg._readers_started = True
        return (
            f"Started background command (pid={proc.pid}). handle: {handle}\n"
            f"Use bash_output(handle={handle!r}) to read stdout/stderr, "
            f"bash_kill(handle={handle!r}) to stop."
        )

    def bash_output(handle: str, stderr: bool = False) -> str:
        bg = state["bg_procs"].get(handle)
        if bg is None:
            return f"Error: no background process with handle {handle!r}"
        chunks = bg.stderr_chunks if stderr else bg.stdout_chunks
        offset_name = "stderr_offset" if stderr else "stdout_offset"
        offset = getattr(bg, offset_name)
        # Join all buffered chunks, slice from offset, then advance offset.
        full = "".join(chunks)
        new_text = full[offset:]
        setattr(bg, offset_name, len(full))
        status = "running" if bg.proc.poll() is None else f"exited rc={bg.proc.returncode}"
        header = f"[{handle} {('stderr' if stderr else 'stdout')} — {status}]"
        if not new_text:
            return header + "\n(no new output)"
        return header + "\n" + new_text

    def bash_kill(handle: str, sigterm_grace_s: float = 2.0) -> str:
        bg = state["bg_procs"].get(handle)
        if bg is None:
            return f"Error: no background process with handle {handle!r}"
        if bg.proc.poll() is not None:
            return f"{handle} already exited (rc={bg.proc.returncode})"
        try:
            os.killpg(os.getpgid(bg.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as e:
            return f"Error sending SIGTERM to {handle}: {e}"
        try:
            bg.proc.wait(timeout=max(0.1, sigterm_grace_s))
            return f"{handle} terminated cleanly (rc={bg.proc.returncode})"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(bg.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                return f"SIGKILL failed for {handle}: {e}"
            try:
                bg.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                return f"{handle} did not exit even after SIGKILL"
            return f"{handle} SIGKILL-ed after grace (rc={bg.proc.returncode})"

    bg_def = {
        "name": "bash_background",
        "description": (
            "Start a shell command in the background and return a handle for "
            "later polling. Use for anything that might run longer than 30 s "
            "(e.g. a simulation subprocess you want to tail while continuing "
            "to edit code). The command runs in its own process group so "
            "bash_kill can reach child processes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "cwd_override": {"type": "string", "description": "Override working directory."},
            },
            "required": ["command"],
        },
    }
    out_def = {
        "name": "bash_output",
        "description": (
            "Read new stdout (or stderr) since the last read from a background "
            "handle. Returns everything that accumulated since your previous "
            "bash_output call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "stderr": {"type": "boolean", "description": "Read stderr instead of stdout. Default false."},
            },
            "required": ["handle"],
        },
    }
    kill_def = {
        "name": "bash_kill",
        "description": (
            "Terminate a background process (SIGTERM, then SIGKILL after a "
            "grace period). Use to stop a background command you started "
            "with bash_background."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "sigterm_grace_s": {"type": "number", "description": "Seconds to wait after SIGTERM before SIGKILL. Default 2."},
            },
            "required": ["handle"],
        },
    }
    return [(bg_def, bash_background), (out_def, bash_output), (kill_def, bash_kill)]


# --- spawn_subagent ---------------------------------------------------------


def _make_spawn_subagent_tool(
    state: Dict[str, Any],
    *,
    default_tools: Optional[List[Dict[str, Any]]] = None,
    default_executors: Optional[Dict[str, Callable]] = None,
) -> Tuple[Dict[str, Any], Callable]:
    """Spawn a fresh sub-agent with its own small context.

    The subagent runs the same ``run_tool_loop`` infrastructure as the
    parent but with its own system prompt, user prompt, and a scoped tool
    set. It returns a single string (the subagent's final text response),
    which keeps the parent context small — perfect for research questions
    like "what's the signature of SCMTerrain.SetSoilParameters?".

    The subagent is created lazily from a factory in ``state['subagent_factory']``
    so we avoid circular imports between this module and the main agent
    class. The parent agent calls this at tool-build time to wire it up.
    """
    default_tools = default_tools or []
    default_executors = default_executors or {}

    async def spawn_subagent(
        task: str,
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        max_steps: int = 10,
    ) -> str:
        factory = state.get("subagent_factory")
        if factory is None:
            return (
                "Error: subagent support not initialized "
                "(state['subagent_factory'] is unset)."
            )
        if not task or not task.strip():
            return "Error: task must be a non-empty description of what the subagent should do."

        sub_system = (system_prompt or
            "You are a focused research sub-agent. Your job is to answer the "
            "task below using the tools provided. Keep tool use minimal; "
            "return a concise plain-text answer when done. Do NOT modify "
            "files unless explicitly told to in the task.")
        # Filter tools
        if allowed_tools:
            allowed = set(allowed_tools)
            tools = [t for t in default_tools if t["name"] in allowed]
            executors = {k: v for k, v in default_executors.items() if k in allowed}
        else:
            # Read-only defaults: skill/file/grep/glob, nothing that mutates.
            safe_names = {
                "read_file", "read_file_content", "read_skill", "read_skill_section",
                "search_skills", "grep_code", "find_files", "list_directory",
                "glob", "web_fetch", "find_assets", "list_chrono_assets",
            }
            tools = [t for t in default_tools if t["name"] in safe_names]
            executors = {k: v for k, v in default_executors.items() if k in safe_names}

        try:
            subagent = factory()
        except Exception as e:
            return f"Error: subagent factory raised: {e}"

        try:
            result = await subagent.run_tool_loop(
                system_prompt=sub_system,
                user_prompt=task,
                tools=tools,
                tool_executors=executors,
                max_steps=max_steps,
            )
        except Exception as e:
            return f"Subagent failed: {e}"

        return f"[subagent task: {task[:80]!r}]\n{result}"

    tool_def = {
        "name": "spawn_subagent",
        "description": (
            "Delegate a self-contained research task to a fresh sub-agent. "
            "Use this to keep the main context small when you need to explore "
            "multiple files/skills just to answer one question (e.g. 'find the "
            "exact signature of ChMaterialSurfaceSMC() constructor'). The "
            "subagent gets read-only tools by default (read_file, read_skill, "
            "grep_code, glob, web_fetch, ...); it cannot mutate the workspace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained prompt for the subagent. Include everything it needs — it has no memory of this conversation.",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional custom system prompt. Defaults to a 'focused research' preset.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional whitelist of tool names. Defaults to safe read-only set.",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Max tool-loop iterations in the subagent. Default 10.",
                },
            },
            "required": ["task"],
        },
    }
    return tool_def, spawn_subagent


# --- Public builder ---------------------------------------------------------


def make_claude_code_tools(
    *,
    state: Optional[Dict[str, Any]] = None,
    default_tools: Optional[List[Dict[str, Any]]] = None,
    default_executors: Optional[Dict[str, Callable]] = None,
    bg_cwd: Optional[str] = None,
    bg_env: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """Bundle Claude-Code-style tools (glob / todo / web / bg-bash / subagent).

    ``state`` is an in-process dict for todo list + background process table;
    pass your existing code-agent context dict so state is shared.

    ``default_tools`` / ``default_executors`` are forwarded to
    ``spawn_subagent`` as the tool set the subagent inherits. Pass the main
    code agent's tools here for full research capability; omit for read-only.
    """
    state = state if state is not None else {}
    tool_defs: List[Dict[str, Any]] = []
    executors: Dict[str, Callable] = {}

    glob_def, glob_fn = _make_glob_tool()
    tool_defs.append(glob_def)
    executors["glob"] = glob_fn

    for td, fn in _make_todo_tools(state):
        tool_defs.append(td)
        executors[td["name"]] = fn

    web_def, web_fn = _make_web_fetch_tool()
    tool_defs.append(web_def)
    executors["web_fetch"] = web_fn

    for td, fn in _make_background_bash_tools(state, cwd=bg_cwd, env=bg_env):
        tool_defs.append(td)
        executors[td["name"]] = fn

    subagent_def, subagent_fn = _make_spawn_subagent_tool(
        state,
        default_tools=default_tools,
        default_executors=default_executors,
    )
    tool_defs.append(subagent_def)
    executors["spawn_subagent"] = subagent_fn

    return tool_defs, executors
