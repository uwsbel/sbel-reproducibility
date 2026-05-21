# README.md

This file provides guidance when working with code in this repository.

## What This Project Does

Chrono-Agent-Claude is a multi-agent system that translates natural language descriptions into executable PyChrono physics simulations. It uses the Anthropic Claude SDK directly (no LangGraph/LangChain) with four specialized agents: Planning, Code Generation, Execution, and Visual Review.

This is a fork of [Chrono-Agent](../Chrono-Agent/) rebuilt on the native Claude SDK.

## Commands

### Install

`pychrono` is **not on PyPI** — it is only distributed via the `projectchrono`
conda channel. We therefore offer two installation paths, mirroring
[chrono-ray](../chrono-ray/ReadMe.md).

#### System requirements (sensor module)

The example scenes (e.g. `prompts/assets_robot/`) and the bundled
`history/iteration_*/simulation.py` files use `pychrono.sensor`, which
links against NVIDIA OptiX at runtime. The conda-distributed
`projectchrono::pychrono >= 10.0.0` is built against **OptiX 9.1**, which
requires an **NVIDIA driver R590 or newer**.

Check your driver:
```bash
nvidia-smi | head -3
```
If `Driver Version` is below `590`, upgrade before continuing. On a driver
older than R590 you will see:
```
OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: Unsupported ABI version
```
the first time anything constructs `ChOptixEngine` (camera / lidar / radar).

References:
- [Chrono::Sensor installation](https://api.projectchrono.org/module_sensor_installation.html)
- [OptiX 9.1 release notes — R590 driver required](https://videocardz.com/newz/nvidia-optix-9-1-0-released-r590-driver-required-optix-6-no-longer-supported)

#### Option 1 — You already have PyChrono

Use this if `pychrono` is already importable in your active environment.

```bash
python -m pip install -e .
```

This installs `chrono-agent` and its remaining PyPI dependencies in-place.
Use `python -m pip` (not bare `pip`) so the install targets the active
Python; a stray `~/.local/bin/pip` on `PATH` will otherwise write into the
system Python.

#### Option 2 — You need everything (including PyChrono)

Use this if you are starting fresh. This creates a full conda environment
(`chrono-code`) with PyChrono and all dependencies.

> **Start from a clean shell.** Open a new terminal and ensure no other
> Chrono-related conda env is active (`conda deactivate` until you're back
> at `base` or no env). Some Chrono installs ship activate scripts that
> prepend `PYTHONPATH` / `LD_LIBRARY_PATH` / `LD_PRELOAD` to point at their
> own `pychrono` build; those entries can leak into the next env you
> activate and cause `ImportError: ... undefined symbol` from a mismatched
> `_core.so`. Verify with `echo $PYTHONPATH $LD_LIBRARY_PATH $LD_PRELOAD` —
> all three should be empty (or at least free of any chrono path) before
> proceeding.

1. Create the environment (miniconda or equivalent required):
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate chrono-code
   ```
3. Install chrono-agent (use `python -m pip` to make sure pip targets the
   conda env, not a system-wide pip on `PATH`):
   ```bash
   python -m pip install -e .
   ```
4. Sanity-check the install:
   ```bash
   python -c "import pychrono.core, pychrono.sensor, chrono_agent; print('ok')"
   ```
   If this raises `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION`, your NVIDIA driver
   is older than R590 — see *System requirements* above.

### Run

Make sure `conda activate chrono-code` (or whatever env you installed into)
is active in the current shell — every command below runs in that env.

```bash
chrono-agent generate "Simulate a bouncing ball"
python -m chrono_agent.main generate --mode detailed "Double pendulum"
```

Example prompt file:
```bash
chrono-agent generate --prompt-file prompts/assets_robot/prompt.md
```

`prompts/assets_robot/prompt.md` asks for a Unitree Go2 robot dog in a
furnished room:

```text
Put a Unitree Go2 robot dog in a 12m x 12m x 3m room with 2 tables and 7 chairs and let it walk around using its RL locomotion policy, bumping into these furnitures.
```

### Lint / Format
```bash
black chrono_agent
ruff check chrono_agent
```

The dev tools (`black`, `ruff`, `mypy`) are listed under the `dev`
dependency group in [pyproject.toml](pyproject.toml). Install them with:
```bash
python -m pip install -e ".[dev]"
```
or, equivalently, `python -m pip install black ruff mypy`.

## Architecture

### Four-Agent Pipeline

| Agent | File | Role |
|-------|------|------|
| Planning | `agents/planning_agent.py` | Parse prompt into SimulationPlan |
| Code Generation | `agents/code_generation_agent.py` | Skills-backed PyChrono code synthesis |
| Execution | `agents/execution_agent.py` | Run simulation subprocess, capture output |
| Review | `agents/review_agent.py` | VLM visual review of frames |

### Workflow (Plain Async Python)

Root workflow (`workflow/engine.py`):
```
planning -> plan_approval -> step_router -> codegen -> run_simulation -> visual_review -> physics_analysis -> END
                                  ^__________failed execution________________________|
                                  ^__________physics issues__________________________|
```

- No LangGraph - uses explicit async control flow
- State is a plain dict (WorkflowState TypedDict)
- Routing via condition functions in `workflow/conditions.py`
- Event system via callbacks in `workflow/events.py`

### LLM Integration

Uses `anthropic` and `openai` Python SDKs directly:
- `BaseAgent._init_client()` creates SDK client instances
- `BaseAgent.invoke_llm()` dispatches to provider-specific API calls
- `BaseAgent.run_tool_loop()` implements Claude's native tool_use API
- Multi-provider: Anthropic, OpenAI, Google Gemini, Ollama, MiniMax, NGC

### Code Generation Tools

Tools use Claude's native JSON schema format (no LangChain @tool decorator):
- `make_code_agent_tools()` returns `(tool_definitions, tool_executors)`
- Tool definitions are dicts with `name`, `description`, `input_schema`
- Tool executors are plain Python functions

### Skills System

Same as original Chrono-Agent - framework-independent SKILL.md files with YAML frontmatter.

## Key File Locations

| Concern | Path |
|---------|------|
| Entry point (CLI) | `chrono_agent/main.py` |
| Workflow engine | `chrono_agent/workflow/engine.py` |
| Workflow state | `chrono_agent/workflow/state.py` |
| Routing conditions | `chrono_agent/workflow/conditions.py` |
| Base agent (SDK) | `chrono_agent/agents/base.py` |
| Pydantic models | `chrono_agent/models/` |
| Prompt templates | `chrono_agent/agents/prompts/` |
| Tool definitions | `chrono_agent/tools/` |
| Skills | `chrono_agent/skills/` |
| Configuration | `chrono_agent/config.py` |

## Production provider recommendations

Each agent has an independent provider/model configured via `.env`
(`PLANNING_AGENT_PROVIDER`, `PLANNING_AGENT_MODEL`, etc. — see
`chrono_agent/config.py`). Provider choice materially affects reliability
because different endpoints have different levels of support for
schema-constrained output. The failure modes and what they trigger:

| Agent | Recommended provider / model | Why |
|-------|------------------------------|-----|
| `PlanningAgent` | `anthropic` / `claude-sonnet-4-6` (preferred) or `anthropic` / `claude-opus-4-7` | Anthropic's `tool_use` path gives server-side schema validation of plan output, preventing the nested shape errors (`milestones[].constraints` as string, `verify.csv_cols` as list-of-strings, etc.) that trigger `PlanModificationValidationError`. |
| `PlanningAgent` (alt.) | **real** `openai` / `gpt-4.1` (not a MiniMax / DeepSeek / other compat endpoint) | Real OpenAI honors `response_format={"type":"json_schema", "strict":true}` at token-decode time. OpenAI-compat endpoints mostly do not. |
| `CodeGenerationAgent` | `anthropic` / `claude-sonnet-4-6` or `openai` / `gpt-4.1` (real OpenAI) | Codegen needs to faithfully follow the skill-required / utils-required instructions in the system prompt. Weaker instruction-following models skip the required `read_skill(...)` call and forget `setup_preview_camera(...)`. |
| `ExecutionAgent` | any capable model (e.g. `openai` / `gpt-4o`) | Thin orchestration layer; just runs subprocess and formats output. |
| `ReviewAgent` / VLM | `google` / `gemini-2.5-pro` or `google` / `gemini-2.5-flash` (video) / `anthropic` / `claude-sonnet-4-6` (image) | Needs vision. Gemini is cost-effective and handles video directly; Claude has slightly better spatial reasoning on images. |

**What not to use for `PlanningAgent`** (observed to trigger silent plan
corruption or `PlanModificationValidationError` chains):
- `openai` base_url pointed at MiniMax / DeepSeek / NGC / Moonshot / other
  non-OpenAI endpoints — these silently ignore `response_format={"type":"json_schema"}`
  and return markdown-wrapped JSON that breaks nested schema validation.
- Small (<13B) local models via `ollama` — instruction-following on nested
  Pydantic schemas is too brittle.
- Older GPT-3.5-class models — same reason.

`modify_plan` now retries once with the validation errors fed back into
the prompt (see `PlanningAgent._build_validation_repair_suffix`), but the
retry cannot recover from a provider that has fundamentally weak schema
compliance. Pick a strong provider for `PlanningAgent` first; retry is
a safety net, not a substitute.

When to accept the tradeoff: if `PlanningAgent` is stuck on MiniMax /
compat for cost reasons, keep the retry as your backstop and expect
occasional user-visible red "plan validation error" panels. Switch the
provider if you see this more than once per session.
