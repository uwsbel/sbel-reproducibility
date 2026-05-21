# Chrono-Code

Chrono-Code is a multi-agent system that turns natural-language descriptions into runnable [PyChrono](https://projectchrono.org/pychrono/) physics simulations. Four LLM agents (planning, code generation, execution, visual review) cooperate in an async pipeline that loops on failure.

## Install

### 1. PyChrono (required)

`pychrono` is **not on PyPI** — it ships only via the `projectchrono` conda channel. See the official guide: <https://api.projectchrono.org/pychrono_installation.html>.

Quick install:
```bash
conda install -c projectchrono pychrono
```

> **Sensor module note.** Demos under `prompts/` use `pychrono.sensor`, which links against NVIDIA OptiX 9.1 and requires NVIDIA driver **R590+**. Older drivers fail with `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION` the first time a camera/lidar is created. Check with `nvidia-smi | head -3`. Reference: <https://api.projectchrono.org/module_sensor_installation.html>.

### 2. chrono-code

**Option A — you already have PyChrono.** Install into your active environment:
```bash
python -m pip install -e .
```

**Option B — start fresh.** Build the bundled conda env:
```bash
conda env create -f environment.yml
conda activate chrono-code
python -m pip install -e .
```

> Open a new terminal first and `conda deactivate` until no other Chrono env is active — stray `PYTHONPATH` / `LD_LIBRARY_PATH` entries from a previous PyChrono install can cause `ImportError: undefined symbol` against the new `_core.so`.

Sanity check:
```bash
python -c "import pychrono.core, chrono_code; print('ok')"
```

### 3. API keys

```bash
cp .env.example .env
```
Then fill in the providers you use. `.env.example` lists everything supported.

## Commands

Activate the env in every new shell (`conda activate chrono-code`), then:

```bash
# Generate from an inline prompt
chrono-code generate "Simulate a bouncing ball"

# Detailed planning mode
chrono-code generate --mode detailed "Double pendulum"

# Generate from a prompt file (multimodal — images in the .md are picked up)
chrono-code generate --prompt-file prompts/assets_robot/prompt.md
```

Equivalent: `python -m chrono_code.main generate ...`.

Bundled example prompts live in [prompts/](prompts/): `assets_robot`, `custom_assets_scene`, `demo_SEN_HMMWV_offroad_vsg`, `veh_fsi_floating`.

## Architecture

Chrono-Code wires four LLM agents into an async pipeline: **planning → code generation → execution → visual review**, with automatic loop-back on build failures or physics issues. State is a plain `TypedDict` and routing lives in [chrono_code/workflow/conditions.py](chrono_code/workflow/conditions.py) — no LangGraph / LangChain.

| Agent      | File                                                                                       | Role                              |
|------------|--------------------------------------------------------------------------------------------|-----------------------------------|
| Planning   | [chrono_code/agents/planning_agent.py](chrono_code/agents/planning_agent.py)             | Prompt → `SimulationPlan`         |
| CodeGen    | [chrono_code/agents/code_generation_agent.py](chrono_code/agents/code_generation_agent.py) | Skills-backed PyChrono code      |
| Execution  | [chrono_code/agents/execution_agent.py](chrono_code/agents/execution_agent.py)           | Runs sim subprocess               |
| Review     | [chrono_code/agents/review_agent.py](chrono_code/agents/review_agent.py)                 | VLM review of rendered frames     |

Entry point: [chrono_code/main.py](chrono_code/main.py) · Engine: [chrono_code/workflow/engine.py](chrono_code/workflow/engine.py) · Skills (framework-independent `SKILL.md` library): [chrono_code/skills/](chrono_code/skills/).

## LLM Providers

Each agent's provider and model are configured independently in `.env`. Anthropic, OpenAI, Google Gemini, MiniMax, DeepSeek, Ollama, OpenRouter, and NGC are all supported — see [.env.example](.env.example) for the variable names and recommended defaults.

