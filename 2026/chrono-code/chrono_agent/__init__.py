"""Chrono-Agent-Claude package."""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PlanningAgent",
    "CodeGenerationAgent",
    "ReviewAgent",
    "ExecutionAgent",
    "run_workflow",
    "get_settings",
]

_LAZY_IMPORTS = {
    "PlanningAgent": ("chrono_agent.agents", "PlanningAgent"),
    "CodeGenerationAgent": ("chrono_agent.agents", "CodeGenerationAgent"),
    "ReviewAgent": ("chrono_agent.agents", "ReviewAgent"),
    "ExecutionAgent": ("chrono_agent.agents", "ExecutionAgent"),
    "run_workflow": ("chrono_agent.workflow.engine", "run_workflow"),
    "get_settings": ("chrono_agent.config", "get_settings"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
