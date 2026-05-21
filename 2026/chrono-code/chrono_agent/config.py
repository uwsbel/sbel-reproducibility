"""
Configuration management for Chrono-Agent using Pydantic Settings.

Loads configuration from environment variables and .env file.
"""

import os
from typing import Literal, Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    All settings can be overridden via environment variables.
    """

    # LLM Provider Configuration
    default_llm_provider: Literal["anthropic", "openai", "ngc", "huggingface", "ollama", "minimax", "deepseek", "google"] = "anthropic"
    anthropic_api_key: Optional[str] = None
    # Optional. Omit for official api.anthropic.com (use ANTHROPIC_API_KEY + ANTHROPIC_MODEL only).
    # Set only when provider=anthropic but the base URL must point elsewhere (Messages API compatible).
    anthropic_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    # OpenRouter (OpenAI-compatible): use with provider=openai + openai_api_base
    openrouter_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None  # e.g. https://openrouter.ai/api/v1 for OpenRouter

    # Model Selection
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    openai_model: str = "gpt-4-turbo"

    # NGC Configuration (for VLM models)
    ngc_api_key: Optional[str] = None
    ngc_api_base: str = "https://integrate.api.nvidia.com"  # Default NGC API base URL
    ngc_model: str = "meta/llama-3.1-8b-instruct"  # Default NGC model

    # HuggingFace Configuration
    huggingface_api_token: Optional[str] = None
    huggingface_api_base: str = "https://api-inference.huggingface.co/models"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"  # Fallback when AGENTx_OLLAMA_MODEL / AGENTx_MODEL unset

    # MiniMax Configuration (Native API - uses OpenAI-compatible endpoint)
    minimax_api_key: Optional[str] = None
    minimax_api_base: Optional[str] = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M2.7"

    # DeepSeek Configuration (OpenAI-compatible endpoint)
    deepseek_api_key: Optional[str] = None
    deepseek_api_base: Optional[str] = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"

    # Google Gemini Configuration (native multimodal including video)
    google_api_key: Optional[str] = None
    google_model: str = "gemini-2.0-flash-exp"

    # Individual Agent Configuration
    agent1_llm_provider: Optional[str] = None
    agent1_model: Optional[str] = None
    agent1_hf_model: Optional[str] = None
    agent1_hf_token: Optional[str] = None
    agent1_ollama_model: Optional[str] = None
    agent1_ollama_base_url: Optional[str] = None
    agent1_ngc_model: Optional[str] = None
    agent1_ngc_api_base: Optional[str] = None
    agent1_openai_api_base: Optional[str] = None
    agent1_anthropic_api_base: Optional[str] = None
    agent1_temperature: float = 0.7
    agent1_max_tokens: int = 32768  # PlanningAgent - mbs_in_scene plans with SCM terrain, topology, milestones, and assets can exceed 8k tokens; 4k truncates mid-JSON and manifests as PlanModificationIncompleteError
    agent1_minimax_api_key: Optional[str] = None
    agent1_minimax_api_base: Optional[str] = None
    agent1_minimax_model: Optional[str] = None
    agent1_deepseek_api_key: Optional[str] = None
    agent1_deepseek_api_base: Optional[str] = None
    agent1_deepseek_model: Optional[str] = None

    agent2_llm_provider: Optional[str] = None
    agent2_model: Optional[str] = None
    agent2_hf_model: Optional[str] = None
    agent2_hf_token: Optional[str] = None
    agent2_ollama_model: Optional[str] = None
    agent2_ollama_base_url: Optional[str] = None
    agent2_ngc_model: Optional[str] = None
    agent2_ngc_api_base: Optional[str] = None
    agent2_openai_api_base: Optional[str] = None
    agent2_anthropic_api_base: Optional[str] = None
    agent2_temperature: float = 0.3
    agent2_max_tokens: int = 32768  # CodeGenerationAgent - needs long output for complete code
    agent2_minimax_api_key: Optional[str] = None
    agent2_minimax_api_base: Optional[str] = None
    agent2_minimax_model: Optional[str] = None
    agent2_deepseek_api_key: Optional[str] = None
    agent2_deepseek_api_base: Optional[str] = None
    agent2_deepseek_model: Optional[str] = None

    agent3_llm_provider: Optional[str] = None
    agent3_model: Optional[str] = None
    agent3_hf_model: Optional[str] = None
    agent3_hf_token: Optional[str] = None
    agent3_ollama_model: Optional[str] = None
    agent3_ollama_base_url: Optional[str] = None
    agent3_ngc_model: Optional[str] = None
    agent3_ngc_api_base: Optional[str] = None
    agent3_openai_api_base: Optional[str] = None
    agent3_anthropic_api_base: Optional[str] = None
    agent3_temperature: float = 0.5
    agent3_max_tokens: int = 16384  # ReviewAgent - medium length reviews
    agent3_minimax_api_key: Optional[str] = None
    agent3_minimax_api_base: Optional[str] = None
    agent3_minimax_model: Optional[str] = None
    agent3_deepseek_api_key: Optional[str] = None
    agent3_deepseek_api_base: Optional[str] = None
    agent3_deepseek_model: Optional[str] = None
    agent3_google_model: Optional[str] = None

    agent4_llm_provider: Optional[str] = None
    agent4_model: Optional[str] = None
    agent4_hf_model: Optional[str] = None
    agent4_hf_token: Optional[str] = None
    agent4_ollama_model: Optional[str] = None
    agent4_ollama_base_url: Optional[str] = None
    agent4_ngc_model: Optional[str] = None
    agent4_ngc_api_base: Optional[str] = None
    agent4_openai_api_base: Optional[str] = None
    agent4_anthropic_api_base: Optional[str] = None
    agent4_temperature: float = 0.5
    agent4_max_tokens: int = 2048  # ExecutionAgent - short outputs
    agent4_minimax_api_key: Optional[str] = None
    agent4_minimax_api_base: Optional[str] = None
    agent4_minimax_model: Optional[str] = None
    agent4_deepseek_api_key: Optional[str] = None
    agent4_deepseek_api_base: Optional[str] = None
    agent4_deepseek_model: Optional[str] = None

    # Workflow Settings
    max_retries: int = 3
    max_compilation_retries: int = 4
    execution_timeout: int = 90  # Seconds; VSG + sim loop often need 60–90s
    plan_mode: Literal["simple", "detailed", "auto"] = "auto"
    use_prompt_cache: bool = True

    # Token Usage Observability (baseline for cost-reduction work; see
    # plan file system-work-tokens-workflow-cached-spindle.md)
    log_token_usage: bool = True  # Emit per-call and per-loop token usage stats

    # Tool Output Truncation (head + tail elision, SWE-agent-style).
    # Controls how large bash / stderr / backtrace / validator output is
    # compressed before being fed back to codegen as tool_result. Never
    # applied to read_skill / read_file content (those are intentional
    # model-requested full documents).
    tool_output_truncate_enabled: bool = True
    tool_output_head_chars: int = 1500
    tool_output_tail_chars: int = 500

    # Haiku Skill-Query Sub-agent: shifts full SKILL.md reads off the main
    # (Sonnet-class) codegen model onto a cheaper Haiku model that returns
    # a focused 300-800 token answer instead of the 3-5K-token full doc.
    # When enabled, the ``query_skill(name, question)`` tool is registered
    # alongside the existing ``read_skill`` / ``read_skill_section``.
    skill_query_subagent_enabled: bool = True
    skill_query_subagent_provider: Literal["anthropic"] = "anthropic"
    skill_query_subagent_model: str = "claude-haiku-4-5-20251001"
    skill_query_subagent_max_tokens: int = 1024

    # LLM-based skill router. Replaces the static "Required Skills"
    # markdown table in core skills (and the keyword-based gate in
    # tools/code_agent_tools.py) with one LLM call that picks the
    # minimal skill set from descriptions + plan content. Result is
    # cached per (plan, skill-set) hash so a CLI run pays one call.
    # When False, falls back to the legacy static-table + keyword path.
    #
    # Defaults to MiniMax via OpenAI-compatible API — same credentials as
    # the codegen agent (cheap, low latency). Override per-deployment via
    # SKILL_ROUTER_MODEL / SKILL_ROUTER_API_BASE / SKILL_ROUTER_API_KEY.
    skill_router_enabled: bool = True
    skill_router_model: Optional[str] = None  # falls back to minimax_model
    skill_router_api_base: Optional[str] = None  # falls back to minimax_api_base
    skill_router_api_key: Optional[str] = None  # falls back to minimax_api_key
    # MiniMax-M2 emits a <think> block in `content` before the tool_call —
    # set generous so reasoning + JSON args both fit. The router runs once
    # per CLI run (cached) so the headroom costs nothing in practice.
    skill_router_max_tokens: int = 8192

    # Skill-gate relaxation (cost-reduction Phase 1): pre-injected skill
    # content is already visible in the system prompt, so forcing the LLM
    # to also emit a read_skill() tool call for those same skills burns
    # one loop turn for pure ceremony. When True, the gate treats any
    # pre-injected skill as "already read" and only nudges for skills
    # that were budget-skipped from pre-injection. Set False to restore
    # the original belt-and-suspenders behavior.
    skill_gate_treat_preinjected_as_read: bool = True

    # In-session memoization of query_skill answers. Keyed on
    # (skill_name, normalized_question) so repeated identical queries
    # within one codegen session return cached answers without
    # re-invoking the Haiku sub-agent.
    skill_query_cache_enabled: bool = True
    skill_query_cache_max_entries: int = 32

    # PyChrono API RAG retrieval (chrono_agent.tools.api_rag).
    # - embedding_model: sentence-transformers model used at build time by
    #   scripts/build_api_index.py and at query time to encode the question.
    #   Must match across both; otherwise the VectorIndex self-disables.
    # - retrieval_mode: "hybrid" (keyword + vector, RRF-fused — default),
    #   "keyword" (legacy, no local torch dep needed), "vector" (pure
    #   semantic, useful for debugging the vector ranker alone).
    # - retrieval_top_k: number of chunks passed to Haiku for grounding.
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_mode: Literal["hybrid", "keyword", "vector"] = "hybrid"
    retrieval_top_k: int = 5

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str = "./chrono_agent.log"

    # Data Paths
    demo_data_path: str = "./demo_data"
    checkpoint_path: str = "./data/checkpoints"

    # Code Validation Settings
    enable_syntax_validation: bool = True
    enable_compilation_validation: bool = True
    enable_physics_validation: bool = True

    # Execution Settings
    enable_visualization: bool = True
    visualization_output_path: str = "./outputs"
    results_path: str = "./results"
    max_simulation_time: int = 60  # seconds

    # VLM Video Analysis Settings
    vlm_video_num_frames: Optional[int] = None  # None = analyze entire video (no frame sampling)

    # Prefer Gemini's native-video path over the per-frame PNG loop: when an
    # mp4 is present in cam/ AND the provider is Google, drop the PNG
    # sequence (Gemini already samples the mp4 internally at 5 fps; sending
    # both is literal duplication of the same content). Default True. Flip
    # to False to restore the legacy "send both" behavior.
    vlm_prefer_native_video: bool = True

    # PNG-sequence frame sampling for non-mp4 / non-Google review paths.
    # Disabled by default — cutting frames trades review cost for the chance
    # of missing a single-frame physics anomaly. Turn on only after
    # measuring that sampled review catches the same failures as full review.
    vlm_frame_sampling_enabled: bool = False
    vlm_max_frames: int = 12
    vlm_frame_stride: str = "auto"  # "auto" → ceil(n / vlm_max_frames), or a positive int as string

    # DGX Spark Remote Access (for Cosmos VLM deployed on remote DGX)
    dgx_spark_host: Optional[str] = None  # Remote DGX Spark host (e.g., "dgx-spark.example.com")
    dgx_spark_port: int = 18000  # Port for NGC/Cosmos API on remote host
    dgx_spark_use_ssh_tunnel: bool = False  # Whether to use SSH tunnel to reach DGX Spark

    # Headless Execution Settings
    enable_headless_mode: bool = False  # Use virtual display (Xvfb) for headless rendering
    xvfb_display: str = ":99"           # Virtual display number for Xvfb
    xvfb_resolution: str = "1280x720"   # Virtual screen resolution

    # noVNC Display Server Settings (used when enable_headless_mode=True)
    novnc_port: int = 6080              # WebSocket port for noVNC browser access
    x11vnc_port: int = 5900             # VNC server port for x11vnc

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def history_output_path(self) -> Path:
        """Base path for code history."""
        return Path("history")

    def dialog_output_path(self) -> Path:
        """Base path for dialog/sessions."""
        return Path("dialog")

    def validate_api_keys(self) -> None:
        """
        Validate that at least one LLM provider API key is configured.

        Raises:
            ValueError: If no API keys are configured
        """
        sku_providers = ("anthropic", "openai", "ngc", "minimax", "deepseek", "google")
        needs_cloud_key = self.default_llm_provider in sku_providers
        has_cloud_key = bool(
            self.anthropic_api_key
            or self.openai_api_key
            or (self.openrouter_api_key and self.openrouter_api_key.strip())
            or self.ngc_api_key
            or self.minimax_api_key
            or self.deepseek_api_key
            or self.google_api_key
        )
        if needs_cloud_key and not has_cloud_key:
            raise ValueError(
                "At least one LLM API key must be configured. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, NGC_API_KEY, MINIMAX_API_KEY, "
                "DEEPSEEK_API_KEY, or GOOGLE_API_KEY in your .env file."
            )

        if self.default_llm_provider == "ollama" and not self.ollama_model:
            raise ValueError(
                "OLLAMA_MODEL must be set when using Ollama as default provider."
            )

        if self.default_llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set when using Anthropic as default provider."
            )

        if self.default_llm_provider == "openai" and not (
            (self.openai_api_key and self.openai_api_key.strip())
            or (self.openrouter_api_key and self.openrouter_api_key.strip())
        ):
            raise ValueError(
                "OPENAI_API_KEY or OPENROUTER_API_KEY must be set when using OpenAI as default provider."
            )

        if self.default_llm_provider == "ngc" and not self.ngc_api_key:
            raise ValueError(
                "NGC_API_KEY must be set when using NGC as default provider."
            )

        if self.default_llm_provider == "minimax" and not self.minimax_api_key:
            raise ValueError(
                "MINIMAX_API_KEY must be set when using MiniMax as default provider."
            )

        if self.default_llm_provider == "deepseek" and not self.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY must be set when using DeepSeek as default provider."
            )

        if self.default_llm_provider == "google" and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY must be set when using Google (Gemini) as default provider."
            )

    def ensure_directories(self) -> None:
        """
        Ensure all required directories exist.

        Note: visualization_output_path is NOT pre-created here. The CLI
        rebinds it to history/ via _activate_cli_runtime_run() before any
        agent writes; pre-creating the default ("./outputs") only leaves
        a misleading empty directory on disk that codegen's find_files()
        later trips over.
        """
        directories = [
            self.checkpoint_path,
            self.results_path,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_agent_config(self, agent_number: int) -> dict:
        """
        Get LLM configuration for a specific agent.

        Supports: anthropic, openai, huggingface, ollama, minimax, deepseek

        Args:
            agent_number: Agent number (1, 2, 3, or 4)

        Returns:
            Dictionary with agent-specific configuration
        """
        # Get agent-specific provider from .env, fallback to default
        agent_provider_attr = f"agent{agent_number}_llm_provider"
        agent_provider = getattr(self, agent_provider_attr, None)
        provider = agent_provider or self.default_llm_provider

        # Normalize "openrouter" → "openai"; auto-set OpenRouter base URL if not already set
        if provider == "openrouter":
            provider = "openai"
            agent_base_attr = f"agent{agent_number}_openai_api_base"
            if not getattr(self, agent_base_attr, None):
                object.__setattr__(self, agent_base_attr, "https://openrouter.ai/api/v1")

        # Get agent-specific temperature from .env
        agent_temp_attr = f"agent{agent_number}_temperature"
        temperature = getattr(self, agent_temp_attr, 0.5)

        # Get agent-specific max_tokens from .env
        agent_max_tokens_attr = f"agent{agent_number}_max_tokens"
        max_tokens = getattr(self, agent_max_tokens_attr, 8192)  # Default 8192 if not set

        config = {
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Provider-specific configuration
        if provider in ["anthropic", "openai", "ngc", "google"]:
            # Get agent-specific model from .env (AGENT{N}_MODEL)
            agent_model_attr = f"agent{agent_number}_model"
            agent_model = getattr(self, agent_model_attr, None)

            # Always get LLM config (includes API key validation)
            llm_config = self.get_llm_config(provider)
            config.update(llm_config)

            # Override model if agent-specific model is set in .env
            if agent_model:
                config["model"] = agent_model

            # For NGC, also check agent-specific NGC settings
            if provider == "ngc":
                agent_ngc_model = getattr(self, f"agent{agent_number}_ngc_model", None)
                agent_ngc_api_base = getattr(self, f"agent{agent_number}_ngc_api_base", None)
                if agent_ngc_model:
                    config["model"] = agent_ngc_model
                if agent_ngc_api_base:
                    config["api_base"] = agent_ngc_api_base
            # Google (Gemini): per-agent model override
            if provider == "google":
                agent_google_model = getattr(self, f"agent{agent_number}_google_model", None)
                if agent_google_model:
                    config["model"] = agent_google_model
            # Anthropic-compatible (e.g. MiniMax): per-agent base overrides global ANTHROPIC_API_BASE
            if provider == "anthropic":
                agent_anthropic_base = getattr(self, f"agent{agent_number}_anthropic_api_base", None)
                if agent_anthropic_base and str(agent_anthropic_base).strip():
                    config["api_base"] = str(agent_anthropic_base).strip()
            # OpenAI / OpenRouter: per-agent API base overrides global OPENAI_API_BASE
            if provider == "openai":
                agent_openai_base = getattr(self, f"agent{agent_number}_openai_api_base", None)
                if agent_openai_base and str(agent_openai_base).strip():
                    config["api_base"] = str(agent_openai_base).strip()
                merged_base = (config.get("api_base") or "").lower()
                or_k = (self.openrouter_api_key or "").strip()
                oa_k = (self.openai_api_key or "").strip()
                if "openrouter.ai" in merged_base:
                    # OpenRouter requires Bearer token; prefer OPENROUTER_API_KEY, else OPENAI_API_KEY (sk-or-v1- works in either)
                    if or_k:
                        config["api_key"] = or_k
                    elif oa_k:
                        config["api_key"] = oa_k
                    if not (config.get("api_key") or "").strip():
                        raise ValueError(
                            f"Agent {agent_number}: API base is OpenRouter but no key is set. "
                            "Set OPENROUTER_API_KEY=sk-or-v1-... or put the same key in OPENAI_API_KEY."
                        )
            # Otherwise, use default model from llm_config (ANTHROPIC_MODEL, OPENAI_MODEL, or NGC_MODEL)

        elif provider == "huggingface":
            hf_model = getattr(self, f"agent{agent_number}_hf_model", None)
            hf_token = getattr(self, f"agent{agent_number}_hf_token", None) or self.huggingface_api_token

            if not hf_model:
                raise ValueError(
                    f"Agent {agent_number}: HuggingFace model not specified. "
                    f"Set AGENT{agent_number}_HF_MODEL in .env"
                )

            config.update({
                "model": hf_model,
                "hf_model": hf_model,
                "hf_token": hf_token,
                "hf_api_base": self.huggingface_api_base,
            })

        elif provider == "ollama":
            # Resolution order: AGENTx_OLLAMA_MODEL → OLLAMA_MODEL → AGENTx_MODEL
            # (many users set AGENTx_MODEL for other providers; reuse it for Ollama)
            ag_ollama = getattr(self, f"agent{agent_number}_ollama_model", None)
            ag_generic = getattr(self, f"agent{agent_number}_model", None)
            ollama_model = (
                (ag_ollama.strip() if isinstance(ag_ollama, str) else None)
                or (self.ollama_model.strip() if self.ollama_model else None)
                or (ag_generic.strip() if isinstance(ag_generic, str) else None)
            )
            ollama_base_url = getattr(self, f"agent{agent_number}_ollama_base_url", None) or self.ollama_base_url

            if not ollama_model:
                raise ValueError(
                    f"Agent {agent_number}: Ollama model not specified. "
                    f"Set AGENT{agent_number}_OLLAMA_MODEL, AGENT{agent_number}_MODEL, or OLLAMA_MODEL in .env"
                )

            config.update({
                "model": ollama_model,
                "ollama_model": ollama_model,
                "ollama_base_url": ollama_base_url,
            })

        elif provider == "minimax":
            # MiniMax uses OpenAI-compatible API
            minimax_key = getattr(self, f"agent{agent_number}_minimax_api_key", None) or self.minimax_api_key
            if not minimax_key:
                raise ValueError(
                    f"Agent {agent_number}: MINIMAX_API_KEY is not configured. "
                    f"Set MINIMAX_API_KEY or AGENT{agent_number}_MINIMAX_API_KEY in .env"
                )
            minimax_base = getattr(self, f"agent{agent_number}_minimax_api_base", None) or self.minimax_api_base
            minimax_model = getattr(self, f"agent{agent_number}_minimax_model", None) or self.minimax_model

            config.update({
                "provider": "openai",  # MiniMax uses OpenAI-compatible client
                "api_key": minimax_key,
                "api_base": minimax_base or "https://api.minimaxi.com/v1",
                "model": minimax_model,
            })

        elif provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API with isolated credentials.
            deepseek_key = getattr(self, f"agent{agent_number}_deepseek_api_key", None) or self.deepseek_api_key
            if not deepseek_key:
                raise ValueError(
                    f"Agent {agent_number}: DEEPSEEK_API_KEY is not configured. "
                    f"Set DEEPSEEK_API_KEY or AGENT{agent_number}_DEEPSEEK_API_KEY in .env"
                )
            deepseek_base = getattr(self, f"agent{agent_number}_deepseek_api_base", None) or self.deepseek_api_base
            deepseek_model = (
                getattr(self, f"agent{agent_number}_deepseek_model", None)
                or getattr(self, f"agent{agent_number}_model", None)
                or self.deepseek_model
            )

            config.update({
                "provider": "openai",  # DeepSeek uses OpenAI-compatible client
                "api_key": str(deepseek_key).strip(),
                "api_base": deepseek_base or "https://api.deepseek.com",
                "model": deepseek_model,
            })

        else:
            raise ValueError(f"Unsupported LLM provider for Agent {agent_number}: {provider}. Supported: anthropic, openai, ngc, huggingface, ollama, minimax, deepseek, google")

        return config

    def get_llm_config(self, provider: Optional[str] = None) -> dict:
        """
        Get LLM configuration for the specified provider.

        Args:
            provider: LLM provider ("anthropic", "openai", "ngc", "ollama", "minimax", "deepseek").
                     If None, uses default_llm_provider.

        Returns:
            Dictionary with provider configuration

        Raises:
            ValueError: If provider is invalid or API key is missing
        """
        provider = provider or self.default_llm_provider
        if provider == "openrouter":
            provider = "openai"

        if provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not configured. "
                    "Please set ANTHROPIC_API_KEY in your .env file."
                )
            out = {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
            }
            ab = (self.anthropic_api_base or "").strip()
            if ab:
                out["api_base"] = ab
            return out
        elif provider == "openai":
            oa_key = (self.openai_api_key or "").strip()
            or_key = (self.openrouter_api_key or "").strip()
            # Default: prefer OPENAI_API_KEY; per-agent OpenRouter override in get_agent_config
            api_key = oa_key or or_key
            if not api_key:
                raise ValueError(
                    "OpenAI-compatible API key not configured. "
                    "Set OPENAI_API_KEY and/or OPENROUTER_API_KEY in .env"
                )
            if not api_key.startswith("sk-"):
                raise ValueError(
                    "OpenAI/OpenRouter API key format appears invalid (should start with 'sk-'). "
                    f"Current key starts with: {api_key[:5]}..."
                )
            out = {
                "provider": "openai",
                "api_key": api_key,
                "model": self.openai_model,
            }
            base = (self.openai_api_base or "").strip()
            if base:
                out["api_base"] = base
            return out
        elif provider == "ngc":
            if not self.ngc_api_key:
                raise ValueError(
                    "NGC_API_KEY is not configured. "
                    "Please set NGC_API_KEY in your .env file."
                )
            return {
                "provider": "ngc",
                "api_key": self.ngc_api_key,
                "api_base": self.ngc_api_base,
                "model": self.ngc_model,
            }
        elif provider == "ollama":
            ollama_model = self.ollama_model.strip() if self.ollama_model else ""
            if not ollama_model:
                raise ValueError(
                    "OLLAMA_MODEL not configured. Set OLLAMA_MODEL in .env (e.g. llama3.2:latest)"
                )
            return {
                "provider": "ollama",
                "model": ollama_model,
                "ollama_model": ollama_model,
                "ollama_base_url": self.ollama_base_url,
            }
        elif provider == "minimax":
            if not self.minimax_api_key:
                raise ValueError(
                    "MINIMAX_API_KEY is not configured. "
                    "Please set MINIMAX_API_KEY in your .env file."
                )
            return {
                "provider": "openai",  # MiniMax uses OpenAI-compatible client
                "api_key": self.minimax_api_key,
                "api_base": self.minimax_api_base or "https://api.minimaxi.com/v1",
                "model": self.minimax_model,
            }
        elif provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is not configured. "
                    "Please set DEEPSEEK_API_KEY in your .env file."
                )
            return {
                "provider": "openai",  # DeepSeek uses OpenAI-compatible client
                "api_key": self.deepseek_api_key,
                "api_base": self.deepseek_api_base or "https://api.deepseek.com",
                "model": self.deepseek_model,
            }
        elif provider == "google":
            if not self.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is not configured. "
                    "Please set GOOGLE_API_KEY in your .env file."
                )
            return {
                "provider": "google",
                "api_key": self.google_api_key,
                "model": self.google_model,
            }
        else:
            raise ValueError(
                f"Invalid LLM provider: {provider}. Supported: anthropic, openai, ngc, ollama, minimax, deepseek, google"
            )


# Global settings instance
_settings: Optional[Settings] = None


def _build_settings() -> Settings:
    """Create settings and run post-load validation/setup."""
    settings = Settings()
    settings.validate_api_keys()
    return settings


def get_settings() -> Settings:
    """
    Get or create the global settings instance.

    Returns:
        Settings instance loaded from environment
    """
    global _settings
    if _settings is None:
        _settings = _build_settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).

    Returns:
        New Settings instance
    """
    global _settings
    _settings = _build_settings()
    return _settings


def init_directories_sync(settings: Optional[Settings] = None) -> None:
    """
    Ensure required directories exist (synchronous).

    Use this in CLI / non-ASGI contexts.
    """
    (settings or get_settings()).ensure_directories()


async def init_directories_async(settings: Optional[Settings] = None) -> None:
    """
    Ensure required directories exist without blocking the event loop.

    Use this in LangGraph Studio/ASGI startup hooks.
    """
    from chrono_agent.utils.async_fs import ensure_dir

    s = settings or get_settings()
    # visualization_output_path is intentionally omitted — it is rebound
    # to history/ by the CLI before use; see ensure_directories() comment.
    for directory in [
        s.checkpoint_path,
        s.results_path,
    ]:
        await ensure_dir(Path(directory))
