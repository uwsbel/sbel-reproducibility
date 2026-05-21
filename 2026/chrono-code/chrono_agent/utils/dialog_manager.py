"""
Dialog Manager for tracking LLM conversations and Chain of Thought.

This module provides functionality to record all LLM interactions including:
- Prompts sent to LLMs
- LLM responses
- Chain of Thought reasoning
- API validation processes
- Code extraction attempts
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DialogManager:
    """
    Manages dialog/conversation logging for all agents.

    Records all LLM interactions to disk for debugging and analysis.
    """

    def __init__(self, base_dir: str = "dialog"):
        """
        Initialize Dialog Manager.

        Args:
            base_dir: Base directory for dialog storage
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"

        self.current_session: Optional[Path] = None
        self.session_data: Dict[str, Any] = {}

        # Track conversation counts per agent
        self.conversation_counts: Dict[str, int] = {}

        logger.debug(f"DialogManager initialized with base directory: {self.base_dir}")

    def ensure_dirs_sync(self) -> None:
        """Ensure base directories exist (synchronous)."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    async def ensure_dirs_async(self) -> None:
        """Ensure base directories exist without blocking the event loop."""
        from chrono_agent.utils.async_fs import ensure_dir

        await ensure_dir(self.base_dir)
        await ensure_dir(self.sessions_dir)

    def create_session(self, user_prompt: str) -> Path:
        """
        Create a new session directory.

        Args:
            user_prompt: Initial user prompt for this session

        Returns:
            Path to the session directory
        """
        self.ensure_dirs_sync()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{timestamp}"

        self.current_session = self.sessions_dir / session_name
        self.current_session.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each agent
        for agent_dir in ["planning", "codegen", "review", "execution", "compilation"]:
            (self.current_session / agent_dir).mkdir(exist_ok=True)

        # Initialize session data
        self.session_data = {
            "session_id": session_name,
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "agents": {},
            "created_at": datetime.now().isoformat()
        }

        # Save initial session info
        self._save_session_summary()

        # Update index
        self._update_index()

        logger.info(f"Session: {session_name}")
        return self.current_session

    def log_prompt(
        self,
        agent_name: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a prompt sent to an LLM.

        Args:
            agent_name: Name of the agent sending the prompt
            prompt: The prompt text
            metadata: Additional metadata (e.g., temperature, model)

        Returns:
            Filename where the prompt was saved
        """
        self.ensure_dirs_sync()
        if not self.current_session:
            logger.warning("No active session, creating one")
            self.create_session("unknown")

        # Get agent directory
        agent_dir = self._get_agent_dir(agent_name)

        # Get conversation number
        conv_num = self._get_next_conversation_number(agent_name)

        # Create prompt file
        filename = f"{conv_num:03d}_prompt.txt"
        filepath = agent_dir / filename

        # Save prompt
        with open(filepath, 'w') as f:
            f.write(f"# Agent: {agent_name}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            if metadata:
                f.write(f"# Metadata: {json.dumps(metadata, indent=2)}\n")
            f.write("#" * 80 + "\n\n")
            f.write(prompt)

        logger.debug(f"Logged prompt for {agent_name} to {filename}")

        # Update session data
        self._update_agent_data(agent_name, "prompts", filename)

        return filename

    def log_response(
        self,
        agent_name: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an LLM response.

        Args:
            agent_name: Name of the agent receiving the response
            response: The response text
            metadata: Additional metadata (e.g., tokens used, model)

        Returns:
            Filename where the response was saved
        """
        self.ensure_dirs_sync()
        if not self.current_session:
            logger.warning("No active session")
            return ""

        # Get agent directory
        agent_dir = self._get_agent_dir(agent_name)

        # Use current conversation number (don't increment)
        conv_num = self.conversation_counts.get(agent_name, 1)

        # Create response file
        filename = f"{conv_num:03d}_response.txt"
        filepath = agent_dir / filename

        # Save response
        with open(filepath, 'w') as f:
            f.write(f"# Agent: {agent_name}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            if metadata:
                f.write(f"# Metadata: {json.dumps(metadata, indent=2)}\n")
            f.write("#" * 80 + "\n\n")
            f.write(response)

        logger.debug(f"Logged response for {agent_name} to {filename}")

        # Update session data
        self._update_agent_data(agent_name, "responses", filename)

        return filename

    def log_chain_of_thought(
        self,
        agent_name: str,
        chain_of_thought: Dict[str, Any],
        iteration: Optional[int] = None
    ) -> str:
        """
        Log Chain of Thought reasoning.

        Args:
            agent_name: Name of the agent
            chain_of_thought: Structured Chain of Thought data
            iteration: Optional iteration number

        Returns:
            Filename where the CoT was saved
        """
        self.ensure_dirs_sync()
        if not self.current_session:
            logger.warning("No active session")
            return ""

        # Get agent directory
        agent_dir = self._get_agent_dir(agent_name)

        # Create filename
        conv_num = self.conversation_counts.get(agent_name, 1)
        if iteration is not None:
            filename = f"{conv_num:03d}_chain_of_thought_iter{iteration}.json"
        else:
            filename = f"{conv_num:03d}_chain_of_thought.json"

        filepath = agent_dir / filename

        # Add metadata
        cot_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "iteration": iteration,
            "chain_of_thought": chain_of_thought
        }

        # Save Chain of Thought
        with open(filepath, 'w') as f:
            json.dump(cot_data, f, indent=2)

        logger.info(f"Logged Chain of Thought for {agent_name} to {filename}")

        # Update session data
        self._update_agent_data(agent_name, "chain_of_thought", filename)

        return filename

    def log_code_extraction(
        self,
        agent_name: str,
        raw_response: str,
        extracted_code: str,
        extraction_method: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log code extraction attempts and results.

        Args:
            agent_name: Name of the agent
            raw_response: Original LLM response
            extracted_code: Code that was extracted (or empty if failed)
            extraction_method: Method used for extraction
            success: Whether extraction succeeded
            error_message: Error message if extraction failed

        Returns:
            Filename where the extraction log was saved
        """
        self.ensure_dirs_sync()
        if not self.current_session:
            logger.warning("No active session")
            return ""

        # Get agent directory
        agent_dir = self._get_agent_dir(agent_name)

        # Create filename
        conv_num = self.conversation_counts.get(agent_name, 1)
        filename = f"{conv_num:03d}_code_extraction.json"
        filepath = agent_dir / filename

        # Create extraction log
        extraction_log = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "extraction_method": extraction_method,
            "success": success,
            "error_message": error_message,
            "raw_response_length": len(raw_response),
            "extracted_code_length": len(extracted_code),
            "raw_response_preview": raw_response[:500] + "..." if len(raw_response) > 500 else raw_response,
            "extracted_code_preview": extracted_code[:500] + "..." if len(extracted_code) > 500 else extracted_code
        }

        # Save extraction log
        with open(filepath, 'w') as f:
            json.dump(extraction_log, f, indent=2)

        # Also save full extracted code if successful
        if success and extracted_code:
            code_filename = f"{conv_num:03d}_extracted_code.py"
            code_filepath = agent_dir / code_filename
            with open(code_filepath, 'w') as f:
                f.write(f"# Extracted using method: {extraction_method}\n")
                f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                f.write("#" * 80 + "\n\n")
                f.write(extracted_code)

        log_level = logging.INFO if success else logging.WARNING
        logger.log(log_level, f"Code extraction {'succeeded' if success else 'failed'} for {agent_name}")

        # Update session data
        self._update_agent_data(agent_name, "code_extractions", filename)

        return filename

    def log_api_validation(
        self,
        agent_name: str,
        api_plan: Dict[str, List[str]],
        verification_results: Dict[str, Any],
        corrections: List[Dict[str, str]]
    ) -> str:
        """
        Log API validation and correction process.

        Args:
            agent_name: Name of the agent
            api_plan: Planned APIs to use
            verification_results: Results of API verification
            corrections: List of corrections made

        Returns:
            Filename where the validation log was saved
        """
        self.ensure_dirs_sync()
        if not self.current_session:
            logger.warning("No active session")
            return ""

        # Get agent directory
        agent_dir = self._get_agent_dir(agent_name)

        # Create filename
        conv_num = self.conversation_counts.get(agent_name, 1)
        filename = f"{conv_num:03d}_api_validation.json"
        filepath = agent_dir / filename

        # Create validation log
        validation_log = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "api_plan": api_plan,
            "verification_results": verification_results,
            "corrections": corrections,
            "total_apis_planned": sum(len(v) for v in api_plan.values()),
            "apis_needing_correction": len(corrections)
        }

        # Save validation log
        with open(filepath, 'w') as f:
            json.dump(validation_log, f, indent=2)

        logger.info(f"Logged API validation for {agent_name}: {len(corrections)} corrections needed")

        # Update session data
        self._update_agent_data(agent_name, "api_validations", filename)

        return filename

    def log_session_stats(
        self,
        agent_name: str,
        session_kind: str,
        elapsed: float,
        usage: Dict[str, int],
        calls: int,
        turns: int = 0,
        provider: str = "",
        model: str = "",
    ) -> str:
        """Append a per-agent-session token/time stats record.

        Writes a numbered ``NNN_stats.json`` next to the agent's prompt /
        response artifacts and registers the entry in ``summary.json``
        under ``agents.<agent>.sessions``. ``calls`` counts raw LLM API
        responses observed; ``turns`` is non-zero only for tool_loop
        sessions and matches the loop's iteration count.
        """
        if not self.current_session:
            return ""

        agent_dir = self._get_agent_dir(agent_name)

        existing = sorted(
            p for p in agent_dir.glob("*_stats.json") if p.is_file()
        )
        next_num = len(existing) + 1
        filename = f"{next_num:03d}_stats.json"
        filepath = agent_dir / filename

        record = {
            "agent": agent_name,
            "session_kind": session_kind,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": float(elapsed),
            "usage": {
                "input": int(usage.get("input", 0) or 0),
                "output": int(usage.get("output", 0) or 0),
                "cache_read": int(usage.get("cache_read", 0) or 0),
                "cache_creation": int(usage.get("cache_creation", 0) or 0),
            },
            "calls": int(calls),
            "turns": int(turns),
            "provider": provider,
            "model": model,
        }
        record["usage"]["total"] = (
            record["usage"]["input"]
            + record["usage"]["output"]
            + record["usage"]["cache_read"]
            + record["usage"]["cache_creation"]
        )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        if agent_name not in self.session_data.get("agents", {}):
            self.session_data.setdefault("agents", {})[agent_name] = {}
        agent_entry = self.session_data["agents"][agent_name]
        agent_entry.setdefault("sessions", []).append({
            "file": filename,
            "session_kind": session_kind,
            "elapsed_seconds": record["elapsed_seconds"],
            "usage": record["usage"],
            "calls": record["calls"],
            "turns": record["turns"],
        })
        self._save_session_summary()

        logger.debug(
            f"Logged session stats for {agent_name}: {filename} "
            f"({record['elapsed_seconds']:.2f}s, total_tokens={record['usage']['total']})"
        )
        return filename

    def log_pipeline_stats(
        self,
        elapsed: float,
        usage: Dict[str, int],
        per_agent: Dict[str, Dict[str, Any]],
        sessions: int = 0,
        calls: int = 0,
    ) -> str:
        """Write the pipeline-level token / time summary.

        Stored at ``dialog/sessions/<session>/pipeline_stats.json`` and
        mirrored into ``summary.json`` under ``pipeline_stats`` so the
        existing post-mortem tooling sees it without having to load a
        second file.
        """
        if not self.current_session:
            return ""

        usage_norm = {
            "input": int(usage.get("input", 0) or 0),
            "output": int(usage.get("output", 0) or 0),
            "cache_read": int(usage.get("cache_read", 0) or 0),
            "cache_creation": int(usage.get("cache_creation", 0) or 0),
        }
        usage_norm["total"] = (
            usage_norm["input"]
            + usage_norm["output"]
            + usage_norm["cache_read"]
            + usage_norm["cache_creation"]
        )

        record = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": float(elapsed),
            "usage": usage_norm,
            "sessions": int(sessions),
            "calls": int(calls),
            "per_agent": per_agent,
        }

        filepath = self.current_session / "pipeline_stats.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        self.session_data["pipeline_stats"] = record
        self._save_session_summary()

        logger.info(
            f"Pipeline stats: {record['elapsed_seconds']:.2f}s, "
            f"total_tokens={usage_norm['total']}, agents={len(per_agent)}"
        )
        return str(filepath)

    def generate_session_report(self) -> str:
        """
        Generate an HTML report for the current session.

        Returns:
            Path to the generated HTML report
        """
        if not self.current_session:
            logger.warning("No active session")
            return ""

        report_path = self.current_session / "session_report.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Session Report - {self.session_data['session_id']}</title>
            <style>
                body {{ font-family: monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }}
                h1 {{ color: #569cd6; }}
                h2 {{ color: #4ec9b0; }}
                .prompt {{ background: #2d2d2d; padding: 10px; border-left: 3px solid #569cd6; margin: 10px 0; }}
                .response {{ background: #2d2d2d; padding: 10px; border-left: 3px solid #4ec9b0; margin: 10px 0; }}
                .cot {{ background: #2d2d2d; padding: 10px; border-left: 3px solid #ce9178; margin: 10px 0; }}
                .error {{ color: #f48771; }}
                .success {{ color: #4ec9b0; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            </style>
        </head>
        <body>
            <h1>Session Report: {self.session_data['session_id']}</h1>
            <p>Created: {self.session_data['created_at']}</p>
            <p>User Prompt: {self.session_data['user_prompt']}</p>

            <h2>Agents Activity</h2>
        """

        # Add agent activity details
        for agent_name, agent_data in self.session_data.get('agents', {}).items():
            html_content += f"""
            <h3>{agent_name}</h3>
            <ul>
                <li>Prompts: {len(agent_data.get('prompts', []))}</li>
                <li>Responses: {len(agent_data.get('responses', []))}</li>
                <li>Chain of Thought: {len(agent_data.get('chain_of_thought', []))}</li>
                <li>Code Extractions: {len(agent_data.get('code_extractions', []))}</li>
                <li>API Validations: {len(agent_data.get('api_validations', []))}</li>
            </ul>
            """

        html_content += """
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Generated session report: {report_path}")
        return str(report_path)

    def save_session_transcript(self, messages: List[Any]) -> str:
        """
        Write full session dialogue to current_session/transcript.txt and transcript.jsonl.

        Args:
            messages: List of AgentMessage (or dict with agent, content, timestamp, metadata)

        Returns:
            Path to transcript.txt, or "" if no session
        """
        if not self.current_session or not messages:
            return ""

        # Human-readable transcript
        transcript_path = self.current_session / "transcript.txt"
        lines = [
            f"# Session: {self.session_data.get('session_id', 'unknown')}",
            f"# Created: {self.session_data.get('created_at', '')}",
            f"# User prompt: {self.session_data.get('user_prompt', '')[:200]}...",
            "",
        ]
        for msg in messages:
            agent = msg.get("agent", "?") if isinstance(msg, dict) else getattr(msg, "agent", "?")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            ts = msg.get("timestamp", "") if isinstance(msg, dict) else getattr(msg, "timestamp", "")
            lines.append(f"--- [{ts}] {agent} ---")
            lines.append(content.strip())
            lines.append("")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # JSONL for programmatic use
        jsonl_path = self.current_session / "transcript.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for msg in messages:
                if isinstance(msg, dict):
                    rec = msg
                else:
                    rec = msg.model_dump() if hasattr(msg, "model_dump") else {"agent": str(msg)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.debug(f"Saved session transcript: {transcript_path}")
        return str(transcript_path)

    def _get_agent_dir(self, agent_name: str) -> Path:
        """Get the directory for a specific agent."""
        if not self.current_session:
            raise ValueError("No active session")

        # Map agent names to directories
        dir_mapping = {
            "PlanningAgent": "planning",
            "CodeGenerationAgent": "codegen",
            "ReviewAgent": "review",
            "ExecutionAgent": "execution",
            "CompilationChecker": "compilation"
        }

        dir_name = dir_mapping.get(agent_name, agent_name.lower())
        agent_dir = self.current_session / dir_name
        agent_dir.mkdir(exist_ok=True)

        return agent_dir

    def _get_next_conversation_number(self, agent_name: str) -> int:
        """Get the next conversation number for an agent."""
        current = self.conversation_counts.get(agent_name, 0)
        next_num = current + 1
        self.conversation_counts[agent_name] = next_num
        return next_num

    def _update_agent_data(self, agent_name: str, data_type: str, filename: str):
        """Update session data for an agent."""
        if agent_name not in self.session_data['agents']:
            self.session_data['agents'][agent_name] = {}

        if data_type not in self.session_data['agents'][agent_name]:
            self.session_data['agents'][agent_name][data_type] = []

        self.session_data['agents'][agent_name][data_type].append(filename)

        # Save updated summary
        self._save_session_summary()

    def _save_session_summary(self):
        """Save session summary to JSON."""
        if not self.current_session:
            return

        summary_path = self.current_session / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)

    def _update_index(self):
        """Update the global index of sessions."""
        index_path = self.base_dir / "index.json"

        # Load existing index
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"sessions": []}

        # Add current session
        session_entry = {
            "id": self.session_data['session_id'],
            "timestamp": self.session_data['timestamp'],
            "user_prompt": self.session_data['user_prompt'],
            "path": str(self.current_session)
        }

        index["sessions"].append(session_entry)

        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)