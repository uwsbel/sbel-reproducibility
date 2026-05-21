"""
Terminal UI utilities for Chrono-Code.

Provides collapsible/expandable output display using Rich library,
with optional Textual TUI mode for full interactivity.
"""

import uuid
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.live import Live

from chrono_code.models.thinking import ThinkingBlock, AgentThinkingSession


# Agent color mapping
AGENT_COLORS = {
    "PlanningAgent": "blue",
    "CodeGenerationAgent": "green",
    "ReviewAgent": "yellow",
    "ExecutionAgent": "magenta",
    "System": "cyan",
}

# Thinking type icons
THINKING_ICONS = {
    "reasoning": "💭",
    "plan": "📋",
    "analysis": "🔍",
    "error": "❌",
    "success": "✅",
    "warning": "⚠️",
    "code": "💻",
    "api": "🔧",
}


class AgentOutputManager:
    """
    Manages collapsible/expandable agent output in the terminal.

    Supports three detail levels:
    - minimal: Only show agent names and final results
    - normal: Show agent activities with summaries
    - verbose: Show full thinking process with expandable details
    """

    def __init__(
        self,
        detail_level: str = "normal",
        console: Optional[Console] = None,
        interactive: bool = False,
    ):
        """
        Initialize the output manager.

        Args:
            detail_level: Output detail level (minimal/normal/verbose)
            console: Rich console instance (creates new one if not provided)
            interactive: Whether to use interactive Textual TUI mode
        """
        self.detail_level = detail_level
        self.console = console or Console()
        self.interactive = interactive

        # Create thinking session
        self.session = AgentThinkingSession(
            session_id=str(uuid.uuid4())[:8]
        )

        # Track expanded agents for display
        self.expanded_agents: set = set()

        # Live display context (for real-time updates)
        self._live: Optional[Live] = None

    def add_thinking(
        self,
        agent: str,
        title: str,
        content: str,
        thinking_type: str = "reasoning",
        metadata: Optional[Dict[str, Any]] = None,
        display: bool = True,
    ) -> ThinkingBlock:
        """
        Add a thinking block and optionally display it.

        Args:
            agent: Agent name
            title: Short title
            content: Full content
            thinking_type: Type of thinking
            metadata: Additional metadata
            display: Whether to display immediately

        Returns:
            The created ThinkingBlock
        """
        block = self.session.add_thinking(
            agent=agent,
            title=title,
            content=content,
            thinking_type=thinking_type,
            metadata=metadata
        )

        if display:
            self._display_block(block)

        return block

    def _display_block(self, block: ThinkingBlock):
        """Display a single thinking block based on detail level."""
        if self.detail_level == "minimal":
            return  # Don't display in minimal mode

        color = AGENT_COLORS.get(block.agent, "white")
        icon = THINKING_ICONS.get(block.thinking_type, "💬")

        if self.detail_level == "normal":
            # Show compact summary
            timestamp = block.timestamp.strftime("%H:%M:%S")
            self.console.print(
                f"[dim]{timestamp}[/dim] [{color}]{block.agent}[/{color}] {icon} {block.title}"
            )
        elif self.detail_level == "verbose":
            # Show full content with panel
            timestamp = block.timestamp.strftime("%H:%M:%S")
            header = f"{icon} {block.title}"

            # Truncate very long content with expand hint
            content = block.content
            if len(content) > 500:
                content = content[:500] + "\n[dim]... (truncated)[/dim]"

            self.console.print(Panel(
                content,
                title=f"[{color}]{block.agent}[/{color}] - {header}",
                subtitle=f"[dim]{timestamp}[/dim]",
                border_style=color,
                padding=(0, 1)
            ))

    def render_tree(
        self,
        expanded_agents: Optional[List[str]] = None
    ) -> Tree:
        """
        Render all thinking blocks as a Rich Tree.

        Args:
            expanded_agents: List of agent names to show expanded

        Returns:
            Rich Tree object
        """
        if expanded_agents is None:
            expanded_agents = list(self.expanded_agents)

        tree = Tree(
            f"[bold cyan]Workflow Session[/bold cyan] ({self.session.session_id})",
            guide_style="dim"
        )

        # Group blocks by agent
        agents_blocks: Dict[str, List[ThinkingBlock]] = {}
        for block in self.session.thinking_blocks:
            if block.agent not in agents_blocks:
                agents_blocks[block.agent] = []
            agents_blocks[block.agent].append(block)

        # Build tree structure
        for agent, blocks in agents_blocks.items():
            color = AGENT_COLORS.get(agent, "white")
            is_expanded = agent in expanded_agents

            # Agent branch
            expand_marker = "▼" if is_expanded else "▶"
            agent_branch = tree.add(
                f"[{color}]{expand_marker} {agent}[/{color}] ({len(blocks)} items)"
            )

            if is_expanded:
                for block in blocks:
                    icon = THINKING_ICONS.get(block.thinking_type, "💬")
                    timestamp = block.timestamp.strftime("%H:%M:%S")

                    if self.detail_level == "verbose":
                        # Show content preview
                        preview = block.content[:100].replace("\n", " ")
                        if len(block.content) > 100:
                            preview += "..."
                        agent_branch.add(
                            f"[dim]{timestamp}[/dim] {icon} {block.title}\n"
                            f"[dim]{preview}[/dim]"
                        )
                    else:
                        agent_branch.add(
                            f"[dim]{timestamp}[/dim] {icon} {block.title}"
                        )

        return tree

    def display_tree(self, expanded_agents: Optional[List[str]] = None):
        """Display the thinking tree."""
        tree = self.render_tree(expanded_agents)
        self.console.print(tree)

    def toggle_agent(self, agent: str):
        """Toggle expanded state for an agent."""
        if agent in self.expanded_agents:
            self.expanded_agents.discard(agent)
        else:
            self.expanded_agents.add(agent)

    def expand_all(self):
        """Expand all agents."""
        for block in self.session.thinking_blocks:
            self.expanded_agents.add(block.agent)

    def collapse_all(self):
        """Collapse all agents."""
        self.expanded_agents.clear()

    def start_live(self):
        """Start live display mode for real-time updates."""
        self._live = Live(
            self.render_tree(),
            console=self.console,
            refresh_per_second=2,
            transient=True
        )
        self._live.start()

    def update_live(self):
        """Update the live display."""
        if self._live:
            self._live.update(self.render_tree())

    def stop_live(self):
        """Stop live display mode."""
        if self._live:
            self._live.stop()
            self._live = None

    def display_agent_header(self, agent: str, action: str):
        """Display a header when an agent starts working."""
        color = AGENT_COLORS.get(agent, "white")

        if self.detail_level == "minimal":
            self.console.print(f"[{color}]{agent}[/{color}]: {action}")
        else:
            self.console.print(Panel.fit(
                f"[bold]{action}[/bold]",
                title=f"[{color}]{agent}[/{color}]",
                border_style=color
            ))

    def display_error(self, agent: str, error: str, details: Optional[str] = None):
        """Display an error message."""
        self.add_thinking(
            agent=agent,
            title=f"Error: {error[:50]}",
            content=details or error,
            thinking_type="error"
        )

        self.console.print(f"[red]Error from {agent}:[/red] {error}")
        if details and self.detail_level == "verbose":
            self.console.print(Panel(details, border_style="red"))

    def display_success(self, agent: str, message: str):
        """Display a success message."""
        self.add_thinking(
            agent=agent,
            title=message,
            content=message,
            thinking_type="success"
        )

        color = AGENT_COLORS.get(agent, "white")
        self.console.print(f"[{color}]{agent}[/{color}] [green]✓[/green] {message}")


def create_output_manager(
    detail_level: str = "normal",
    interactive: bool = False
) -> AgentOutputManager:
    """
    Factory function to create an output manager.

    Args:
        detail_level: Output detail level
        interactive: Whether to use interactive mode

    Returns:
        AgentOutputManager instance
    """
    return AgentOutputManager(
        detail_level=detail_level,
        interactive=interactive
    )
