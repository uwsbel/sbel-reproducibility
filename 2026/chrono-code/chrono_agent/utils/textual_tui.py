"""
Textual TUI application for Chrono-Agent.

Provides an interactive terminal interface with:
- Collapsible/expandable thinking blocks
- Real-time workflow status updates
- Keyboard navigation
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Tree, Static
    from textual.containers import Container, ScrollableContainer
    from textual.binding import Binding
    from textual.reactive import reactive

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

from chrono_agent.models.thinking import ThinkingBlock, AgentThinkingSession


# Agent color mapping for Textual CSS
AGENT_STYLES = {
    "PlanningAgent": "blue",
    "CodeGenerationAgent": "green",
    "ReviewAgent": "yellow",
    "ExecutionAgent": "magenta",
    "System": "cyan",
}


if TEXTUAL_AVAILABLE:

    class ThinkingTreeNode(Tree):
        """Custom tree widget for displaying thinking blocks."""

        def __init__(self, session: AgentThinkingSession, **kwargs):
            super().__init__("Workflow", **kwargs)
            self.session = session
            self._build_tree()

        def _build_tree(self):
            """Build tree from thinking session."""
            self.clear()
            self.root.expand()

            # Group blocks by agent
            agents_blocks: Dict[str, List[ThinkingBlock]] = {}
            for block in self.session.thinking_blocks:
                if block.agent not in agents_blocks:
                    agents_blocks[block.agent] = []
                agents_blocks[block.agent].append(block)

            # Build tree structure
            for agent, blocks in agents_blocks.items():
                agent_node = self.root.add(f"{agent} ({len(blocks)})", expand=False)

                for block in blocks:
                    timestamp = block.timestamp.strftime("%H:%M:%S")
                    node_label = f"[{timestamp}] {block.title}"
                    block_node = agent_node.add(node_label, expand=False)

                    # Add content as child if verbose
                    if len(block.content) > 0:
                        # Truncate long content
                        content = block.content[:200]
                        if len(block.content) > 200:
                            content += "..."
                        block_node.add_leaf(content)

        def refresh_tree(self):
            """Refresh tree from session data."""
            self._build_tree()


    class StatusPanel(Static):
        """Panel showing current workflow status."""

        status = reactive("Initializing...")
        current_agent = reactive("")

        def render(self) -> str:
            agent_display = f" [{self.current_agent}]" if self.current_agent else ""
            return f"Status: {self.status}{agent_display}"


    class ChronoAgentTUI(App):
        """Main Textual TUI application for Chrono-Agent."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 2;
            grid-columns: 1fr 2fr;
        }

        #sidebar {
            width: 100%;
            height: 100%;
            border: solid green;
        }

        #main {
            width: 100%;
            height: 100%;
            border: solid blue;
        }

        #status-panel {
            height: 3;
            border: solid cyan;
            padding: 0 1;
        }

        #thinking-tree {
            height: 100%;
        }

        #output-log {
            height: 1fr;
            border: solid white;
            padding: 1;
        }

        .agent-planning {
            color: blue;
        }

        .agent-codegen {
            color: green;
        }

        .agent-review {
            color: yellow;
        }

        .agent-execution {
            color: magenta;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("e", "expand_all", "Expand All"),
            Binding("c", "collapse_all", "Collapse All"),
            Binding("r", "refresh", "Refresh"),
            Binding("space", "toggle_node", "Toggle"),
        ]

        def __init__(self, session: Optional[AgentThinkingSession] = None, **kwargs):
            super().__init__(**kwargs)
            self.session = session or AgentThinkingSession(session_id="tui")
            self.thinking_tree: Optional[ThinkingTreeNode] = None
            self.status_panel: Optional[StatusPanel] = None
            self.output_log: Optional[Static] = None
            self._log_lines: List[str] = []

        def compose(self) -> ComposeResult:
            """Compose the TUI layout."""
            yield Header()

            with Container(id="sidebar"):
                yield StatusPanel(id="status-panel")
                yield ThinkingTreeNode(self.session, id="thinking-tree")

            with Container(id="main"):
                yield ScrollableContainer(
                    Static("Workflow output will appear here...", id="output-log")
                )

            yield Footer()

        def on_mount(self) -> None:
            """Called when app is mounted."""
            self.thinking_tree = self.query_one("#thinking-tree", ThinkingTreeNode)
            self.status_panel = self.query_one("#status-panel", StatusPanel)
            self.output_log = self.query_one("#output-log", Static)

        def action_quit(self) -> None:
            """Quit the application."""
            self.exit()

        def action_expand_all(self) -> None:
            """Expand all tree nodes."""
            if self.thinking_tree:
                self.thinking_tree.root.expand_all()

        def action_collapse_all(self) -> None:
            """Collapse all tree nodes."""
            if self.thinking_tree:
                self.thinking_tree.root.collapse_all()

        def action_refresh(self) -> None:
            """Refresh the tree display."""
            if self.thinking_tree:
                self.thinking_tree.refresh_tree()

        def action_toggle_node(self) -> None:
            """Toggle the currently selected node."""
            if self.thinking_tree:
                cursor_node = self.thinking_tree.cursor_node
                if cursor_node:
                    cursor_node.toggle()

        def update_status(self, status: str, agent: str = "") -> None:
            """Update the status panel."""
            if self.status_panel:
                self.status_panel.status = status
                self.status_panel.current_agent = agent

        def add_thinking(
            self,
            agent: str,
            title: str,
            content: str,
            thinking_type: str = "reasoning"
        ) -> None:
            """Add a thinking block and refresh display."""
            self.session.add_thinking(
                agent=agent,
                title=title,
                content=content,
                thinking_type=thinking_type
            )

            if self.thinking_tree:
                self.thinking_tree.refresh_tree()

        def log_output(self, message: str) -> None:
            """Add a line to the output log."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_lines.append(f"[{timestamp}] {message}")

            # Keep only last 100 lines
            if len(self._log_lines) > 100:
                self._log_lines = self._log_lines[-100:]

            if self.output_log:
                self.output_log.update("\n".join(self._log_lines))


    def run_tui(session: Optional[AgentThinkingSession] = None) -> ChronoAgentTUI:
        """
        Create and return a TUI app instance.

        Args:
            session: Optional thinking session to display

        Returns:
            ChronoAgentTUI instance
        """
        return ChronoAgentTUI(session=session)


else:
    # Fallback when Textual is not available

    class ChronoAgentTUI:
        """Placeholder when Textual is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Textual is not installed. Add it with `uv add textual` if you need TUI mode."
            )

    def run_tui(*args, **kwargs):
        raise ImportError(
            "Textual is not installed. Add it with `uv add textual` if you need TUI mode."
        )


def is_textual_available() -> bool:
    """Check if Textual is available."""
    return TEXTUAL_AVAILABLE
