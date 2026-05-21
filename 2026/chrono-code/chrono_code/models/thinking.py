"""
Models for tracking agent thinking process and reasoning.
Used for terminal UI collapsible thinking display.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ThinkingBlock(BaseModel):
    """
    Represents a block of agent thinking/reasoning.

    Used to capture and display agent thought processes in the terminal UI.
    Supports hierarchical structure for nested thoughts.
    """

    agent: str = Field(description="Name of the agent that produced this thought")
    title: str = Field(description="Short title/summary of the thinking block")
    content: str = Field(description="Full content of the thinking")
    timestamp: datetime = Field(default_factory=datetime.now)
    collapsed: bool = Field(default=True, description="Whether this block is collapsed in UI")
    thinking_type: str = Field(
        default="reasoning",
        description="Type of thinking: reasoning, plan, analysis, error, etc."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["ThinkingBlock"] = Field(
        default_factory=list,
        description="Nested thinking blocks"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentThinkingSession(BaseModel):
    """
    Collection of thinking blocks for a workflow session.

    Tracks all agent thoughts throughout a simulation generation workflow.
    """

    session_id: str = Field(description="Unique session identifier")
    started_at: datetime = Field(default_factory=datetime.now)
    thinking_blocks: List[ThinkingBlock] = Field(default_factory=list)
    current_agent: Optional[str] = Field(default=None)

    def add_thinking(
        self,
        agent: str,
        title: str,
        content: str,
        thinking_type: str = "reasoning",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ThinkingBlock:
        """Add a new thinking block to the session."""
        block = ThinkingBlock(
            agent=agent,
            title=title,
            content=content,
            thinking_type=thinking_type,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        self.thinking_blocks.append(block)
        self.current_agent = agent
        return block

    def get_blocks_by_agent(self, agent: str) -> List[ThinkingBlock]:
        """Get all thinking blocks from a specific agent."""
        return [b for b in self.thinking_blocks if b.agent == agent]

    def get_latest_blocks(self, count: int = 5) -> List[ThinkingBlock]:
        """Get the most recent thinking blocks."""
        return self.thinking_blocks[-count:] if self.thinking_blocks else []


# Enable forward references for recursive model
ThinkingBlock.model_rebuild()
