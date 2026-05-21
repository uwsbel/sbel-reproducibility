"""
Logging configuration for Chrono-Code.

Provides structured logging with configurable output levels and formats.
Includes boxed blocks for modification suggestions and error output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List
from rich.logging import RichHandler

# Box drawing for readable blocks in logs (no fancy Unicode for file compatibility)
BOX_WIDTH = 72
BOX_CHAR = "="
BOX_SIDE = "| "


def format_boxed(title: str, body: str, width: int = BOX_WIDTH) -> str:
    """Format a title and multi-line body as a clear box for log output."""
    lines = [title]
    lines.extend(body.strip().split("\n"))
    border = BOX_CHAR * width
    out = [border]
    for line in lines:
        # Wrap long lines
        while len(line) > width - 4:
            out.append(BOX_SIDE + line[: width - 4] + " |")
            line = line[width - 4 :]
        out.append(BOX_SIDE + line.ljust(width - 4) + " |")
    out.append(border)
    return "\n".join(out)


def log_boxed(logger: logging.Logger, level: int, title: str, body: str) -> None:
    """Log a boxed block (e.g. modification suggestions or stderr)."""
    msg = format_boxed(title, body)
    logger.log(level, "\n" + msg)


def setup_logger(
    name: str = "chrono_code",
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_rich: Whether to use Rich for pretty console output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_time=True,
            show_path=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)

    # File handler (if specified): cleaner format, one line per record (message can be multiline)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "chrono_code") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class AgentLogger:
    """
    Wrapper for agent-specific logging with context.
    """

    def __init__(self, agent_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize agent logger.

        Args:
            agent_name: Name of the agent (e.g., "PlanningAgent")
            logger: Optional parent logger
        """
        self.agent_name = agent_name
        self.logger = logger or get_logger()

    def _format_message(self, message: str) -> str:
        """Format message with agent name."""
        return f"[{self.agent_name}] {message}"

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message), *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message), *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message), *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message), *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message), *args, **kwargs)
