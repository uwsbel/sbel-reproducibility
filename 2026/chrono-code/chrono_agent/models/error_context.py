"""
Error Context Models for Chrono-Agent.

This module provides a generic framework for error tracking and strategy escalation.
It does NOT rely on specific error types (like SIGSEGV) - instead, it extracts
comparable "signatures" from any error and detects patterns.

Key Principles:
- Generic: Works with any error type
- Evidence-driven: Strategy decisions based on error history patterns
- Diff-only: Keep incremental unified-diff fixes and fail explicitly when patching cannot proceed
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import hashlib


class ErrorSignature(BaseModel):
    """
    Generic error signature for detecting repeated errors.

    Instead of hardcoding error types like "sigsegv" or "python_exception",
    we extract comparable features from ANY error.
    """
    # Error location (file:line or function name, if extractable)
    location: Optional[str] = None

    # Core error message (truncated)
    core_message: str

    # Auto-inferred category (not hardcoded types)
    category: str  # "syntax", "import", "crash", "runtime", "unknown"

    # Iteration when this error occurred
    iteration: int

    # Timestamp
    timestamp: datetime = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)

    def get_hash(self) -> str:
        """
        Generate a hash for this error signature.

        Used for quick comparison - two errors with the same hash
        are considered the same error pattern.
        """
        content = f"{self.location or 'unknown'}:{self.category}:{self.core_message[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        return f"[{self.category}] {self.location or 'unknown'}: {self.core_message[:50]}..."


class ErrorResolution(BaseModel):
    """Resolution/update entry for a previously recorded error signature."""

    signature: str
    iteration: int
    status: str = "resolved"
    note: str = ""
    timestamp: datetime = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ErrorHistory(BaseModel):
    """
    Error history tracker for prior attempts only.

    The current failure should travel separately in the handoff; this model is
    reserved for anti-repetition memory across previous attempts.
    """
    errors: List[ErrorSignature] = []
    resolved: List[ErrorResolution] = []

    def add(self, error: ErrorSignature):
        """Add a new error to history."""
        self.errors.append(error)

    def mark_resolved(self, signature: str, iteration: int, note: str = "") -> None:
        """Record that a previously seen signature has been addressed in a later attempt."""
        self.resolved.append(
            ErrorResolution(signature=signature, iteration=iteration, note=note)
        )

    def get_signature_counts(self) -> Dict[str, int]:
        """Count occurrences of each signature."""
        counts = {}
        for err in self.errors:
            h = err.get_hash()
            counts[h] = counts.get(h, 0) + 1
        return counts

    def get_repeated_signatures(self, threshold: int = 2) -> List[str]:
        """
        Return signatures that appear threshold or more times.

        This is the key pattern detection method.
        """
        counts = self.get_signature_counts()
        return [h for h, c in counts.items() if c >= threshold]

    def get_location_frequency(self) -> Dict[str, int]:
        """
        Count error frequency by location.

        Useful for identifying problem hotspots.
        """
        freq = {}
        for err in self.errors:
            if err.location:
                freq[err.location] = freq.get(err.location, 0) + 1
        return freq

    def should_escalate(self, current_level: int) -> bool:
        """
        Determine if strategy should be escalated.

        Diff-only mode: escalation is disabled.
        """
        return False

    def get_diagnosis_summary(self) -> str:
        """
        Generate a diagnosis summary for LLM reference.

        This summary helps the LLM understand the error patterns
        without us hardcoding specific error handling.
        """
        if not self.errors:
            return "No errors recorded."

        lines = ["=== Error History Summary ==="]

        # Signature statistics
        counts = self.get_signature_counts()
        repeated = [(h, c) for h, c in counts.items() if c >= 2]

        if repeated:
            lines.append(f"\n**Repeated Error Patterns ({len(repeated)}):**")
            for h, c in sorted(repeated, key=lambda x: -x[1]):
                # Find the corresponding error
                for err in self.errors:
                    if err.get_hash() == h:
                        lines.append(f"  - [{c}x] {err}")
                        break

        # Location hotspots
        loc_freq = self.get_location_frequency()
        hotspots = [(l, f) for l, f in loc_freq.items() if f >= 2]
        if hotspots:
            lines.append(f"\n**Problem Hotspots:**")
            for loc, freq in sorted(hotspots, key=lambda x: -x[1]):
                lines.append(f"  - {loc}: {freq} errors")

        # Recent errors
        lines.append(f"\n**Recent Errors ({min(3, len(self.errors))}):**")
        for err in self.errors[-3:]:
            lines.append(f"  - [{err.iteration}] {err}")

        # Summary statistics
        lines.append(f"\n**Statistics:**")
        lines.append(f"  - Total errors: {len(self.errors)}")
        lines.append(f"  - Unique patterns: {len(counts)}")
        lines.append(f"  - Repeated patterns: {len(repeated)}")

        return "\n".join(lines)

    def get_last_error(self) -> Optional[ErrorSignature]:
        """Get the most recent error."""
        return self.errors[-1] if self.errors else None


class StrategyLevel(BaseModel):
    """
    Current fix strategy level.

    Level definitions:
    - 1: UNIFIED DIFF ONLY - retry patching; no rebuild fallback
    """
    level: int = 1
    reason: str = "Diff-only mode"
    escalation_history: List[str] = []

    def __init__(self, **data):
        data["level"] = 1
        if not data.get("reason"):
            data["reason"] = "Diff-only mode"
        super().__init__(**data)

    def escalate(self, reason: str) -> bool:
        """
        Escalate to the next strategy level.

        Returns True if escalation succeeded, False if already at max level.
        """
        _ = reason
        # Diff-only mode: no further escalation.
        self.level = 1
        return False

    def get_strategy_description(self) -> str:
        """Get a description of the current strategy."""
        return "UNIFIED DIFF ONLY: Retry patching with strict baseline matching."

    def get_strategy_instructions(self) -> str:
        """Get instructions for the LLM based on current strategy level."""
        return """
**Strategy: UNIFIED DIFF ONLY**
- Keep edits incremental and patch-based
- Apply on the exact baseline snapshot only
- If patch retries fail, stop and surface a structured failure
"""
