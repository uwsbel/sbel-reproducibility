"""
Pydantic models for simulation execution (Agent 4 output).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """
    Represents the result of executing the simulation (Agent 4 output).

    Attributes:
        success: Whether execution completed successfully
        output_files: List of generated output files (images, videos, etc.)
        execution_log: Captured stdout/stderr from execution
        runtime_seconds: Total runtime in seconds
        error_message: Error message if execution failed
        performance_metrics: Performance and resource usage metrics
        visualization_count: Number of visualization frames generated
    """

    success: bool = Field(
        description="Whether the simulation executed successfully"
    )

    output_files: List[str] = Field(
        default_factory=list,
        description="Paths to generated output files (images, videos, data)"
    )

    execution_log: str = Field(
        default="",
        description="Captured output from simulation execution"
    )

    runtime_seconds: float = Field(
        default=0.0,
        description="Total execution time in seconds"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )

    error_messages: List[str] = Field(
        default_factory=list,
        description="Deduplicated execution error messages collected during run_simulation"
    )

    return_code: Optional[int] = Field(
        default=None,
        description="Process return code (negative values indicate signals, e.g., -11 = SIGSEGV)"
    )

    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics (memory, CPU, etc.)"
    )

    visualization_count: int = Field(
        default=0,
        description="Number of visualization frames generated"
    )

    csv_data_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics from CSV data files"
    )

    csv_files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected CSV files with metadata and analysis"
    )

    simulation_data_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physics validation metrics derived from CSV data"
    )

    structured_error: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Parsed runtime-error fields for retry feedback. Keys: "
            "error_type, failing_symbol, failing_line, file_line_excerpt, "
            "introspection_hint. Populated only when execution failed and a "
            "traceback (or signal) was detected."
        )
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "output_files": [
                    "/outputs/simulation_0001.png",
                    "/outputs/simulation_0002.png",
                    "/outputs/simulation_animation.gif"
                ],
                "execution_log": "Simulation started...\\nFrame 1/100\\n...\\nSimulation completed successfully",
                "runtime_seconds": 5.2,
                "error_message": None,
                "error_messages": [],
                "performance_metrics": {
                    "peak_memory_mb": 256,
                    "avg_cpu_percent": 45.3,
                    "frames_per_second": 30
                },
                "visualization_count": 100
            }
        }


class RuntimeError(BaseModel):
    """
    Represents a runtime error during simulation execution.

    Attributes:
        error_type: Type of runtime error
        message: Error message
        traceback: Full error traceback
        timestamp: When the error occurred (simulation time)
    """

    error_type: str = Field(
        description="Type of runtime error"
    )

    message: str = Field(
        description="Error message"
    )

    traceback: str = Field(
        description="Full traceback of the error"
    )

    timestamp: float = Field(
        description="Simulation time when error occurred"
    )
