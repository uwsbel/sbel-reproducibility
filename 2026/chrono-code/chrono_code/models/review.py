"""
Pydantic models for code review (Agent 3 output).
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class SimulationDataAnalysis(BaseModel):
    """
    Results of analyzing simulation CSV data.

    Attributes:
        csv_files_found: Number of CSV files found
        data_completeness: Percentage of expected columns found (0-1)
        temporal_coverage: Coverage of temporal data
        anomalies_detected: List of detected anomalies
        physics_metrics: Physics validation metrics
        quality_score: Overall quality score (0-10)
        validation_issues: List of validation issues found
        recommendations: List of recommendations based on analysis
    """

    csv_files_found: int = Field(
        default=0,
        description="Number of CSV files found in output"
    )

    data_completeness: float = Field(
        default=0.0,
        description="Percentage of expected columns found (0-1)"
    )

    temporal_coverage: Literal["full", "partial", "sparse", "none"] = Field(
        default="none",
        description="Coverage of temporal data in simulation"
    )

    anomalies_detected: List[str] = Field(
        default_factory=list,
        description="List of detected anomalies in data"
    )

    physics_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Physics validation metrics (energy conservation, constraints, etc.)"
    )

    quality_score: float = Field(
        default=0.0,
        description="Overall data quality score (0-10)"
    )

    validation_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of physics validation issues found"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations based on data analysis"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "csv_files_found": 1,
                "data_completeness": 0.95,
                "temporal_coverage": "full",
                "anomalies_detected": ["sudden_jump in position at t=2.5s"],
                "physics_metrics": {
                    "max_constraint_violation": 1.2e-7,
                    "energy_drift_percent": 0.02,
                    "stability_score": 0.98
                },
                "quality_score": 8.5,
                "validation_issues": [],
                "recommendations": [
                    "Consider reducing time step for better accuracy",
                    "Energy conservation is excellent"
                ]
            }
        }


class ReviewResult(BaseModel):
    """
    Represents the review result from the Review Agent (Agent 3).

    Attributes:
        approved: Whether the code is approved
        physics_check: Physics correctness assessment
        common_sense_check: Common sense validation
        feedback: Detailed feedback from the reviewer
        issues: List of identified issues
        suggestions: List of improvement suggestions
    """

    approved: bool = Field(
        description="Whether the code passes review"
    )

    physics_check: Literal["valid", "questionable", "invalid"] = Field(
        description="Physics correctness assessment"
    )

    common_sense_check: Literal["valid", "questionable", "invalid"] = Field(
        description="Common sense validation result"
    )

    feedback: str = Field(
        description="Detailed feedback and reasoning"
    )

    issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of identified issues with structured information (issue_type, location, description, correction)"
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )

    data_analysis: Optional[SimulationDataAnalysis] = Field(
        default=None,
        description="Analysis of simulation output data (CSV files)"
    )

    # VLM Visual Review fields
    visual_quality_score: Optional[float] = Field(
        default=None,
        description="Overall visual quality score from VLM analysis (0-10)"
    )

    visual_issues: List["VisualIssue"] = Field(
        default_factory=list,
        description="List of visual issues detected by VLM"
    )

    analyzed_frames: List[str] = Field(
        default_factory=list,
        description="List of frame paths that were analyzed by VLM"
    )

    video_description: Optional[str] = Field(
        default=None,
        description="Pure descriptive summary of what is visible in the reviewed video or frames"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "approved": True,
                "physics_check": "valid",
                "common_sense_check": "valid",
                "feedback": "The simulation correctly implements a bouncing ball with proper physics. Gravity, collision detection, and material properties are appropriately configured.",
                "issues": [],
                "suggestions": [
                    "Consider adding damping for more realistic behavior",
                    "Could visualize velocity vectors for educational purposes"
                ]
            }
        }


class PhysicsIssue(BaseModel):
    """
    Represents a specific physics issue found in the code.

    Attributes:
        category: Category of the issue
        severity: Severity level
        description: Description of the issue
        location: Where in the code the issue occurs
        correction: Suggested correction
    """

    category: Literal[
        "missing_gravity",
        "invalid_mass",
        "impossible_velocity",
        "missing_constraint",
        "invalid_material",
        "energy_violation",
        "other"
    ] = Field(
        description="Category of physics issue"
    )

    severity: Literal["critical", "warning", "suggestion"] = Field(
        description="Severity of the issue"
    )

    description: str = Field(
        description="Detailed description of the issue"
    )

    location: str = Field(
        description="Location in code where issue occurs"
    )

    correction: str = Field(
        description="Suggested correction"
    )


class VisualIssue(BaseModel):
    """
    Represents a visual issue found by VLM analysis of simulation output.

    Attributes:
        category: Freeform category/type of the visual issue (no restrictions)
        severity: Severity level
        description: Description of the visual problem
        reasoning: LLM's reasoning for why this is an issue
        frame_index: Frame number where issue was detected (if applicable)
        time_range: Time range in video where issue occurs (e.g., "0.5-1.2s")
        suggested_fix: Suggested code fix for the issue
    """

    category: str = Field(
        description="Category/type of visual issue (freeform, e.g., 'missing_connection', 'physics_anomaly', 'visualization_incomplete')"
    )

    severity: Literal["critical", "warning", "suggestion"] = Field(
        description="Severity of the issue"
    )

    description: str = Field(
        description="Detailed description of the visual problem"
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="LLM's reasoning for why this is considered an issue"
    )

    frame_index: Optional[int] = Field(
        default=None,
        description="Frame number where issue was detected"
    )

    time_range: Optional[str] = Field(
        default=None,
        description="Time range in video where issue occurs (e.g., '0.5-1.2s')"
    )

    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested code fix for the visual issue"
    )


# Rebuild model to resolve forward reference for VisualIssue in ReviewResult
ReviewResult.model_rebuild()
