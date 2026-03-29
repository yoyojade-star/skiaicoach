"""
Structured coaching payloads (Pydantic).

Used by `ski_gemini.generate_feedback_agent` for Gemini JSON / `response_json_schema`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["SkiCoachingFeedback"]


class SkiCoachingFeedback(BaseModel):
    """Structured multimodal coach output matching the agent API contract.

    This class defines the data structure for ski coaching feedback, including
    identified faults, explanations, scores, and recommended drills. It is used
    as a Pydantic model to ensure type safety and structure for API responses.

    Attributes:
        primary_fault (str): The main technical issue identified in the skier's
            performance.
        biomechanical_explanation (str): An explanation of the fault based on
            the skier's body mechanics and joint angles.
        carving_score_analysis (str): A detailed analysis of the carving score,
            with suggestions for improving edge angle.
        recommended_drill_name (str): The name of a specific drill recommended
            to address the primary fault.
        drill_steps (list[str]): A list of step-by-step instructions for
            performing the recommended drill.
        visual_observations (str): Specific observations made from the video
            analysis, such as posture or timing.
        progression_note (str): A note on the skier's progress over time,
            comparing to past performance if available.
    """

    primary_fault: str = Field(
        description=(
            "The top technical issue identified (e.g., Backseat, A-Frame, "
            "Lack of Angulation)."
        )
    )
    biomechanical_explanation: str = Field(
        description=(
            "A professional explanation of why this fault is occurring based on "
            "joint angles."
        )
    )
    carving_score_analysis: str = Field(
        description=(
            "A detailed breakdown of the 0-100 carving score and how to improve "
            "the edge angle."
        )
    )
    recommended_drill_name: str = Field(
        description=(
            "A specific, catchy name for the training drill (e.g., 'The Thousand "
            "Steps' or 'Sword Fight')."
        )
    )
    drill_steps: list[str] = Field(
        description=(
            "A step-by-step numbered list on how to perform the recommended drill "
            "on the snow."
        )
    )
    visual_observations: str = Field(
        description=(
            "Observations from the video footage, such as hand placement, head "
            "position, or pole plant timing."
        )
    )
    progression_note: str = Field(
        description=(
            "A personalized note comparing current performance to historical "
            "trends if available."
        )
    )