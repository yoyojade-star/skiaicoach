"""Gemini multimodal ski coach (Google GenAI only — no OpenCV / YOLO)."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from google import genai
from google.genai import types

from schema import SkiCoachingFeedback

GEMINI_COACH_SYSTEM_INSTRUCTION = """
You are an elite, PSIA-certified alpine ski instructor and biomechanics expert.
You will be provided with a video of a skier, along with hard kinematic data extracted by a computer vision model.

ANALYSIS PROTOCOL:
1. Cross-reference your visual observations with the hard data provided (Edge angles, Backseat percentage).
2. Ankle Flexion: Check if shins press against boot tongues.
3. Knee Alignment: Check if knees track over toes; watch for A-framing.
4. Hip Position: Ensure hips are centered, not in the "backseat."
5. Upper Body/Core: Look for a neutral spine and slight forward hinge.
6. Arm/Hand Placement: Hands should be forward and wide ("carrying a tray").
7. Head/Gaze: Eyes should be looking down the fall line.

FEEDBACK STRUCTURE:
- Initial Observation: What is the skier doing well?
- The Breakdown: Bulleted list of specific postural faults.
- The Impact: Explain WHY this hinders performance (e.g., loss of leverage).
- Actionable Corrections: 1-2 physical cues (e.g., "crush the grape").
- Targeted Drills: 1-2 specific drills to fix the primary flaw.
"""

DEFAULT_SKILLS_FALLBACK = (
    "Use standard PSIA biomechanics: focus on shin pressure, centered stance, and outside ski control."
)


def _agent_error_payload(message: str) -> dict[str, Any]:
    return {
        "primary_fault": "Analysis Unavailable",
        "biomechanical_explanation": message,
        "carving_score_analysis": "Error during processing.",
        "recommended_drill_name": "Check System Logs",
        "drill_steps": [
            "Ensure your API connection is stable.",
            "Verify video file integrity.",
        ],
        "visual_observations": "N/A",
        "progression_note": "N/A",
    }


class GeminiSkiCoach:
    """Multimodal coaching via Google GenAI. Pass `client` in tests to avoid network calls."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        model_id: str | None = None,
    ):
        if client is not None:
            self.client = client
        else:
            key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "Gemini API key required: pass api_key=... to GeminiSkiCoach() "
                    "or set the GEMINI_API_KEY environment variable."
                )
            self.client = genai.Client(api_key=key)

        self.model_id = model_id or "gemini-3.1-pro-preview"
        self.system_instruction = GEMINI_COACH_SYSTEM_INSTRUCTION

    def _load_active_video_file(self, video_path: str) -> Any:
        """Upload file, wait until ACTIVE; raises RuntimeError on failure."""
        print("Uploading video to Gemini for visual analysis...")
        video_file = self.client.files.upload(file=video_path)
        print("Waiting for video processing to complete...")
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)
        print()
        if video_file.state.name == "FAILED":
            raise RuntimeError("Video processing failed on Gemini's servers.")
        print("Video ready. Generating AI coaching feedback...")
        return video_file

    def generate_feedback(self, video_path: str, run_summary: dict[str, Any] | None) -> str:
        if not run_summary:
            return "No run data available."

        try:
            video_file = self._load_active_video_file(video_path)
        except RuntimeError as e:
            return f"Error: {e}"

        prompt = f"""
        Please review the attached video of my ski run, along with this hard kinematic data:
        - Time in backseat: {run_summary['backseat_percentage']}%
        - Time breaking at waist: {run_summary['breaking_at_waist_percentage']}%
        - Max edge angle achieved: {run_summary['max_edge_inclination_deg']} degrees
        - Carving Score: {run_summary['carving_score']}/100

        Based on what you see in the video and the data provided, give me a master-class level breakdown of my technique and how to improve.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.4,
                ),
            )
            self.client.files.delete(name=video_file.name)
            return response.text

        except Exception as e:
            return f"Error: {str(e)}"

    def generate_feedback_agent(
        self,
        video_path: str,
        summary: dict[str, Any] | None,
        skills: str | None = None,
    ) -> dict[str, Any]:
        """
        Structured JSON coaching with optional skills.md-style knowledge injection.
        Used by the agent backend (`mainagent`); no LangGraph merge in the API layer.
        """
        if not summary:
            return _agent_error_payload("No run data available.")

        skills_block = (skills or "").strip() or DEFAULT_SKILLS_FALLBACK

        system_instruction = f"""
You are an elite, highly observant PSIA-certified Alpine Ski Coach.
Analyze the provided video of a skier, along with kinematic data from computer vision.

STRICT TECHNICAL KNOWLEDGE BASE TO APPLY:
{skills_block}

ANALYSIS PROTOCOL:
1. Cross-reference visual observations with the kinematic summary.
2. Evaluate execution using the KNOWLEDGE BASE above.
3. Recommend specific drills to refine form.

Respond with JSON only, matching the response schema exactly.
"""

        prompt = f"Kinematic summary (JSON): {json.dumps(summary)}"

        try:
            video_file = self._load_active_video_file(video_path)
        except RuntimeError as e:
            return _agent_error_payload(str(e))

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                    response_mime_type="application/json",
                    response_json_schema=SkiCoachingFeedback.model_json_schema(),
                ),
            )
            self.client.files.delete(name=video_file.name)
            raw = response.text
            if not raw:
                return _agent_error_payload("Empty model response.")
            data = json.loads(raw)
            validated = SkiCoachingFeedback.model_validate(data)
            return validated.model_dump()

        except Exception as e:
            return _agent_error_payload(f"The coach could not process the data: {e}")
