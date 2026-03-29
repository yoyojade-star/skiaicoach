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

CHAT_FOLLOWUP_SYSTEM_PREFIX = """
You are the same elite PSIA-certified alpine ski coach who already produced the initial analysis below.
The user is asking follow-up questions about that same run.

Rules:
- Ground answers in the kinematic summary and your initial coaching. Stay consistent with it.
- You cannot see the video again in this chat; do not claim to re-watch footage. If they need another visual pass, say they should run a new analysis upload.
- Be concise, practical, and safety-aware. Prefer specific cues and drills when helpful.
"""


def _initial_feedback_as_text(initial_feedback: Any) -> str:
    """Formats the initial coaching feedback into a consistent string.

    Args:
        initial_feedback (Any): The initial feedback, which could be a dict,
            string, or None.

    Returns:
        str: A string representation of the feedback.
    """
    if initial_feedback is None:
        return "(No initial coaching text.)"
    if isinstance(initial_feedback, dict):
        try:
            return json.dumps(initial_feedback, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(initial_feedback)
    return str(initial_feedback).strip() or "(No initial coaching text.)"


def _agent_error_payload(message: str) -> dict[str, Any]:
    """Creates a standardized JSON payload for agent-based errors.

    Args:
        message (str): The error message to include in the payload.

    Returns:
        dict[str, Any]: A dictionary structured for error reporting.
    """
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
        """Initializes the GeminiSkiCoach.

        Args:
            api_key (str, optional): The Gemini API key. If not provided, the
                GEMINI_API_KEY environment variable is used.
            client (Any, optional): An existing `genai.Client` instance. If provided,
                `api_key` is ignored. Used for testing.
            model_id (str, optional): The specific Gemini model to use. Defaults
                to "gemini-3.1-pro-preview".

        Raises:
            ValueError: If neither `api_key`, `client`, nor the `GEMINI_API_KEY`
                environment variable is provided.
        """
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
        """Uploads a video file to Gemini and waits for it to become active.

        This method handles the file upload and polls the API until the video
        is processed and ready for analysis.

        Args:
            video_path (str): The local path to the video file.

        Returns:
            Any: The active `File` object from the Gemini API.

        Raises:
            RuntimeError: If video processing fails on the server.
        """
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
        """Generates free-form text feedback for a ski run.

        Args:
            video_path (str): The local path to the ski run video.
            run_summary (dict[str, Any] | None): A dictionary containing kinematic
                data about the run.

        Returns:
            str: The generated coaching feedback as a string, or an error message.
        """
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
        """Generates structured JSON coaching feedback.

        This method is used by the agent backend to get a structured response
        that conforms to the `SkiCoachingFeedback` schema. It can optionally
        inject a knowledge base from a `skills.md`-style file.

        Args:
            video_path (str): The local path to the ski run video.
            summary (dict[str, Any] | None): A dictionary containing kinematic
                data about the run.
            skills (str, optional): A string containing a custom knowledge base
                to guide the analysis. Defaults to standard PSIA biomechanics.

        Returns:
            dict[str, Any]: A dictionary containing the structured coaching
                feedback or a standardized error payload.
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

    def chat_followup(
        self,
        *,
        run_summary: dict[str, Any],
        initial_feedback: Any,
        chat_messages: list[dict[str, Any]],
        user_message: str,
        skills: str | None = None,
    ) -> str:
        """Handles text-only follow-up questions after an initial analysis.

        This method uses the stored run summary and initial feedback as context
        to answer user questions without re-analyzing the video.

        Args:
            run_summary (dict[str, Any]): The kinematic summary of the run.
            initial_feedback (Any): The initial coaching feedback (text or dict).
            chat_messages (list[dict[str, Any]]): The history of the conversation,
                where each dict has "role" ('user' or 'assistant') and "content".
            user_message (str): The latest question from the user.
            skills (str, optional): A string containing a custom knowledge base
                to ground the answers.

        Returns:
            str: The model's text response to the user's question.
        """
        feedback_text = _initial_feedback_as_text(initial_feedback)
        skills_block = (skills or "").strip()
        if not skills_block:
            skills_block = "Use standard PSIA biomechanics."

        system_instruction = (
            CHAT_FOLLOWUP_SYSTEM_PREFIX
            + "\nOPTIONAL KNOWLEDGE BASE (from session skills):\n"
            + skills_block
            + "\n\nKINEMATIC SUMMARY (JSON):\n"
            + json.dumps(run_summary, ensure_ascii=False)
            + "\n\nINITIAL COACHING (verbatim from analysis):\n"
            + feedback_text
        )

        contents: list[Any] = []
        for m in chat_messages:
            role = m.get("role")
            text = (m.get("content") or "").strip()
            if not text:
                continue
            api_role = "user" if role == "user" else "model"
            if role not in ("user", "assistant"):
                continue
            contents.append(
                types.Content(
                    role=api_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message.strip())],
            )
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.5,
                ),
            )
            out = (response.text or "").strip()
            return out or "I don't have a response right now. Please try rephrasing."
        except Exception as e:
            return f"Sorry, I could not answer that: {e}"