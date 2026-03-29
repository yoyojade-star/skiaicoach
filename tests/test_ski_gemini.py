"""Unit tests for GeminiSkiCoach (no OpenCV / YOLO)."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from ski_gemini import GeminiSkiCoach


class TestGeminiSkiCoachInit(unittest.TestCase):
    def test_requires_key_when_no_client(self):
        with patch("ski_gemini.os.getenv", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                GeminiSkiCoach()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_uses_injected_client_without_env(self):
        mock_client = MagicMock()
        coach = GeminiSkiCoach(client=mock_client)
        self.assertIs(coach.client, mock_client)


class TestGeminiSkiCoachGenerateFeedback(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.coach = GeminiSkiCoach(client=self.mock_client)

    def test_empty_summary_returns_message(self):
        out = self.coach.generate_feedback("/tmp/fake.mp4", None)
        self.assertEqual(out, "No run data available.")
        self.mock_client.files.upload.assert_not_called()

    def test_happy_path_deletes_uploaded_file(self):
        summary = {
            "backseat_percentage": 10.0,
            "breaking_at_waist_percentage": 5.0,
            "max_edge_inclination_deg": 40.0,
            "carving_score": 60,
        }

        vfile = MagicMock()
        vfile.name = "files/abc"
        vfile.state.name = "ACTIVE"
        self.mock_client.files.upload.return_value = vfile

        mock_resp = MagicMock()
        mock_resp.text = "Great turns!"
        self.mock_client.models.generate_content.return_value = mock_resp

        out = self.coach.generate_feedback("/tmp/fake.mp4", summary)
        self.assertEqual(out, "Great turns!")
        self.mock_client.files.delete.assert_called_once_with(name="files/abc")

    def test_runtime_error_from_upload_returns_error_string(self):
        summary = {
            "backseat_percentage": 0.0,
            "breaking_at_waist_percentage": 0.0,
            "max_edge_inclination_deg": 30.0,
            "carving_score": 40,
        }
        with patch.object(
            self.coach,
            "_load_active_video_file",
            side_effect=RuntimeError("Video processing failed on Gemini's servers."),
        ):
            out = self.coach.generate_feedback("/tmp/fake.mp4", summary)
        self.assertIn("Error", out)

    def test_generate_content_exception_returns_error_string(self):
        summary = {
            "backseat_percentage": 0.0,
            "breaking_at_waist_percentage": 0.0,
            "max_edge_inclination_deg": 30.0,
            "carving_score": 40,
        }
        vfile = MagicMock()
        vfile.name = "files/xyz"
        vfile.state.name = "ACTIVE"
        self.mock_client.files.upload.return_value = vfile
        self.mock_client.models.generate_content.side_effect = OSError("network down")

        out = self.coach.generate_feedback("/tmp/fake.mp4", summary)
        self.assertIn("Error", out)
        # Delete runs only after a successful generate_content; on failure the upload may linger.
        self.mock_client.files.delete.assert_not_called()


class TestGeminiSkiCoachGenerateFeedbackAgent(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.coach = GeminiSkiCoach(client=self.mock_client)

    def test_empty_summary_returns_error_payload(self):
        out = self.coach.generate_feedback_agent("/tmp/x.mp4", None, skills=None)
        self.assertEqual(out["primary_fault"], "Analysis Unavailable")
        self.assertIn("No run data", out["biomechanical_explanation"])

    def test_video_runtime_error_returns_error_payload(self):
        summary = {"carving_score": 50, "max_edge_inclination_deg": 30.0}
        with patch.object(
            self.coach,
            "_load_active_video_file",
            side_effect=RuntimeError("upload failed"),
        ):
            out = self.coach.generate_feedback_agent("/tmp/x.mp4", summary, skills="")
        self.assertEqual(out["primary_fault"], "Analysis Unavailable")
        self.assertIn("upload failed", out["biomechanical_explanation"])

    def test_happy_path_returns_validated_dict(self):
        summary = {
            "backseat_percentage": 1.0,
            "breaking_at_waist_percentage": 2.0,
            "max_edge_inclination_deg": 35.0,
            "carving_score": 55,
        }
        payload = {
            "primary_fault": "Backseat",
            "biomechanical_explanation": "Hips are behind boots.",
            "carving_score_analysis": "Score reflects moderate edge angles.",
            "recommended_drill_name": "Javelin turns",
            "drill_steps": ["Step 1", "Step 2"],
            "visual_observations": "Hands low.",
            "progression_note": "Keep going.",
        }
        vfile = MagicMock()
        vfile.name = "files/agent1"
        vfile.state.name = "ACTIVE"
        self.mock_client.files.upload.return_value = vfile
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(payload)
        self.mock_client.models.generate_content.return_value = mock_resp

        out = self.coach.generate_feedback_agent(
            "/tmp/x.mp4", summary, skills="Custom skill text."
        )
        self.assertEqual(out["primary_fault"], "Backseat")
        self.assertEqual(out["recommended_drill_name"], "Javelin turns")
        self.mock_client.files.delete.assert_called_once_with(name="files/agent1")

    def test_empty_model_response_returns_error_payload(self):
        summary = {"carving_score": 50, "max_edge_inclination_deg": 30.0}
        vfile = MagicMock()
        vfile.name = "files/e"
        vfile.state.name = "ACTIVE"
        self.mock_client.files.upload.return_value = vfile
        mock_resp = MagicMock()
        mock_resp.text = ""
        self.mock_client.models.generate_content.return_value = mock_resp

        out = self.coach.generate_feedback_agent("/tmp/x.mp4", summary, skills=None)
        self.assertEqual(out["primary_fault"], "Analysis Unavailable")
        self.assertIn("Empty model response", out["biomechanical_explanation"])


class TestGeminiSkiCoachChatFollowup(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.coach = GeminiSkiCoach(client=self.mock_client)

    def test_builds_history_and_returns_text(self):
        mock_resp = MagicMock()
        mock_resp.text = "  Do javelin turns.  "
        self.mock_client.models.generate_content.return_value = mock_resp

        out = self.coach.chat_followup(
            run_summary={"carving_score": 50},
            initial_feedback={"primary_fault": "A-frame"},
            chat_messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            user_message="Best drill?",
            skills="Focus on ankles.",
        )
        self.assertEqual(out, "Do javelin turns.")
        self.mock_client.models.generate_content.assert_called_once()
        call_kw = self.mock_client.models.generate_content.call_args.kwargs
        contents = call_kw["contents"]
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[-1].role, "user")

    def test_empty_model_reply_falls_back(self):
        mock_resp = MagicMock()
        mock_resp.text = ""
        self.mock_client.models.generate_content.return_value = mock_resp

        out = self.coach.chat_followup(
            run_summary={"carving_score": 1},
            initial_feedback="Earlier: stay forward.",
            chat_messages=[],
            user_message="Ok?",
            skills=None,
        )
        self.assertIn("rephrasing", out)


if __name__ == "__main__":
    unittest.main()
