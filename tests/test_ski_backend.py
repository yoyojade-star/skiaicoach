"""
Tests for `ski_backend`: `normalize_video_extension`, `run_analysis_task`, and `create_app`.

`create_app` exposes one POST `/upload` that always accepts optional multipart field
`agent_skills`. When `use_agent_feedback=False` (default / `main.py`), skills are ignored
for the pipeline and omitted from the processing job record. When `use_agent_feedback=True`
(`mainagent.py`), `skills_preview` is stored and `run_analysis_task` runs with
`merge_coach_graph=False` and `agent_skills` forwarded.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ski_backend import (
    configure_server_event_loop,
    create_app,
    normalize_video_extension,
    run_analysis_task,
)


class TestConfigureServerEventLoop(unittest.TestCase):
    def test_callable_without_error(self):
        configure_server_event_loop()


class TestNormalizeVideoExtension(unittest.TestCase):
    def test_known_extension_preserved(self):
        self.assertEqual(normalize_video_extension("run.MOV"), ".mov")

    def test_unknown_extension_defaults_to_mp4(self):
        self.assertEqual(normalize_video_extension("clip.exe"), ".mp4")

    def test_none_filename_defaults_to_mp4(self):
        self.assertEqual(normalize_video_extension(None), ".mp4")


class TestRunAnalysisTask(unittest.TestCase):
    def setUp(self):
        self.saved: list[tuple[str, dict]] = []

    def save(self, job_id: str, data: dict) -> None:
        self.saved.append((job_id, data))

    def test_standard_pipeline_requires_graph(self):
        td = Path(tempfile.mkdtemp())
        tmp_in = td / "in.mp4"
        try:
            tmp_in.write_bytes(b"x")
            proc = MagicMock()
            proc.process_video.return_value = []
            run_analysis_task(
                "j-graph",
                str(tmp_in),
                str(td / "out.mp4"),
                save_result=self.save,
                processor_cls=MagicMock(return_value=proc),
                coach_cls=MagicMock(),
                ski_app_graph=None,
                summarize_run_data=lambda _: {"carving_score": 0},
                merge_coach_graph=True,
            )
        finally:
            shutil.rmtree(td, ignore_errors=True)
        err = self.saved[-1][1]["error"]
        self.assertEqual(self.saved[-1][1]["status"], "failed")
        self.assertIn("ski_app_graph", err)
        self.assertIn("standard coaching pipeline", err)

    def test_agent_pipeline_skips_graph(self):
        td = Path(tempfile.mkdtemp())
        tmp_in = td / "in.mp4"
        in_path = str(tmp_in)
        coach_inst = MagicMock()
        coach_inst.generate_feedback_agent.return_value = {"primary_fault": "ok"}
        try:
            tmp_in.write_bytes(b"x")
            proc = MagicMock()
            proc.process_video.return_value = [{"flags": [], "edge_inclination_deg": 20.0}]

            def summarize(data):
                return {
                    "carving_score": 10,
                    "backseat_percentage": 0.0,
                    "breaking_at_waist_percentage": 0.0,
                    "max_edge_inclination_deg": 20.0,
                }

            run_analysis_task(
                "j-agent",
                in_path,
                str(td / "out.mp4"),
                save_result=self.save,
                processor_cls=MagicMock(return_value=proc),
                coach_cls=MagicMock(return_value=coach_inst),
                ski_app_graph=None,
                summarize_run_data=summarize,
                agent_skills="Custom rules",
                merge_coach_graph=False,
            )
        finally:
            shutil.rmtree(td, ignore_errors=True)

        self.assertEqual(self.saved[-1][1]["status"], "completed")
        self.assertEqual(self.saved[-1][1]["feedback"]["primary_fault"], "ok")
        coach_inst.generate_feedback_agent.assert_called_once()
        kw = coach_inst.generate_feedback_agent.call_args.kwargs
        self.assertEqual(kw["video_path"], in_path)
        self.assertEqual(kw["skills"], "Custom rules")
        self.assertIsInstance(kw["summary"], dict)

    def test_missing_components_marks_failed(self):
        run_analysis_task(
            "j1",
            "/no/such/file.mp4",
            "/out.mp4",
            save_result=self.save,
            processor_cls=None,
            coach_cls=None,
            ski_app_graph=None,
            summarize_run_data=None,
        )
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.saved[0][1]["status"], "failed")
        self.assertIn("processor, coach, or summarizer", self.saved[0][1]["error"])

    def test_happy_path_with_mocks(self):
        td = Path(tempfile.mkdtemp())
        tmp_in = td / "in.mp4"
        tmp_out = td / "out.mp4"
        try:
            tmp_in.write_bytes(b"x")
            graph = MagicMock()
            graph.invoke.return_value = {"feedback": "trend + ai"}

            proc = MagicMock()
            proc.process_video.return_value = [{"flags": [], "edge_inclination_deg": 20.0}]

            coach = MagicMock()
            coach.generate_feedback.return_value = "AI text"

            def summarize(data):
                return {
                    "carving_score": 10,
                    "backseat_percentage": 0.0,
                    "breaking_at_waist_percentage": 0.0,
                    "max_edge_inclination_deg": 20.0,
                }

            run_analysis_task(
                "job-a",
                str(tmp_in),
                str(tmp_out),
                save_result=self.save,
                processor_cls=MagicMock(return_value=proc),
                coach_cls=MagicMock(return_value=coach),
                ski_app_graph=graph,
                summarize_run_data=summarize,
                merge_coach_graph=True,
            )
        finally:
            shutil.rmtree(td, ignore_errors=True)

        self.assertEqual(len(self.saved), 1)
        payload = self.saved[0][1]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["feedback"], "trend + ai")
        self.assertIn("video_url", payload)
        proc.process_video.assert_called_once()
        coach.generate_feedback.assert_called_once()
        graph.invoke.assert_called_once()


class _CreateAppHarness(unittest.TestCase):
    """Shared temp dirs + TestClient; subclasses set `use_agent_feedback` and optional `title`."""

    use_agent_feedback: bool = False
    app_title: str | None = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.upload = self.tmp / "up"
        self.data = self.tmp / "data"
        kwargs = dict(
            upload_dir=str(self.upload),
            data_dir=str(self.data),
            use_agent_feedback=self.use_agent_feedback,
            processor_cls=None,
            coach_cls=None,
            ski_app_graph=None,
            summarize_run_data=None,
        )
        if self.app_title is not None:
            kwargs["title"] = self.app_title
        self.app = create_app(**kwargs)
        self.client = TestClient(self.app)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCreateAppStandardRoutes(_CreateAppHarness):
    """`create_app(..., use_agent_feedback=False)` — same as `main.py`."""

    use_agent_feedback = False
    app_title = "SkiAI Standard Test"

    def test_app_title(self):
        self.assertEqual(self.app.title, "SkiAI Standard Test")

    def test_get_result_not_found(self):
        r = self.client.get("/result/does-not-exist")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "not_found")

    def test_get_result_from_disk(self):
        jid = "abc-123"
        payload = {"status": "completed", "job_id": jid}
        self.data.mkdir(parents=True, exist_ok=True)
        (self.data / f"{jid}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        r = self.client.get(f"/result/{jid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], jid)

    def test_list_jobs_reads_json_files(self):
        self.data.mkdir(parents=True, exist_ok=True)
        (self.data / "a.json").write_text(
            json.dumps({"job_id": "a"}), encoding="utf-8"
        )
        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["job_id"], "a")

    def test_upload_multipart_and_schedules_standard_pipeline(self):
        with patch("ski_backend.run_analysis_task") as mock_task:
            files = {"file": ("clip.mov", b"fakevideo", "video/quicktime")}
            r = self.client.post("/upload", files=files)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "processing")
        jid = body["job_id"]
        stored = json.loads((self.data / f"{jid}.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["status"], "processing")
        self.assertNotIn("skills_preview", stored)
        uploaded = list(self.upload.glob(f"{jid}*"))
        self.assertTrue(any(p.suffix.lower() == ".mov" for p in uploaded))
        mock_task.assert_called_once()
        kw = mock_task.call_args.kwargs
        self.assertEqual(kw.get("merge_coach_graph"), True)
        self.assertIsNone(kw.get("agent_skills"))

    def test_standard_mode_ignores_agent_skills_form_field(self):
        """Unified `/upload` accepts `agent_skills` Form; standard app drops it before the task."""
        with patch("ski_backend.run_analysis_task") as mock_task:
            files = {"file": ("x.mp4", b"x", "video/mp4")}
            data = {"agent_skills": "Should be ignored in standard mode."}
            r = self.client.post("/upload", files=files, data=data)
        self.assertEqual(r.status_code, 200)
        jid = r.json()["job_id"]
        stored = json.loads((self.data / f"{jid}.json").read_text(encoding="utf-8"))
        self.assertNotIn("skills_preview", stored)
        kw = mock_task.call_args.kwargs
        self.assertIsNone(kw.get("agent_skills"))
        self.assertEqual(kw.get("merge_coach_graph"), True)


class TestCreateAppAgentRoutes(_CreateAppHarness):
    """`create_app(..., use_agent_feedback=True)` — same as `mainagent.py`."""

    use_agent_feedback = True
    app_title = "SkiAI Agent Test"

    def test_app_title(self):
        self.assertEqual(self.app.title, "SkiAI Agent Test")

    def test_upload_forwards_agent_skills_and_disables_graph_merge(self):
        with patch("ski_backend.run_analysis_task") as mock_task:
            files = {"file": ("clip.mp4", b"fake", "video/mp4")}
            data = {"agent_skills": "Focus on ankle flexion."}
            r = self.client.post("/upload", files=files, data=data)
        self.assertEqual(r.status_code, 200)
        jid = r.json()["job_id"]
        stored = json.loads((self.data / f"{jid}.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["skills_preview"], "Focus on ankle flexion.")
        mock_task.assert_called_once()
        kw = mock_task.call_args.kwargs
        self.assertEqual(kw.get("merge_coach_graph"), False)
        self.assertEqual(kw.get("agent_skills"), "Focus on ankle flexion.")

    def test_skills_preview_default_when_skills_omitted(self):
        with patch("ski_backend.run_analysis_task"):
            files = {"file": ("a.mp4", b"x", "video/mp4")}
            r = self.client.post("/upload", files=files)
        self.assertEqual(r.status_code, 200)
        jid = r.json()["job_id"]
        stored = json.loads((self.data / f"{jid}.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["skills_preview"], "Default Coaching")

    def test_skills_preview_truncates_long_text(self):
        long_skills = "x" * 101
        with patch("ski_backend.run_analysis_task"):
            files = {"file": ("b.mp4", b"x", "video/mp4")}
            data = {"agent_skills": long_skills}
            r = self.client.post("/upload", files=files, data=data)
        self.assertEqual(r.status_code, 200)
        jid = r.json()["job_id"]
        stored = json.loads((self.data / f"{jid}.json").read_text(encoding="utf-8"))
        self.assertEqual(len(stored["skills_preview"]), 103)
        self.assertTrue(stored["skills_preview"].endswith("..."))
        self.assertEqual(stored["skills_preview"][:100], "x" * 100)


if __name__ == "__main__":
    unittest.main()
