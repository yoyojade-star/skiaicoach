"""
Unit tests for `frontend_logic` (no Streamlit).

The Streamlit entry script `frontend.py` is smoke-tested in `test_frontend_py.py`.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from frontend_logic import (
    DEFAULT_SKILLS_FALLBACK,
    analysis_video_url,
    build_upload_multipart_fields,
    chart_rows_from_jobs,
    fetch_jobs,
    fetch_video_bytes,
    job_is_stale_for_current_upload,
    load_skills_md,
    parse_coaching_feedback,
    strip_json_fenced_block,
    upload_identity,
)


class TestLoadSkillsMd(unittest.TestCase):
    def test_reads_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Skills\n")
            path = f.name
        try:
            self.assertEqual(load_skills_md(path).strip(), "# Skills")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_file_returns_fallback(self):
        self.assertEqual(load_skills_md("/nonexistent/skills.md"), DEFAULT_SKILLS_FALLBACK)


class TestStripJsonFencedBlock(unittest.TestCase):
    def test_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(strip_json_fenced_block(raw), '{"a": 1}')

    def test_plain_fence(self):
        raw = "```\nhello\n```"
        self.assertEqual(strip_json_fenced_block(raw), "hello")


class TestParseCoachingFeedback(unittest.TestCase):
    def test_dict_passthrough(self):
        d = {"primary_fault": "X", "drill_steps": ["a"]}
        out = parse_coaching_feedback(d)
        self.assertEqual(out["primary_fault"], "X")
        self.assertIsNot(out, d)

    def test_json_string(self):
        s = json.dumps({"primary_fault": "Backseat", "drill_steps": []})
        out = parse_coaching_feedback(s)
        self.assertEqual(out["primary_fault"], "Backseat")

    def test_fenced_json_string(self):
        inner = json.dumps({"primary_fault": "AFrame"})
        raw = f"```json\n{inner}\n```"
        out = parse_coaching_feedback(raw)
        self.assertEqual(out["primary_fault"], "AFrame")

    def test_plain_text_fallback(self):
        out = parse_coaching_feedback("Just some coach text.")
        self.assertEqual(out["primary_fault"], "Technical Analysis")
        self.assertIn("Just some coach text.", out["biomechanical_explanation"])

    def test_none_fallback(self):
        out = parse_coaching_feedback(None)
        self.assertEqual(out["primary_fault"], "Technical Analysis")


class TestChartRowsFromJobs(unittest.TestCase):
    def test_only_completed(self):
        jobs = [
            {"status": "failed", "job_id": "a"},
            {
                "status": "completed",
                "job_id": "d7ba9abe-9064",
                "summary": {"carving_score": 50, "max_edge_inclination_deg": 40},
            },
        ]
        rows = chart_rows_from_jobs(jobs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Date"], "d7ba9abe")
        self.assertEqual(rows[0]["Score"], 50)
        self.assertEqual(rows[0]["Edge"], 40)


class TestBuildUploadMultipartFields(unittest.TestCase):
    def test_file_only_no_agent_skills_key_when_none(self):
        f = {"x": 1}
        fields = build_upload_multipart_fields("a.mp4", f, "video/mp4", agent_skills=None)
        self.assertEqual(list(fields.keys()), ["file"])
        self.assertEqual(fields["file"], ("a.mp4", f, "video/mp4"))

    def test_default_mime_when_empty_type(self):
        fields = build_upload_multipart_fields("a.mp4", b"", None, agent_skills=None)
        self.assertEqual(fields["file"][2], "video/mp4")

    def test_includes_agent_skills_string(self):
        fields = build_upload_multipart_fields(
            "a.mp4", b"x", "video/mp4", agent_skills="rules"
        )
        self.assertEqual(fields["agent_skills"], "rules")


class TestAnalysisVideoUrl(unittest.TestCase):
    def test_joins_path(self):
        self.assertEqual(
            analysis_video_url("http://localhost:8001", "abc-uuid"),
            "http://localhost:8001/uploads/abc-uuid_analyzed.mp4",
        )

    def test_strips_trailing_slash_on_base(self):
        self.assertEqual(
            analysis_video_url("http://x:1/", "j"),
            "http://x:1/uploads/j_analyzed.mp4",
        )


class TestUploadSessionBinding(unittest.TestCase):
    def test_upload_identity_none(self):
        self.assertIsNone(upload_identity(None))

    def test_upload_identity_from_file_like(self):
        f = MagicMock()
        f.name = "a.mp4"
        f.size = 42
        self.assertEqual(upload_identity(f), ("a.mp4", 42))

    def test_stale_when_bound_file_differs(self):
        self.assertTrue(
            job_is_stale_for_current_upload(
                job_id="j1",
                bound_identity=("a.mp4", 10),
                current_identity=("b.mp4", 20),
                job_from_history=False,
            )
        )

    def test_not_stale_when_same_file(self):
        self.assertFalse(
            job_is_stale_for_current_upload(
                job_id="j1",
                bound_identity=("a.mp4", 10),
                current_identity=("a.mp4", 10),
                job_from_history=False,
            )
        )

    def test_not_stale_when_viewing_history_with_file_in_uploader(self):
        self.assertFalse(
            job_is_stale_for_current_upload(
                job_id="j1",
                bound_identity=None,
                current_identity=("new.mp4", 5),
                job_from_history=True,
            )
        )

    def test_stale_legacy_no_binding_not_from_history(self):
        self.assertTrue(
            job_is_stale_for_current_upload(
                job_id="j1",
                bound_identity=None,
                current_identity=("new.mp4", 5),
                job_from_history=False,
            )
        )


class TestFetchJobs(unittest.TestCase):
    def test_success_returns_list(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"job_id": "1"}]

        def fake_get(url, timeout=30):
            self.assertIn("/jobs", url)
            return mock_resp

        self.assertEqual(fetch_jobs("http://h:1", get=fake_get), [{"job_id": "1"}])

    def test_error_returns_empty(self):
        def boom(*a, **k):
            raise ConnectionError("nope")

        self.assertEqual(fetch_jobs("http://h:1", get=boom), [])


class TestFetchVideoBytes(unittest.TestCase):
    def test_200_returns_content(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"vid"

        def fake_get(url, stream=True, timeout=120):
            return mock_resp

        self.assertEqual(fetch_video_bytes("http://x/v.mp4", get=fake_get), b"vid")

    def test_206_returns_content(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 206
        mock_resp.content = b"partial"

        def fake_get(url, stream=True, timeout=120):
            return mock_resp

        self.assertEqual(fetch_video_bytes("http://x/v.mp4", get=fake_get), b"partial")

    def test_non_200_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        def fake_get(url, stream=True, timeout=120):
            return mock_resp

        self.assertIsNone(fetch_video_bytes("http://x/v.mp4", get=fake_get))


if __name__ == "__main__":
    unittest.main()
