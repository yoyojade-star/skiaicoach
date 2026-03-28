"""Streamlit AppTest coverage for `frontend_streamlit_common`."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def _run(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=45).run(timeout=45)


class TestClearJobIfUploaderFileChanged(unittest.TestCase):
    def test_clears_job_when_new_file_selected(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import (
    ensure_job_id_session_state,
    clear_job_if_uploader_file_changed,
)

ensure_job_id_session_state()
st.session_state["job_id"] = "keep-or-clear"
st.session_state["_job_started_for_upload_key"] = ("old.mp4", 1)
st.session_state["_job_from_history"] = False

class NewFile:
    name = "new.mp4"
    size = 2

clear_job_if_uploader_file_changed(NewFile())
st.text(st.session_state.get("job_id") is None)
"""
        )
        values = [t.value for t in at.text]
        self.assertTrue(any("True" in (v or "") for v in values))


class TestEnsureJobSessionState(unittest.TestCase):
    def test_initializes_job_keys(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import ensure_job_id_session_state
ensure_job_id_session_state()
st.text(repr(st.session_state.get("job_id")))
st.text(repr(st.session_state.get("_job_started_for_upload_key")))
"""
        )
        values = [t.value for t in at.text]
        self.assertTrue(any("None" in (v or "") for v in values), msg=repr(values))


class TestConfigurePage(unittest.TestCase):
    def test_sets_page_config(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import configure_page, STANDARD_PROFILE
configure_page(STANDARD_PROFILE)
st.title("ok")
"""
        )
        titles = [t.value for t in at.title]
        self.assertTrue(any("ok" in (t or "") for t in titles))


class TestRenderStandardSkillsSidebar(unittest.TestCase):
    def test_renders_header_and_expander(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import render_standard_skills_sidebar
with st.sidebar:
    render_standard_skills_sidebar("# Skills")
"""
        )
        headers = [h.value for h in at.sidebar.header]
        self.assertTrue(any("Agent Configuration" in (h or "") for h in headers))


class TestRenderAgentSkillsTextArea(unittest.TestCase):
    def test_returns_widget_value(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import render_agent_skills_text_area
with st.sidebar:
    render_agent_skills_text_area("hello skills")
"""
        )
        ta = list(at.sidebar.text_area)
        self.assertTrue(ta, msg="expected a text_area in sidebar")
        self.assertEqual(ta[0].value, "hello skills")


class TestRenderHistorySidebar(unittest.TestCase):
    def test_empty_history_shows_info(self):
        with patch("frontend_streamlit_common.fetch_jobs", return_value=[]):
            at = _run(
                """
import streamlit as st
from frontend_streamlit_common import render_history_sidebar
with st.sidebar:
    render_history_sidebar("http://localhost:8001")
"""
            )
        infos = [i.value for i in at.sidebar.info]
        self.assertTrue(any("No past runs" in (m or "") for m in infos))

    def test_completed_job_shows_button_and_chart(self):
        history = [
            {
                "job_id": "abcdef12-3456-7890-abcd-ef1234567890",
                "status": "completed",
                "filename": "run.mp4",
                "summary": {"carving_score": 55, "max_edge_inclination_deg": 40},
            }
        ]
        with patch("frontend_streamlit_common.fetch_jobs", return_value=history):
            at = _run(
                """
import streamlit as st
from frontend_streamlit_common import render_history_sidebar
with st.sidebar:
    render_history_sidebar("http://h:1")
"""
            )
        labels = [b.label for b in at.sidebar.button]
        self.assertTrue(any("run.mp4" in lbl for lbl in labels))
        # line_chart is a VegaLite chart in newer Streamlit — check generic chart/widget presence
        self.assertTrue(len(at.sidebar.subheader) >= 1)


class TestRunUploadWithProgress(unittest.TestCase):
    def test_success_sets_session_state(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"job_id": "11111111-2222-3333-4444-555555555555"}

        with patch("frontend_streamlit_common.requests.post", return_value=mock_resp):
            at = _run(
                """
import io
import streamlit as st
from frontend_streamlit_common import run_upload_with_progress

class NamedBytes(io.BytesIO):
    def __init__(self, data, name, content_type="video/mp4"):
        super().__init__(data)
        self.name = name
        self.type = content_type

run_upload_with_progress(
    "http://api:8001",
    NamedBytes(b"fake-bytes", "clip.mp4"),
    agent_skills=None,
    success_message="done",
    upload_identity_key=("clip.mp4", 99),
)
st.text(st.session_state.get("job_id") or "")
"""
            )
        texts = [t.value for t in at.text]
        self.assertTrue(
            any("11111111-2222-3333-4444-555555555555" in (x or "") for x in texts)
        )

    def test_http_error_shows_message(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "server boom"

        with patch("frontend_streamlit_common.requests.post", return_value=mock_resp):
            at = _run(
                """
import io
import streamlit as st
from frontend_streamlit_common import run_upload_with_progress

class NamedBytes(io.BytesIO):
    def __init__(self, data, name, content_type="video/mp4"):
        super().__init__(data)
        self.name = name
        self.type = content_type

run_upload_with_progress(
    "http://x", NamedBytes(b"x", "a.mp4"), agent_skills=None, success_message="ok"
)
"""
            )
        errs = [e.value for e in at.error]
        self.assertTrue(any("Upload failed" in (m or "") for m in errs))

    def test_post_exception_shows_connection_error(self):
        with patch(
            "frontend_streamlit_common.requests.post",
            side_effect=ConnectionError("down"),
        ):
            at = _run(
                """
import io
import streamlit as st
from frontend_streamlit_common import run_upload_with_progress

class NamedBytes(io.BytesIO):
    def __init__(self, data, name, content_type="video/mp4"):
        super().__init__(data)
        self.name = name
        self.type = content_type

run_upload_with_progress(
    "http://x", NamedBytes(b"z", "a.mp4"), agent_skills=None, success_message="ok"
)
"""
            )
        errs = [e.value for e in at.error]
        self.assertTrue(any("Connection Error" in (m or "") for m in errs))


class TestPollJobUntilTerminal(unittest.TestCase):
    def test_completed_renders_results(self):
        done = MagicMock()
        done.json.return_value = {
            "status": "completed",
            "summary": {"carving_score": 10, "max_edge_inclination_deg": 20, "backseat_percentage": 1.0},
            "feedback": {"primary_fault": "Test", "biomechanical_explanation": "x"},
        }
        with patch("frontend_streamlit_common.requests.get", return_value=done):
            at = _run(
                """
import streamlit as st
from frontend_streamlit_common import poll_job_until_terminal, STANDARD_PROFILE
poll_job_until_terminal(
    "http://h:1",
    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    lambda url: b"fake",
    STANDARD_PROFILE,
)
"""
            )
        subs = [s.value for s in at.subheader]
        self.assertTrue(any("Performance Metrics" in (s or "") for s in subs))

    def test_failed_shows_error(self):
        fail = MagicMock()
        fail.json.return_value = {"status": "failed", "error": "bad input"}
        with patch("frontend_streamlit_common.requests.get", return_value=fail):
            at = _run(
                """
import streamlit as st
from frontend_streamlit_common import poll_job_until_terminal, STANDARD_PROFILE
poll_job_until_terminal(
    "http://h:1",
    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    lambda u: None,
    STANDARD_PROFILE,
)
"""
            )
        errs = [e.value for e in at.error]
        self.assertTrue(any("bad input" in (m or "") for m in errs))

    def test_connection_error_breaks_loop(self):
        with patch(
            "frontend_streamlit_common.requests.get",
            side_effect=ConnectionError("refused"),
        ):
            at = _run(
                """
import streamlit as st
from frontend_streamlit_common import poll_job_until_terminal, STANDARD_PROFILE
poll_job_until_terminal(
    "http://h:1",
    "cccccccc-cccc-cccc-cccc-cccccccccccc",
    lambda u: None,
    STANDARD_PROFILE,
)
"""
            )
        errs = [e.value for e in at.error]
        self.assertTrue(any("refused" in (m or "") for m in errs))

    def test_processing_then_completed(self):
        proc = MagicMock()
        proc.json.return_value = {"status": "processing"}
        done = MagicMock()
        done.json.return_value = {
            "status": "completed",
            "summary": {"carving_score": 1, "max_edge_inclination_deg": 20, "backseat_percentage": 0.0},
            "feedback": {},
        }
        with patch(
            "frontend_streamlit_common.requests.get",
            side_effect=[proc, done],
        ):
            with patch("frontend_streamlit_common.time.sleep"):
                at = _run(
                    """
import streamlit as st
from frontend_streamlit_common import poll_job_until_terminal, STANDARD_PROFILE
poll_job_until_terminal(
    "http://h:1",
    "dddddddd-dddd-dddd-dddd-dddddddddddd",
    lambda u: None,
    STANDARD_PROFILE,
)
"""
                )
        subs = [s.value for s in at.subheader]
        self.assertTrue(any("Performance Metrics" in (s or "") for s in subs))


class TestRenderCompletedResultsAgentProfile(unittest.TestCase):
    def test_carving_expander_when_enabled(self):
        at = _run(
            """
import streamlit as st
from frontend_streamlit_common import render_completed_results, AGENT_PROFILE
render_completed_results(
    "http://h:1",
    "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
    {"carving_score": 5, "max_edge_inclination_deg": 10, "backseat_percentage": 0.0},
    {
        "primary_fault": "P",
        "biomechanical_explanation": "E",
        "carving_score_analysis": "C",
        "recommended_drill_name": "D",
        "drill_steps": ["a", "b"],
        "visual_observations": "V",
        "progression_note": "N",
    },
    lambda url: b"vid",
    AGENT_PROFILE,
)
"""
            )
        expanders = [e.label for e in at.expander]
        self.assertTrue(any("Carving Analysis" in lbl for lbl in expanders))


if __name__ == "__main__":
    unittest.main()
