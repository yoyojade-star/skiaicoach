"""
Smoke tests for `frontend.py` (the Streamlit script).

`frontend_logic` is covered in `test_frontend_logic.py`. Here we only verify the
app script runs under Streamlit's `AppTest` and exposes expected chrome.

`get_history()` calls the real backend unless patched — patch `fetch_jobs` so
offline/CI does not hang on `localhost:8001`.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_PY = _ROOT / "frontend.py"


@pytest.mark.skipif(not FRONTEND_PY.is_file(), reason="frontend.py missing")
class TestFrontendPySmoke(unittest.TestCase):
    def test_app_loads_and_main_title_contains_skiai(self):
        with patch("frontend_logic.fetch_jobs", return_value=[]):
            at = AppTest.from_file(str(FRONTEND_PY), default_timeout=20).run(
                timeout=20
            )
        titles = [t.value for t in at.title]
        self.assertTrue(
            any(t and "SkiAI" in t for t in titles),
            msg=f"expected a title containing 'SkiAI', got {titles!r}",
        )


if __name__ == "__main__":
    unittest.main()
