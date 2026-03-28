"""Smoke test that `mainagent` builds the agent FastAPI app."""
from __future__ import annotations

import unittest

import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")


class TestMainAgentModule(unittest.TestCase):
    def test_app_exposed_with_expected_title(self):
        import mainagent

        self.assertEqual(mainagent.app.title, "SkiAI Backend (Agent)")


if __name__ == "__main__":
    unittest.main()
