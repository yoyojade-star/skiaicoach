"""Smoke test that the real `main` module builds the FastAPI app."""
from __future__ import annotations

import unittest

import pytest

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")


class TestMainModule(unittest.TestCase):
    def test_app_exposed_with_expected_title(self):
        import main

        self.assertEqual(main.app.title, "SkiAI Backend")


if __name__ == "__main__":
    unittest.main()
