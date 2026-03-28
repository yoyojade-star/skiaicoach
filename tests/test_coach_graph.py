"""Unit tests for coach_graph pure helpers and compiled LangGraph."""
from __future__ import annotations

import unittest

from coach_graph import (
    PROGRESS_SEPARATOR,
    analyze_progress,
    build_coach_graph,
    merge_feedback_with_trend,
    trend_message,
)


class TestTrendMessage(unittest.TestCase):
    def test_empty_history_first_run(self):
        self.assertEqual(
            trend_message({"carving_score": 50}, []),
            "First run!",
        )

    def test_score_improved(self):
        prev = [{"carving_score": 40}]
        self.assertEqual(
            trend_message({"carving_score": 55}, prev),
            "Score up by 15 points!",
        )

    def test_score_flat_or_down_focus_consistency(self):
        self.assertEqual(
            trend_message({"carving_score": 40}, [{"carving_score": 40}]),
            "Focus on consistency.",
        )
        self.assertEqual(
            trend_message({"carving_score": 30}, [{"carving_score": 50}]),
            "Focus on consistency.",
        )

    def test_missing_carving_score_treated_as_zero(self):
        self.assertEqual(
            trend_message({}, []),
            "First run!",
        )
        self.assertEqual(
            trend_message({}, [{"carving_score": 10}]),
            "Focus on consistency.",
        )


class TestMergeFeedbackWithTrend(unittest.TestCase):
    def test_preserves_ai_and_appends_trend(self):
        out = merge_feedback_with_trend("Great turns!", "First run!")
        self.assertIn("Great turns!", out)
        self.assertIn(PROGRESS_SEPARATOR, out)
        self.assertTrue(out.endswith("First run!"))

    def test_whitespace_only_ai_yields_trend_only(self):
        self.assertEqual(
            merge_feedback_with_trend("   \n", "First run!"),
            "First run!",
        )

    def test_none_ai_yields_trend_only(self):
        self.assertEqual(
            merge_feedback_with_trend(None, "Focus on consistency."),
            "Focus on consistency.",
        )


class TestAnalyzeProgress(unittest.TestCase):
    def test_merges_like_invoke(self):
        state = {
            "current_summary": {"carving_score": 60},
            "previous_summaries": [],
            "feedback": "AI coaching text.",
        }
        out = analyze_progress(state)
        self.assertIn("AI coaching text.", out["feedback"])
        self.assertIn("First run!", out["feedback"])


class TestCompiledGraph(unittest.TestCase):
    def test_build_coach_graph_invoke(self):
        graph = build_coach_graph()
        final = graph.invoke(
            {
                "current_summary": {"carving_score": 70},
                "previous_summaries": [{"carving_score": 50}],
                "feedback": "Nice edge angles.",
            }
        )
        self.assertIn("Nice edge angles.", final["feedback"])
        self.assertIn("Score up by 20 points!", final["feedback"])


if __name__ == "__main__":
    unittest.main()
