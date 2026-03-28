"""
LangGraph workflow that appends a short progress / trend line to multimodal coach feedback.

Pure helpers (`trend_message`, `merge_feedback_with_trend`) are testable without compiling the graph.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

PROGRESS_SEPARATOR = "\n\n--- Progress ---\n"

NODE_ANALYZE_PROGRESS = "analyze_progress"


class CoachState(TypedDict, total=False):
    """State passed to `ski_app_graph.invoke` from `ski_backend.run_analysis_task`."""

    current_summary: dict[str, Any]
    previous_summaries: list[dict[str, Any]]
    feedback: str


def _carving_score(summary: dict[str, Any]) -> int:
    try:
        return int(summary.get("carving_score", 0))
    except (TypeError, ValueError):
        return 0


def trend_message(
    current_summary: dict[str, Any],
    previous_summaries: list[dict[str, Any]],
) -> str:
    """One-line trend from current vs last run carving scores."""
    if not previous_summaries:
        return "First run!"
    diff = _carving_score(current_summary) - _carving_score(previous_summaries[-1])
    if diff > 0:
        return f"Score up by {diff} points!"
    return "Focus on consistency."


def merge_feedback_with_trend(ai_feedback: str | None, trend: str) -> str:
    """
    Preserve Gemini (or other) text and append the progress line.
    Whitespace-only AI feedback is treated as absent.
    """
    text = (ai_feedback or "").strip()
    if text:
        return f"{text}{PROGRESS_SEPARATOR}{trend}"
    return trend


def analyze_progress(state: CoachState) -> dict[str, str]:
    trend = trend_message(
        state["current_summary"],
        state.get("previous_summaries") or [],
    )
    merged = merge_feedback_with_trend(state.get("feedback"), trend)
    return {"feedback": merged}


def build_coach_graph():
    """Compile the coach workflow (useful in tests or multiple instances)."""
    workflow = StateGraph(CoachState)
    workflow.add_node(NODE_ANALYZE_PROGRESS, analyze_progress)
    workflow.set_entry_point(NODE_ANALYZE_PROGRESS)
    workflow.add_edge(NODE_ANALYZE_PROGRESS, END)
    return workflow.compile()


ski_app_graph = build_coach_graph()
