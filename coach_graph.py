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
    """
    Safely extracts and converts the carving score from a summary dictionary.

    Args:
        summary (dict[str, Any]): A dictionary potentially containing a 'carving_score'.

    Returns:
        int: The integer value of 'carving_score', or 0 if it's missing, not a number, or None.
    """
    try:
        return int(summary.get("carving_score", 0))
    except (TypeError, ValueError):
        return 0


def trend_message(
    current_summary: dict[str, Any],
    previous_summaries: list[dict[str, Any]],
) -> str:
    """
    Generates a one-line trend message by comparing current vs. last run carving scores.

    Args:
        current_summary (dict[str, Any]): The summary of the most recent run.
        previous_summaries (list[dict[str, Any]]): A list of historical run summaries.

    Returns:
        str: A message indicating the score trend or "First run!" if no history exists.
    """
    if not previous_summaries:
        return "First run!"
    diff = _carving_score(current_summary) - _carving_score(previous_summaries[-1])
    if diff > 0:
        return f"Score up by {diff} points!"
    return "Focus on consistency."


def merge_feedback_with_trend(ai_feedback: str | None, trend: str) -> str:
    """
    Appends a trend line to existing AI feedback, separated by a standard separator.

    Whitespace-only AI feedback is treated as absent, in which case only the trend
    message is returned.

    Args:
        ai_feedback (str | None): The original feedback text from an AI model.
        trend (str): The progress trend message to append.

    Returns:
        str: The combined feedback and trend message.
    """
    text = (ai_feedback or "").strip()
    if text:
        return f"{text}{PROGRESS_SEPARATOR}{trend}"
    return trend


def analyze_progress(state: CoachState) -> dict[str, str]:
    """
    LangGraph node that calculates the trend and merges it with existing feedback.

    Args:
        state (CoachState): The current state of the graph, containing summaries and feedback.

    Returns:
        dict[str, str]: A dictionary with the updated 'feedback' key to merge into the state.
    """
    trend = trend_message(
        state["current_summary"],
        state.get("previous_summaries") or [],
    )
    merged = merge_feedback_with_trend(state.get("feedback"), trend)
    return {"feedback": merged}


def build_coach_graph():
    """
    Compiles the coach workflow into a runnable LangGraph application.

    The graph has a single node that analyzes progress and appends a trend line
    to the feedback.

    Returns:
        A compiled LangGraph application.
    """
    workflow = StateGraph(CoachState)
    workflow.add_node(NODE_ANALYZE_PROGRESS, analyze_progress)
    workflow.set_entry_point(NODE_ANALYZE_PROGRESS)
    workflow.add_edge(NODE_ANALYZE_PROGRESS, END)
    return workflow.compile()


ski_app_graph = build_coach_graph()