"""
Pure helpers for the Streamlit frontend (URLs, skills file, feedback parsing, history chart).

Keeps `frontend.py` focused on layout and I/O; unit-test without Streamlit.
"""
from __future__ import annotations

import json
from typing import Any

# ``(filename, size_bytes)`` for the file last submitted via the uploader for this job.
UploadIdentity = tuple[str, int]

DEFAULT_SKILLS_FALLBACK = (
    "Standard Biomechanical Coaching Mode (skills.md not found)"
)


def load_skills_md(path: str = "skills.md", *, encoding: str = "utf-8") -> str:
    """Read skills markdown; on missing file return `DEFAULT_SKILLS_FALLBACK`."""
    try:
        with open(path, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        return DEFAULT_SKILLS_FALLBACK


def upload_identity(file_obj: Any | None) -> UploadIdentity | None:
    """Stable key for ``st.file_uploader`` output; ``None`` if no file."""
    if file_obj is None:
        return None
    name = str(getattr(file_obj, "name", "") or "")
    size = int(getattr(file_obj, "size", 0) or 0)
    return (name, size)


def job_is_stale_for_current_upload(
    *,
    job_id: str | None,
    bound_identity: UploadIdentity | None,
    current_identity: UploadIdentity | None,
    job_from_history: bool = False,
) -> bool:
    """
    True when ``job_id`` should be dropped because the uploader no longer matches
    the run being shown (new file selected, or legacy session without binding).
    """
    if not job_id or current_identity is None:
        return False
    if bound_identity is not None:
        return current_identity != bound_identity
    # No upload binding: history restores ``job_id`` with binding None — keep it.
    return not job_from_history


def strip_json_fenced_block(text: str) -> str:
    """Remove optional ```json ... ``` or ``` ... ``` wrappers from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _plain_text_feedback_dict(text: str) -> dict[str, Any]:
    return {
        "primary_fault": "Technical Analysis",
        "biomechanical_explanation": text,
        "carving_score_analysis": "",
        "recommended_drill_name": "Basic Fundamentals",
        "drill_steps": ["Focus on balanced stance", "Maintain forward pressure"],
        "visual_observations": "N/A",
        "progression_note": "Trend data updated.",
    }


def parse_coaching_feedback(feedback_raw: Any) -> dict[str, Any]:
    """
    Normalize `/result` `feedback` (structured dict, JSON string, fenced JSON, or plain text)
    into a dict suitable for the UI.
    """
    if feedback_raw is None:
        return _plain_text_feedback_dict("")

    if isinstance(feedback_raw, dict):
        return dict(feedback_raw)

    if not isinstance(feedback_raw, str):
        return _plain_text_feedback_dict(str(feedback_raw))

    cleaned = strip_json_fenced_block(feedback_raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return dict(parsed)
    except json.JSONDecodeError:
        pass

    return _plain_text_feedback_dict(feedback_raw)


def chart_rows_from_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rows for the sidebar line chart (completed jobs only)."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status") != "completed":
            continue
        summary = job.get("summary") or {}
        jid = str(job.get("job_id") or "")
        rows.append(
            {
                "Date": jid[:8] if len(jid) >= 8 else jid,
                "Score": summary.get("carving_score", 0),
                "Edge": summary.get("max_edge_inclination_deg", 0),
            }
        )
    return rows


def build_upload_multipart_fields(
    file_name: str,
    file_obj: Any,
    content_type: str | None,
    *,
    agent_skills: str | None = None,
) -> dict[str, Any]:
    """
    Fields dict for ``MultipartEncoder`` / video upload POST.

    When ``agent_skills`` is set, includes form field ``agent_skills`` for
    ``mainagent`` / ``use_agent_feedback`` backends. Omit by passing ``None``.
    """
    mime = (content_type or "").strip() or "video/mp4"
    fields: dict[str, Any] = {"file": (file_name, file_obj, mime)}
    if agent_skills is not None:
        fields["agent_skills"] = str(agent_skills)
    return fields


def analysis_video_url(base_url: str, job_id: str) -> str:
    """Public URL for the analyzed MP4 served under `/uploads`."""
    return f"{base_url.rstrip('/')}/uploads/{job_id}_analyzed.mp4"


def fetch_jobs(base_url: str, get: Any = None, timeout: float = 30.0) -> list[dict[str, Any]]:
    """
    GET ``{base_url}/jobs``. `get` defaults to ``requests.get`` (inject mock in tests).
    """
    import requests

    getter = get or requests.get
    try:
        response = getter(f"{base_url.rstrip('/')}/jobs", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def fetch_video_bytes(video_url: str, get: Any = None, timeout: float = 120.0) -> bytes | None:
    """
    Download full video body for in-app replay / download. `get` defaults to ``requests.get``.

    Accepts ``200`` or ``206`` (some stacks use partial responses for media).
    """
    import requests

    getter = get or requests.get
    try:
        response = getter(video_url, stream=True, timeout=timeout)
        if response.status_code in (200, 206):
            return response.content
    except Exception:
        return None
    return None
