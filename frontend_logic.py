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
    """Remove optional