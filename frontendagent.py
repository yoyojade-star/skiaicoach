"""
Streamlit UI for SkiAI agent backend (`mainagent.py`): uploads include `agent_skills`.

Shared UI lives in `frontend_streamlit_common`; HTTP/skills helpers in `frontend_logic`.
"""
from __future__ import annotations

import streamlit as st

from frontend_logic import fetch_video_bytes, load_skills_md, upload_identity
from frontend_streamlit_common import (
    AGENT_PROFILE,
    clear_job_if_uploader_file_changed,
    configure_page,
    ensure_job_id_session_state,
    poll_job_until_terminal,
    render_agent_skills_text_area,
    render_history_sidebar,
    run_upload_with_progress,
)

BACKEND_URL = "http://localhost:8001"

configure_page(AGENT_PROFILE)
ensure_job_id_session_state()


@st.cache_data
def load_agent_skills() -> str:
    return load_skills_md("skills.md")


def get_video_bytes(url: str) -> bytes | None:
    return fetch_video_bytes(url)


with st.sidebar:
    skills_content = render_agent_skills_text_area(load_agent_skills())
    st.divider()
    render_history_sidebar(BACKEND_URL)

st.title(AGENT_PROFILE.main_title)
st.markdown(AGENT_PROFILE.main_markdown)

uploaded_file = st.file_uploader("Choose a ski video...", type=["mp4", "mov", "avi"])
clear_job_if_uploader_file_changed(uploaded_file)

if uploaded_file is not None:
    if st.button("🚀 Start AI Analysis"):
        run_upload_with_progress(
            BACKEND_URL,
            uploaded_file,
            agent_skills=skills_content,
            success_message=AGENT_PROFILE.upload_success,
            upload_identity_key=upload_identity(uploaded_file),
        )

if st.session_state["job_id"]:
    poll_job_until_terminal(
        BACKEND_URL,
        st.session_state["job_id"],
        get_video_bytes,
        AGENT_PROFILE,
    )
