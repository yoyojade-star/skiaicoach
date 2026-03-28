"""Streamlit UI for SkiAI (standard backend). Shared UI: `frontend_streamlit_common`."""
from __future__ import annotations

import streamlit as st

from frontend_logic import fetch_video_bytes, load_skills_md, upload_identity
from frontend_streamlit_common import (
    STANDARD_PROFILE,
    clear_job_if_uploader_file_changed,
    configure_page,
    ensure_job_id_session_state,
    poll_job_until_terminal,
    render_history_sidebar,
    render_standard_skills_sidebar,
    run_upload_with_progress,
)

BACKEND_URL = "http://localhost:8001"

configure_page(STANDARD_PROFILE)
ensure_job_id_session_state()


@st.cache_data
def load_agent_skills() -> str:
    return load_skills_md("skills.md")


def get_video_bytes(url: str) -> bytes | None:
    # Do not cache None: a transient failure would lock replay for the whole TTL.
    return fetch_video_bytes(url)


with st.sidebar:
    render_standard_skills_sidebar(load_agent_skills())
    st.divider()
    render_history_sidebar(BACKEND_URL)

st.title(STANDARD_PROFILE.main_title)
st.markdown(STANDARD_PROFILE.main_markdown)

uploaded_file = st.file_uploader("Choose a ski video...", type=["mp4", "mov", "avi"])
clear_job_if_uploader_file_changed(uploaded_file)

if uploaded_file is not None:
    if st.button("🚀 Start AI Analysis"):
        run_upload_with_progress(
            BACKEND_URL,
            uploaded_file,
            agent_skills=None,
            success_message=STANDARD_PROFILE.upload_success,
            upload_identity_key=upload_identity(uploaded_file),
        )

if st.session_state["job_id"]:
    poll_job_until_terminal(
        BACKEND_URL,
        st.session_state["job_id"],
        get_video_bytes,
        STANDARD_PROFILE,
    )
