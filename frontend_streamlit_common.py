"""
Shared Streamlit layout for `frontend.py` and `frontendagent.py`.

Business logic stays in `frontend_logic.py`; this module only wires Streamlit widgets.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import requests
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

from frontend_logic import (
    analysis_video_url,
    build_upload_multipart_fields,
    chart_rows_from_jobs,
    fetch_jobs,
    job_is_stale_for_current_upload,
    parse_coaching_feedback,
    post_job_chat,
    upload_identity,
)

# ``st.video`` defaults to width="stretch", which fills the whole row on wide layout.
REPLAY_VIDEO_WIDTH_PX = 720


@dataclass(frozen=True)
class FrontendAppProfile:
    page_title: str
    main_title: str
    main_markdown: str
    upload_success: str
    download_label: str
    download_file_stem: str
    show_carving_analysis_expander: bool
    failed_detail_fmt: str  # e.g. "Error: {detail}"
    connection_lost_fmt: str  # e.g. "Connection lost: {detail}"


STANDARD_PROFILE = FrontendAppProfile(
    page_title="SkiAI Pro Coach",
    main_title="⛷️ SkiAI: Advanced Biomechanical Coaching",
    main_markdown="Upload your skiing video to get **instant AI analysis**.",
    upload_success="Upload successful!",
    download_label="💾 Download Analysis",
    download_file_stem="ski_analysis",
    show_carving_analysis_expander=False,
    failed_detail_fmt="Error: {detail}",
    connection_lost_fmt="Connection lost: {detail}",
)

AGENT_PROFILE = FrontendAppProfile(
    page_title="SkiAI Pro Coach (Agent)",
    main_title="⛷️ SkiAI: Advanced Biomechanical Coaching",
    main_markdown=(
        "Upload your skiing video to get **instant AI analysis** on your carving form."
    ),
    upload_success="Upload successful! Skills injected into Agent.",
    download_label="💾 Download Analyzed Video",
    download_file_stem="skiai_analysis",
    show_carving_analysis_expander=True,
    failed_detail_fmt="Reason: {detail}",
    connection_lost_fmt="Lost connection to backend: {detail}",
)


def ensure_job_id_session_state() -> None:
    """Initializes job-related keys in the Streamlit session state."""
    if "job_id" not in st.session_state:
        st.session_state["job_id"] = None
    if "_job_started_for_upload_key" not in st.session_state:
        st.session_state["_job_started_for_upload_key"] = None


def clear_job_if_uploader_file_changed(uploaded_file: Any | None) -> None:
    """Clears the current job ID from session state if the uploaded file changes.

    This prevents displaying results for a previous file when a new file is
    selected in the uploader but the analysis has not been re-run.

    Args:
        uploaded_file (Any | None): The file object from `st.file_uploader`.
    """
    current = upload_identity(uploaded_file)
    if not job_is_stale_for_current_upload(
        job_id=st.session_state.get("job_id"),
        bound_identity=st.session_state.get("_job_started_for_upload_key"),
        current_identity=current,
        job_from_history=bool(st.session_state.get("_job_from_history")),
    ):
        return
    st.session_state["job_id"] = None


def configure_page(profile: FrontendAppProfile) -> None:
    """Configures the Streamlit page settings like title, icon, and layout.

    Args:
        profile (FrontendAppProfile): A dataclass containing page configuration strings.
    """
    st.set_page_config(page_title=profile.page_title, page_icon="⛷️", layout="wide")


def render_standard_skills_sidebar(skills_markdown: str) -> None:
    """Renders the agent's skills in a non-editable expander in the sidebar.

    Args:
        skills_markdown (str): The markdown content of the agent's skills.
    """
    st.header("⛷️ Agent Configuration")
    with st.expander("🛠️ Active Skills (skills.md)"):
        st.markdown(skills_markdown)


def render_agent_skills_text_area(initial_skills: str) -> str:
    """Renders an editable text area for agent skills and returns its current content.

    This allows users to modify the agent's system prompt on the fly.

    Args:
        initial_skills (str): The initial markdown content to display in the text area.

    Returns:
        str: The current, possibly edited, content of the text area.
    """
    st.header("⛷️ Agent Configuration")
    return st.text_area(
        "Edit Agent Knowledge (skills.md)",
        value=initial_skills,
        height=300,
        help=(
            "This markdown content is injected into the AI's system prompt to guide "
            "the analysis."
        ),
    )


def render_history_sidebar(backend_url: str) -> None:
    """Fetches and displays the job history in the Streamlit sidebar.

    Renders a button for each completed job to reload its results and a line chart
    showing the carving score progression over time.

    Args:
        backend_url (str): The base URL of the backend API service.
    """
    st.header("📜 Session History")
    history = fetch_jobs(backend_url)
    if not history:
        st.info("No past runs found.")
        return
    chart_data = chart_rows_from_jobs(history)
    for job in history:
        if job.get("status") == "completed":
            if st.button(
                f"Run: {job.get('filename', 'Video')}", key=job["job_id"]
            ):
                st.session_state["job_id"] = job["job_id"]
                st.session_state["_job_started_for_upload_key"] = None
                st.session_state["_job_from_history"] = True
    if chart_data:
        st.divider()
        st.subheader("Carving Progress")
        df = pd.DataFrame(chart_data)
        st.line_chart(df.set_index("Date")["Score"])


def run_upload_with_progress(
    backend_url: str,
    uploaded_file: Any,
    *,
    agent_skills: str | None,
    success_message: str,
    upload_identity_key: tuple[str, int] | None = None,
) -> None:
    """Uploads a file to the backend with a progress bar.

    On success, it updates the session state with the new job ID.

    Args:
        backend_url (str): The base URL of the backend API.
        uploaded_file (Any): The file object from `st.file_uploader`.
        agent_skills (str | None): Optional markdown skills to be sent with the upload.
        success_message (str): The message to display in `st.success` on completion.
        upload_identity_key (tuple[str, int] | None, optional): A unique identifier
            for the uploaded file to track state. Defaults to None.
    """
    progress_bar = st.progress(0, text="Preparing upload...")

    def progress_callback(monitor: Any) -> None:
        pct = monitor.bytes_read / monitor.len
        progress_bar.progress(pct, text=f"Uploading: {int(pct * 100)}%")

    fields = build_upload_multipart_fields(
        uploaded_file.name,
        uploaded_file,
        getattr(uploaded_file, "type", None),
        agent_skills=agent_skills,
    )
    encoder = MultipartEncoder(fields=fields)
    monitor = MultipartEncoderMonitor(encoder, progress_callback)
    try:
        response = requests.post(
            f"{backend_url.rstrip('/')}/upload",
            data=monitor,
            headers={"Content-Type": monitor.content_type},
        )
        if response.status_code == 200:
            st.session_state["job_id"] = response.json().get("job_id")
            if upload_identity_key is not None:
                st.session_state["_job_started_for_upload_key"] = upload_identity_key
            st.session_state["_job_from_history"] = False
            st.success(success_message)
        else:
            st.error(f"Upload failed: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")


def render_followup_chat(
    backend_url: str,
    job_id: str,
    server_chat_messages: list[dict[str, Any]],
    profile: FrontendAppProfile,
) -> None:
    """Renders a chat interface for follow-up questions about a completed analysis.

    Manages chat history within a Streamlit fragment to prevent full page reloads
    on message submission.

    Args:
        backend_url (str): The base URL of the backend API.
        job_id (str): The ID of the job being discussed.
        server_chat_messages (list[dict[str, Any]]): The initial chat history
            from the server.
        profile (FrontendAppProfile): The app profile for UI strings.
    """

    @st.fragment
    def _chat_fragment() -> None:
        anchor = "_coach_chat_active_job_id"
        buf_key = f"_coach_chat_history_{job_id}"
        if st.session_state.get(anchor) != job_id:
            st.session_state[anchor] = job_id
            st.session_state[buf_key] = [
                dict(m)
                for m in server_chat_messages
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]

        messages: list[dict[str, Any]] = st.session_state[buf_key]

        st.divider()
        st.subheader("💬 Ask the coach")
        st.caption(
            "Follow-up questions use your run metrics and the analysis above "
            "(the model does not re-watch the video in this chat)."
        )
        for m in messages:
            role = m.get("role", "user")
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(m.get("content", ""))

        if prompt := st.chat_input("Ask a follow-up about this run…"):
            with st.spinner("Coach is thinking…"):
                try:
                    out = post_job_chat(backend_url, job_id, prompt)
                    hist = out.get("chat_messages")
                    if isinstance(hist, list):
                        st.session_state[buf_key] = [
                            dict(x)
                            for x in hist
                            if isinstance(x, dict)
                            and x.get("role") in ("user", "assistant")
                        ]
                    try:
                        st.rerun(scope="fragment")
                    except TypeError:
                        st.rerun()
                except Exception as e:
                    st.error(profile.failed_detail_fmt.format(detail=e))

    _chat_fragment()


def render_completed_results(
    backend_url: str,
    job_id: str,
    summary: dict[str, Any],
    feedback: dict[str, Any],
    get_video_bytes: Callable[[str], bytes | None],
    profile: FrontendAppProfile,
    *,
    server_chat_messages: list[dict[str, Any]] | None = None,
) -> None:
    """Displays the complete results for a successful job.

    This includes performance metrics, AI coaching feedback, the synthesized
    replay video, a download button, and the follow-up chat interface.

    Args:
        backend_url (str): The base URL of the backend API.
        job_id (str): The ID of the completed job.
        summary (dict[str, Any]): A dictionary of performance metrics.
        feedback (dict[str, Any]): A dictionary of parsed coaching feedback.
        get_video_bytes (Callable[[str], bytes | None]): A function that takes a
            video URL and returns its byte content.
        profile (FrontendAppProfile): The app profile for UI strings.
        server_chat_messages (list[dict[str, Any]] | None, optional): The initial
            chat history from the server. Defaults to None.
    """
    col_stats, col_coach = st.columns([1, 2])
    with col_stats:
        st.subheader("📊 Performance Metrics")
        st.metric("Carving Score", f"{summary.get('carving_score')}/100")
        st.markdown(
            f"- **Max Edge Angle:** `{summary.get('max_edge_inclination_deg')}°`\n"
            f"- **Backseat Time:** `{summary.get('backseat_percentage')}%`"
        )
        st.divider()
        st.subheader("🔎 Observations")
        st.write(feedback.get("visual_observations", "N/A"))
        st.subheader("📈 Progression")
        st.write(feedback.get("progression_note", "Trend data updated."))
    with col_coach:
        st.subheader(
            f"🤖 AI Coach: {feedback.get('primary_fault', 'Technical Feedback')}"
        )
        with st.expander("🔬 Biomechanical Explanation", expanded=True):
            st.write(feedback.get("biomechanical_explanation", ""))
        if profile.show_carving_analysis_expander:
            with st.expander("📉 Carving Analysis", expanded=False):
                st.write(feedback.get("carving_score_analysis", ""))
        st.success(
            f"🛠️ Recommended Drill: **{feedback.get('recommended_drill_name', 'Drill')}**"
        )
        st.markdown("### 📋 Execution Steps")
        steps = feedback.get("drill_steps", [])
        if isinstance(steps, list):
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")
        else:
            st.write(steps)

    st.divider()
    st.subheader("📽️ Synthesized Replay")
    video_url = analysis_video_url(backend_url, job_id)
    # Bytes + BytesIO: same-origin for the player; H.264 from backend ffmpeg is required
    # for Chromium/Edge (OpenCV mp4v often will not decode in <video>).
    replay_key = f"_analysis_video_bytes_{job_id}"
    v_bytes = st.session_state.get(replay_key)
    if v_bytes is None:
        fetched = get_video_bytes(video_url)
        if fetched:
            st.session_state[replay_key] = fetched
            v_bytes = fetched
    if v_bytes:
        st.video(
            io.BytesIO(v_bytes),
            format="video/mp4",
            width=REPLAY_VIDEO_WIDTH_PX,
        )
    else:
        st.video(video_url, width=REPLAY_VIDEO_WIDTH_PX)
        st.caption(
            "Could not load video into the app (fetch failed). "
            "Check that the API is reachable, or open the file URL in a new tab."
        )
    if v_bytes:
        st.download_button(
            label=profile.download_label,
            data=v_bytes,
            file_name=f"{profile.download_file_stem}_{job_id[:8]}.mp4",
            mime="video/mp4",
        )

    render_followup_chat(
        backend_url,
        job_id,
        server_chat_messages if server_chat_messages is not None else [],
        profile,
    )


def _render_completed_job_view(
    backend_url: str,
    job_id: str,
    data: dict[str, Any],
    get_video_bytes: Callable[[str], bytes | None],
    profile: FrontendAppProfile,
) -> None:
    """Helper to render the view for a completed job.

    Shows celebratory balloons once per job and then delegates to
    `render_completed_results` to display the detailed analysis.

    Args:
        backend_url (str): The base URL of the backend API.
        job_id (str): The ID of the completed job.
        data (dict[str, Any]): The full job result data from the API.
        get_video_bytes (Callable[[str], bytes | None]): A function to fetch
            the analysis video bytes.
        profile (FrontendAppProfile): The app profile for UI strings.
    """
    balloons_key = f"_analysis_balloons_shown_{job_id}"
    if not st.session_state.get(balloons_key):
        st.balloons()
        st.session_state[balloons_key] = True
    feedback = parse_coaching_feedback(data.get("feedback"))
    raw_chat = data.get("chat_messages")
    chat_hist = raw_chat if isinstance(raw_chat, list) else []
    render_completed_results(
        backend_url,
        job_id,
        data.get("summary", {}),
        feedback,
        get_video_bytes,
        profile,
        server_chat_messages=chat_hist,
    )


def poll_job_until_terminal(
    backend_url: str,
    job_id: str,
    get_video_bytes: Callable[[str], bytes | None],
    profile: FrontendAppProfile,
) -> None:
    """Polls the backend API for a job's status until it is completed or failed.

    Displays a `st.status` indicator during polling and renders the final result
    or an error message upon completion.

    Args:
        backend_url (str): The base URL of the backend API.
        job_id (str): The ID of the job to poll.
        get_video_bytes (Callable[[str], bytes | None]): A function to fetch
            the analysis video bytes upon job completion.
        profile (FrontendAppProfile): The app profile for UI strings.
    """
    base = backend_url.rstrip("/")
    url = f"{base}/result/{job_id}"

    try:
        res = requests.get(url)
        data = res.json()
    except Exception as e:
        st.error(profile.connection_lost_fmt.format(detail=e))
        return

    st_val = data.get("status")
    if st_val == "completed":
        _render_completed_job_view(backend_url, job_id, data, get_video_bytes, profile)
        return
    if st_val == "failed":
        st.error(profile.failed_detail_fmt.format(detail=data.get("error") or ""))
        return

    with st.status(
        f"Analyzing Biomechanics (Job: {job_id[:8]})...", expanded=True
    ) as status:
        while True:
            try:
                st_val = data.get("status")
                if st_val == "completed":
                    status.update(
                        label="Analysis Complete!",
                        state="complete",
                        expanded=False,
                    )
                    break
                if st_val == "failed":
                    status.update(label="Analysis Failed", state="error")
                    st.error(
                        profile.failed_detail_fmt.format(
                            detail=data.get("error") or ""
                        )
                    )
                    return
                time.sleep(3)
                res = requests.get(url)
                data = res.json()
            except Exception as e:
                st.error(profile.connection_lost_fmt.format(detail=e))
                return

    if data.get("status") == "completed":
        _render_completed_job_view(
            backend_url, job_id, data, get_video_bytes, profile
        )