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
    if "job_id" not in st.session_state:
        st.session_state["job_id"] = None
    if "_job_started_for_upload_key" not in st.session_state:
        st.session_state["_job_started_for_upload_key"] = None


def clear_job_if_uploader_file_changed(uploaded_file: Any | None) -> None:
    """Drop ``job_id`` when the user picks a new file before starting another run."""
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
    st.set_page_config(page_title=profile.page_title, page_icon="⛷️", layout="wide")


def render_standard_skills_sidebar(skills_markdown: str) -> None:
    st.header("⛷️ Agent Configuration")
    with st.expander("🛠️ Active Skills (skills.md)"):
        st.markdown(skills_markdown)


def render_agent_skills_text_area(initial_skills: str) -> str:
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


def render_completed_results(
    backend_url: str,
    job_id: str,
    summary: dict[str, Any],
    feedback: dict[str, Any],
    get_video_bytes: Callable[[str], bytes | None],
    profile: FrontendAppProfile,
) -> None:
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


def poll_job_until_terminal(
    backend_url: str,
    job_id: str,
    get_video_bytes: Callable[[str], bytes | None],
    profile: FrontendAppProfile,
) -> None:
    with st.status(
        f"Analyzing Biomechanics (Job: {job_id[:8]})...", expanded=True
    ) as status:
        completed = False
        while not completed:
            try:
                res = requests.get(f"{backend_url.rstrip('/')}/result/{job_id}")
                data = res.json()
                if data.get("status") == "completed":
                    status.update(
                        label="Analysis Complete!", state="complete", expanded=False
                    )
                    st.balloons()
                    feedback = parse_coaching_feedback(data.get("feedback"))
                    render_completed_results(
                        backend_url,
                        job_id,
                        data.get("summary", {}),
                        feedback,
                        get_video_bytes,
                        profile,
                    )
                    completed = True
                elif data.get("status") == "failed":
                    status.update(label="Analysis Failed", state="error")
                    st.error(
                        profile.failed_detail_fmt.format(
                            detail=data.get("error") or ""
                        )
                    )
                    completed = True
                else:
                    time.sleep(3)
            except Exception as e:
                st.error(profile.connection_lost_fmt.format(detail=e))
                break
