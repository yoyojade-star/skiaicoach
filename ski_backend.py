"""
FastAPI application factory, job pipeline, and upload helpers (testable, injectable paths/deps).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VALID_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


class JobChatRequest(BaseModel):
    """Pydantic model for a chat request associated with a job."""
    message: str = Field(..., min_length=1, max_length=12000)


def configure_server_event_loop() -> None:
    """
    Windows + default ProactorEventLoop: clients closing mid-request on ranged static
    files (video ``206 Partial Content``) can trigger ``asyncio`` ERROR logs from
    ``_ProactorBasePipeTransport._call_connection_lost`` / ``socket.shutdown``.

    Entry modules import ``ski_entry_bootstrap`` first so policy is set before
    FastAPI/Starlette touch asyncio. This function remains for tests or custom
    launchers. On non-Windows this is a no-op. Safe to call more than once.
    """
    if sys.platform != "win32":
        return
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.debug("Using WindowsSelectorEventLoopPolicy for static file streaming.")


def normalize_video_extension(filename: str | None) -> str:
    """Return a safe file extension (with leading dot) for the uploaded video."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in VALID_VIDEO_EXTENSIONS else ".mp4"


def run_analysis_task(
    job_id: str,
    input_path: str,
    output_path: str,
    *,
    save_result: Callable[[str, dict[str, Any]], None],
    processor_cls: type | None,
    coach_cls: type | None,
    ski_app_graph: Any | None,
    summarize_run_data: Callable[[list[Any]], dict[str, Any] | None] | None,
    agent_skills: str | None = None,
    merge_coach_graph: bool = True,
) -> None:
    """
    Background pipeline: YOLO → summary → Gemini → (optional LangGraph) → persist.

    Standard API (`merge_coach_graph=True`): text feedback + `ski_app_graph` trend line.
    Agent API (`merge_coach_graph=False`): structured dict from `generate_feedback_agent`
    with optional `agent_skills` (no LangGraph merge).
    """
    try:
        logger.info("Starting analysis for Job: %s", job_id)
        logger.info("Targeting video file: %s", input_path)

        if processor_cls is None or coach_cls is None or summarize_run_data is None:
            raise RuntimeError(
                "SkiAI logic components (processor, coach, or summarizer) are missing."
            )
        if merge_coach_graph and ski_app_graph is None:
            raise RuntimeError(
                "ski_app_graph is required for the standard coaching pipeline."
            )

        if input_path.lower().endswith(".json"):
            raise RuntimeError(
                f"System Error: Attempted to analyze a metadata file ({input_path}) instead of a video."
            )

        if not os.path.exists(input_path):
            raise RuntimeError(f"Input video file not found at {input_path}")

        if os.path.getsize(input_path) == 0:
            raise RuntimeError(f"Input video file is empty at {input_path}")

        processor = processor_cls()
        run_data = processor.process_video(input_path, output_path)
        summary = summarize_run_data(run_data)

        coach = coach_cls()
        if merge_coach_graph:
            ai_feedback = coach.generate_feedback(input_path, summary)
            initial_state = {
                "current_summary": summary,
                "previous_summaries": [],
                "feedback": ai_feedback,
            }
            final_state = ski_app_graph.invoke(initial_state)
            feedback_out: Any = final_state["feedback"]
        else:
            feedback_out = coach.generate_feedback_agent(
                video_path=input_path,
                summary=summary,
                skills=agent_skills,
            )

        result_data = {
            "status": "completed",
            "job_id": job_id,
            "summary": summary,
            "feedback": feedback_out,
            "video_url": f"/uploads/{os.path.basename(output_path)}",
            "chat_messages": [],
        }
        if agent_skills is not None:
            result_data["agent_skills"] = agent_skills
        save_result(job_id, result_data)
        logger.info("Analysis completed successfully for %s", job_id)

    except Exception as e:
        logger.error("Analysis failed for %s: %s", job_id, str(e))
        save_result(job_id, {"status": "failed", "error": str(e), "job_id": job_id})


def create_app(
    upload_dir: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    *,
    title: str = "SkiAI Backend",
    use_agent_feedback: bool = False,
    processor_cls: type | None = None,
    coach_cls: type | None = None,
    ski_app_graph: Any | None = None,
    summarize_run_data: Callable[[list[Any]], dict[str, Any] | None] | None = None,
) -> FastAPI:
    """
    Creates and configures a FastAPI application instance.

    This factory function sets up storage directories, CORS middleware, static
    file serving, and all API endpoints for video upload, result retrieval,
    and chat interaction. It injects dependencies for the analysis pipeline.

    Args:
        upload_dir (str | os.PathLike[str] | None, optional): Directory to store
            uploaded and processed videos. Defaults to "uploads".
        data_dir (str | os.PathLike[str] | None, optional): Directory to store
            job metadata and results as JSON files. Defaults to "data_store".
        title (str, optional): The title of the FastAPI application.
            Defaults to "SkiAI Backend".
        use_agent_feedback (bool, optional): If True, uses the agent-based
            feedback generation. Defaults to False.
        processor_cls (type | None, optional): The class responsible for video
            processing (e.g., pose estimation).
        coach_cls (type | None, optional): The class responsible for generating
            AI feedback and handling chat.
        ski_app_graph (Any | None, optional): The LangGraph instance for stateful
            coaching analysis.
        summarize_run_data (Callable | None, optional): A function to summarize
            the raw data from the processor.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    upload_path = Path(upload_dir or os.path.abspath("uploads")).resolve()
    data_path = Path(data_dir or os.path.abspath("data_store")).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    upload_dir_str = str(upload_path)
    data_dir_str = str(data_path)

    results_db: dict[str, Any] = {}

    def save_result(job_id: str, data: dict[str, Any]) -> None:
        """
        Persists job data to a JSON file and caches it in memory.

        Args:
            job_id (str): The unique identifier for the job.
            data (dict[str, Any]): The job data to save.
        """
        file_path = os.path.join(data_dir_str, f"{job_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        results_db[job_id] = data

    app = FastAPI(title=title)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/uploads", StaticFiles(directory=upload_dir_str), name="uploads")

    @app.post("/upload")
    async def upload_video(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        agent_skills: str | None = Form(None),
    ):
        """
        Handles video file uploads and starts the analysis pipeline.

        This endpoint saves the uploaded video, creates a new job ID, and
        schedules the `run_analysis_task` to execute in the background.

        Args:
            background_tasks (BackgroundTasks): FastAPI dependency to manage
                background tasks.
            file (UploadFile): The video file being uploaded.
            agent_skills (str, optional): A string of specific skills for the
                AI coach to focus on.

        Returns:
            dict: A dictionary containing the new job's ID and its initial
                processing status.

        Raises:
            HTTPException: A 500 error if the file cannot be saved or the
                background task cannot be started.
        """
        skills_for_task = agent_skills if use_agent_feedback else None
        try:
            job_id = str(uuid.uuid4())
            file_ext = normalize_video_extension(file.filename)
            input_path = os.path.join(upload_dir_str, f"{job_id}{file_ext}")
            output_path = os.path.join(upload_dir_str, f"{job_id}_analyzed.mp4")

            logger.info("Receiving file: %s -> %s", file.filename, input_path)

            with open(input_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):
                    buffer.write(content)

            initial_status: dict[str, Any] = {
                "status": "processing",
                "job_id": job_id,
                "filename": file.filename,
                "input_file": input_path,
            }
            if use_agent_feedback:
                initial_status["skills_preview"] = (
                    (agent_skills[:100] + "...")
                    if agent_skills and len(agent_skills) > 100
                    else (agent_skills or "Default Coaching")
                )

            save_result(job_id, initial_status)

            background_tasks.add_task(
                run_analysis_task,
                job_id,
                input_path,
                output_path,
                save_result=save_result,
                processor_cls=processor_cls,
                coach_cls=coach_cls,
                ski_app_graph=ski_app_graph,
                summarize_run_data=summarize_run_data,
                agent_skills=skills_for_task,
                merge_coach_graph=not use_agent_feedback,
            )

            return {"job_id": job_id, "status": "processing"}

        except Exception as e:
            logger.error("Upload Route Error: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/result/{job_id}")
    async def get_result(job_id: str):
        """
        Retrieves the status and results for a specific analysis job.

        It first checks an in-memory cache, then falls back to reading the
        result from the corresponding JSON file on disk.

        Args:
            job_id (str): The unique identifier of the job to retrieve.

        Returns:
            dict: The job's data, or a dictionary with status "not_found"
                if the job does not exist.
        """
        result = results_db.get(job_id)
        if not result:
            file_path = os.path.join(data_dir_str, f"{job_id}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        results_db[job_id] = result
                except (json.JSONDecodeError, OSError):
                    pass

        if not result:
            return {"status": "not_found"}
        return result

    @app.get("/jobs")
    async def list_jobs():
        """
        Lists all available job results from the data store directory.

        Returns:
            list[Any]: A list of job data dictionaries loaded from JSON files.
                Returns an empty list if the directory doesn't exist or is empty.
        """
        jobs: list[Any] = []
        if not os.path.exists(data_dir_str):
            return jobs
        for name in os.listdir(data_dir_str):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(data_dir_str, name), "r", encoding="utf-8") as fp:
                    jobs.append(json.load(fp))
            except (json.JSONDecodeError, OSError):
                continue
        return jobs

    @app.post("/jobs/{job_id}/chat")
    def post_job_chat(job_id: str, body: JobChatRequest) -> dict[str, Any]:
        """
        Handles a follow-up chat message for a completed analysis job.

        It retrieves the job's context, invokes the AI coach's chat function,
        and persists the updated conversation history.

        Args:
            job_id (str): The identifier of the job to chat with.
            body (JobChatRequest): The request body containing the user's message.

        Returns:
            dict[str, Any]: A dictionary containing the AI's reply and the
                complete, updated chat history.

        Raises:
            HTTPException: Various errors for conditions like job not found (404),
                coach not configured (503), job not completed (400), or internal
                chat errors (500).
        """
        if coach_cls is None:
            raise HTTPException(status_code=503, detail="Coach is not configured.")
        user_text = body.message.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="Message is empty.")

        result: dict[str, Any] | None = results_db.get(job_id)
        if not result:
            file_path = os.path.join(data_dir_str, f"{job_id}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        results_db[job_id] = result
                except (json.JSONDecodeError, OSError):
                    result = None

        if not result:
            raise HTTPException(status_code=404, detail="Job not found.")
        if result.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail="Analysis is not complete yet.",
            )

        summary = result.get("summary")
        if not isinstance(summary, dict):
            raise HTTPException(status_code=400, detail="Job has no run summary.")

        raw_history = result.get("chat_messages")
        chat_messages: list[dict[str, Any]] = (
            list(raw_history) if isinstance(raw_history, list) else []
        )

        coach = coach_cls()
        chat_fn = getattr(coach, "chat_followup", None)
        if chat_fn is None:
            raise HTTPException(
                status_code=501,
                detail="The configured coach does not support follow-up chat.",
            )

        try:
            reply = chat_fn(
                run_summary=summary,
                initial_feedback=result.get("feedback"),
                chat_messages=chat_messages,
                user_message=user_text,
                skills=result.get("agent_skills"),
            )
        except Exception as e:
            logger.exception("Chat failed for job %s", job_id)
            raise HTTPException(status_code=500, detail=str(e)) from e

        reply_text = reply if isinstance(reply, str) else str(reply)

        updated = dict(result)
        history = list(chat_messages)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply_text})
        updated["chat_messages"] = history
        save_result(job_id, updated)

        return {"reply": reply_text, "chat_messages": history}

    return app