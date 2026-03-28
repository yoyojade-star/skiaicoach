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

logger = logging.getLogger(__name__)

VALID_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


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
        }
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
    upload_path = Path(upload_dir or os.path.abspath("uploads")).resolve()
    data_path = Path(data_dir or os.path.abspath("data_store")).resolve()
    upload_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    upload_dir_str = str(upload_path)
    data_dir_str = str(data_path)

    results_db: dict[str, Any] = {}

    def save_result(job_id: str, data: dict[str, Any]) -> None:
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

    return app
