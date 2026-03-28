# SkiAI Coach

Ski run analysis pipeline: **pose estimation (YOLO)** on video, **kinematic summaries**, and **Gemini** multimodal coaching. Includes a **FastAPI** backend and **Streamlit** frontends (standard and agent modes).

## Features

- Upload a ski video; backend runs pose tracking, edge/backseat heuristics, and optional **LangGraph**-merged coaching (standard mode).
- **Agent mode** accepts editable **skills** (markdown) and returns **structured JSON** coaching from Gemini.
- Analyzed video is re-encoded to **H.264** when possible (system `ffmpeg` or bundled **`imageio-ffmpeg`**) for reliable in-browser replay in Streamlit.

## Requirements

- **Python** 3.11+ recommended (3.14 may work; see upstream wheels for torch/ultralytics).
- **`GEMINI_API_KEY`** — [Google AI Studio](https://aistudio.google.com/apikey) or Google Cloud GenAI key; never commit the value (use `.env` or your shell; `.env` is gitignored).
- **GPU** optional but faster for YOLO; CPU runs with smaller models.

## Setup

```bash
cd skiaicoach
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key (example for PowerShell):

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

Ultralytics will download **YOLO pose** weights (e.g. `yolov8n-pose.pt`) on first use; `*.pt` / `*.pth` are listed in `.gitignore`.

## Run locally

Default API port is **8001** (matches Streamlit `BACKEND_URL`).

**Standard backend** (LangGraph trend merge + text feedback):

```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8001
```

**Agent backend** (structured JSON + `agent_skills` form field):

```bash
python mainagent.py
```

**Streamlit UI** (in a second terminal):

```bash
streamlit run frontend.py
# Agent UI + skills editor:
streamlit run frontendagent.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Ensure the backend is up before uploading.

> **Note:** `main.py` and `mainagent.py` both default to port **8001**. Run only one API at a time, or change the port in code / uvicorn and update `BACKEND_URL` in the frontend.

## Tests and coverage

```bash
pytest -q
pytest -q --cov=. --cov-config=.coveragerc --cov-report=term-missing
```

## Project layout (main modules)

| Path | Role |
|------|------|
| `main.py` / `mainagent.py` | FastAPI entrypoints |
| `ski_backend.py` | App factory, upload, jobs, static `/uploads` |
| `ski_logic.py` | Video pipeline, YOLO, overlay render, ffmpeg re-encode |
| `ski_analysis.py` | Pure kinematics + smoothing + run summaries |
| `ski_gemini.py` | Gemini client (text + agent JSON) |
| `frontend.py` / `frontendagent.py` | Streamlit apps |
| `frontend_streamlit_common.py` | Shared Streamlit layout |
| `frontend_logic.py` | URL helpers, feedback parsing, upload helpers |
| `coach_graph.py` | LangGraph coaching flow (standard backend) |
| `schema.py` | Pydantic schema for agent JSON responses |
| `ski_entry_bootstrap.py` | Windows asyncio policy before FastAPI imports |
| `tests/` | Pytest suite |
| `android/` | Kotlin + Jetpack Compose app (see `android/README.md`) |

## Android app

A **Kotlin + Jetpack Compose** client lives under **`android/`**. It uploads a video to `POST /upload`, polls `GET /result/{job_id}`, and plays the analyzed MP4 from `/uploads/…` with **ExoPlayer**.

- Open the **`android`** folder in **Android Studio** (not the repo root).
- Emulator default base URL: **`http://10.0.2.2:8001`** (see `BuildConfig.DEFAULT_BASE_URL`).
- Physical device: set base URL to `http://<your-pc-lan-ip>:8001` in the app.

Details: **`android/README.md`**.

## Docker

`docker-compose.yml` defines a GPU-oriented backend and a Streamlit service. You need a **`Dockerfile`** in the repo root that matches the compose `build` context; compose currently references `app_frontend.py` — align that command with `frontend.py` / `frontendagent.py` if your image uses this repository layout as-is.

## License

Add a `LICENSE` file if you distribute the project.
