"""SkiAI FastAPI entrypoint. Application logic lives in `ski_backend`."""
from __future__ import annotations

import ski_entry_bootstrap  # noqa: F401 — Windows asyncio policy before other imports

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from coach_graph import ski_app_graph
    from ski_logic import GeminiSkiCoach, SkiVideoProcessor, summarize_run_data
except ImportError:
    ski_app_graph = None
    SkiVideoProcessor = None
    GeminiSkiCoach = None
    summarize_run_data = None
    logger.error(
        "Could not import ski_logic or coach_graph. Ensure modules are on PYTHONPATH."
    )

from ski_backend import create_app

app = create_app(
    processor_cls=SkiVideoProcessor,
    coach_cls=GeminiSkiCoach,
    ski_app_graph=ski_app_graph,
    summarize_run_data=summarize_run_data,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
