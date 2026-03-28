"""
Import this module **first** in process entrypoints (before FastAPI / Uvicorn).

On Windows, the default Proactor loop + ranged static files (video 206) often logs
``ConnectionResetError`` in ``_ProactorBasePipeTransport``. Selector policy avoids it.
"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
