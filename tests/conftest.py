"""Put the project root on sys.path so `ski_analysis`, `ski_logic`, etc. resolve."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_root_str = str(_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
