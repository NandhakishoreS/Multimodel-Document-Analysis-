"""
Phase 11 — Root Entry-Point.

Launch with:
    streamlit run app_phase11.py

This file mirrors the existing Phase 10 app.py pattern:
it ensures the project root is on sys.path and then exec()-delegates
to app/phase11_app.py so Streamlit sees the correct __file__ context.

The existing Phase 10 app (app.py) is completely unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_app_path = _PROJECT_ROOT / "app" / "phase11_app.py"
with open(_app_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_app_path), "exec"), {"__file__": str(_app_path), "__name__": "__main__"})
