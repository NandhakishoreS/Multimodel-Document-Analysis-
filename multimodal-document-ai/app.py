"""
Root entry point for the Streamlit application.

Usage:
    streamlit run app.py

This file re-exports the full Streamlit application defined in app/app.py.
Running `streamlit run app.py` from the project root is the canonical way to
launch the web interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all project imports resolve correctly
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import and execute app/app.py contents (Streamlit reads this module directly)
# Using exec with the file contents allows `streamlit run app.py` to work correctly.
_app_path = _PROJECT_ROOT / "app" / "app.py"
with open(_app_path, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_app_path), "exec"), {"__file__": str(_app_path), "__name__": "__main__"})
