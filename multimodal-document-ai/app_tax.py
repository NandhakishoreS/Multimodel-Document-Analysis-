"""
Root entry point for the AI-Powered Indian Income Tax Return (ITR-1) Preparation System.

Usage:
    streamlit run app_tax.py

Provides the full 8-screen tax preparation workflow: onboarding, document ingestion,
cross-document reconciliation, deterministic tax computation, ITR-1 eligibility,
and official-format JSON export with offline utility filing instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from tax_module.ui.tax_dashboard import render_tax_system

# Page configuration
st.set_page_config(
    page_title="Indian Income Tax Return (ITR-1) Preparation Platform",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

if __name__ == "__main__" or True:
    render_tax_system(project_root=_PROJECT_ROOT)
