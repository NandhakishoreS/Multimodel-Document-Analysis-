"""
Phase 11 — Institutional Financial Document Intelligence & Searchable Reporting
Streamlit Multi-Page Application.

Pages:
    1. Process Documents   — upload/batch process → extract → classify → save
    2. Document Registry   — search, filter, edit, verify
    3. Financial Analytics — interactive Plotly charts and KPI cards
    4. Anomaly Detection   — rule-based flag browser
    5. Export & Reports    — JSON / CSV / Excel / PDF download

Entry-point: app_phase11.py  (mirrors the Phase 10 app.py pattern)
This file must NOT be run directly with `streamlit run` from the app/ directory.
Use `streamlit run app_phase11.py` from the project root.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Phase 10 validated pipeline (read-only) ───────────────────────────────
from inference.final_inference import UnseenDocumentInferencePipeline

# ── Phase 11 & 12 business-intelligence & ingestion layer ─────────────────
from phase11.schema import FinancialDocument, VERIFICATION_STATUSES
from phase11.classifier import BillCategoryClassifier
from phase11.extractor import FinancialDocumentExtractor
from phase11.database import FinancialDocumentDB
from phase11.export import (
    DocumentExporter,
    export_institutional_pdf,
    export_institutional_excel,
    export_institutional_csv,
    export_institutional_json,
)
from phase11.reporting import (
    InstitutionalReportEngine,
    ReportType,
    REPORT_TITLES,
    format_inr,
    format_currency,
)
from phase11.ingestion import (
    DocumentIngestionService,
    IngestionConfig,
    IngestionItem,
    BatchSummary,
    ProcessingStatus,
    DuplicatePolicy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phase11_app")

# ─────────────────────────────────────────────────────────────────────────
# Global singletons (cached across all reruns)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_inference_pipeline(dataset_type: str, use_robust: bool) -> UnseenDocumentInferencePipeline:
    return UnseenDocumentInferencePipeline(
        dataset_type=dataset_type,
        use_robust_model=use_robust,
        project_root=_PROJECT_ROOT,
    )

@st.cache_resource(show_spinner=False)
def get_db() -> FinancialDocumentDB:
    return FinancialDocumentDB(project_root=_PROJECT_ROOT)

@st.cache_resource(show_spinner=False)
def get_extractor() -> FinancialDocumentExtractor:
    return FinancialDocumentExtractor()

@st.cache_resource(show_spinner=False)
def get_classifier() -> BillCategoryClassifier:
    return BillCategoryClassifier()

@st.cache_resource(show_spinner=False)
def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService()

@st.cache_resource(show_spinner=False)
def get_exporter() -> DocumentExporter:
    return DocumentExporter()

@st.cache_resource(show_spinner=False)
def get_reporting_engine() -> InstitutionalReportEngine:
    return InstitutionalReportEngine(db=get_db())

# ─────────────────────────────────────────────────────────────────────────
# Page config & global CSS
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Document Intelligence — Enterprise Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.ph11-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1d4ed8 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    color: white;
}
.ph11-hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.ph11-hero p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.95rem; }

.kpi-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #1e293b; margin: 0.2rem 0 0; }

.status-pending  { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.status-verified { background:#d1fae5; color:#065f46; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.status-modified { background:#e0e7ff; color:#3730a3; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.status-flagged  { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }
.status-rejected { background:#f1f5f9; color:#475569; border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600; }

.sev-critical { background:#7f1d1d; color:#ffffff; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
.sev-high     { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
.sev-medium   { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
.sev-low      { background:#e0f2fe; color:#075985; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
.sev-info     { background:#f1f5f9; color:#475569; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700; }

.section-title { font-size:1.15rem; font-weight:700; color:#1e293b; margin:1.2rem 0 0.6rem; border-left:4px solid #1d4ed8; padding-left:0.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────

PAGES = [
    "🏠 Process Documents",
    "📋 Document Registry",
    "📊 Financial Analytics",
    "⚠️ Anomaly Detection",
    "📤 Export & Reports",
]

with st.sidebar:
    st.markdown("## 🏦 Financial Document AI")
    st.markdown("<small style='color:#64748b'>Enterprise Financial Intelligence</small>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='color:#94a3b8'>Powered by validated Phase 10 ML pipeline<br>SROIE F1: 0.8837 | FUNSD: 0.7179 | CORD: 0.9733</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    cls = f"status-{status}"
    return f'<span class="{cls}">{status.upper()}</span>'

def _severity_badge(sev: str) -> str:
    s = (sev or "INFO").lower()
    return f'<span class="sev-{s}">{sev.upper()}</span>'

def _currency_fmt(val: Optional[float], currency: str = "INR") -> str:
    if val is None:
        return "—"
    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, currency + " ")
    return f"{symbol}{val:,.2f}"

def _init_filter_state():
    defaults = {
        "p13_q": "",
        "p13_cat": "All",
        "p13_vendor": "",
        "p13_inv_num": "",
        "p13_date_mode": "All",
        "p13_date_from": "",
        "p13_date_to": "",
        "p13_date_exact": "",
        "p13_month": "",
        "p13_year": "",
        "p13_fiscal_year": "",
        "p13_min_amt": 0.0,
        "p13_max_amt": 0.0,
        "p13_stat": "All",
        "p13_domain": "All",
        "p13_rev": "All",
        "p13_modified": "All",
        "p13_sort_by": "created_at",
        "p13_sort_order": "desc",
        "p13_page": 1,
        "p13_page_size": 25,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _clear_filters():
    st.session_state["p13_q"] = ""
    st.session_state["p13_cat"] = "All"
    st.session_state["p13_vendor"] = ""
    st.session_state["p13_inv_num"] = ""
    st.session_state["p13_date_mode"] = "All"
    st.session_state["p13_date_from"] = ""
    st.session_state["p13_date_to"] = ""
    st.session_state["p13_date_exact"] = ""
    st.session_state["p13_month"] = ""
    st.session_state["p13_year"] = ""
    st.session_state["p13_fiscal_year"] = ""
    st.session_state["p13_min_amt"] = 0.0
    st.session_state["p13_max_amt"] = 0.0
    st.session_state["p13_stat"] = "All"
    st.session_state["p13_domain"] = "All"
    st.session_state["p13_rev"] = "All"
    st.session_state["p13_modified"] = "All"
    st.session_state["p13_sort_by"] = "created_at"
    st.session_state["p13_sort_order"] = "desc"
    st.session_state["p13_page"] = 1

def _get_active_filter_kwargs():
    kwargs = {}
    if st.session_state.get("p13_q"):
        kwargs["query"] = st.session_state["p13_q"]
    if st.session_state.get("p13_cat") and st.session_state["p13_cat"] != "All":
        kwargs["category"] = st.session_state["p13_cat"]
    if st.session_state.get("p13_vendor"):
        kwargs["vendor"] = st.session_state["p13_vendor"]
    if st.session_state.get("p13_inv_num"):
        kwargs["invoice_number"] = st.session_state["p13_inv_num"]
    if st.session_state.get("p13_stat") and st.session_state["p13_stat"] != "All":
        kwargs["verification_status"] = st.session_state["p13_stat"]
    if st.session_state.get("p13_domain") and st.session_state["p13_domain"] != "All":
        kwargs["benchmark_domain"] = st.session_state["p13_domain"]

    date_mode = st.session_state.get("p13_date_mode", "All")
    if date_mode == "Exact Date" and st.session_state.get("p13_date_exact"):
        kwargs["date_exact"] = st.session_state["p13_date_exact"]
    elif date_mode == "Date Range":
        if st.session_state.get("p13_date_from"):
            kwargs["date_from"] = st.session_state["p13_date_from"]
        if st.session_state.get("p13_date_to"):
            kwargs["date_to"] = st.session_state["p13_date_to"]
    elif date_mode == "Month" and st.session_state.get("p13_month"):
        kwargs["month"] = st.session_state["p13_month"]
    elif date_mode == "Year" and st.session_state.get("p13_year"):
        kwargs["year"] = st.session_state["p13_year"]
    elif date_mode == "Fiscal Year (FY)" and st.session_state.get("p13_fiscal_year"):
        kwargs["fiscal_year"] = st.session_state["p13_fiscal_year"]

    if st.session_state.get("p13_min_amt", 0.0) > 0:
        kwargs["min_amount"] = st.session_state["p13_min_amt"]
    if st.session_state.get("p13_max_amt", 0.0) > 0:
        kwargs["max_amount"] = st.session_state["p13_max_amt"]

    rev = st.session_state.get("p13_rev", "All")
    if rev == "Requires Review Only":
        kwargs["requires_review"] = True
    elif rev == "No Review Needed":
        kwargs["requires_review"] = False

    mod = st.session_state.get("p13_modified", "All")
    if mod == "Modified Only":
        kwargs["is_modified"] = True
    elif mod == "Unmodified Only":
        kwargs["is_modified"] = False

    return kwargs

# ─────────────────────────────────────────────────────────────────────────
# PAGE 1 — Process Documents
# ─────────────────────────────────────────────────────────────────────────

if page == PAGES[0]:
    import pandas as pd
    st.markdown("""
    <div class="ph11-hero">
        <h1>🏠 Process Documents</h1>
        <p>Institutional multi-document batch ingestion, cryptographic deduplication, multimodal extraction, and intelligent bill classification.</p>
    </div>
    """, unsafe_allow_html=True)

    col_cfg, col_up = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="section-title">⚙️ Processing Configuration</div>', unsafe_allow_html=True)
        doc_domain = st.selectbox(
            "Document Domain",
            ["SROIE (Receipts & Bills)", "CORD (Dining Receipts)", "FUNSD (Forms)"],
            help="Select the benchmark domain that best matches your document type."
        )
        domain_key = doc_domain.split()[0].lower()
        use_robust = st.checkbox("Use Phase 9 Robust Model (SROIE only)", value=True)

        dup_policy_str = st.selectbox(
            "Exact Duplicate Policy",
            ["Skip Duplicate (Recommended)", "Process Anyway", "Warn & Flag"],
            help="How to handle files whose SHA-256 hash already exists in the database."
        )
        dup_policy_map = {
            "Skip Duplicate (Recommended)": DuplicatePolicy.SKIP,
            "Process Anyway": DuplicatePolicy.PROCESS_ANYWAY,
            "Warn & Flag": DuplicatePolicy.WARN,
        }
        dup_policy = dup_policy_map[dup_policy_str]

        review_threshold = st.slider(
            "Classification Review Threshold",
            0.0, 1.0, 0.60, 0.05,
            help="Classifications with confidence below this threshold are flagged for review."
        )
        currency_sel = st.selectbox("Document Currency", ["INR", "USD", "EUR", "GBP", "Other"])

    with col_up:
        st.markdown('<div class="section-title">📂 Upload Documents (Multi-Format Batch)</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or more document files (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    if uploaded_files:
        db = get_db()
        ingestion_service = get_ingestion_service()
        ingestion_service.config.confidence_threshold = review_threshold
        ingestion_service.config.duplicate_policy = dup_policy
        ingestion_service.config.default_currency = currency_sel

        # Build IngestionItems
        items: List[IngestionItem] = []
        preview_rows = []
        for uf in uploaded_files:
            item = IngestionItem.from_bytes(uf.name, uf.getvalue())
            is_valid = ingestion_service.validate_item(item)
            is_dup, dup_doc = ingestion_service.check_duplicate(item, db)
            items.append(item)

            status_str = "✅ Valid" if is_valid else f"❌ {item.error_message or 'Rejected'}"
            dup_str = f"⚠️ Duplicate ({dup_doc.document_id[:8]}…)" if is_dup and dup_doc else "✨ New"

            preview_rows.append({
                "Filename": uf.name,
                "Format": item.file_extension.upper().replace(".", ""),
                "Size": f"{item.file_size / 1024:.1f} KB",
                "Validation": status_str,
                "Duplicate Status": dup_str,
                "SHA-256": f"{item.file_hash[:12]}…",
            })

        st.markdown(f"**{len(uploaded_files)} file(s) queued for ingestion.**")
        with st.expander("🔍 Ingestion Queue Preview", expanded=True):
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        run_btn = st.button("🚀 Process Documents Batch", type="primary", use_container_width=True)

        if run_btn:
            pipeline = get_inference_pipeline(domain_key, use_robust)
            extractor = get_extractor()
            classifier = get_classifier()

            progress_bar = st.progress(0, text="Initializing batch processing…")
            status_text = st.empty()

            def on_progress(current: int, total: int, it: IngestionItem):
                pct = current / total
                status_text.markdown(f"**Processing {current}/{total}:** `{it.filename}` — *Stage: {it.stage}*")
                progress_bar.progress(pct)

            summary = ingestion_service.process_batch(
                items=items,
                pipeline=pipeline,
                extractor=extractor,
                classifier=classifier,
                db=db,
                progress_callback=on_progress,
            )

            progress_bar.progress(1.0, text="Batch processing complete!")
            status_text.empty()

            # Batch Summary KPI Cards
            st.markdown('<div class="section-title">📊 Batch Execution Summary</div>', unsafe_allow_html=True)
            bk1, bk2, bk3, bk4, bk5, bk6 = st.columns(6)
            bk1.metric("Total Uploaded", summary.total_uploaded)
            bk2.metric("Completed", summary.completed)
            bk3.metric("Duplicates", summary.duplicates)
            bk4.metric("Failed", summary.failed)
            bk5.metric("Rejected", summary.rejected)
            bk6.metric("Requires Review", summary.requires_review_count)

            st.caption(f"⏱️ Total batch time: **{summary.total_processing_time:.2f}s** | Average per document: **{summary.average_processing_time:.2f}s**")

            # Category distribution badges
            if summary.category_counts:
                cat_pills = " ".join([f"`{k}: {v}`" for k, v in summary.category_counts.items()])
                st.markdown(f"**Category Breakdown:** {cat_pills}")

            # Detailed Processed Documents Accordion
            st.markdown('<div class="section-title">📄 Processed Document Records</div>', unsafe_allow_html=True)
            for it in summary.items:
                fin_doc = it.document
                if fin_doc:
                    rev_badge = "⚠️ Needs Review" if fin_doc.requires_review else "✅ Confident"
                    title = f"{it.filename} — {fin_doc.category} ({rev_badge})"
                    with st.expander(title, expanded=(summary.completed == 1)):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Category", fin_doc.category)
                        c2.metric("Total Amount", _currency_fmt(fin_doc.total_amount, fin_doc.currency))
                        c3.metric("Vendor", fin_doc.vendor_name or "—")
                        c4.metric("AI Confidence", f"{fin_doc.classification_confidence:.1%}")

                        if fin_doc.requires_review:
                            st.warning(f"⚠️ **Review Notice**: {fin_doc.classification_reason}")
                        else:
                            st.info(f"ℹ️ **Explainability**: {fin_doc.classification_reason}")

                        if fin_doc.classification_evidence:
                            st.markdown(f"**Evidence Keywords:** `{'`, `'.join(fin_doc.classification_evidence)}`")

                        # In-line classification override form
                        with st.form(key=f"correct_{fin_doc.document_id}"):
                            st.markdown("**✏️ Manual Classification Review & Correction**")
                            mc1, mc2 = st.columns(2)
                            cat_options = [
                                "Electricity","Water","Internet/Telecom","Gas",
                                "Office Supplies","Equipment","Maintenance","Transportation","Travel",
                                "Accommodation","Training","Professional Services","Food/Catering","Miscellaneous","Unknown"
                            ]
                            cur_idx = cat_options.index(fin_doc.category) if fin_doc.category in cat_options else 0
                            new_cat = mc1.selectbox("Override Category", options=cat_options, index=cur_idx)
                            c_notes = mc2.text_input("Correction Reason / Audit Note", placeholder="e.g. Verified against physical invoice")

                            if st.form_submit_button("💾 Apply Manual Category Override", type="primary"):
                                db.update_category(
                                    document_id=fin_doc.document_id,
                                    new_category=new_cat,
                                    modified_by="reviewer",
                                    notes=c_notes,
                                )
                                st.success(f"✅ Category updated to {new_cat}. AI baseline ({fin_doc.ai_category}) preserved.")
                                st.rerun()
                else:
                    with st.expander(f"❌ {it.filename} — {it.status.value}", expanded=True):
                        st.error(f"Status: **{it.status.value}** | Stage: **{it.stage}**")
                        st.write(f"Reason: {it.error_message or 'No specific error details provided.'}")

# ─────────────────────────────────────────────────────────────────────────
# PAGE 2 — Document Registry
# ─────────────────────────────────────────────────────────────────────────

elif page == PAGES[1]:
    import pandas as pd
    _init_filter_state()

    st.markdown("""
    <div class="ph11-hero">
        <h1>📋 Document Registry</h1>
        <p>Advanced search, multi-criteria filtering, audit-aware record editing, and document lifecycle management.</p>
    </div>
    """, unsafe_allow_html=True)

    db = get_db()

    # Search & Filter Bar
    with st.expander("🔍 Search & Structured Filters", expanded=True):
        fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
        q_in = fcol1.text_input("Keyword Search (vendor, invoice #, text, filename, category)", value=st.session_state["p13_q"], placeholder="e.g. Electricity, TANGEDCO, INV-2026-001...")
        cat_options = ["All"] + [
            "Electricity", "Water", "Internet/Telecom", "Gas",
            "Office Supplies", "Equipment", "Maintenance", "Transportation", "Travel",
            "Accommodation", "Training", "Professional Services", "Food/Catering", "Miscellaneous", "Unknown"
        ]
        cat_idx = cat_options.index(st.session_state["p13_cat"]) if st.session_state["p13_cat"] in cat_options else 0
        cat_in = fcol2.selectbox("Bill Category", cat_options, index=cat_idx)
        stat_options = ["All"] + list(VERIFICATION_STATUSES)
        stat_idx = stat_options.index(st.session_state["p13_stat"]) if st.session_state["p13_stat"] in stat_options else 0
        stat_in = fcol3.selectbox("Verification Status", stat_options, index=stat_idx)

        # Second row: Vendor, Invoice Number, Date Mode
        r2_1, r2_2, r2_3 = st.columns([1.5, 1.5, 1.5])
        vendor_in = r2_1.text_input("Vendor Name (exact or partial)", value=st.session_state["p13_vendor"], placeholder="e.g. TANGEDCO, Bharat Petroleum...")
        inv_num_in = r2_2.text_input("Invoice Number (exact or partial)", value=st.session_state["p13_inv_num"], placeholder="e.g. INV-2026-042...")
        date_modes = ["All", "Date Range", "Exact Date", "Month", "Year", "Fiscal Year (FY)"]
        dmode_idx = date_modes.index(st.session_state["p13_date_mode"]) if st.session_state["p13_date_mode"] in date_modes else 0
        date_mode_in = r2_3.selectbox("Date Filter Type", date_modes, index=dmode_idx)

        # Dynamic Date Controls
        date_from_val = st.session_state["p13_date_from"]
        date_to_val = st.session_state["p13_date_to"]
        date_exact_val = st.session_state["p13_date_exact"]
        month_val = st.session_state["p13_month"]
        year_val = st.session_state["p13_year"]
        fy_val = st.session_state["p13_fiscal_year"]

        if date_mode_in == "Date Range":
            dc1, dc2 = st.columns(2)
            date_from_val = dc1.text_input("Date From (YYYY-MM-DD)", value=st.session_state["p13_date_from"], placeholder="2026-03-01")
            date_to_val = dc2.text_input("Date To (YYYY-MM-DD)", value=st.session_state["p13_date_to"], placeholder="2026-03-31")
        elif date_mode_in == "Exact Date":
            date_exact_val = st.text_input("Exact Date (YYYY-MM-DD)", value=st.session_state["p13_date_exact"], placeholder="2026-03-15")
        elif date_mode_in == "Month":
            month_val = st.text_input("Month (YYYY-MM)", value=st.session_state["p13_month"], placeholder="2026-03")
        elif date_mode_in == "Year":
            year_val = st.text_input("Year (YYYY)", value=st.session_state["p13_year"], placeholder="2026")
        elif date_mode_in == "Fiscal Year (FY)":
            fy_val = st.text_input("Fiscal Year (e.g. FY 2025-26, 2025-2026)", value=st.session_state["p13_fiscal_year"], placeholder="FY 2025-26")

        # Third row: Amount range, Review filter, Domain, Sort controls
        r3_1, r3_2, r3_3, r3_4 = st.columns(4)
        min_amt_in = r3_1.number_input("Min Amount (₹)", min_value=0.0, value=float(st.session_state["p13_min_amt"]), step=500.0)
        max_amt_in = r3_2.number_input("Max Amount (₹, 0 = no limit)", min_value=0.0, value=float(st.session_state["p13_max_amt"]), step=500.0)
        rev_options = ["All", "Requires Review Only", "No Review Needed"]
        rev_idx = rev_options.index(st.session_state["p13_rev"]) if st.session_state["p13_rev"] in rev_options else 0
        rev_in = r3_3.selectbox("Review Queue", rev_options, index=rev_idx)
        domain_options = ["All", "SROIE", "CORD", "FUNSD", "Other"]
        dom_idx = domain_options.index(st.session_state["p13_domain"]) if st.session_state["p13_domain"] in domain_options else 0
        domain_in = r3_4.selectbox("Document Domain", domain_options, index=dom_idx)

        # Fourth row: Sorting and Actions
        s_col1, s_col2, s_col3, s_col4 = st.columns([1.5, 1, 1, 1])
        sort_field_map = {
            "created_at": "Processing Date",
            "invoice_date": "Invoice Date",
            "total_amount": "Total Amount",
            "vendor_name": "Vendor Name",
            "category": "Category",
            "confidence": "Classification Confidence",
            "verification_status": "Verification Status",
        }
        rev_sort_map = {v: k for k, v in sort_field_map.items()}
        curr_sort_label = sort_field_map.get(st.session_state["p13_sort_by"], "Processing Date")
        sort_by_label = s_col1.selectbox("Sort By", list(sort_field_map.values()), index=list(sort_field_map.values()).index(curr_sort_label))
        sort_by_in = rev_sort_map[sort_by_label]

        order_options = ["Descending (Newest / Highest)", "Ascending (Oldest / Lowest)"]
        order_idx = 0 if st.session_state["p13_sort_order"].lower() == "desc" else 1
        sort_order_label = s_col2.selectbox("Order", order_options, index=order_idx)
        sort_order_in = "desc" if "Descending" in sort_order_label else "asc"

        # Apply state updates from inputs
        st.session_state["p13_q"] = q_in
        st.session_state["p13_cat"] = cat_in
        st.session_state["p13_vendor"] = vendor_in
        st.session_state["p13_inv_num"] = inv_num_in
        st.session_state["p13_stat"] = stat_in
        st.session_state["p13_date_mode"] = date_mode_in
        st.session_state["p13_date_from"] = date_from_val
        st.session_state["p13_date_to"] = date_to_val
        st.session_state["p13_date_exact"] = date_exact_val
        st.session_state["p13_month"] = month_val
        st.session_state["p13_year"] = year_val
        st.session_state["p13_fiscal_year"] = fy_val
        st.session_state["p13_min_amt"] = min_amt_in
        st.session_state["p13_max_amt"] = max_amt_in
        st.session_state["p13_rev"] = rev_in
        st.session_state["p13_domain"] = domain_in
        st.session_state["p13_sort_by"] = sort_by_in
        st.session_state["p13_sort_order"] = sort_order_in

        s_col3.write("")
        s_col3.write("")
        if s_col3.button("🔄 Refresh"):
            st.session_state["p13_page"] = 1
            st.rerun()

        s_col4.write("")
        s_col4.write("")
        if s_col4.button("🧹 Clear Filters", type="secondary"):
            _clear_filters()
            st.rerun()

    active_filter_kwargs = _get_active_filter_kwargs()
    total_matching = db.count_search_documents(**active_filter_kwargs)

    # Pagination calculations
    page_size = st.session_state.get("p13_page_size", 25)
    total_pages = max(1, (total_matching + page_size - 1) // page_size)
    current_page = min(max(1, st.session_state.get("p13_page", 1)), total_pages)
    st.session_state["p13_page"] = current_page
    offset = (current_page - 1) * page_size

    # Query matching documents
    docs = db.search_documents(
        limit=page_size,
        offset=offset,
        sort_by=st.session_state["p13_sort_by"],
        sort_order=st.session_state["p13_sort_order"],
        **active_filter_kwargs,
    )

    # Pagination header bar
    start_record = offset + 1 if total_matching > 0 else 0
    end_record = min(offset + len(docs), total_matching)

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([3, 1, 1, 1, 1.5])
    p_col1.markdown(f"**Showing {start_record}–{end_record} of {total_matching:,} documents** (Page {current_page} of {total_pages})")

    if p_col2.button("⏮ First", disabled=(current_page <= 1)):
        st.session_state["p13_page"] = 1
        st.rerun()
    if p_col3.button("◀ Prev", disabled=(current_page <= 1)):
        st.session_state["p13_page"] = current_page - 1
        st.rerun()
    if p_col4.button("Next ▶", disabled=(current_page >= total_pages)):
        st.session_state["p13_page"] = current_page + 1
        st.rerun()

    new_psize = p_col5.selectbox("Per page", [10, 25, 50, 100], index=[10, 25, 50, 100].index(page_size) if page_size in [10, 25, 50, 100] else 1, key="psize_select")
    if new_psize != page_size:
        st.session_state["p13_page_size"] = new_psize
        st.session_state["p13_page"] = 1
        st.rerun()

    if docs:
        table_data = []
        for d in docs:
            missing_items = d.detect_missing_fields()
            table_data.append({
                "Document ID": d.document_id[:8] + "…",
                "Filename":    d.filename[:22],
                "Category":    d.category,
                "AI Cat":      d.ai_category or d.category,
                "Vendor":      (d.vendor_name or "—")[:20],
                "Invoice #":   (d.invoice_number or "—")[:15],
                "Date":        d.invoice_date or "—",
                "Total":       _currency_fmt(d.total_amount, d.currency),
                "Confidence":  f"{d.classification_confidence:.1%}",
                "Status":      d.verification_status.upper(),
                "Review":      "⚠️ Review" if d.requires_review else ("⚠️ Missing" if missing_items else "✅ Clean"),
                "Modified":    "✏️ Yes" if d.is_modified else "No",
                "_id":         d.document_id,
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df.drop(columns=["_id"]), use_container_width=True, height=360)

        # Bulk Operations Toolbar
        with st.expander("⚡ Safe Bulk Operations on Current Page"):
            b_c1, b_c2, b_c3 = st.columns([1.5, 1.5, 3])
            if b_c1.button(f"✅ Mark Current Page as Verified ({len(docs)} docs)"):
                for d in docs:
                    d.verification_status = "verified"
                    d.requires_review = False
                    db.update_document(d)
                st.success(f"Verified {len(docs)} documents on current page.")
                st.rerun()

            if b_c2.button(f"🚩 Mark Current Page as Flagged ({len(docs)} docs)"):
                for d in docs:
                    d.verification_status = "flagged"
                    db.update_document(d)
                st.warning(f"Flagged {len(docs)} documents on current page.")
                st.rerun()

        # Detailed Document Record & Edit Section
        st.markdown('<div class="section-title">📄 Document Detail, Verification & Change History</div>', unsafe_allow_html=True)
        doc_names = [
            f"[{d.verification_status.upper()}] {d.filename} — {d.vendor_name or 'Unknown'} ({_currency_fmt(d.total_amount, d.currency)})"
            for d in docs
        ]
        selected_idx = st.selectbox("Select document to inspect / edit", range(len(docs)), format_func=lambda i: doc_names[i])
        sel_doc = docs[selected_idx]

        # Missing Fields Detection Banner
        missing_fields = sel_doc.detect_missing_fields()
        if missing_fields:
            st.warning(f"⚠️ **Missing Required Financial Fields:** {', '.join(missing_fields)}")

        tab_view, tab_edit, tab_history = st.tabs(["👁️ View & Traceability", "✏️ Edit Financial Record", "📜 Audit History Timeline"])

        with tab_view:
            v_col1, v_col2, v_col3, v_col4 = st.columns(4)
            v_col1.metric("Final Category", sel_doc.final_category or sel_doc.category)
            v_col2.metric("AI Extraction Category", sel_doc.ai_category or sel_doc.category)
            v_col3.metric("AI Confidence", f"{(sel_doc.ai_confidence or sel_doc.classification_confidence):.1%}")
            v_col4.metric("Verification Status", sel_doc.verification_status.upper())

            # Side-by-side AI prediction vs Verified Value comparison
            st.markdown("##### 🔍 AI Prediction vs Final Verified Value")
            ai_c1, ai_c2, ai_c3, ai_c4 = st.columns(4)
            ai_c1.markdown(f"**Vendor**<br><small style='color:#64748b'>AI:</small> `{sel_doc.ai_vendor_name or sel_doc.vendor_name or '—'}`<br><small style='color:#1e4ed8'>Final:</small> **{sel_doc.vendor_name or '—'}**", unsafe_allow_html=True)
            ai_c2.markdown(f"**Invoice Date**<br><small style='color:#64748b'>AI:</small> `{sel_doc.ai_invoice_date or sel_doc.invoice_date or '—'}`<br><small style='color:#1e4ed8'>Final:</small> **{sel_doc.invoice_date or '—'}**", unsafe_allow_html=True)
            ai_c3.markdown(f"**Invoice Number**<br><small style='color:#64748b'>AI:</small> `{sel_doc.ai_invoice_number or sel_doc.invoice_number or '—'}`<br><small style='color:#1e4ed8'>Final:</small> **{sel_doc.invoice_number or '—'}**", unsafe_allow_html=True)
            ai_c4.markdown(f"**Total Amount**<br><small style='color:#64748b'>AI:</small> `{_currency_fmt(sel_doc.ai_total_amount or sel_doc.total_amount, sel_doc.currency)}`<br><small style='color:#1e4ed8'>Final:</small> **{_currency_fmt(sel_doc.total_amount, sel_doc.currency)}**", unsafe_allow_html=True)

            if sel_doc.classification_reason:
                st.info(f"ℹ️ **Classification Evidence & Reason:** {sel_doc.classification_reason}")

            st.caption(
                f"**Document ID:** `{sel_doc.document_id}` | "
                f"**File Hash (SHA-256):** `{sel_doc.file_hash or '—'}` | "
                f"**Domain:** `{sel_doc.benchmark_domain or 'Institutional'}` | "
                f"**Processing Duration:** `{sel_doc.processing_duration or 0:.2f}s` | "
                f"**Created At:** `{sel_doc.created_at}`"
            )

            # Line Items Table
            if sel_doc.line_items:
                st.markdown("##### 🛒 Extracted Line Items")
                items_data = [{
                    "Description": it.description,
                    "Quantity": it.quantity if it.quantity is not None else "—",
                    "Unit Price": _currency_fmt(it.unit_price, sel_doc.currency),
                    "Total": _currency_fmt(it.total, sel_doc.currency),
                } for it in sel_doc.line_items]
                st.dataframe(pd.DataFrame(items_data), use_container_width=True)

            # Raw OCR Text Expander
            if sel_doc.ocr_raw_text:
                with st.expander("📄 View Extracted Raw OCR Text"):
                    st.text(sel_doc.ocr_raw_text)

        with tab_edit:
            st.markdown("##### ✏️ Correct Extracted Financial Information")
            st.caption("Editing updates final verified fields and automatically preserves immutable audit history and original AI predictions.")

            with st.form(f"edit_form_{sel_doc.document_id}"):
                ef1, ef2, ef3 = st.columns(3)
                ev_vendor  = ef1.text_input("Vendor Name", value=sel_doc.vendor_name or "")
                ev_date    = ef2.text_input("Invoice Date (YYYY-MM-DD)", value=sel_doc.invoice_date or "")
                ev_inv_num = ef3.text_input("Invoice Number", value=sel_doc.invoice_number or "")

                ef4, ef5, ef6 = st.columns(3)
                ev_total = ef4.number_input("Total Amount", value=float(sel_doc.total_amount or 0.0), step=1.0)
                ev_sub   = ef5.number_input("Subtotal", value=float(sel_doc.subtotal or 0.0), step=1.0)
                ev_tax   = ef6.number_input("Tax Amount", value=float(sel_doc.tax_amount or 0.0), step=1.0)

                ef7, ef8, ef9 = st.columns(3)
                cat_choices = [
                    "Electricity", "Water", "Internet/Telecom", "Gas",
                    "Office Supplies", "Equipment", "Maintenance", "Transportation", "Travel",
                    "Accommodation", "Training", "Professional Services", "Food/Catering", "Miscellaneous", "Unknown"
                ]
                sel_cat_idx = cat_choices.index(sel_doc.category) if sel_doc.category in cat_choices else 0
                ev_cat   = ef7.selectbox("Category", cat_choices, index=sel_cat_idx)
                curr_choices = ["INR", "USD", "EUR", "GBP", "Other"]
                sel_curr_idx = curr_choices.index(sel_doc.currency) if sel_doc.currency in curr_choices else 0
                ev_curr  = ef8.selectbox("Currency", curr_choices, index=sel_curr_idx)
                stat_choices = list(VERIFICATION_STATUSES)
                sel_stat_idx = stat_choices.index(sel_doc.verification_status) if sel_doc.verification_status in stat_choices else 0
                ev_stat  = ef9.selectbox("Status", stat_choices, index=sel_stat_idx)

                ef10, ef11 = st.columns([1, 2])
                ev_user   = ef10.text_input("Modified By (User / Role)", value="institution_admin")
                ev_reason = ef11.text_input("Reason for Change (Audit Comment)", placeholder="e.g. Corrected OCR date parsing error")
                ev_notes  = st.text_area("Additional Notes", value=sel_doc.notes or "", height=60)

                b_col1, b_col2 = st.columns([3, 1])
                btn_save = b_col1.form_submit_button("💾 Save Changes & Record Audit Event", type="primary")
                btn_del  = b_col2.form_submit_button("🗑️ Delete Document", type="secondary")

            if btn_save:
                validation_errors = []
                if ev_total < 0:
                    validation_errors.append("Total amount cannot be negative.")
                if ev_sub < 0:
                    validation_errors.append("Subtotal cannot be negative.")
                if ev_tax < 0:
                    validation_errors.append("Tax amount cannot be negative.")

                if validation_errors:
                    for err in validation_errors:
                        st.error(f"❌ {err}")
                else:
                    updates_dict = {
                        "vendor_name": ev_vendor.strip() if ev_vendor.strip() else None,
                        "invoice_date": ev_date.strip() if ev_date.strip() else None,
                        "invoice_number": ev_inv_num.strip() if ev_inv_num.strip() else None,
                        "total_amount": ev_total if ev_total != 0.0 else None,
                        "subtotal": ev_sub if ev_sub != 0.0 else None,
                        "tax_amount": ev_tax if ev_tax != 0.0 else None,
                        "category": ev_cat,
                        "currency": ev_curr,
                        "verification_status": ev_stat,
                    }
                    try:
                        db.edit_financial_document(
                            document_id=sel_doc.document_id,
                            updates=updates_dict,
                            modified_by=ev_user.strip() or "user",
                            reason=ev_reason.strip() or "Manual record modification",
                            notes=ev_notes,
                        )
                        st.success("✅ Changes saved to database and recorded in the audit log.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error saving changes: {exc}")

            if btn_del:
                db.delete_document(sel_doc.document_id)
                st.warning(f"🗑️ Deleted document {sel_doc.document_id[:8]}…")
                st.rerun()

        with tab_history:
            st.markdown("##### 📜 Chronological Field-Level Change History")
            history = db.get_audit_history(sel_doc.document_id)
            if not history:
                st.info("No audit history events recorded for this document.")
            else:
                for ev in reversed(history):
                    action = ev["action"]
                    ts = ev["timestamp"]
                    det = ev["details"]

                    if action == "FIELD_UPDATED" and isinstance(det, dict):
                        f_name = det.get("field", "field")
                        old_v = det.get("old_value", "—")
                        new_v = det.get("new_value", "—")
                        u = det.get("modified_by", "user")
                        r = det.get("reason", "")
                        st.markdown(f"""
                        <div style="background:#f8fafc; border-left:4px solid #3b82f6; padding:0.6rem 1rem; margin-bottom:0.6rem; border-radius:4px;">
                            <span style="font-size:0.8rem; color:#64748b;">🕒 {ts}</span> &nbsp;|&nbsp;
                            <span style="font-weight:600; color:#1e40af;">FIELD UPDATED: <code>{f_name}</code></span><br>
                            <span style="color:#64748b;">Old:</span> <code>{old_v}</code> &nbsp;➔&nbsp; <span style="color:#059669; font-weight:600;">New:</span> <code>{new_v}</code><br>
                            <span style="font-size:0.85rem; color:#475569;">👤 <b>{u}</b> {f'— <i>"{r}"</i>' if r else ''}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif action == "CLASSIFICATION_MODIFIED" and isinstance(det, dict):
                        st.markdown(f"""
                        <div style="background:#f8fafc; border-left:4px solid #8b5cf6; padding:0.6rem 1rem; margin-bottom:0.6rem; border-radius:4px;">
                            <span style="font-size:0.8rem; color:#64748b;">🕒 {ts}</span> &nbsp;|&nbsp;
                            <span style="font-weight:600; color:#6d28d9;">CATEGORY OVERRIDE</span><br>
                            <span style="color:#64748b;">Previous:</span> <code>{det.get('previous_category')}</code> &nbsp;➔&nbsp; <span style="color:#7c3aed; font-weight:600;">New:</span> <code>{det.get('new_category')}</code><br>
                            <span style="font-size:0.85rem; color:#475569;">👤 <b>{det.get('modified_by', 'reviewer')}</b> | Initial AI: {det.get('ai_category')} ({det.get('ai_confidence', 0):.1%})</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif action == "RECORD_EDITED" and isinstance(det, dict):
                        f_list = ", ".join(det.get("fields_changed", []))
                        st.markdown(f"""
                        <div style="background:#f1f5f9; border-left:4px solid #64748b; padding:0.6rem 1rem; margin-bottom:0.6rem; border-radius:4px;">
                            <span style="font-size:0.8rem; color:#64748b;">🕒 {ts}</span> &nbsp;|&nbsp;
                            <span style="font-weight:600; color:#334155;">RECORD EDITED</span><br>
                            <span style="font-size:0.85rem; color:#475569;">Fields changed: <code>{f_list}</code> | User: <b>{det.get('modified_by', 'user')}</b> {f'— <i>"{det.get("reason")}"</i>' if det.get("reason") else ""}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#f8fafc; border-left:4px solid #10b981; padding:0.6rem 1rem; margin-bottom:0.6rem; border-radius:4px;">
                            <span style="font-size:0.8rem; color:#64748b;">🕒 {ts}</span> &nbsp;|&nbsp;
                            <span style="font-weight:600; color:#047857;">{action.upper()}</span><br>
                            <span style="font-size:0.85rem; color:#475569;">Initial ingestion & creation event</span>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        st.info("No documents match your active search and filter criteria. Try adjusting or clearing filters.")

# ─────────────────────────────────────────────────────────────────────────
# PAGE 3 — Financial Analytics
# ─────────────────────────────────────────────────────────────────────────

elif page == PAGES[2]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        PLOTLY_OK = True
    except ImportError:
        PLOTLY_OK = False

    st.markdown("""
    <div class="ph11-hero">
        <h1>📊 Financial Analytics Dashboard</h1>
        <p>Institutional-grade KPIs, Financial Year analysis, trend intelligence, category &amp; vendor insights, tax compliance, and anomaly detection.</p>
    </div>
    """, unsafe_allow_html=True)

    _init_filter_state()
    active_filters = _get_active_filter_kwargs()
    db = get_db()

    # Active filter banner
    if active_filters:
        filter_desc = []
        if "category" in active_filters: filter_desc.append(f"Category: **{active_filters['category']}**")
        if "vendor" in active_filters: filter_desc.append(f"Vendor: **{active_filters['vendor']}**")
        if "fiscal_year" in active_filters: filter_desc.append(f"FY: **{active_filters['fiscal_year']}**")
        if "date_from" in active_filters or "date_to" in active_filters:
            filter_desc.append(f"Date: **{active_filters.get('date_from','—')} → {active_filters.get('date_to','—')}**")
        if "month" in active_filters: filter_desc.append(f"Month: **{active_filters['month']}**")
        if "verification_status" in active_filters: filter_desc.append(f"Status: **{active_filters['verification_status'].upper()}**")
        if "query" in active_filters: filter_desc.append(f"Query: **'{active_filters['query']}'**")
        desc_text = " | ".join(filter_desc) if filter_desc else f"{len(active_filters)} active filter(s)"
        c_a1, c_a2 = st.columns([4, 1])
        c_a1.info(f"🔍 **Metrics computed on filtered dataset:** {desc_text}")
        if c_a2.button("🧹 Clear Filters", key="clear_dash_filters"):
            _clear_filters()
            st.rerun()

    # ── 8 Dashboard Tabs ─────────────────────────────────────────────────────
    (tab_kpi, tab_fy, tab_trend, tab_cat, tab_vendor,
     tab_tax, tab_anomaly, tab_export) = st.tabs([
        "🎯 KPI Overview",
        "📅 Financial Year",
        "📈 Trend Analysis",
        "🍰 Category Intelligence",
        "🏢 Vendor Intelligence",
        "💰 Tax & Compliance",
        "⚠️ Anomaly Analytics",
        "📥 Executive Export",
    ])

    # =========================================================================
    # TAB A — KPI Overview
    # =========================================================================
    with tab_kpi:
        with st.spinner("Computing KPIs…"):
            kpi = db.kpi_summary(**active_filters)

        if not kpi:
            st.info("No documents in the database yet.")
        else:
            st.markdown('<div class="section-title">Core Performance Indicators</div>', unsafe_allow_html=True)

            # Row 1: Volume & Spend
            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, "Invoice Count",   f"{kpi['invoice_count']:,}"),
                (c2, "Total Spend",     f"₹{kpi['total_spend']:,.0f}"),
                (c3, "Avg Invoice Value", f"₹{kpi['avg_invoice_value']:,.0f}" if kpi.get("avg_invoice_value") else "—"),
                (c4, "Verified Rate",   f"{kpi['verified_rate']:.1%}"),
            ]:
                col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

            st.markdown("")

            # Row 2: Tax & Compliance
            c5, c6, c7, c8 = st.columns(4)
            tax_display = (f"₹{kpi['tax_total']:,.0f}" if kpi.get("tax_total") is not None else "Tax data unavailable")
            eff_rate_display = (f"{kpi['effective_tax_rate']:.2%}" if kpi.get("effective_tax_rate") is not None else "—")
            for col, label, val in [
                (c5, "Tax Collected",       tax_display),
                (c6, "Effective Tax Rate",  eff_rate_display),
                (c7, "Avg AI Confidence",   f"{kpi['avg_confidence']:.1%}" if kpi.get("avg_confidence") else "—"),
                (c8, "Modified Records",    str(kpi.get("modified_count", 0))),
            ]:
                col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

            st.markdown("")

            # Status breakdown
            st.markdown('<div class="section-title">Verification Status Breakdown</div>', unsafe_allow_html=True)
            if PLOTLY_OK:
                status_items = [
                    ("Verified",  kpi["verified_count"],  "#10b981"),
                    ("Pending",   kpi["pending_count"],   "#f59e0b"),
                    ("Modified",  kpi["modified_count"],  "#6366f1"),
                    ("Flagged",   kpi["flagged_count"],   "#ef4444"),
                    ("Rejected",  kpi["rejected_count"],  "#94a3b8"),
                ]
                df_status = pd.DataFrame([(s, c, col) for s, c, col in status_items if c > 0],
                                         columns=["Status", "Count", "Color"])
                if not df_status.empty:
                    fig_stat = go.Figure(go.Bar(
                        x=df_status["Status"], y=df_status["Count"],
                        marker_color=df_status["Color"].tolist(),
                        text=df_status["Count"], textposition="outside",
                    ))
                    fig_stat.update_layout(showlegend=False, margin=dict(t=20, b=10),
                                           plot_bgcolor="rgba(0,0,0,0)", height=320)
                    st.plotly_chart(fig_stat, use_container_width=True)
            else:
                st.json({"verified": kpi["verified_count"], "pending": kpi["pending_count"],
                         "flagged": kpi["flagged_count"], "modified": kpi["modified_count"]})

            if kpi.get("tax_unavailable_count", 0) > 0:
                st.caption(f"ℹ️ {kpi['tax_unavailable_count']} document(s) have no tax data recorded. "
                           f"Tax figures above exclude those documents.")

    # =========================================================================
    # TAB B — Financial Year Analysis
    # =========================================================================
    with tab_fy:
        with st.spinner("Computing Financial Year data…"):
            fy_data = db.spend_by_fiscal_year(**active_filters)

        st.markdown('<div class="section-title">Spend by Indian Financial Year (April–March)</div>', unsafe_allow_html=True)

        if not fy_data:
            st.info("No dated documents available for Financial Year analysis.")
        else:
            if PLOTLY_OK:
                df_fy = pd.DataFrame(fy_data)
                fig_fy = px.bar(
                    df_fy, x="fiscal_year", y="total_spend",
                    text=df_fy["total_spend"].apply(lambda x: f"₹{x:,.0f}"),
                    color="total_spend", color_continuous_scale="Blues",
                    labels={"fiscal_year": "Financial Year", "total_spend": "Total Spend (₹)"},
                )
                fig_fy.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10),
                                     plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fy, use_container_width=True)

                # FY Summary table
                fy_table = [{
                    "Financial Year": r["fiscal_year"],
                    "Invoices": r["count"],
                    "Total Spend": f"₹{r['total_spend']:,.0f}",
                    "Avg Invoice": f"₹{r['avg_amount']:,.0f}" if r.get("avg_amount") else "—",
                } for r in fy_data]
                st.dataframe(pd.DataFrame(fy_table), use_container_width=True, hide_index=True)

            # YoY Comparison selector
            if len(fy_data) >= 2:
                st.markdown('<div class="section-title">Year-over-Year Comparison</div>', unsafe_allow_html=True)
                fy_labels = [r["fiscal_year"] for r in fy_data]
                yoy_c1, yoy_c2 = st.columns(2)
                fy1_sel = yoy_c1.selectbox("Baseline FY", fy_labels, index=0, key="yoy_fy1")
                fy2_sel = yoy_c2.selectbox("Comparison FY", fy_labels, index=len(fy_labels)-1, key="yoy_fy2")

                if fy1_sel != fy2_sel:
                    yoy = db.yoy_comparison(fy1_sel, fy2_sel)
                    m1, m2, m3 = st.columns(3)
                    d1, d2 = yoy["fy1_data"], yoy["fy2_data"]

                    def _delta_str(delta):
                        if delta is None: return "—"
                        arrow = "▲" if delta > 0 else "▼"
                        color = "#10b981" if delta > 0 else "#ef4444"
                        return f'<span style="color:{color}; font-weight:700;">{arrow} {abs(delta):.1f}%</span>'

                    m1.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">Total Spend</div>
                        <div class="kpi-value" style="font-size:1.3rem">₹{d2['total_spend']:,.0f}</div>
                        <div style="font-size:0.85rem;color:#64748b;">vs ₹{d1['total_spend']:,.0f} &nbsp; {_delta_str(yoy['delta_spend_pct'])}</div>
                    </div>""", unsafe_allow_html=True)
                    m2.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">Invoice Count</div>
                        <div class="kpi-value" style="font-size:1.3rem">{d2['count']}</div>
                        <div style="font-size:0.85rem;color:#64748b;">vs {d1['count']} &nbsp; {_delta_str(yoy['delta_count_pct'])}</div>
                    </div>""", unsafe_allow_html=True)
                    avg2 = d2.get("avg_amount") or 0
                    avg1 = d1.get("avg_amount") or 0
                    m3.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">Avg Invoice</div>
                        <div class="kpi-value" style="font-size:1.3rem">₹{avg2:,.0f}</div>
                        <div style="font-size:0.85rem;color:#64748b;">vs ₹{avg1:,.0f} &nbsp; {_delta_str(yoy['delta_avg_pct'])}</div>
                    </div>""", unsafe_allow_html=True)

    # =========================================================================
    # TAB C — Trend Analysis
    # =========================================================================
    with tab_trend:
        with st.spinner("Computing monthly trends…"):
            trend_data = db.monthly_trend_with_delta(**active_filters)

        st.markdown('<div class="section-title">Monthly Spend Trend with MoM Delta</div>', unsafe_allow_html=True)

        if not trend_data:
            st.info("No dated documents available for trend analysis.")
        elif not PLOTLY_OK:
            st.warning("Install plotly for interactive charts.")
            st.json(trend_data[:5])
        else:
            df_t = pd.DataFrame(trend_data)

            # Dual-axis: bars for spend, line for MoM delta
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(
                x=df_t["month"], y=df_t["total_spend"],
                name="Monthly Spend",
                marker_color="#1d4ed8",
                opacity=0.85,
            ))
            # Rolling average overlay
            if df_t["rolling_avg_3m"].notna().any():
                fig_trend.add_trace(go.Scatter(
                    x=df_t["month"], y=df_t["rolling_avg_3m"],
                    name="3-Month Rolling Avg",
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=2.5, dash="dot"),
                    yaxis="y",
                ))
            # MoM delta on secondary axis
            if df_t["mom_delta_pct"].notna().any():
                fig_trend.add_trace(go.Scatter(
                    x=df_t["month"], y=df_t["mom_delta_pct"],
                    name="MoM Δ (%)",
                    mode="lines+markers",
                    line=dict(color="#10b981", width=2),
                    yaxis="y2",
                ))
            fig_trend.update_layout(
                xaxis_tickangle=-30,
                yaxis=dict(title="Spend (₹)", titlefont_color="#1d4ed8"),
                yaxis2=dict(title="MoM Delta (%)", overlaying="y", side="right",
                            titlefont_color="#10b981", showgrid=False),
                legend=dict(orientation="h", y=-0.25),
                margin=dict(t=10, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # MoM table
            st.markdown('<div class="section-title">Month-by-Month Detail</div>', unsafe_allow_html=True)
            tbl = []
            for r in trend_data:
                delta_str = f"{r['mom_delta_pct']:+.1f}%" if r.get("mom_delta_pct") is not None else "—"
                rolling_str = f"₹{r['rolling_avg_3m']:,.0f}" if r.get("rolling_avg_3m") is not None else "—"
                tbl.append({
                    "Month": r["month"],
                    "Invoices": r["count"],
                    "Spend": f"₹{r['total_spend']:,.0f}",
                    "MoM Δ": delta_str,
                    "3m Rolling Avg": rolling_str,
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB D — Category Intelligence
    # =========================================================================
    with tab_cat:
        with st.spinner("Analyzing category concentration…"):
            cc = db.category_concentration(**active_filters)
            spend_cat = db.spend_by_category(**active_filters)

        if not cc["categories"]:
            st.info("No spend data available for category analysis.")
        else:
            # HHI Concentration banner
            hhi_color = {"Highly Concentrated": "#ef4444", "Moderate Concentration": "#f59e0b", "Diversified": "#10b981"}.get(cc["hhi_interpretation"], "#64748b")
            st.markdown(f"""
            <div style="background:{hhi_color}22; border-left:4px solid {hhi_color}; padding:0.8rem 1.2rem;
                        border-radius:8px; margin-bottom:1rem;">
                <b style="color:{hhi_color};">📊 Concentration: {cc['hhi_interpretation']}</b>
                &nbsp;|&nbsp; HHI Score: <b>{cc['hhi']:.0f}</b>
                &nbsp;|&nbsp; Top Category: <b>{cc['top1_share_pct']:.1f}%</b>
                &nbsp;|&nbsp; Top 3 Categories: <b>{cc['top3_share_pct']:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="section-title">Spend by Category (Treemap)</div>', unsafe_allow_html=True)
                if PLOTLY_OK and cc["categories"]:
                    df_cc = pd.DataFrame(cc["categories"])
                    fig_tree = px.treemap(
                        df_cc, path=["category"], values="total_spend",
                        color="share_pct",
                        color_continuous_scale=["#dbeafe", "#1d4ed8"],
                        hover_data={"share_pct": ":.1f"},
                    )
                    fig_tree.update_layout(margin=dict(t=10, b=10), height=380)
                    fig_tree.update_traces(texttemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{customdata[0]:.1f}%")
                    st.plotly_chart(fig_tree, use_container_width=True)

            with col_right:
                st.markdown('<div class="section-title">Category Share Table</div>', unsafe_allow_html=True)
                cat_tbl = [{
                    "Category": r["category"],
                    "Spend": f"₹{r['total_spend']:,.0f}",
                    "Share": f"{r['share_pct']:.1f}%",
                } for r in cc["categories"]]
                st.dataframe(pd.DataFrame(cat_tbl), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB E — Vendor Intelligence
    # =========================================================================
    with tab_vendor:
        vend_n = st.slider("Top N Vendors", 5, 20, 10, key="vend_n_slider")
        with st.spinner("Analyzing vendor data…"):
            vendors = db.vendor_summary(n=vend_n, **active_filters)

        if not vendors:
            st.info("No vendor data available.")
        else:
            # Concentration warning
            concentration_vendors = [v for v in vendors if v["concentration_risk"]]
            if concentration_vendors:
                names = ", ".join(v["vendor_name"] for v in concentration_vendors)
                st.warning(f"⚠️ **Single-Vendor Concentration Risk:** {names} account for >30% of total spend. Diversification recommended.")

            st.markdown('<div class="section-title">Top Vendors by Total Spend</div>', unsafe_allow_html=True)
            if PLOTLY_OK:
                df_v = pd.DataFrame(vendors)
                colors = ["#ef4444" if r else "#1d4ed8" for r in df_v["concentration_risk"]]
                fig_v = go.Figure(go.Bar(
                    x=df_v["total_spend"], y=df_v["vendor_name"],
                    orientation="h",
                    marker_color=colors,
                    text=df_v["share_pct"].apply(lambda s: f"{s:.1f}%"),
                    textposition="inside",
                ))
                fig_v.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=max(350, vend_n * 35),
                    xaxis_title="Total Spend (₹)",
                )
                st.plotly_chart(fig_v, use_container_width=True)

            # Vendor table
            vend_tbl = [{
                "Vendor": v["vendor_name"],
                "Invoices": v["invoice_count"],
                "Total Spend": f"₹{v['total_spend']:,.0f}",
                "Share": f"{v['share_pct']:.1f}%",
                "Avg Invoice": f"₹{v['avg_invoice']:,.0f}" if v.get("avg_invoice") else "—",
                "Last Invoice": v.get("last_invoice_date") or "—",
                "Conc. Risk": "⚠️ YES" if v["concentration_risk"] else "✅ NO",
            } for v in vendors]
            st.dataframe(pd.DataFrame(vend_tbl), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB F — Tax & Compliance
    # =========================================================================
    with tab_tax:
        with st.spinner("Analyzing tax data…"):
            tax = db.tax_analysis(**active_filters)

        st.markdown('<div class="section-title">Tax Data Coverage & Effective Rates</div>', unsafe_allow_html=True)

        tx1, tx2, tx3, tx4 = st.columns(4)
        total_tax_display = f"₹{tax['total_tax']:,.0f}" if tax.get("total_tax") is not None else "Tax data unavailable"
        avg_rate_display = f"{tax['avg_effective_rate']:.2%}" if tax.get("avg_effective_rate") is not None else "—"
        for col, label, val in [
            (tx1, "Tax Coverage",     f"{tax['coverage_pct']:.1f}%"),
            (tx2, "Total Tax",        total_tax_display),
            (tx3, "Avg Effective Rate", avg_rate_display),
            (tx4, "Zero-Tax Documents", str(tax["zero_tax_count"])),
        ]:
            col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        if tax.get("min_effective_rate") is not None and tax.get("max_effective_rate") is not None:
            st.caption(f"🔢 Effective tax rate range: **{tax['min_effective_rate']:.2%}** – **{tax['max_effective_rate']:.2%}**")

        if tax["null_tax_count"] > 0:
            st.info(f"ℹ️ {tax['null_tax_count']} document(s) have no tax data recorded — excluded from rate calculations.")
        if tax["zero_tax_count"] > 0:
            st.warning(f"⚠️ {tax['zero_tax_count']} document(s) have ₹0 tax on non-zero invoices — may require compliance review.")

        # Tax by category
        if tax.get("tax_by_category"):
            st.markdown('<div class="section-title">Tax Collected by Category</div>', unsafe_allow_html=True)
            if PLOTLY_OK:
                df_tax_cat = pd.DataFrame(tax["tax_by_category"])
                fig_tax = px.bar(
                    df_tax_cat, x="category", y="total_tax",
                    color="total_tax", color_continuous_scale="Greens",
                    labels={"category": "Category", "total_tax": "Total Tax (₹)"},
                    text=df_tax_cat["total_tax"].apply(lambda x: f"₹{x:,.0f}"),
                )
                fig_tax.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10),
                                      plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tax, use_container_width=True)

    # =========================================================================
    # TAB G — Advanced Anomaly Analytics
    # =========================================================================
    with tab_anomaly:
        sigma_sel = st.slider("Sigma Threshold (σ)", 1.5, 4.0, 2.0, 0.5, key="sigma_slider",
                               help="Documents with amount > mean + σ×std are flagged as outliers.")
        with st.spinner("Running anomaly analysis…"):
            ar = db.advanced_anomaly_report(sigma_threshold=sigma_sel, **active_filters)

        a1, a2, a3, a4 = st.columns(4)
        mean_str = f"₹{ar['mean_amount']:,.0f}" if ar.get("mean_amount") is not None else "—"
        std_str  = f"₹{ar['std_amount']:,.0f}"  if ar.get("std_amount")  is not None else "—"
        for col, label, val in [
            (a1, "Mean Amount",          mean_str),
            (a2, "Std Dev",              std_str),
            (a3, "Sigma Outliers",       str(len(ar["sigma_outliers"]))),
            (a4, "Duplicate Inv. Groups", str(len(ar["duplicate_invoices"]))),
        ]:
            col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Sigma scatter plot
        if ar.get("sigma_outliers") and PLOTLY_OK:
            st.markdown('<div class="section-title">⚡ Sigma Outlier Documents</div>', unsafe_allow_html=True)
            df_out = pd.DataFrame(ar["sigma_outliers"])
            fig_scatter = px.scatter(
                df_out, x="invoice_date", y="total_amount",
                color="sigma_distance", color_continuous_scale="Reds",
                size="total_amount", size_max=25,
                hover_data=["vendor_name", "verification_status"],
                labels={"total_amount": "Amount (₹)", "invoice_date": "Invoice Date",
                        "sigma_distance": "σ Distance"},
            )
            if ar.get("mean_amount"):
                fig_scatter.add_hline(y=ar["mean_amount"], line_dash="dot",
                                      annotation_text="Mean", line_color="#64748b")
            fig_scatter.update_layout(margin=dict(t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Duplicate invoice table
        if ar["duplicate_invoices"]:
            st.markdown('<div class="section-title">🔁 Duplicate Invoice Numbers</div>', unsafe_allow_html=True)
            dup_tbl = [{
                "Invoice Number": d["invoice_number"],
                "Occurrences": d["count"],
                "Document IDs": ", ".join(d["document_ids"][:3]) + ("…" if len(d["document_ids"]) > 3 else ""),
            } for d in ar["duplicate_invoices"]]
            if PLOTLY_OK:
                st.dataframe(pd.DataFrame(dup_tbl), use_container_width=True, hide_index=True)

        # Missing data matrix
        st.markdown('<div class="section-title">📋 Data Completeness Matrix</div>', unsafe_allow_html=True)
        matrix = ar["missing_data_matrix"]
        if matrix and PLOTLY_OK:
            total = ar.get("total_count", 1) or 1
            field_labels = {"vendor_name": "Vendor Name", "invoice_date": "Invoice Date",
                            "invoice_number": "Invoice #", "total_amount": "Total Amount",
                            "tax_amount": "Tax Amount", "subtotal": "Subtotal", "address": "Address"}
            matrix_rows = [{
                "Field": field_labels.get(f, f),
                "Missing Count": v["null_count"],
                "Missing %": f"{v['null_pct']:.1f}%",
                "Available": total - v["null_count"],
            } for f, v in matrix.items()]
            df_matrix = pd.DataFrame(matrix_rows)
            fig_matrix = px.bar(
                df_matrix, x="Field",
                y="Missing Count",
                color="Missing Count", color_continuous_scale="RdYlGn_r",
                text="Missing %",
                labels={"Missing Count": "# Missing"},
            )
            fig_matrix.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10),
                                     plot_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_matrix, use_container_width=True)
            st.dataframe(df_matrix, use_container_width=True, hide_index=True)

        if ar["round_amount_count"] > 0:
            st.info(f"🎯 **{ar['round_amount_count']}** document(s) have amounts that are exact multiples of ₹1,000 — potential placeholder values.")

    # =========================================================================
    # TAB H — Executive Export
    # =========================================================================
    with tab_export:
        st.markdown('<div class="section-title">📥 Download Institutional Dashboard Report</div>', unsafe_allow_html=True)
        st.markdown("""
        Export a comprehensive **7-sheet Excel workbook** of the current dashboard data including:
        - **Summary KPIs** — all key metrics
        - **Category Analysis** — spend & share by category
        - **Vendor Analysis** — top vendors with concentration risk
        - **Tax Compliance** — coverage, rates, by-category breakdown
        - **Monthly Trend** — spend with MoM delta and rolling average
        - **Anomaly Report** — sigma outliers and duplicate invoices
        - **FY Comparison** — year-over-year fiscal data
        """)

        if active_filters:
            filter_label = " | ".join([f"{k}: {v}" for k, v in list(active_filters.items())[:3]])
        else:
            filter_label = "All Documents"

        if st.button("⬇️ Generate & Download Dashboard Excel", type="primary", key="gen_excel_btn"):
            with st.spinner("Building Excel workbook…"):
                try:
                    kpi_e    = db.kpi_summary(**active_filters)
                    cat_e    = db.category_concentration(**active_filters)["categories"]
                    vend_e   = db.vendor_summary(n=20, **active_filters)
                    tax_e    = db.tax_analysis(**active_filters)
                    trend_e  = db.monthly_trend_with_delta(**active_filters)
                    anomaly_e = db.advanced_anomaly_report(**active_filters)
                    fy_e     = db.spend_by_fiscal_year(**active_filters)

                    exporter = get_exporter()
                    excel_bytes = exporter.export_dashboard_excel(
                        kpi_data=kpi_e,
                        category_data=cat_e,
                        vendor_data=vend_e,
                        tax_data=tax_e,
                        monthly_data=trend_e,
                        anomaly_data=anomaly_e,
                        fy_data=fy_e,
                        filter_description=filter_label,
                    )
                    from datetime import datetime
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📊 Download Dashboard.xlsx",
                        data=excel_bytes,
                        file_name=f"financial_dashboard_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_dashboard_btn",
                    )
                    st.success("✅ Dashboard workbook ready for download!")
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

        st.markdown("---")
        st.markdown("**Standard document-level exports** are available on the 📤 Export & Reports page.")

# ─────────────────────────────────────────────────────────────────────────
# PAGE 4 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────

elif page == PAGES[3]:
    import pandas as pd
    try:
        import plotly.express as px
        PLOTLY_OK = True
    except ImportError:
        PLOTLY_OK = False

    from phase11.anomaly_engine import ANOMALY_NAMES, AnomalyType
    from phase11.duplicate_detector import DuplicateType

    st.markdown("""
    <div class="ph11-hero">
        <h1>⚠️ Duplicate & Anomaly Intelligence Center</h1>
        <p>Institutional multi-signal duplicate detection, deterministic anomaly scoring, explainable evidence, and human-in-the-loop review workflow.</p>
    </div>
    """, unsafe_allow_html=True)

    db = get_db()
    summary = db.anomaly_dashboard_summary()

    # Rescan Toolbar
    r_col1, r_col2, r_col3 = st.columns([2, 2, 3])
    with r_col1:
        if st.button("🔄 Scan All Documents", key="btn_scan_all", type="primary"):
            with st.spinner("Executing full institutional anomaly & duplicate scan…"):
                res = db.rescan_anomalies()
                st.success(f"Scan complete: {res['scanned_documents']} documents analyzed, {res['findings_detected']} findings, {res['duplicates_detected']} duplicate pairs.")
                st.rerun()
    with r_col2:
        if st.button("🔍 Scan Filtered Records", key="btn_scan_filt"):
            active_filt = _get_active_filter_kwargs()
            f_docs = db.search_documents(limit=5000, **active_filt)
            with st.spinner(f"Analyzing {len(f_docs)} filtered records…"):
                res = db.rescan_anomalies(document_ids=[d.document_id for d in f_docs])
                st.success(f"Scan complete: {res['scanned_documents']} analyzed, {res['findings_detected']} findings, {res['duplicates_detected']} duplicates.")
                st.rerun()
    with r_col3:
        st.caption("ℹ️ *Note: Duplicate candidates & anomaly signals are decision-support alerts for human review and do not constitute proof of wrongdoing.*")

    st.markdown("")

    # Summary KPI Cards
    crit_high = summary["severity_counts"].get("CRITICAL", 0) + summary["severity_counts"].get("HIGH", 0)
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    cards = [
        (k1, "Total Findings", str(summary["total_findings"])),
        (k2, "High/Critical", str(crit_high)),
        (k3, "Open Queue", str(summary["open_findings"])),
        (k4, "Under Review", str(summary["under_review_findings"])),
        (k5, "Resolved", str(summary["resolved_findings"])),
        (k6, "Dismissed", str(summary["dismissed_findings"])),
        (k7, "Duplicate Pairs", str(summary["total_duplicate_candidates"])),
    ]
    for col, lbl, val in cards:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab_queue, tab_dups, tab_analytics = st.tabs([
        "📋 Anomaly Review Queue",
        "🔁 Duplicate Candidates & Match Comparison",
        "📊 Integrity Analytics & Executive Export",
    ])

    # =========================================================================
    # TAB 1: Anomaly Review Queue
    # =========================================================================
    with tab_queue:
        st.markdown('<div class="section-title">📋 Active Anomaly Review Queue</div>', unsafe_allow_html=True)

        q_f1, q_f2, q_f3, q_f4 = st.columns(4)
        with q_f1:
            q_sev = st.selectbox("Filter Severity", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"], key="q_sev")
        with q_f2:
            q_stat = st.selectbox("Filter Status", ["All", "OPEN", "UNDER_REVIEW", "RESOLVED", "DISMISSED"], key="q_stat")
        with q_f3:
            all_types = ["All"] + list(ANOMALY_NAMES.keys())
            q_type = st.selectbox("Filter Anomaly Type", all_types, format_func=lambda t: "All Types" if t == "All" else f"{t} ({ANOMALY_NAMES.get(t, t)})", key="q_type")
        with q_f4:
            q_min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05, key="q_min_score")

        findings = db.get_anomaly_findings(
            severity=None if q_sev == "All" else q_sev,
            status=None if q_stat == "All" else q_stat,
            anomaly_type=None if q_type == "All" else q_type,
            min_score=q_min_score,
            limit=500,
        )

        if not findings:
            st.info("No anomaly findings matching the selected filters.")
        else:
            st.caption(f"Showing **{len(findings)}** finding(s). Select a finding below to inspect and disposition.")
            rows = []
            for f in findings:
                rows.append({
                    "#": f["id"],
                    "Filename": (f.get("filename") or "—")[:25],
                    "Vendor": (f.get("vendor_name") or "—")[:20],
                    "Invoice #": f.get("invoice_number") or "—",
                    "Amount": _currency_fmt(f.get("total_amount"), f.get("currency") or "INR"),
                    "Type": f["anomaly_type"],
                    "Severity": f["severity"],
                    "Score": f"{f['score']:.2f}",
                    "Status": f["status"],
                    "Explanation": (f["explanation"][:60] + "…") if len(f["explanation"]) > 60 else f["explanation"],
                })
            df_findings = pd.DataFrame(rows)
            st.dataframe(df_findings, use_container_width=True, hide_index=True)

            st.markdown("### 🔍 Document Investigation & Review Action")
            finding_options = [f"{f['id']} — {f['severity']} [{f['anomaly_type']}] {f.get('filename', 'doc')} (₹{f.get('total_amount', 0):,.0f})" for f in findings]
            selected_idx = st.selectbox("Select Finding to Investigate", range(len(findings)), format_func=lambda i: finding_options[i], key="sel_finding_idx")

            cur_f = findings[selected_idx]
            doc_id = cur_f["document_id"]

            inv_col1, inv_col2 = st.columns([1, 1])
            with inv_col1:
                st.markdown(f"""
                **Target Document Particulars:**
                - **Document ID:** `{doc_id}`
                - **Filename:** `{cur_f.get('filename', '—')}`
                - **Vendor Name:** **{cur_f.get('vendor_name', '—')}**
                - **Invoice Number:** `{cur_f.get('invoice_number', '—')}`
                - **Invoice Date:** `{cur_f.get('invoice_date', '—')}`
                - **Category:** `{cur_f.get('category', '—')}`
                - **Total Amount:** **{_currency_fmt(cur_f.get('total_amount'), cur_f.get('currency') or 'INR')}**
                """)

            with inv_col2:
                sev_html = _severity_badge(cur_f['severity'])
                stat_html = _status_badge(cur_f['status'].lower())
                st.markdown(f"""
                **Integrity Signal Details:**
                - **Severity:** {sev_html}
                - **Current Status:** {stat_html}
                - **Anomaly Code:** `{cur_f['anomaly_type']}` ({ANOMALY_NAMES.get(cur_f['anomaly_type'], cur_f['anomaly_type'])})
                - **Anomaly Score:** **{cur_f['score']:.2f}** | **Confidence:** **{cur_f['confidence']:.2f}**
                - **Detected At:** `{cur_f['detected_at']}`
                - **Explanation:** *{cur_f['explanation']}*
                """, unsafe_allow_html=True)

            # Disposition Form
            st.markdown("#### ✍️ Human-in-the-Loop Review Action")
            rev_c1, rev_c2 = st.columns([1, 2])
            with rev_c1:
                rev_user = st.text_input("Reviewer Name", value="Finance Reviewer", key="rev_user_queue")
            with rev_c2:
                rev_comment = st.text_input("Review Notes / Reason", placeholder="e.g., Verified valid seasonal variance with procurement department", key="rev_comm_queue")

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("🔍 Mark Under Review", key="btn_q_under_rev"):
                    db.update_anomaly_status(cur_f["id"], "UNDER_REVIEW", rev_user, rev_comment)
                    st.success(f"Finding #{cur_f['id']} marked as UNDER_REVIEW.")
                    st.rerun()
            with b2:
                if st.button("✅ Mark Resolved", key="btn_q_resolve", type="primary"):
                    db.update_anomaly_status(cur_f["id"], "RESOLVED", rev_user, rev_comment)
                    st.success(f"Finding #{cur_f['id']} marked as RESOLVED.")
                    st.rerun()
            with b3:
                if st.button("🚫 Dismiss Finding", key="btn_q_dismiss"):
                    db.update_anomaly_status(cur_f["id"], "DISMISSED", rev_user, rev_comment)
                    st.info(f"Finding #{cur_f['id']} DISMISSED (audit history preserved).")
                    st.rerun()
            with b4:
                if st.button("🔁 Confirm Duplicate", key="btn_q_conf_dup"):
                    db.update_anomaly_status(cur_f["id"], "CONFIRMED_DUPLICATE", rev_user, rev_comment)
                    st.warning(f"Finding #{cur_f['id']} marked as CONFIRMED_DUPLICATE.")
                    st.rerun()

            # Document Audit History
            hist = db.get_anomaly_review_history(document_id=doc_id, limit=20)
            if hist:
                st.markdown("##### 📜 Document Review History")
                for h in hist:
                    st.markdown(f"- `{h['timestamp'][:19]}` — **{h['reviewer']}** changed status from **{h['old_status']}** to **{h['new_status']}**: *{h.get('comment') or 'No comment'}*")

    # =========================================================================
    # TAB 2: Duplicate Candidates & Comparison
    # =========================================================================
    with tab_dups:
        st.markdown('<div class="section-title">🔁 Multi-Signal Duplicate Candidates</div>', unsafe_allow_html=True)

        d_f1, d_f2 = st.columns(2)
        with d_f1:
            dup_types = ["All", DuplicateType.EXACT_FILE_DUPLICATE, DuplicateType.STRONG_DUPLICATE_CANDIDATE,
                         DuplicateType.DUPLICATE_INVOICE_ID, DuplicateType.DUPLICATE_FINANCIAL_SIGNATURE,
                         DuplicateType.NEAR_DUPLICATE_RECORD]
            sel_dup_type = st.selectbox("Filter Duplicate Signal", dup_types, key="sel_dup_type")
        with d_f2:
            sel_dup_stat = st.selectbox("Filter Status", ["All", "OPEN", "UNDER_REVIEW", "RESOLVED", "DISMISSED", "CONFIRMED_DUPLICATE"], key="sel_dup_stat")

        dup_candidates = db.get_duplicate_candidates(
            duplicate_type=None if sel_dup_type == "All" else sel_dup_type,
            status=None if sel_dup_stat == "All" else sel_dup_stat,
            limit=500,
        )

        if not dup_candidates:
            st.info("No duplicate candidates found matching the selected filters.")
        else:
            st.caption(f"Showing **{len(dup_candidates)}** duplicate candidate pair(s). Select a pair below to compare side-by-side.")
            dup_rows = []
            for d in dup_candidates:
                dup_rows.append({
                    "#": d["id"],
                    "Source Doc": (d.get("filename") or "—")[:20],
                    "Matched Doc": (d.get("matched_filename") or "—")[:20],
                    "Vendor": (d.get("vendor_name") or "—")[:18],
                    "Source Inv #": d.get("invoice_number") or "—",
                    "Matched Inv #": d.get("matched_invoice_number") or "—",
                    "Amount": _currency_fmt(d.get("total_amount")),
                    "Duplicate Type": d["duplicate_type"],
                    "Score": f"{d['score']:.2f}",
                    "Status": d["status"],
                })
            st.dataframe(pd.DataFrame(dup_rows), use_container_width=True, hide_index=True)

            st.markdown("### 🔍 Side-by-Side Duplicate Comparison")
            dup_options = [f"Pair #{d['id']}: {d.get('filename', 'doc')} vs {d.get('matched_filename', 'doc')} [{d['duplicate_type']}]" for d in dup_candidates]
            sel_dup_idx = st.selectbox("Select Duplicate Pair to Compare", range(len(dup_candidates)), format_func=lambda i: dup_options[i], key="sel_dup_pair_idx")

            pair = dup_candidates[sel_dup_idx]

            # Side-by-side comparison table
            comp_data = [
                {"Field": "Filename", "Source Document": pair.get("filename", "—"), "Matched Document": pair.get("matched_filename", "—"), "Comparison": "MATCH" if pair.get("filename") == pair.get("matched_filename") else "DIFFERENT"},
                {"Field": "Vendor Name", "Source Document": pair.get("vendor_name", "—"), "Matched Document": pair.get("matched_vendor_name", "—"), "Comparison": "MATCH" if (pair.get("vendor_name") or "").lower() == (pair.get("matched_vendor_name") or "").lower() else "DIFFERENT"},
                {"Field": "Invoice #", "Source Document": pair.get("invoice_number", "—"), "Matched Document": pair.get("matched_invoice_number", "—"), "Comparison": "MATCH" if pair.get("invoice_number") == pair.get("matched_invoice_number") else "DIFFERENT"},
                {"Field": "Invoice Date", "Source Document": pair.get("invoice_date", "—"), "Matched Document": pair.get("matched_invoice_date", "—"), "Comparison": "MATCH" if pair.get("invoice_date") == pair.get("matched_invoice_date") else "DIFFERENT"},
                {"Field": "Total Amount", "Source Document": _currency_fmt(pair.get("total_amount")), "Matched Document": _currency_fmt(pair.get("matched_total_amount")), "Comparison": "MATCH" if pair.get("total_amount") == pair.get("matched_total_amount") else "DIFFERENT"},
                {"Field": "Category", "Source Document": pair.get("category", "—"), "Matched Document": pair.get("matched_category", "—"), "Comparison": "MATCH" if pair.get("category") == pair.get("matched_category") else "DIFFERENT"},
            ]
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            st.markdown(f"**Explanation:** *{pair['explanation']}*")
            st.markdown(f"**Score:** `{pair['score']:.2f}` | **Confidence:** `{pair['confidence']:.2f}` | **Status:** `{pair['status']}`")

            # Duplicate Actions
            st.markdown("#### ✍️ Duplicate Resolution Action")
            d_rev1, d_rev2 = st.columns([1, 2])
            with d_rev1:
                d_user = st.text_input("Reviewer Name", value="Finance Reviewer", key="dup_rev_user")
            with d_rev2:
                d_comm = st.text_input("Review Notes", placeholder="e.g., Confirmed exact duplicate submission by vendor", key="dup_rev_comm")

            db1, db2, db3 = st.columns(3)
            with db1:
                if st.button("🔁 Confirm Duplicate Record", key="btn_conf_dup_pair", type="primary"):
                    db.update_duplicate_status(pair["id"], "CONFIRMED_DUPLICATE", d_user, d_comm)
                    st.success(f"Duplicate pair #{pair['id']} confirmed as CONFIRMED_DUPLICATE.")
                    st.rerun()
            with db2:
                if st.button("🔍 Mark Under Review", key="btn_under_rev_pair"):
                    db.update_duplicate_status(pair["id"], "UNDER_REVIEW", d_user, d_comm)
                    st.info(f"Duplicate pair #{pair['id']} marked as UNDER_REVIEW.")
                    st.rerun()
            with db3:
                if st.button("🚫 Dismiss Duplicate Candidate", key="btn_dismiss_pair"):
                    db.update_duplicate_status(pair["id"], "DISMISSED", d_user, d_comm)
                    st.info(f"Duplicate pair #{pair['id']} marked as DISMISSED (evidence preserved).")
                    st.rerun()

    # =========================================================================
    # TAB 3: Integrity Analytics & Executive Export
    # =========================================================================
    with tab_analytics:
        st.markdown('<div class="section-title">📊 Financial Integrity Analytics & Executive Export</div>', unsafe_allow_html=True)

        if PLOTLY_OK and summary["total_findings"] > 0:
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("##### Findings by Severity")
                sev_df = pd.DataFrame([{"Severity": k, "Count": v} for k, v in summary["severity_counts"].items()])
                fig_sev = px.bar(
                    sev_df, x="Severity", y="Count", color="Severity",
                    color_discrete_map={"CRITICAL": "#991b1b", "HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#0284c7", "INFO": "#64748b"},
                )
                fig_sev.update_layout(showlegend=False, margin=dict(t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sev, use_container_width=True)

            with c_p2:
                st.markdown("##### Findings by Anomaly Category")
                type_df = pd.DataFrame([{"Code": k, "Name": ANOMALY_NAMES.get(k, k), "Count": v} for k, v in summary["anomaly_type_counts"].items()])
                fig_type = px.bar(
                    type_df, x="Count", y="Name", orientation="h",
                    color="Count", color_continuous_scale="Reds",
                )
                fig_type.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Download Institutional Integrity Reports")
        st.markdown("""
        Export complete audit reports for compliance, internal audit, or finance committee review:
        - **Comprehensive Excel Workbook**: Multi-sheet workbook covering all Anomaly Findings, Duplicate Candidates, and Review History Audit Logs.
        - **CSV / JSON Exports**: Standard machine-readable feeds for ERP reconciliation or SIEM integration.
        """)

        exp_c1, exp_c2, exp_c3 = st.columns(3)
        exporter = get_exporter()
        all_f = db.get_anomaly_findings(limit=5000)
        all_d = db.get_duplicate_candidates(limit=5000)
        all_h = db.get_anomaly_review_history(limit=5000)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        with exp_c1:
            try:
                excel_bytes = exporter.export_anomalies_excel(all_f, all_d, all_h)
                st.download_button(
                    label="📊 Download Anomaly Report.xlsx",
                    data=excel_bytes,
                    file_name=f"financial_anomaly_report_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_anom_excel",
                    type="primary",
                )
            except Exception as exc:
                st.error(f"Excel export unavailable: {exc}")

        with exp_c2:
            csv_str = exporter.export_anomalies_csv(all_f)
            st.download_button(
                label="📄 Download Findings.csv",
                data=csv_str,
                file_name=f"anomaly_findings_{ts}.csv",
                mime="text/csv",
                key="dl_anom_csv",
            )

        with exp_c3:
            json_str = exporter.export_anomalies_json(all_f, all_d)
            st.download_button(
                label="📦 Download Integrity.json",
                data=json_str,
                file_name=f"financial_integrity_{ts}.json",
                mime="application/json",
                key="dl_anom_json",
            )

# ─────────────────────────────────────────────────────────────────────────
# PAGE 5 — Export & Reports
# ─────────────────────────────────────────────────────────────────────────

elif page == PAGES[4]:
    st.markdown("""
    <div class="ph11-hero">
        <h1>📤 Dynamic Institutional Reporting Center</h1>
        <p>Compile executive-grade financial reports, regulatory audit dossiers, and multi-format data exports with deterministic intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_inst, tab_raw = st.tabs(["🏛️ Institutional Intelligence Reports", "📦 Raw Document Data Dumps"])

    # ═════════════════════════════════════════════════════════════════════
    # TAB 1: Phase 16 Institutional Intelligence Reports
    # ═════════════════════════════════════════════════════════════════════
    with tab_inst:
        _init_filter_state()
        active_filters = _get_active_filter_kwargs()
        db = get_db()
        reporting_engine = get_reporting_engine()

        st.markdown('<div class="section-title">1. Report Archetype & Scope Configuration</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns([1, 1])

        report_type_options = {
            "Executive Financial Summary": ReportType.EXECUTIVE_SUMMARY,
            "Fiscal Year Performance & Trends": ReportType.FISCAL_YEAR,
            "Vendor Exposure & Concentration Intelligence": ReportType.VENDOR_INTELLIGENCE,
            "Category Distribution & HHI Diversity": ReportType.CATEGORY_INTELLIGENCE,
            "Tax Compliance & Data Quality": ReportType.TAX_COMPLIANCE,
            "Anomaly & Duplicate Investigation Dossier": ReportType.ANOMALY_DUPLICATE,
            "Document Lifecycle & Decision Audit Trail": ReportType.AUDIT,
        }

        with rc1:
            sel_report_name = st.selectbox("Report Archetype", list(report_type_options.keys()), index=0, key="inst_report_type")
            sel_report_type = report_type_options[sel_report_name]
            default_title = REPORT_TITLES[sel_report_type]
            report_title = st.text_input("Report Title", value=default_title, key="inst_report_title")

        with rc2:
            org_name = st.text_input("Issuing Entity / Organization", value="Institutional Finance Operations", key="inst_org_name")
            notes_input = st.text_input("Auditor / Executive Remarks", value="Formal institutional review copy.", key="inst_notes")

        st.markdown("**Select Analytical Scope:**")
        scope_choice = st.radio(
            "Report Scope Selection",
            ["Use Active Filters from Document Registry", "Full Database Population", "Custom Ad-Hoc Filter"],
            horizontal=True,
            key="inst_scope_choice",
        )

        filter_kwargs = {}
        if scope_choice == "Use Active Filters from Document Registry":
            filter_kwargs = dict(active_filters)
        elif scope_choice == "Custom Ad-Hoc Filter":
            with st.expander("🎛️ Ad-Hoc Scope Filters", expanded=True):
                ac1, ac2, ac3 = st.columns(3)
                adhoc_cat = ac1.selectbox("Category", ["All"] + [
                    "Electricity", "Water", "Internet/Telecom", "Gas",
                    "Office Supplies", "Equipment", "Maintenance", "Transportation", "Travel",
                    "Accommodation", "Training", "Professional Services", "Food/Catering", "Miscellaneous", "Unknown"
                ], key="inst_adhoc_cat")
                adhoc_stat = ac2.selectbox("Status", ["All"] + list(VERIFICATION_STATUSES), key="inst_adhoc_stat")
                adhoc_dom = ac3.selectbox("Domain", ["All", "SROIE", "CORD", "FUNSD"], key="inst_adhoc_dom")
                if adhoc_cat != "All":
                    filter_kwargs["category"] = adhoc_cat
                if adhoc_stat != "All":
                    filter_kwargs["verification_status"] = adhoc_stat
                if adhoc_dom != "All":
                    filter_kwargs["benchmark_domain"] = adhoc_dom

        # Compile report data via central engine
        report_data = reporting_engine.build_report(
            report_type=sel_report_type,
            report_title=report_title,
            organization=org_name,
            notes=notes_input,
            **filter_kwargs,
        )

        meta = report_data["metadata"]
        kpis = report_data["kpis"]
        narratives = report_data["narratives"]
        consistency = report_data["consistency"]

        # Scope status chips
        st.markdown(
            f"<div style='display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;'>"
            f"<span style='background:#f1f5f9;padding:4px 10px;border-radius:6px;font-size:0.85rem;font-weight:600;'>🆔 {meta['report_id']}</span>"
            f"<span style='background:#eff6ff;color:#1e40af;padding:4px 10px;border-radius:6px;font-size:0.85rem;font-weight:600;'>📁 {meta['document_count']} Document(s)</span>"
            f"<span style='background:#f8fafc;padding:4px 10px;border-radius:6px;font-size:0.85rem;font-weight:500;'>🌐 {meta['scope_description']}</span>"
            f"<span style='background:#fdf4ff;color:#86198f;padding:4px 10px;border-radius:6px;font-size:0.85rem;font-weight:500;'>💱 Currency: {meta['currency']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Mathematical Reconciliation Banner
        if consistency["is_consistent"]:
            st.success("✅ **Reconciliation Status: PASS** — Segment totals mathematically reconcile with aggregate expenditure.")
        else:
            st.warning(f"⚠️ **Reconciliation Status: VARIANCE DETECTED** — {consistency['reconciliation_notes']}")

        # Executive Observations
        st.markdown('<div class="section-title">2. Executive Summary & Deterministic Observations</div>', unsafe_allow_html=True)
        st.markdown("<small style='color:#64748b;'>Algorithmic observations computed strictly from verified database aggregations (Zero LLM hallucinations).</small>", unsafe_allow_html=True)
        for bullet in narratives:
            st.markdown(f"• **{bullet}**")

        # KPI Metrics Grid
        st.markdown('<div class="section-title">3. Key Performance Indicators</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Spend", format_inr(kpis.get("total_spend", 0.0)))
        k2.metric("Invoices Scoped", f"{kpis.get('invoice_count', 0):,}")
        k3.metric("Avg Invoice Value", format_inr(kpis.get("avg_invoice_value")))
        ver_rate = kpis.get("verified_rate", 0.0)
        k4.metric("Verification Rate", f"{ver_rate:.1%}")

        # Data Section Preview
        st.markdown('<div class="section-title">4. Institutional Dataset Preview</div>', unsafe_allow_html=True)
        with st.expander("🔍 View Scoped Analytical Tables", expanded=False):
            import pandas as pd
            if report_data["category_analysis"]["breakdown"]:
                st.markdown("**Category Spend & Concentration**")
                st.dataframe(pd.DataFrame(report_data["category_analysis"]["breakdown"]), use_container_width=True)
            if report_data["vendor_analysis"]["top_vendors"]:
                st.markdown("**Top Vendor Pareto Distribution**")
                st.dataframe(pd.DataFrame(report_data["vendor_analysis"]["top_vendors"]), use_container_width=True)
            if report_data["integrity_analysis"]["findings"]:
                st.markdown("**Open Anomaly Findings**")
                st.dataframe(pd.DataFrame(report_data["integrity_analysis"]["findings"]), use_container_width=True)

        # Actionable Download Center
        st.markdown('<div class="section-title">5. Institutional Export Artifacts</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.markdown("**📑 Executive PDF Brief**")
            st.caption("Publication-grade formatted PDF with running headers, page numbering, KPI table, and audit disclosures.")
            try:
                pdf_bytes = export_institutional_pdf(report_data)
                st.download_button(
                    "📥 Download Institutional PDF",
                    data=pdf_bytes,
                    file_name=f"{meta['report_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_inst_pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        with d2:
            st.markdown("**📗 Institutional Excel (11 Sheets)**")
            st.caption("Formula-injection protected multi-sheet workbook covering Executive, Records, Vendors, FY, Tax, Anomalies, and Audit History.")
            try:
                xlsx_bytes = export_institutional_excel(report_data)
                st.download_button(
                    "📥 Download Institutional Excel",
                    data=xlsx_bytes,
                    file_name=f"{meta['report_id']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="dl_inst_xlsx",
                )
            except Exception as e:
                st.error(f"Excel generation failed: {e}")

        with d3:
            st.markdown("**📊 Scoped CSV Table**")
            st.caption("Normalized flat record export of scoped documents with UTF-8 encoding.")
            csv_str = export_institutional_csv(report_data, section="documents")
            st.download_button(
                "📥 Download Scoped CSV",
                data=csv_str.encode("utf-8"),
                file_name=f"{meta['report_id']}_documents.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_inst_csv",
            )

        with d4:
            st.markdown("**📄 Institutional JSON**")
            st.caption("Machine-readable JSON schema containing full metadata, aggregations, findings, and records.")
            json_str = export_institutional_json(report_data)
            st.download_button(
                "📥 Download Full JSON",
                data=json_str.encode("utf-8"),
                file_name=f"{meta['report_id']}_dataset.json",
                mime="application/json",
                use_container_width=True,
                key="dl_inst_json",
            )

    # ═════════════════════════════════════════════════════════════════════
    # TAB 2: Phase 11 Raw Document Registry Data Dumps (Backward Compatibility)
    # ═════════════════════════════════════════════════════════════════════
    with tab_raw:
        _init_filter_state()
        active_filters = _get_active_filter_kwargs()
        db = get_db()
        exporter = get_exporter()

        exp_mode = st.radio(
            "Export Scope",
            ["Use Active Filters from Document Registry", "Export All Database Records", "Custom Ad-Hoc Filter"],
            horizontal=True,
            key="raw_exp_mode",
        )

        if exp_mode == "Use Active Filters from Document Registry":
            export_docs = db.search_documents(limit=10000, **active_filters)
            spend_by_cat = db.spend_by_category(**active_filters)
            scope_info = f"Filtered ({len(export_docs)} documents matching active registry filters)"
        elif exp_mode == "Export All Database Records":
            export_docs = db.search_documents(limit=10000)
            spend_by_cat = db.spend_by_category()
            scope_info = f"Complete Database ({len(export_docs)} total documents)"
        else:
            # Custom Ad-Hoc Filter expander
            with st.expander("🎛️ Custom Export Filters", expanded=True):
                ec1, ec2, ec3 = st.columns(3)
                exp_cat   = ec1.selectbox("Category", ["All"] + [
                    "Electricity", "Water", "Internet/Telecom", "Gas",
                    "Office Supplies", "Equipment", "Maintenance", "Transportation", "Travel",
                    "Accommodation", "Training", "Professional Services", "Food/Catering", "Miscellaneous", "Unknown"
                ], key="adhoc_cat_raw")
                exp_stat  = ec2.selectbox("Status", ["All"] + list(VERIFICATION_STATUSES), key="adhoc_stat_raw")
                exp_domain = ec3.selectbox("Domain", ["All", "SROIE", "CORD", "FUNSD"], key="adhoc_dom_raw")
            export_docs = db.search_documents(
                category=None if exp_cat == "All" else exp_cat,
                verification_status=None if exp_stat == "All" else exp_stat,
                benchmark_domain=None if exp_domain == "All" else exp_domain,
                limit=5000,
            )
            spend_by_cat = db.spend_by_category(
                category=None if exp_cat == "All" else exp_cat,
                verification_status=None if exp_stat == "All" else exp_stat,
                benchmark_domain=None if exp_domain == "All" else exp_domain,
            )
            scope_info = f"Custom Filter ({len(export_docs)} documents)"

        st.info(f"**{len(export_docs)} document(s)** selected for export ({scope_info}).")

        if not export_docs:
            st.warning("No documents match the selected filters.")
        else:
            st.markdown('<div class="section-title">Download Options</div>', unsafe_allow_html=True)
            dl1, dl2, dl3, dl4 = st.columns(4)

            # JSON
            with dl1:
                st.markdown("**📄 JSON**")
                st.markdown("Structured array of all document fields including line items and metadata.")
                json_str = exporter.to_json(export_docs)
                st.download_button(
                    "📥 Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name="financial_documents.json",
                    mime="application/json",
                    use_container_width=True,
                    key="dl_raw_json",
                )

            # CSV
            with dl2:
                st.markdown("**📊 CSV**")
                st.markdown("Flat tabular format compatible with Excel, Google Sheets, and BI tools.")
                csv_str = exporter.to_csv(export_docs)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_str.encode("utf-8"),
                    file_name="financial_documents.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_raw_csv",
                )

            # Excel
            with dl3:
                st.markdown("**📗 Excel**")
                st.markdown("Multi-sheet workbook: Documents, Line Items, Category Summary.")
                try:
                    xlsx_bytes = exporter.to_excel(export_docs)
                    st.download_button(
                        "📥 Download Excel",
                        data=xlsx_bytes,
                        file_name="financial_documents.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="dl_raw_xlsx",
                    )
                except RuntimeError as e:
                    st.error(str(e))

            # PDF
            with dl4:
                st.markdown("**📑 PDF Report**")
                st.markdown("Formatted summary report with KPIs, category table, and document registry.")
                try:
                    analytics = {"spend_by_category": spend_by_cat}
                    pdf_bytes = exporter.to_pdf_report(export_docs, analytics)
                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_bytes,
                        file_name="financial_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_raw_pdf",
                    )
                except RuntimeError as e:
                    st.error(str(e))

            # Preview JSON
            st.markdown('<div class="section-title">JSON Preview (first 2 documents)</div>', unsafe_allow_html=True)
            preview = json.loads(exporter.to_json(export_docs[:2]))
            st.json(preview)
