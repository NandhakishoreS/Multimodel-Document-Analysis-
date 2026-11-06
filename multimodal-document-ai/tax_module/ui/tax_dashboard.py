"""
Tax Module — Streamlit Income Tax Return Preparation Interface (Phase 9 & 10).

Provides an 8-screen interactive workflow for taxpayer onboarding, document review,
cross-document reconciliation, deterministic tax computation, ITR-1 eligibility,
and official-format JSON export with offline utility filing instructions.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from tax_module.calculator import TaxCalculatorEngine
from tax_module.constants import (
    DEFAULT_ASSESSMENT_YEAR,
    SUPPORTED_ASSESSMENT_YEARS,
    FilingStatus,
    FormType,
    IncomeHead,
    ResidentialStatus,
    TaxDocumentCategory,
    TaxRegime,
    VerificationStatus,
    get_fy_from_ay,
    mask_bank_account,
    mask_pan,
)
from tax_module.db import TaxDatabase
from tax_module.eligibility import ITR1EligibilityEngine
from tax_module.ingestion import TaxDocumentIngestionService
from tax_module.itr_json.generators import ITR1JSONGenerator
from tax_module.models import (
    BankAccount,
    TaxAssessment,
    TaxpayerProfile,
)
from tax_module.onboarding import TaxpayerOnboardingService
from tax_module.reconciliation import IncomeReconciliationEngine

logger = logging.getLogger(__name__)


def render_tax_system(project_root: Optional[Path] = None):
    """Main rendering entry point for the Income Tax Return Preparation System."""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    db = TaxDatabase(project_root=root)
    onboarding_service = TaxpayerOnboardingService(db=db)
    ingestion_service = TaxDocumentIngestionService(db=db)
    recon_engine = IncomeReconciliationEngine(db=db)
    eligibility_engine = ITR1EligibilityEngine(db=db)
    calc_engine = TaxCalculatorEngine(db=db)
    json_generator = ITR1JSONGenerator(db=db)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%); padding: 1.5rem 2rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700;">🇮🇳 Indian Income Tax Return (ITR-1 SAHAJ) Preparation System</h1>
        <p style="margin-top: 0.5rem; opacity: 0.9; font-size: 1rem;">AI-Assisted Document Understanding • Source Reconciliation • Deterministic Computation • Official JSON Export</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Taxpayer & Assessment Selector
    taxpayers = db.list_taxpayers()

    with st.sidebar:
        st.header("👤 Active Taxpayer")
        if taxpayers:
            taxpayer_options = {f"{t.full_name} ({t.masked_pan})": t.taxpayer_id for t in taxpayers}
            taxpayer_options["➕ Onboard New Taxpayer"] = "NEW"
            selected_label = st.selectbox("Select Taxpayer Profile", list(taxpayer_options.keys()))
            selected_taxpayer_id = taxpayer_options[selected_label]
        else:
            st.info("No taxpayers onboarded yet.")
            selected_taxpayer_id = "NEW"

        # Assessment Year Selector
        st.header("📅 Assessment Year")
        selected_ay = st.selectbox("Assessment Year (AY)", SUPPORTED_ASSESSMENT_YEARS, index=0)
        st.caption(f"Corresponding Financial Year: **FY {get_fy_from_ay(selected_ay)}**")

        st.markdown("---")
        screen = st.radio(
            "📋 Workflow Screens",
            [
                "1. Tax Dashboard & Onboarding",
                "2. Document Ingestion & OCR",
                "3. Income & Deductions Summary",
                "4. Financial Reconciliation",
                "5. Tax Calculation & Regimes",
                "6. ITR-1 Statutory Eligibility",
                "7. Return Review & Validation",
                "8. Export & Offline Handoff",
            ],
        )

    # -----------------------------------------------------------------------
    # SCREEN 1: Tax Dashboard & Onboarding
    # -----------------------------------------------------------------------
    if screen == "1. Tax Dashboard & Onboarding":
        st.subheader("Taxpayer Onboarding & Session Overview")

        if selected_taxpayer_id == "NEW":
            st.markdown("### Onboard Individual Taxpayer")
            with st.form("onboard_form"):
                col1, col2 = st.columns(2)
                with col1:
                    pan = st.text_input("Permanent Account Number (PAN)", placeholder="e.g. ABCPE1234F", max_chars=10).upper()
                    first_name = st.text_input("First Name")
                    middle_name = st.text_input("Middle Name (Optional)")
                    last_name = st.text_input("Last Name / Surname")
                    dob = st.text_input("Date of Birth (YYYY-MM-DD)", placeholder="1990-01-15")
                    gender = st.selectbox("Gender", ["M", "F", "O"])
                with col2:
                    res_status = st.selectbox("Residential Status", [ResidentialStatus.RESIDENT.value, ResidentialStatus.RNOR.value, ResidentialStatus.NON_RESIDENT.value])
                    mobile = st.text_input("Mobile Number", placeholder="9876543210", max_chars=10)
                    email = st.text_input("Email Address", placeholder="user@example.com")
                    city = st.text_input("City / District", placeholder="Bengaluru")
                    state = st.text_input("State", placeholder="Karnataka")
                    pincode = st.text_input("PIN Code", placeholder="560001", max_chars=6)

                st.markdown("#### Primary Bank Account for Tax Refund")
                b_col1, b_col2, b_col3 = st.columns(3)
                with b_col1:
                    bank_acc = st.text_input("Account Number", placeholder="123456789012")
                with b_col2:
                    bank_ifsc = st.text_input("IFSC Code", placeholder="SBIN0001234").upper()
                with b_col3:
                    bank_name = st.text_input("Bank Name", placeholder="State Bank of India")

                submitted = st.submit_button("Complete Onboarding & Initialize AY")

                if submitted:
                    data = {
                        "pan": pan,
                        "first_name": first_name,
                        "middle_name": middle_name,
                        "last_name": last_name,
                        "dob": dob,
                        "gender": gender,
                        "residential_status": res_status,
                        "mobile": mobile,
                        "email": email,
                        "address_city": city,
                        "address_state": state,
                        "address_pincode": pincode,
                    }
                    prof, errs = onboarding_service.onboard_taxpayer(data)
                    if errs:
                        for e in errs:
                            st.error(e)
                    elif prof:
                        if bank_acc and bank_ifsc:
                            onboarding_service.add_bank_account(prof.taxpayer_id, bank_acc, bank_ifsc, bank_name)
                        onboarding_service.initialize_assessment(prof.taxpayer_id, selected_ay)
                        st.success(f"Taxpayer {prof.full_name} ({prof.masked_pan}) onboarded successfully!")
                        st.rerun()

        else:
            profile = db.get_taxpayer_profile(selected_taxpayer_id)
            assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
            if not assessment:
                assessment, _ = onboarding_service.initialize_assessment(selected_taxpayer_id, selected_ay)

            if profile and assessment:
                status = onboarding_service.get_onboarding_status(profile.taxpayer_id, assessment.assessment_id)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Taxpayer", profile.full_name)
                col2.metric("PAN", profile.masked_pan)
                col3.metric("Filing Status", assessment.filing_status.value)
                col4.metric("Onboarding Completion", f"{status['completion_percentage']}%")

                st.progress(status["completion_percentage"] / 100)

                if not status["is_ready"]:
                    st.warning(f"Incomplete profile items: {', '.join(status['missing_items'])}")
                else:
                    st.success("Taxpayer profile is 100% complete and ready for document ingestion.")

                # Bank Accounts
                st.markdown("#### Registered Bank Accounts")
                for idx, b in enumerate(profile.bank_accounts, 1):
                    refund_badge = "✅ **Nominated for Electronic Refund**" if b.is_refund_account else ""
                    st.write(f"{idx}. **{b.bank_name}** | A/C: `{b.masked_account_number()}` | IFSC: `{b.ifsc_code}` | {refund_badge}")

    # -----------------------------------------------------------------------
    # SCREEN 2: Document Ingestion & OCR
    # -----------------------------------------------------------------------
    elif screen == "2. Document Ingestion & OCR":
        st.subheader("Tax Document Ingestion & Field Extraction")
        if selected_taxpayer_id == "NEW":
            st.warning("Please onboard or select an active taxpayer first.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning(f"No active assessment found for AY {selected_ay}.")
            return

        st.markdown("#### Upload Tax Document")
        st.caption("Supported: Form 16 (Part A & B), Salary Slips, Bank Interest Certificates, AIS/TIS Statements, Form 26AS, Investment Receipts.")

        uploaded_file = st.file_uploader("Upload Document (PDF, Image, or Text)", type=["pdf", "png", "jpg", "jpeg", "txt"])

        if uploaded_file:
            content = uploaded_file.read()
            text_content = ""
            if uploaded_file.name.endswith(".txt"):
                text_content = content.decode("utf-8", errors="ignore")
            else:
                # Text extraction fallback for uploaded docs
                text_content = f"Uploaded file: {uploaded_file.name}\nSize: {len(content)} bytes."

            if st.button("Process & Ingest Document"):
                doc, errs = ingestion_service.ingest_document_text(
                    assessment_id=assessment.assessment_id,
                    text=text_content,
                    filename=uploaded_file.name,
                    file_bytes=content,
                )
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    st.success(f"Ingested '{uploaded_file.name}' successfully as **{doc.document_category.value}**.")
                    st.rerun()

        # Ingested Document Browser
        docs = db.list_documents_for_assessment(assessment.assessment_id)
        st.markdown(f"#### Ingested Documents ({len(docs)})")
        if not docs:
            st.info("No documents uploaded for this assessment session.")
        else:
            for d in docs:
                with st.expander(f"📄 {d.filename} — {d.document_category.value} (SHA-256: {d.file_hash[:8]}...)"):
                    st.write(f"**Category:** {d.document_category.value} | **Confidence:** {d.ocr_confidence:.2f}")
                    fields = db.list_extracted_fields_for_doc(d.doc_id)
                    st.write(f"**Extracted Fields:** {len(fields)}")
                    for f in fields:
                        st.write(f"- `{f.field_name}`: **₹{f.effective_value}** (Raw OCR: `{f.original_ocr_text}`)")

    # -----------------------------------------------------------------------
    # SCREEN 3: Income & Deductions Summary
    # -----------------------------------------------------------------------
    elif screen == "3. Income & Deductions Summary":
        st.subheader("Verified Income, Deductions & Tax Credits")
        if selected_taxpayer_id == "NEW":
            st.warning("Please onboard or select an active taxpayer first.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No active assessment session.")
            return

        incomes = db.list_income_records(assessment.assessment_id)
        deductions = db.list_deduction_records(assessment.assessment_id)
        credits = db.list_tax_credit_records(assessment.assessment_id)

        tab1, tab2, tab3 = st.tabs(["💰 Income Heads", "📉 Deductions (Chap VI-A)", "💳 Prepaid Taxes / TDS"])

        with tab1:
            st.markdown("#### Declared & Extracted Incomes")
            if not incomes:
                st.info("No income records found.")
            else:
                for inc in incomes:
                    st.write(f"• **{inc.income_head.value}** from *{inc.source_name}*: Gross: **₹{inc.gross_amount}** | Exempt: **₹{inc.exempt_allowances}** | Taxable: **₹{inc.net_taxable_amount}**")

        with tab2:
            st.markdown("#### Deductions Claimed")
            if not deductions:
                st.info("No deductions recorded.")
            else:
                for ded in deductions:
                    st.write(f"• **Section {ded.section_code}** ({ded.description}): Claimed: **₹{ded.claimed_amount}** | Eligible: **₹{ded.eligible_amount}**")

        with tab3:
            st.markdown("#### Tax Deducted at Source & Tax Credits")
            if not credits:
                st.info("No TDS or prepaid taxes recorded.")
            else:
                for cr in credits:
                    st.write(f"• **{cr.credit_type}** by *{cr.deductor_name}* (TAN: `{cr.deductor_tan}`): Deducted: **₹{cr.total_tax_deducted}** | Claimed: **₹{cr.claimed_amount}**")

    # -----------------------------------------------------------------------
    # SCREEN 4: Financial Reconciliation
    # -----------------------------------------------------------------------
    elif screen == "4. Financial Reconciliation":
        st.subheader("Cross-Document Reconciliation & Overlap Protection")
        if selected_taxpayer_id == "NEW":
            st.warning("Please select a taxpayer.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No assessment session.")
            return

        if st.button("🔄 Execute Multi-Source Reconciliation"):
            records = recon_engine.reconcile_assessment(assessment.assessment_id)
            st.success(f"Reconciliation completed with {len(records)} audit checks.")

        recon_list = db.list_reconciliation_records(assessment.assessment_id)
        if not recon_list:
            st.info("Reconciliation has not been executed yet. Click above to run.")
        else:
            for r in recon_list:
                status_color = "🟢" if r.status == "MATCHED" else ("🟡" if r.status == "REVIEW_REQUIRED" else "🔵")
                st.markdown(f"#### {status_color} {r.field_name} — Status: `{r.status}`")
                st.write(f"**Primary Source:** {r.primary_source} (₹{r.primary_amount}) | **Secondary Source:** {r.secondary_source} (₹{r.secondary_amount})")
                st.write(f"**Difference:** ₹{r.difference} | **Notes:** {r.resolution_notes}")
                st.markdown("---")

    # -----------------------------------------------------------------------
    # SCREEN 5: Tax Calculation & Regimes
    # -----------------------------------------------------------------------
    elif screen == "5. Tax Calculation & Regimes":
        st.subheader("Deterministic Tax Computation & Regime Comparison")
        if selected_taxpayer_id == "NEW":
            st.warning("Please select a taxpayer.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No active assessment session.")
            return

        if st.button("⚡ Calculate Taxes (Old vs New Regime)"):
            c_new, c_old, rec_regime, savings = calc_engine.calculate_assessment_taxes(assessment.assessment_id)
            st.success(f"Calculation complete! Recommended: **{rec_regime.value} Regime** (Tax Savings: ₹{savings})")

        calc_new = db.get_latest_tax_calculation(assessment.assessment_id, TaxRegime.NEW)
        calc_old = db.get_latest_tax_calculation(assessment.assessment_id, TaxRegime.OLD)

        if calc_new and calc_old:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🆕 New Regime (Section 115BAC)")
                st.write(f"• **Gross Total Income:** ₹{calc_new.gross_total_income}")
                st.write(f"• **Deductions:** ₹{calc_new.total_deductions}")
                st.write(f"• **Taxable Income:** ₹{calc_new.total_taxable_income}")
                st.write(f"• **Tax on Slabs:** ₹{calc_new.tax_on_taxable_income}")
                st.write(f"• **Rebate u/s 87A:** ₹{calc_new.rebate_87a}")
                st.write(f"• **Health & Edu Cess (4%):** ₹{calc_new.cess}")
                st.metric("Total Tax Liability", f"₹{calc_new.total_tax_liability}")
                st.metric("Net Balance / Refund", f"₹{calc_new.refund_or_payable}")

            with col2:
                st.markdown("### 🏛️ Old Regime")
                st.write(f"• **Gross Total Income:** ₹{calc_old.gross_total_income}")
                st.write(f"• **Chapter VI-A Deductions:** ₹{calc_old.total_deductions}")
                st.write(f"• **Taxable Income:** ₹{calc_old.total_taxable_income}")
                st.write(f"• **Tax on Slabs:** ₹{calc_old.tax_on_taxable_income}")
                st.write(f"• **Rebate u/s 87A:** ₹{calc_old.rebate_87a}")
                st.write(f"• **Health & Edu Cess (4%):** ₹{calc_old.cess}")
                st.metric("Total Tax Liability", f"₹{calc_old.total_tax_liability}")
                st.metric("Net Balance / Refund", f"₹{calc_old.refund_or_payable}")
        else:
            st.info("Click 'Calculate Taxes' above to execute dual-regime calculation.")

    # -----------------------------------------------------------------------
    # SCREEN 6: ITR-1 Statutory Eligibility
    # -----------------------------------------------------------------------
    elif screen == "6. ITR-1 Statutory Eligibility":
        st.subheader("Statutory Eligibility Evaluation for ITR-1 (SAHAJ)")
        if selected_taxpayer_id == "NEW":
            st.warning("Please select a taxpayer.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No assessment session.")
            return

        res = eligibility_engine.evaluate(selected_taxpayer_id, assessment.assessment_id)

        if res.eligible:
            st.success(f"✅ **Eligible for ITR-1 (SAHAJ)**: {res.explanation}")
        else:
            st.error(f"❌ **Ineligible for ITR-1**: {res.explanation}")
            st.info(f"Recommended Form: **{res.recommended_form.value}**")

        st.markdown("#### Passed Statutory Conditions")
        for p in res.passed_conditions:
            st.write(f"✅ {p}")

        if res.failed_conditions:
            st.markdown("#### Disqualifications")
            for f in res.failed_conditions:
                st.write(f"❌ {f}")

    # -----------------------------------------------------------------------
    # SCREEN 7: Return Review & Validation
    # -----------------------------------------------------------------------
    elif screen == "7. Return Review & Validation":
        st.subheader("Official Return Schedule Review")
        if selected_taxpayer_id == "NEW":
            st.warning("Please select a taxpayer.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No assessment session.")
            return

        calc = db.get_latest_tax_calculation(assessment.assessment_id, assessment.selected_regime)
        if not calc:
            st.warning("Please calculate tax first in Screen 5.")
            return

        st.markdown("### Return Summary Preview")
        st.write(f"• **Assessment Year:** {assessment.assessment_year} (FY {assessment.financial_year})")
        st.write(f"• **Regime:** {assessment.selected_regime.value} Regime")
        st.write(f"• **Gross Total Income:** ₹{calc.gross_total_income}")
        st.write(f"• **Total Taxable Income:** ₹{calc.total_taxable_income}")
        st.write(f"• **Tax Liability:** ₹{calc.total_tax_liability}")
        st.write(f"• **Prepaid Taxes (TDS):** ₹{calc.total_tax_credits}")
        st.write(f"• **Net Refund / Payable:** ₹{calc.refund_or_payable}")

    # -----------------------------------------------------------------------
    # SCREEN 8: Export & Offline Handoff
    # -----------------------------------------------------------------------
    elif screen == "8. Export & Offline Handoff":
        st.subheader("Generate Official ITR-1 JSON & Official Utility Handoff")
        if selected_taxpayer_id == "NEW":
            st.warning("Please select a taxpayer.")
            return

        assessment = db.get_assessment_for_year(selected_taxpayer_id, selected_ay)
        if not assessment:
            st.warning("No assessment session.")
            return

        col1, col2 = st.columns(2)
        with col1:
            place = st.text_input("Place of Filing", value="Bengaluru")
        with col2:
            father_name = st.text_input("Father's Name (for Verification Declaration)", value="")

        if st.button("🚀 Generate Official ITR-1 JSON"):
            itr_ret, errs = json_generator.generate_itr1_json(
                assessment_id=assessment.assessment_id,
                place_of_filing=place,
                father_name=father_name,
            )
            if errs:
                st.error("Validation errors encountered:")
                for e in errs:
                    st.write(f"- {e}")
            elif itr_ret and itr_ret.is_valid:
                st.success("Official ITR-1 JSON generated and passed 100% schema validation!")
                st.download_button(
                    label="📥 Download Official ITR-1 JSON",
                    data=itr_ret.official_json,
                    file_name=f"ITR1_{selected_ay}_{assessment.assessment_id[:8]}.json",
                    mime="application/json",
                )

        st.markdown("""
        ---
        ### 📋 Official Income Tax Department Offline Utility Filing Handoff
        
        1. **Download the Prepared JSON:** Click the download button above to obtain your verified return JSON.
        2. **Launch Official Utility:** Download & open the Income Tax Department's official offline utility from [incometax.gov.in](https://www.incometax.gov.in/iec/foportal/downloads/income-tax-returns).
        3. **Import Return:** Select *Import Pre-filled Data / JSON* and upload your generated file.
        4. **Review & Verify:** Confirm that all schedules match the computations displayed in this system.
        5. **Submit & e-Verify:** Complete the official submission using Aadhaar OTP or Net Banking e-verification.
        6. **Record Acknowledgement:** Enter your ITR-V acknowledgement number below for permanent audit tracking.
        """)

        with st.form("ack_form"):
            ack_no = st.text_input("ITR-V Acknowledgement Number (15 Digits)")
            ack_submit = st.submit_button("Record Submission Status")
            if ack_submit and ack_no:
                assessment.acknowledgement_number = ack_no
                db.save_assessment(assessment)
                db.update_filing_status(
                    assessment_id=assessment.assessment_id,
                    status=FilingStatus.SUBMITTED_BY_TAXPAYER,
                    notes=f"Taxpayer confirmed official submission with Ack No: {ack_no}",
                )
                st.success(f"Acknowledgement #{ack_no} recorded! Filing status updated to SUBMITTED_BY_TAXPAYER.")
                st.rerun()
