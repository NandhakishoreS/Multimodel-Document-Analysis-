# Phase 13: Known Limitations & Future Expansion Scope
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Date:** November 2026

---

## 1. Supported Scope

1. **Form Scope:** Form **ITR-1 (SAHAJ)** for resident salaried individuals under the Indian Income Tax Act.
2. **Assessment Years Supported:** Assessment Year **2024-25** (Financial Year 2023-24) and **2025-26** (Financial Year 2024-25).
3. **Filing Method:** Official Offline Utility handoff via schema-compliant JSON generation.

---

## 2. Documented Statutory Limitations

1. **No Direct E-Filing Automation (ERIs / Direct Portal Submission):** The Income Tax Department of India requires certified e-Return Intermediary (ERI) registration and digital cryptographic keys (DSC/XML signing) to submit returns directly via REST APIs. Because this application operates on a local sandbox, returns are prepared and exported as official JSON files for handoff into the official offline utility.
2. **Non-Individual Taxpayers Excluded:** Companies (ITR-6), Partnership Firms (ITR-5), HUFs (ITR-2/3), and Trusts (ITR-7) are not supported by ITR-1.
3. **Complex Income Heads:** Taxpayers with Capital Gains (ITR-2), Business/Professional income (ITR-3 / ITR-4 SUGAM), multiple house properties, or foreign assets are detected and flagged as ineligible, with clear recommendations to file ITR-2 or ITR-3.
4. **Scanned PDF OCR Fidelity:** While RapidOCR extracts text from high-resolution scans, handwritten physical receipts or low-resolution faxes may produce lower OCR confidence, triggering human-in-the-loop review on Screen 2.
