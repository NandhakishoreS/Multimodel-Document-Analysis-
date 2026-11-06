# Phase 10: Official Offline Utility Filing Workflow & State Machine
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module`  
**Date:** November 2026  
**Status:** Completed & Tested

---

## 1. Supported Filing Workflow (Official Offline Handoff)

The application adheres strictly to the official **Offline Utility Handoff** mechanism established by the Income Tax Department of India (ITD). 

> [!IMPORTANT]
> **No Direct/Automated Credential Filing:** The system never asks for or stores taxpayer e-filing portal passwords or OTPs, which would breach ITD security terms. Instead, it generates a validated, schema-compliant JSON return file ready to import into the official Income Tax Department offline utility.

### Step-by-Step Filing Journey

```mermaid
sequenceDiagram
    autonumber
    actor Taxpayer
    participant App as ITR Preparation System
    participant Utility as ITD Official Offline Utility
    participant Portal as Income Tax Portal (CPC)

    Taxpayer->>App: 1. Complete Onboarding (PAN, Bank, Bio)
    Taxpayer->>App: 2. Ingest Financial Documents (Form 16, AIS, Bank Cert)
    App->>App: 3. Cross-Document Reconciliation & Duplicate Check
    App->>App: 4. Statutory Eligibility (ITR-1 vs ITR-2)
    App->>App: 5. Deterministic Tax Computation (Old vs New Regime)
    Taxpayer->>App: 6. Review & Trigger JSON Generation
    App->>App: 7. Validate against Official JSON Schema
    App-->>Taxpayer: 8. Download Verified ITR-1 JSON
    Taxpayer->>Utility: 9. Import JSON into Official ITD Utility
    Utility-->>Taxpayer: 10. Review Schedules & Generate JSON Payload
    Taxpayer->>Portal: 11. Upload & Submit Return on e-filing portal
    Taxpayer->>Portal: 12. Complete e-Verification (Aadhaar OTP)
    Portal-->>Taxpayer: 13. Receive 15-digit Acknowledgement Number (ITR-V)
    Taxpayer->>App: 14. Record Acknowledgement Number & Confirm Status
```

---

## 2. Statutory Filing Lifecycle State Machine

The session moves deterministically through 14 explicit states defined in `FilingStatus`:

1. `NOT_STARTED`: Onboarding initialized; waiting for document uploads.
2. `DOCUMENTS_UPLOADED`: Documents ingested and hashed (SHA-256).
3. `EXTRACTION_IN_PROGRESS`: OCR text extraction and field normalization.
4. `AWAITING_USER_REVIEW`: Extracted financial fields awaiting taxpayer verification.
5. `RECONCILIATION_REQUIRED`: Cross-source discrepancy flags detected requiring review.
6. `TAX_CALCULATED`: Deterministic tax computation executed for Old and New regimes.
7. `ITR_VALIDATION_PENDING`: Candidate return constructed; awaiting schema validation.
8. `ITR_READY`: Return schedules verified; ready for JSON serialization.
9. `JSON_GENERATED`: Validated official JSON produced and downloaded.
10. `IMPORTED_IN_OFFICIAL_UTILITY`: Taxpayer confirmed import into official offline utility.
11. `SUBMITTED_BY_TAXPAYER`: Taxpayer confirmed submission on e-filing portal with acknowledgement number.
12. `VERIFICATION_PENDING`: Return submitted; awaiting Aadhaar OTP e-verification.
13. `VERIFIED`: Return e-verified and CPC acknowledgement confirmed.
14. `REJECTED_OR_CORRECTION_REQUIRED`: CPC defect notice or user correction required.
