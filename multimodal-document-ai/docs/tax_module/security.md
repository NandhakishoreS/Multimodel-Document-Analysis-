# Phase 11: Security, Privacy, Masking & Audit Trail
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module`  
**Date:** November 2026  
**Status:** Completed & Tested

---

## 1. Statutory Privacy & Sensitive Identifier Masking

Under Indian data privacy principles and Income Tax Department guidelines, tax returns handle sensitive Personal Identifiable Information (PII):

- **PAN Masking:** All user interface displays and logs display only the first two and last two characters of the PAN:
  $$\text{ABCPE1234F} \longrightarrow \text{ABXXXXXX4F}$$
- **Bank Account Number Masking:** Bank account numbers mask all leading digits, displaying only the last 4 digits:
  $$\text{50100234567890} \longrightarrow \text{XXXXXXXXXX7890}$$
- **Zero Credential Retention:** The system never prompts for, collects, or stores the taxpayer's e-filing portal password, banking net-banking credentials, or OTPs.

---

## 2. Multi-Taxpayer Data Isolation

- **Database-Level Isolation:** The SQLite persistence layer (`tax_module/db.py`) enforces strict multi-taxpayer segregation. Every query for assessments, documents, extracted fields, income, deductions, and calculations requires an explicit `assessment_id` or `taxpayer_id` parameter.
- **Physical Document Isolation:** Ingested files are stored with unique UUID-based identifiers, and duplicate file prevention ensures one taxpayer's documents cannot collide with another's.

---

## 3. Cryptographic Hashing & Tamper-Evident Audit Logging

- **SHA-256 Document Fingerprinting:** Every ingested tax document (PDF, image, text) has its SHA-256 cryptographic hash computed upon upload. Re-uploading the identical file in the same session is blocked.
- **Immutable Audit Trail:** All critical state changes (profile updates, document uploads, manual corrections, discrepancy resolutions, calculations, and JSON generations) write an immutable record to the `tax_audit_log` table with timestamps, actions, and detailed explanation payloads.
