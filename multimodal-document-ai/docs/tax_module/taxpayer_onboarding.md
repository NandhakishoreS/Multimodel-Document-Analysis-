# Phase 2: Taxpayer Onboarding Module
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.onboarding`  
**Date:** November 2026  
**Status:** Completed & Tested (6 unit tests passed, 17 total tax tests passing)

---

## 1. Overview & Statutory Requirements

The **Taxpayer Onboarding Module** manages taxpayer identity, statutory validations, bank account verification for refund processing, and Assessment Year binding under the Indian Income Tax Act.

### Supported Taxpayer Category
- **Status:** Individual (Natural Persons only)
- **PAN 4th Character Rule:** Must be `'P'` (Section 139A). Entities with `'C'` (Company), `'F'` (Firm), `'A'` (AOP), or `'H'` (HUF) are rejected as ineligible for ITR-1 natural individual filing.
- **Residential Status:** Sourced from `ResidentialStatus` (`RESIDENT`, `RNOR`, `NON_RESIDENT`).

---

## 2. Statutory Validation Rules

| Field | Format / Pattern | Statutory Rationale |
|---|---|---|
| **PAN** | `^[A-Z]{5}[0-9]{4}[A-Z]{1}$` | Mandatory 10-char alphanumeric identifier; 4th char must be `'P'`. |
| **Date of Birth** | ISO `YYYY-MM-DD` | Past calendar date; used for senior citizen slab determination. |
| **Mobile Number** | `^[6-9]\d{9}$` | 10-digit Indian telecommunications standard. |
| **Email Address** | Standard RFC 5322 regex | Required for CPC notices and communication. |
| **PIN Code** | `^\d{6}$` | 6-digit postal index code. |
| **IFSC Code** | `^[A-Z]{4}0[A-Z0-9]{6}$` | RBI 11-char alphanumeric branch identifier with 5th char `'0'`. |
| **Bank Account** | 9 to 18 alphanumeric chars | Validated for electronic refund mandate credit. |

---

## 3. Privacy & Masking Controls

To protect taxpayer PII:
- **PAN Masking:** `mask_pan("ABCPE1234F")` → `"ABXXXXXX4F"` (only first 2 and last 2 characters displayed in logs and UI headers).
- **Bank Account Masking:** `mask_bank_account("123456789012")` → `"XXXXXXXX9012"` (only last 4 digits visible).
- **No Sensitive Credential Storage:** The system never requests or stores income tax e-filing passwords or bank OTPs.

---

## 4. Assessment Year Binding & Readiness Check

The `TaxpayerOnboardingService.get_onboarding_status()` method evaluates a 9-point readiness checklist:
1. Valid Individual PAN
2. Full Name
3. Date of Birth
4. Residential Status
5. Mobile Contact
6. Email Contact
7. Address Details (City & PIN code)
8. At least one Bank Account recorded
9. Explicitly nominated Electronic Refund Bank Account

---

## 5. Test Suite Verification

- **Suite:** `tests/test_tax_onboarding.py`
- **Result:** 6 passed in 0.60s
- **Scenarios Tested:**
  - Valid individual taxpayer onboarding with PAN masking
  - Rejection of corporate / non-individual PANs (e.g. 4th letter 'C')
  - Rejection of future dates, invalid mobile, malformed email, and bad PIN code
  - Bank account IFSC regex validation and refund nomination
  - Supported Assessment Year binding (`2024-25` → `2023-24`) and rejection of unsupported years
  - Readiness checklist completion percentage calculation
