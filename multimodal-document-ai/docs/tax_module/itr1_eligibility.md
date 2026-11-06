# Phase 5: ITR-1 (SAHAJ) Statutory Eligibility Engine
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.eligibility`  
**Date:** November 2026  
**Status:** Completed & Tested (5 unit tests passed, 27 total tax tests passing)

---

## 1. Statutory Rules & Legal Basis

Form **ITR-1 (SAHAJ)** is governed by Rule 12 of the Income Tax Rules, 1962. It is a simplified return intended strictly for resident individual taxpayers with uncomplicated earnings. 

The **ITR-1 Eligibility Engine** enforces the exact legal boundaries established under the Income Tax Act:

| Statutory Condition | Legal Rule for ITR-1 | Action on Violation |
|---|---|---|
| **Taxpayer Status** | Natural Individuals only (PAN 4th character must be `'P'`). | Ineligible. HUFs/Firms/Companies are barred. |
| **Residential Status** | Resident and Ordinarily Resident (ROR). | Ineligible. RNOR and Non-Residents must file **ITR-2**. |
| **Total Income Ceiling** | Total income must not exceed **₹50,00,000** (50 Lakhs). | Ineligible. Must file **ITR-2**. |
| **Permitted Income Heads** | Salary/Pension, One House Property, Other Sources (Interest, Dividend). | Ineligible if other heads present. |
| **Capital Gains** | Short-term or Long-term capital gains are prohibited. | Ineligible. Must file **ITR-2** (Schedule CG). |
| **Business / Profession** | Profits & Gains from Business or Profession (PGBP) prohibited. | Ineligible. Must file **ITR-3** or **ITR-4 SUGAM**. |
| **House Property Count** | Maximum of **one single house property** (Self-occupied or Let out). | Ineligible if owning multiple properties (Requires ITR-2). |
| **Agricultural Income** | Exempt agricultural income up to **₹5,000** only. | Ineligible if > ₹5,000. Requires **ITR-2**. |
| **Company Directorship** | Must not have been a director in any company at any time. | Ineligible. Requires **ITR-2**. |
| **Unlisted Equity** | Must not have held unlisted equity shares at any time. | Ineligible. Requires **ITR-2**. |
| **Foreign Assets & Income** | Prohibited from having any foreign asset, accounts, or foreign income. | Ineligible. Requires **ITR-2** (Schedule FA). |
| **Loss Carry-Forward** | Cannot claim brought-forward or carried-forward house property losses. | Ineligible. Requires **ITR-2** (Schedule CFL). |
| **Lottery / VDA / Crypto** | Section 115BB / 115BBH winnings & Virtual Digital Assets are barred. | Ineligible. Requires **ITR-2**. |

---

## 2. Audit Trail & Form Recommendations

- When a taxpayer is disqualified, the engine never forces them into ITR-1.
- The engine identifies each statutory condition that failed, provides a plain-English explanation citing the statutory reason, and identifies the recommended form (`ITR-2` or `ITR-3`).
- All eligibility outcomes are recorded in the `tax_audit_log` table.

---

## 3. Test Suite Verification

- **Suite:** `tests/test_tax_eligibility.py`
- **Result:** 5 passed in 0.54s
- **Scenarios Tested:**
  - Standard eligible resident salaried taxpayer (income <= 50L)
  - Income exceeding ₹50,00,000 threshold recommending ITR-2
  - Non-Resident / RNOR disqualification recommending ITR-2
  - Business/professional income disqualification recommending ITR-3
  - Statutory exclusion combination (multiple house properties, agricultural income > ₹5k, directorship, foreign assets)
