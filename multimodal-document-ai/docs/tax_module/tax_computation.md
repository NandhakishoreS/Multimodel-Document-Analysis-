# Phase 6: Versioned Tax Calculation & Dual-Regime Optimizer
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.tax_rules`, `tax_module.calculator`  
**Date:** November 2026  
**Status:** Completed & Tested (5 unit tests passed, 32 total tax tests passing)

---

## 1. Statutory Architecture & Precision Guarantee

Final tax liability under the Income Tax Act must be **100% deterministic, explainable, and verified against statutory rate schedules**. 

- **No LLM Computation:** The LLM is prohibited from performing final tax arithmetic. All calculations are executed deterministically in Python using `Decimal` arithmetic quantized to 2 decimal places.
- **Assessment Year Versioning:** Separate statutory modules exist for `AY 2024-25` (Finance Act 2023) and `AY 2025-26` (Finance (No. 2) Act 2024).

---

## 2. Tax Slabs & Rate Schedules

### 2.1 AY 2024-25 New Regime (Section 115BAC — Default)
- **Standard Deduction:** ₹50,000
- **Slabs:**
  - ₹0 to ₹3,00,000: 0%
  - ₹3,00,001 to ₹6,00,000: 5%
  - ₹6,00,001 to ₹9,00,000: 10%
  - ₹9,00,001 to ₹12,00,000: 15%
  - ₹12,00,001 to ₹15,00,000: 20%
  - Above ₹15,00,000: 30%
- **Rebate u/s 87A:** Full rebate (up to ₹25,000) for taxable income <= ₹7,00,000 (tax liability becomes ₹0.00).
- **Marginal Relief u/s 87A:** For taxable income marginally exceeding ₹7,00,000, tax liability cannot exceed the income in excess of ₹7,00,000.
- **Health & Education Cess:** 4% on tax after rebate.

### 2.2 AY 2024-25 Old Regime (Optional)
- **Standard Deduction:** ₹50,000
- **Slabs (Individual < 60 yrs):**
  - ₹0 to ₹2,50,000: 0%
  - ₹2,50,001 to ₹5,00,000: 5%
  - ₹5,00,001 to ₹10,00,000: 20%
  - Above ₹10,00,000: 30%
- **Rebate u/s 87A:** Up to ₹12,500 if taxable income <= ₹5,00,000.
- **Statutory Deductions:** 80C (up to ₹1.5L), 80D (up to ₹25k/₹50k), 80TTA (up to ₹10k).
- **Health & Education Cess:** 4% on tax after rebate.

### 2.3 AY 2025-26 New Regime Amendments
- **Standard Deduction:** ₹75,000 under New Regime.
- **Amended Slabs:** 0-3L 0%, 3-7L 5%, 7-10L 10%, 10-12L 15%, 12-15L 20%, >15L 30%.

---

## 3. Dual-Regime Optimizer

`TaxCalculatorEngine.calculate_assessment_taxes()` runs both regimes in parallel, compares total tax liabilities, determines the optimal regime, calculates exact taxpayer savings, and logs the explainable breakdown JSON to SQLite.

---

## 4. Test Suite Verification

- **Suite:** `tests/test_tax_calculator.py`
- **Result:** 5 passed in 0.95s
- **Scenarios Tested:**
  - Section 87A zero-tax outcome for ₹7.5L gross salary (7.0L taxable income)
  - Section 87A marginal relief capping tax at excess income (₹7.1L taxable income)
  - Old Regime multi-tier deductions (80C, 80D) and slab computation
  - AY 2025-26 revised standard deduction (₹75k) and updated slabs
  - Dual-regime optimizer selecting the optimal regime and recording savings
