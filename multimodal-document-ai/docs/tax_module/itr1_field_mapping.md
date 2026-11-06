# Phase 7 & 8: ITR-1 Field Mapping & Official JSON Generation
**Project:** Indian Income Tax Return (ITR-1 SAHAJ) Preparation System  
**Module:** `tax_module.itr_json`  
**Date:** November 2026  
**Status:** Completed & Tested (2 unit tests passed, 34 total tax tests passing)

---

## 1. Official JSON Schema & Hierarchy

The Income Tax Department (ITD) e-filing portal requires returns to be structured in a strict, versioned JSON hierarchy matching the offline utility specification:

```
{
  "ITR": {
    "ITR1": {
      "CreationInfo": { ... },
      "Form_ITR1": { ... },
      "PersonalInfo": { ... },
      "FilingStatus": { ... },
      "IncomeDeductions": { ... },
      "TaxComputation": { ... },
      "TaxPaid": { ... },
      "ScheduleTDS1": [ ... ],
      "BankAccountDtls": [ ... ],
      "Verification": { ... }
    }
  }
}
```

---

## 2. Field Mapping Table (Internal Model → Official ITR-1)

| Internal Domain Field | Official ITR-1 JSON Field | Type & Constraints |
|---|---|---|
| `profile.pan` | `ITR.ITR1.PersonalInfo.PAN` | String, regex `^[A-Z]{5}[0-9]{4}[A-Z]{1}$` |
| `profile.first_name` | `ITR.ITR1.PersonalInfo.AssesseeName.FirstName` | String |
| `profile.last_name` | `ITR.ITR1.PersonalInfo.AssesseeName.SurNameOrOrgName` | String |
| `profile.dob` | `ITR.ITR1.PersonalInfo.DOB` | String, ISO `YYYY-MM-DD` |
| `profile.residential_status` | `ITR.ITR1.FilingStatus.ResidentialStatus` | Enum `"RES"` |
| `assessment.selected_regime` | `ITR.ITR1.FilingStatus.NewTaxRegime` | Enum `"Y"` (New) / `"N"` (Old) |
| `inc.gross_amount` (Salary) | `ITR.ITR1.IncomeDeductions.GrossSalary` | Integer (Section 288A rounded) |
| Standard deduction | `ITR.ITR1.IncomeDeductions.DeductionUs16.StandardDeductionUs16ia` | Integer (₹50,000 for AY 2024-25) |
| `inc.net_taxable_amount` | `ITR.ITR1.IncomeDeductions.IncomeFromSal` | Integer |
| `other_sources` (Interest) | `ITR.ITR1.IncomeDeductions.IncomeOthSrc` | Integer |
| Chapter VI-A deductions | `ITR.ITR1.IncomeDeductions.UsrDeductUndChapVIA` | Object (`Section80C`, `Section80D`, etc.) |
| `calc.total_taxable_income` | `ITR.ITR1.IncomeDeductions.TotalIncome` | Integer |
| `calc.rebate_87a` | `ITR.ITR1.TaxComputation.Rebate87A` | Integer |
| `calc.cess` (4%) | `ITR.ITR1.TaxComputation.EducationCess` | Integer |
| `calc.total_tax_liability` | `ITR.ITR1.TaxComputation.NetTaxLiability` | Integer |
| `tc.total_tax_deducted` | `ITR.ITR1.ScheduleTDS1[].TotalTDSSal` | Integer |
| `tc.deductor_tan` | `ITR.ITR1.ScheduleTDS1[].TAN` | String, regex `^[A-Z]{4}[0-9]{5}[A-Z]{1}$` |
| `bank.account_number` | `ITR.ITR1.BankAccountDtls[].BankAccountNo` | String, 9-18 chars |
| `bank.ifsc_code` | `ITR.ITR1.BankAccountDtls[].IFSC` | String, regex `^[A-Z]{4}0[A-Z0-9]{6}$` |
| `bank.is_refund_account` | `ITR.ITR1.BankAccountDtls[].RefundFlag` | Enum `"Y"` / `"N"` |

---

## 3. Schema Validation & Audit Enforcement

- **Draft-07 JSON Schema:** The schema specification resides at `tax_module/itr_json/schemas/AY_2024_25/itr1_schema.json`.
- **Validation Engine:** `ITR1SchemaValidator` evaluates every generated return against the schema before storing in `itr_returns`.
- **Filing Status:** Returns passing schema validation transition to `JSON_GENERATED`. Returns with schema errors are flagged and prevented from final export.

---

## 4. Test Suite Verification

- **Suite:** `tests/test_tax_itr_json.py`
- **Result:** 2 passed in 0.74s
- **Scenarios Tested:**
  - Automated rejection of malformed / incomplete return structures
  - End-to-end official ITR-1 JSON generation for salaried individual with Form 16, interest income, TDS, and bank accounts passing 100% schema validation
