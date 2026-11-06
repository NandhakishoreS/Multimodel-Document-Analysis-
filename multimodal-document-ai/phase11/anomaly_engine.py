"""
Phase 15 — Centralized Anomaly Detection Engine.

Implements rule-based, deterministic anomaly evaluation across 10 defined categories:
- ANOMALY_001: Statistical outlier (sigma-based with sample size guard N >= 3)
- ANOMALY_002: Unusually high invoice (relative to category/vendor baseline)
- ANOMALY_003: Round-number amount (multiples of Rs 1,000)
- ANOMALY_004: Duplicate invoice number
- ANOMALY_005: Missing critical financial fields
- ANOMALY_006: Tax inconsistency (subtotal + tax != total with tolerance)
- ANOMALY_007: Category/value distribution mismatch
- ANOMALY_008: Vendor concentration (>25% spend exposure)
- ANOMALY_009: Temporal spending surge (>50% MoM surge and >2x rolling average)
- ANOMALY_010: Repeated financial signature (vendor + date + amount)

Provides multi-signal aggregation, deterministic severity scoring (INFO, LOW, MEDIUM,
HIGH, CRITICAL), and explainable recommendations.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from phase11.schema import FinancialDocument


# ---------------------------------------------------------------------------
# Anomaly Type Constants & Severities
# ---------------------------------------------------------------------------

class AnomalyType:
    STATISTICAL_OUTLIER = "ANOMALY_001"
    HIGH_INVOICE_BASELINE = "ANOMALY_002"
    ROUND_AMOUNT = "ANOMALY_003"
    DUPLICATE_INVOICE_NUM = "ANOMALY_004"
    MISSING_CRITICAL_FIELDS = "ANOMALY_005"
    TAX_INCONSISTENCY = "ANOMALY_006"
    CATEGORY_MISMATCH = "ANOMALY_007"
    VENDOR_CONCENTRATION = "ANOMALY_008"
    TEMPORAL_SPEND_SURGE = "ANOMALY_009"
    REPEATED_SIGNATURE = "ANOMALY_010"


ANOMALY_NAMES = {
    AnomalyType.STATISTICAL_OUTLIER: "Statistical Outlier",
    AnomalyType.HIGH_INVOICE_BASELINE: "Unusually High Invoice",
    AnomalyType.ROUND_AMOUNT: "Round Amount",
    AnomalyType.DUPLICATE_INVOICE_NUM: "Duplicate Invoice Number",
    AnomalyType.MISSING_CRITICAL_FIELDS: "Missing Critical Fields",
    AnomalyType.TAX_INCONSISTENCY: "Tax Inconsistency",
    AnomalyType.CATEGORY_MISMATCH: "Category Distribution Mismatch",
    AnomalyType.VENDOR_CONCENTRATION: "Vendor Concentration Exposure",
    AnomalyType.TEMPORAL_SPEND_SURGE: "Temporal Spend Surge",
    AnomalyType.REPEATED_SIGNATURE: "Repeated Financial Signature",
}

SEVERITY_LEVELS = ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")
SEVERITY_ORDER = {"INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFinding:
    """Represents a single atomic anomaly signal identified on a document."""
    document_id: str
    anomaly_type: str
    severity: str
    score: float
    confidence: float
    explanation: str
    status: str = "OPEN"
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finding_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anomaly_name"] = ANOMALY_NAMES.get(self.anomaly_type, self.anomaly_type)
        return d


@dataclass
class DocumentAnomalyReport:
    """Aggregated anomaly assessment for a document combining all detected signals."""
    document_id: str
    overall_score: float
    severity: str
    anomaly_count: int
    primary_anomaly_type: str
    anomaly_types: List[str]
    findings: List[AnomalyFinding]
    confidence: float
    explanations: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["findings"] = [f.to_dict() for f in self.findings]
        return d


# ---------------------------------------------------------------------------
# Population Statistics Context
# ---------------------------------------------------------------------------

@dataclass
class PopulationContext:
    """Pre-computed population statistics to support fast batch anomaly checks."""
    global_mean: Optional[float] = None
    global_std: Optional[float] = None
    sample_size: int = 0
    category_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vendor_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    vendor_spend_shares: Dict[str, float] = field(default_factory=dict)
    invoice_number_counts: Dict[str, int] = field(default_factory=dict)
    financial_signature_counts: Dict[Tuple[str, str, float], int] = field(default_factory=dict)
    monthly_spends: Dict[str, float] = field(default_factory=dict)
    rolling_3m_averages: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Anomaly Detection Engine
# ---------------------------------------------------------------------------

class CentralizedAnomalyEngine:
    """
    Evaluates individual documents or collections against 10 deterministic anomaly rules.
    """

    def __init__(self, sigma_threshold: float = 2.0, min_sample_size: int = 3):
        self.sigma_threshold = sigma_threshold
        self.min_sample_size = min_sample_size

    def build_population_context(self, documents: List[FinancialDocument]) -> PopulationContext:
        """Compute statistical baselines across the provided document collection."""
        ctx = PopulationContext()
        amounts = [d.total_amount for d in documents if d.total_amount is not None and d.total_amount > 0]
        ctx.sample_size = len(amounts)
        if ctx.sample_size >= self.min_sample_size:
            ctx.global_mean = statistics.mean(amounts)
            ctx.global_std = statistics.stdev(amounts) if ctx.sample_size > 1 else 0.0

        # Category and Vendor breakdowns
        cat_amounts: Dict[str, List[float]] = {}
        vend_amounts: Dict[str, List[float]] = {}
        vend_totals: Dict[str, float] = {}
        total_global_spend = sum(amounts) if amounts else 0.0

        for d in documents:
            amt = d.total_amount
            if amt is not None and amt > 0:
                if d.category and d.category != "Unknown":
                    cat_amounts.setdefault(d.category, []).append(amt)
                if d.vendor_name:
                    vend_amounts.setdefault(d.vendor_name, []).append(amt)
                    vend_totals[d.vendor_name] = vend_totals.get(d.vendor_name, 0.0) + amt

            # Invoice number count
            if d.invoice_number and d.invoice_number.strip():
                inv_norm = d.invoice_number.strip().upper()
                ctx.invoice_number_counts[inv_norm] = ctx.invoice_number_counts.get(inv_norm, 0) + 1

            # Financial signature count
            if d.vendor_name and d.invoice_date and amt is not None and amt > 0:
                sig_key = (d.vendor_name.strip().lower(), d.invoice_date.strip()[:10], round(amt, 2))
                ctx.financial_signature_counts[sig_key] = ctx.financial_signature_counts.get(sig_key, 0) + 1

            # Monthly spend
            if d.invoice_date and len(d.invoice_date.strip()) >= 7 and amt is not None and amt > 0:
                m_key = d.invoice_date.strip()[:7]
                ctx.monthly_spends[m_key] = ctx.monthly_spends.get(m_key, 0.0) + amt

        for cat, vals in cat_amounts.items():
            if len(vals) >= self.min_sample_size:
                ctx.category_stats[cat] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "count": len(vals),
                }

        for vend, vals in vend_amounts.items():
            ctx.vendor_stats[vend] = {
                "mean": statistics.mean(vals),
                "count": len(vals),
            }
            if total_global_spend > 0:
                ctx.vendor_spend_shares[vend] = (vend_totals[vend] / total_global_spend) * 100

        # Compute preceding rolling average (baseline from prior months)
        sorted_months = sorted(ctx.monthly_spends.keys())
        for idx, m in enumerate(sorted_months):
            if idx >= 3:
                prev_3 = [ctx.monthly_spends[sorted_months[idx - 1 - i]] for i in range(3)]
                ctx.rolling_3m_averages[m] = statistics.mean(prev_3)
            elif idx >= 1:
                prev = [ctx.monthly_spends[sorted_months[idx - 1 - i]] for i in range(idx)]
                ctx.rolling_3m_averages[m] = statistics.mean(prev)

        return ctx

    def evaluate_document(
        self,
        doc: FinancialDocument,
        ctx: Optional[PopulationContext] = None,
    ) -> DocumentAnomalyReport:
        """
        Evaluate a single document across all 10 anomaly rules and aggregate the findings.
        """
        findings: List[AnomalyFinding] = []
        amt = doc.total_amount

        # -------------------------------------------------------------------
        # ANOMALY_001: Statistical Outlier (Sigma-based)
        # -------------------------------------------------------------------
        if ctx and ctx.global_mean is not None and ctx.global_std and ctx.global_std > 0 and amt is not None:
            if ctx.sample_size >= self.min_sample_size:
                z_score = (amt - ctx.global_mean) / ctx.global_std
                if z_score >= self.sigma_threshold:
                    if z_score >= 3.0:
                        sev = "HIGH"
                        score = min(0.95, 0.85 + (z_score - 3.0) * 0.05)
                    else:
                        sev = "MEDIUM"
                        score = 0.70 + (z_score - 2.0) * 0.15
                    findings.append(
                        AnomalyFinding(
                            document_id=doc.document_id,
                            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                            severity=sev,
                            score=round(score, 3),
                            confidence=0.90,
                            explanation=(
                                f"Statistical outlier: Amount ₹{amt:,.2f} is {z_score:.2f}σ above "
                                f"mean (₹{ctx.global_mean:,.2f} ± ₹{ctx.global_std:,.2f}, N={ctx.sample_size})."
                            ),
                        )
                    )

        # -------------------------------------------------------------------
        # ANOMALY_002: Unusually High Invoice (Category or Vendor Baseline)
        # -------------------------------------------------------------------
        if ctx and amt is not None and amt > 0:
            cat_stat = ctx.category_stats.get(doc.category)
            if cat_stat and cat_stat["mean"] > 0 and amt > 3.0 * cat_stat["mean"]:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.HIGH_INVOICE_BASELINE,
                        severity="MEDIUM",
                        score=0.75,
                        confidence=0.85,
                        explanation=(
                            f"Unusually high for category '{doc.category}': Amount ₹{amt:,.2f} is "
                            f"{amt / cat_stat['mean']:.1f}x the category average (₹{cat_stat['mean']:,.2f})."
                        ),
                    )
                )
            elif doc.vendor_name and doc.vendor_name in ctx.vendor_stats:
                vend_stat = ctx.vendor_stats[doc.vendor_name]
                if vend_stat["count"] >= 2 and vend_stat["mean"] > 0 and amt > 3.0 * vend_stat["mean"]:
                    findings.append(
                        AnomalyFinding(
                            document_id=doc.document_id,
                            anomaly_type=AnomalyType.HIGH_INVOICE_BASELINE,
                            severity="MEDIUM",
                            score=0.70,
                            confidence=0.80,
                            explanation=(
                                f"Unusually high for vendor '{doc.vendor_name}': Amount ₹{amt:,.2f} is "
                                f"{amt / vend_stat['mean']:.1f}x the vendor average (₹{vend_stat['mean']:,.2f})."
                            ),
                        )
                    )

        # -------------------------------------------------------------------
        # ANOMALY_003: Round-Number Amount
        # -------------------------------------------------------------------
        if amt is not None and amt > 0 and (int(amt) % 1000 == 0) and abs(amt - round(amt)) < 0.01:
            if amt >= 50000:
                sev = "LOW"
                score = 0.50
            else:
                sev = "INFO"
                score = 0.30
            findings.append(
                AnomalyFinding(
                    document_id=doc.document_id,
                    anomaly_type=AnomalyType.ROUND_AMOUNT,
                    severity=sev,
                    score=score,
                    confidence=0.95,
                    explanation=f"Round-number amount: ₹{amt:,.2f} is an exact multiple of ₹1,000 (potential placeholder).",
                )
            )

        # -------------------------------------------------------------------
        # ANOMALY_004: Duplicate Invoice Number
        # -------------------------------------------------------------------
        if ctx and doc.invoice_number and doc.invoice_number.strip():
            inv_norm = doc.invoice_number.strip().upper()
            count = ctx.invoice_number_counts.get(inv_norm, 0)
            if count > 1:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.DUPLICATE_INVOICE_NUM,
                        severity="HIGH",
                        score=0.85,
                        confidence=0.95,
                        explanation=f"Duplicate invoice number '{doc.invoice_number}' appears on {count} documents.",
                    )
                )

        # -------------------------------------------------------------------
        # ANOMALY_005: Missing Critical Financial Fields
        # -------------------------------------------------------------------
        missing_fields = []
        if not doc.vendor_name or not doc.vendor_name.strip():
            missing_fields.append("Vendor Name")
        if not doc.invoice_date or not doc.invoice_date.strip():
            missing_fields.append("Invoice Date")
        if doc.total_amount is None:
            missing_fields.append("Total Amount")
        if not doc.currency or not doc.currency.strip():
            missing_fields.append("Currency")

        if missing_fields:
            sev = "MEDIUM" if len(missing_fields) >= 2 else "LOW"
            score = 0.65 if len(missing_fields) >= 2 else 0.40
            findings.append(
                AnomalyFinding(
                    document_id=doc.document_id,
                    anomaly_type=AnomalyType.MISSING_CRITICAL_FIELDS,
                    severity=sev,
                    score=score,
                    confidence=1.00,
                    explanation=f"Missing essential financial field(s): {', '.join(missing_fields)}.",
                )
            )

        # -------------------------------------------------------------------
        # ANOMALY_006: Tax Inconsistency (subtotal + tax != total)
        # -------------------------------------------------------------------
        if doc.subtotal is not None and doc.tax_amount is not None and amt is not None:
            expected_total = doc.subtotal + doc.tax_amount
            discrepancy = abs(expected_total - amt)
            if discrepancy > 1.0:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.TAX_INCONSISTENCY,
                        severity="MEDIUM",
                        score=0.60,
                        confidence=0.90,
                        explanation=(
                            f"Tax calculation mismatch: Subtotal (₹{doc.subtotal:,.2f}) + "
                            f"Tax (₹{doc.tax_amount:,.2f}) = ₹{expected_total:,.2f}, but Total is ₹{amt:,.2f} "
                            f"(diff: ₹{discrepancy:,.2f})."
                        ),
                    )
                )

        # -------------------------------------------------------------------
        # ANOMALY_007: Category/Value Distribution Mismatch
        # -------------------------------------------------------------------
        if ctx and amt is not None and doc.category:
            cat_stat = ctx.category_stats.get(doc.category)
            if cat_stat and cat_stat["std"] > 0:
                cat_z = (amt - cat_stat["mean"]) / cat_stat["std"]
                if cat_z >= 2.5:
                    findings.append(
                        AnomalyFinding(
                            document_id=doc.document_id,
                            anomaly_type=AnomalyType.CATEGORY_MISMATCH,
                            severity="MEDIUM",
                            score=0.70,
                            confidence=0.85,
                            explanation=(
                                f"Category distribution anomaly: Amount ₹{amt:,.2f} deviates {cat_z:.2f}σ "
                                f"from category '{doc.category}' historical mean (₹{cat_stat['mean']:,.2f})."
                            ),
                        )
                    )

        # -------------------------------------------------------------------
        # ANOMALY_008: Vendor Concentration Risk (>25% share)
        # -------------------------------------------------------------------
        if ctx and doc.vendor_name:
            share = ctx.vendor_spend_shares.get(doc.vendor_name, 0.0)
            if share > 25.0:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.VENDOR_CONCENTRATION,
                        severity="LOW",
                        score=0.50,
                        confidence=0.90,
                        explanation=(
                            f"High vendor concentration: Vendor '{doc.vendor_name}' commands "
                            f"{share:.1f}% of total institutional expenditure (threshold: 25.0%)."
                        ),
                    )
                )

        # -------------------------------------------------------------------
        # ANOMALY_009: Temporal Spending Surge
        # -------------------------------------------------------------------
        if ctx and doc.invoice_date and len(doc.invoice_date.strip()) >= 7:
            m_key = doc.invoice_date.strip()[:7]
            m_spend = ctx.monthly_spends.get(m_key, 0.0)
            roll_avg = ctx.rolling_3m_averages.get(m_key)
            if roll_avg and roll_avg > 0 and m_spend >= 1.5 * roll_avg:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.TEMPORAL_SPEND_SURGE,
                        severity="MEDIUM",
                        score=0.65,
                        confidence=0.80,
                        explanation=(
                            f"Temporal spending surge in {m_key}: Monthly expenditure (₹{m_spend:,.2f}) "
                            f"is {m_spend / roll_avg:.1f}x above the preceding baseline average (₹{roll_avg:,.2f})."
                        ),
                    )
                )

        # -------------------------------------------------------------------
        # ANOMALY_010: Repeated Financial Signature
        # -------------------------------------------------------------------
        if ctx and doc.vendor_name and doc.invoice_date and amt is not None and amt > 0:
            sig_key = (doc.vendor_name.strip().lower(), doc.invoice_date.strip()[:10], round(amt, 2))
            sig_count = ctx.financial_signature_counts.get(sig_key, 0)
            if sig_count > 1:
                findings.append(
                    AnomalyFinding(
                        document_id=doc.document_id,
                        anomaly_type=AnomalyType.REPEATED_SIGNATURE,
                        severity="HIGH",
                        score=0.80,
                        confidence=0.90,
                        explanation=(
                            f"Repeated financial signature: Same vendor ('{doc.vendor_name}'), "
                            f"date ({doc.invoice_date[:10]}), and amount (₹{amt:,.2f}) appear on {sig_count} documents."
                        ),
                    )
                )

        return self._aggregate_findings(doc.document_id, findings)

    def _aggregate_findings(
        self,
        document_id: str,
        findings: List[AnomalyFinding],
    ) -> DocumentAnomalyReport:
        """Combine multiple anomaly signals into a unified document-level report."""
        if not findings:
            return DocumentAnomalyReport(
                document_id=document_id,
                overall_score=0.0,
                severity="INFO",
                anomaly_count=0,
                primary_anomaly_type="NONE",
                anomaly_types=[],
                findings=[],
                confidence=1.0,
                explanations=["No anomalies or financial integrity signals detected."],
                recommendation="Document appears regular; no review required.",
            )

        # Multi-signal overall score: max score + 0.1 for each additional signal (capped at 1.0)
        max_score = max(f.score for f in findings)
        overall_score = min(1.0, max_score + 0.08 * (len(findings) - 1))

        # Determine overall severity
        crit_count = sum(1 for f in findings if f.severity == "CRITICAL")
        high_count = sum(1 for f in findings if f.severity == "HIGH")
        med_count = sum(1 for f in findings if f.severity == "MEDIUM")
        low_count = sum(1 for f in findings if f.severity == "LOW")

        if crit_count > 0 or high_count >= 2:
            agg_severity = "CRITICAL"
        elif high_count > 0 or med_count >= 2:
            agg_severity = "HIGH"
        elif med_count > 0 or low_count >= 2:
            agg_severity = "MEDIUM"
        elif low_count > 0:
            agg_severity = "LOW"
        else:
            agg_severity = "INFO"

        # Primary anomaly type is highest score
        primary_finding = max(findings, key=lambda f: f.score)
        avg_confidence = statistics.mean([f.confidence for f in findings])

        if agg_severity in ("CRITICAL", "HIGH"):
            recommendation = "Immediate investigation recommended — multiple or high-confidence integrity signals detected."
        elif agg_severity == "MEDIUM":
            recommendation = "Operational review recommended — one or more financial signals require human sign-off."
        else:
            recommendation = "Informational review only — minor or single low-severity signal detected."

        return DocumentAnomalyReport(
            document_id=document_id,
            overall_score=round(overall_score, 3),
            severity=agg_severity,
            anomaly_count=len(findings),
            primary_anomaly_type=primary_finding.anomaly_type,
            anomaly_types=[f.anomaly_type for f in findings],
            findings=findings,
            confidence=round(avg_confidence, 2),
            explanations=[f.explanation for f in findings],
            recommendation=recommendation,
        )
