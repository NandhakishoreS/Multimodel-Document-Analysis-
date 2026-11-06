"""
Generate reports/final_results.csv and reports/final_results.md.
Run after all training and evaluation is complete.

    python scripts/final_results_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS = PROJECT_ROOT / "reports"
EXP_DIR = PROJECT_ROOT / "experiments"


def load_experiment_summaries() -> List[Dict[str, Any]]:
    summaries = []
    for exp_dir in sorted(EXP_DIR.glob("layoutlmv3_*_full_*")):
        s_path = exp_dir / "training_summary.json"
        if s_path.exists():
            with open(s_path) as f:
                summaries.append(json.load(f))
    return summaries


def build_final_results() -> pd.DataFrame:
    rows = []

    # Phase 3 baselines from existing CSV
    baseline_csv = REPORTS / "baseline_results.csv"
    if baseline_csv.exists():
        df_b = pd.read_csv(baseline_csv)
        for _, r in df_b.iterrows():
            rows.append({
                "Model":       r.get("model_type", ""),
                "Dataset":     r.get("dataset", ""),
                "Split":       r.get("split", "test"),
                "Macro F1":    round(float(r.get("macro_f1", 0) or 0), 4),
                "Micro F1":    round(float(r.get("micro_f1", 0) or 0), 4),
                "Precision":   round(float(r.get("precision", 0) or 0), 4),
                "Recall":      round(float(r.get("recall", 0) or 0), 4),
                "Epochs":      "—",
                "LR":          "—",
                "Batch":       "—",
                "GPU":         "—",
                "Time (s)":    "—",
            })

    # LayoutLMv3 results
    for ds in ["sroie", "funsd", "cord"]:
        per_entity = REPORTS / f"layoutlmv3_{ds}_per_entity.json"
        if per_entity.exists():
            with open(per_entity) as f:
                data = json.load(f)
            seq = data.get("seqeval", {})
            # Find matching experiment summary
            exp_meta = {}
            for exp_dir in sorted(EXP_DIR.glob(f"layoutlmv3_{ds}_full_*")):
                sp = exp_dir / "training_summary.json"
                if sp.exists():
                    with open(sp) as f2:
                        exp_meta = json.load(f2)
                    break

            rows.append({
                "Model":     f"LayoutLMv3-base (full)",
                "Dataset":   ds.upper(),
                "Split":     "test",
                "Macro F1":  seq.get("macro_f1", 0.0),
                "Micro F1":  seq.get("micro_f1", 0.0),
                "Precision": seq.get("precision", 0.0),
                "Recall":    seq.get("recall", 0.0),
                "Epochs":    exp_meta.get("epochs_trained", "—"),
                "LR":        exp_meta.get("config", {}).get("learning_rate", "—"),
                "Batch":     exp_meta.get("config", {}).get("effective_batch_size", "—"),
                "GPU":       exp_meta.get("hardware", {}).get("gpu_name", "cpu"),
                "Time (s)":  exp_meta.get("training_time_s", "—"),
            })

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Final Results — Multimodal Document Understanding",
        "",
        "## Table 1 — Overall Model Comparison (Test Set)",
        "",
        df.to_markdown(index=False),
        "",
        "## Table 2 — Per-Class Performance (LayoutLMv3)",
        "",
    ]

    for ds in ["sroie", "funsd", "cord"]:
        per_entity = REPORTS / f"layoutlmv3_{ds}_per_entity.json"
        if not per_entity.exists():
            continue
        with open(per_entity) as f:
            data = json.load(f)
        pe = data.get("seqeval", {}).get("per_entity", {})
        if not pe:
            pe = data.get("token_level", {}).get("per_class", {})
        if pe:
            lines.append(f"### {ds.upper()}")
            lines.append("")
            rows = [{"Class": c, **{k.capitalize(): round(v, 4) for k, v in m.items()}}
                    for c, m in pe.items()]
            lines.append(pd.DataFrame(rows).to_markdown(index=False))
            lines.append("")

    lines += [
        "## Table 3 — Ablation Study",
        "",
        "| Input Modality | Model | SROIE Macro F1 | FUNSD Macro F1 | CORD Macro F1 |",
        "|---|---|---|---|---|",
        "| Text only | TF-IDF + LR | 0.262 | 0.351 | 0.327 |",
        "| Text + Layout (bboxes) | TF-IDF + LR | 0.266 | 0.500 | 0.377 |",
        "| Text + Layout + Image | LayoutLMv3-base | *(see Table 1)* | *(see Table 1)* | *(see Table 1)* |",
        "",
        "> Note: Text-only and Text+Layout are token-level sklearn F1.",
        "> LayoutLMv3 uses seqeval entity-span F1. Different metrics are not directly comparable.",
        "  ",
        "## Table 4 — Training Configuration",
        "",
    ]

    cfg_rows = []
    for exp_dir in sorted(EXP_DIR.glob("layoutlmv3_*_full_*")):
        sp = exp_dir / "training_summary.json"
        if not sp.exists():
            continue
        with open(sp) as f:
            s = json.load(f)
        c = s.get("config", {})
        h = s.get("hardware", {})
        cfg_rows.append({
            "Dataset":  s.get("dataset", ""),
            "Epochs":   s.get("epochs_trained", ""),
            "LR":       c.get("learning_rate", ""),
            "Eff.Batch":c.get("effective_batch_size", ""),
            "fp16":     c.get("fp16", ""),
            "GPU":      h.get("gpu_name", "cpu"),
            "Time(s)":  s.get("training_time_s", ""),
            "Best Val F1": s.get("best_val_macro_f1", ""),
        })
    if cfg_rows:
        lines.append(pd.DataFrame(cfg_rows).to_markdown(index=False))
    else:
        lines.append("*(Run full training to populate this table)*")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Building final results report...")
    df = build_final_results()
    df.to_csv(REPORTS / "final_results.csv", index=False)
    print(f"Saved: {REPORTS / 'final_results.csv'}")

    md_text = write_markdown(df)
    md_path = REPORTS / "final_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved: {md_path}")

    print("\nFull results table:")
    print(df.to_string(index=False))
