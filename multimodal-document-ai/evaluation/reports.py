"""Report compilation and visualization for baseline evaluations."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from utils.visualization import plot_f1_comparison

logger = logging.getLogger(__name__)


def save_baseline_results(
    results_list: List[Dict[str, Any]],
    output_dir: str = "reports",
) -> Path:
    """
    Save baseline evaluation results to CSV and JSON formats.

    Args:
        results_list: List of result records with keys:
            ['dataset', 'model_type', 'split', 'macro_f1', 'micro_f1', 'weighted_f1', 'precision', 'recall', 'cer', 'wer']
        output_dir: Destination directory.

    Returns:
        Path to generated CSV report.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df = pd.DataFrame(results_list)
    csv_file = out_path / "baseline_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved baseline results CSV to {csv_file}")

    # Save JSON
    json_file = out_path / "baseline_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    logger.info(f"Saved baseline results JSON to {json_file}")

    # Generate F1 comparison plot
    f1_comparison_dict: Dict[str, Dict[str, float]] = {}
    for res in results_list:
        ds = res.get("dataset", "Unknown").upper()
        m_type = res.get("model_type", "")
        f1_val = res.get("macro_f1", 0.0)

        if ds not in f1_comparison_dict:
            f1_comparison_dict[ds] = {}

        if "Text-Only" in m_type or "text_baseline" in m_type.lower():
            f1_comparison_dict[ds]["Text-Only"] = f1_val
        elif "Layout" in m_type or "layout_baseline" in m_type.lower():
            f1_comparison_dict[ds]["Text+Layout"] = f1_val

    if f1_comparison_dict:
        plot_f1_comparison(
            f1_comparison_dict,
            title="Baseline Performance: Text-Only vs. Text+Layout",
            save_path=out_path / "baseline_f1_comparison.png",
        )

    return csv_file
