import os
import sys
import importlib
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root))

print("=== 1. TESTING IMPORTABILITY OF ALL CORE MODULES ===")
modules_to_test = [
    "models.layoutlmv3.config",
    "models.layoutlmv3.model",
    "models.layoutlmv3.tokenizer",
    "models.layoutlmv3.dataset",
    "preprocessing.bbox_utils",
    "preprocessing.text_utils",
    "preprocessing.cord_processor",
    "preprocessing.funsd_processor",
    "preprocessing.sroie_processor",
    "preprocessing.sroie_corrected_processor",
    "preprocessing.unified_dataset",
    "inference.schemas",
    "inference.postprocessing",
    "inference.final_inference",
    "visualization.draw_predictions",
    "evaluation.metrics",
    "training.phase9_augmentation",
]

failed_imports = []
for m in modules_to_test:
    try:
        importlib.import_module(m)
        print(f"  [OK] {m}")
    except Exception as e:
        print(f"  [FAIL] {m}: {e}")
        failed_imports.append((m, str(e)))

print(f"\nTotal import failures: {len(failed_imports)}")

print("\n=== 2. SCANNING FOR HARD-CODED MACHINE PATHS IN CODE ===")
py_files = []
for d in ["models", "preprocessing", "inference", "visualization", "evaluation", "training", "app"]:
    py_files.extend(list(Path(d).glob("**/*.py")))

hardcoded_findings = []
for p in py_files:
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if "C:\\Users\\" in line or "c:\\mda" in line.lower() or "/home/" in line:
            # Check if it's just a docstring or comment
            hardcoded_findings.append((str(p.as_posix()), idx + 1, line.strip()[:100]))

print(f"Hard-coded path occurrences in core modules: {len(hardcoded_findings)}")
for f in hardcoded_findings[:15]:
    print(f"  {f[0]}:{f[1]} -> {f[2]}")
