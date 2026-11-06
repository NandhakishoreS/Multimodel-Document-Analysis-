"""Verify SROIE label distribution after preprocessing fix."""
import json, sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for split in ["train", "val", "test"]:
    counter = Counter()
    n_samples = 0
    n_with_company = 0
    n_with_address = 0
    n_with_date = 0
    n_with_total = 0

    path = PROJECT_ROOT / f"processed_data/sroie/{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            labels = s.get("labels", [])
            counter.update(labels)
            n_samples += 1
            if "B-COMPANY" in labels: n_with_company += 1
            if "B-ADDRESS" in labels: n_with_address += 1
            if "B-DATE"    in labels: n_with_date    += 1
            if "B-TOTAL"   in labels: n_with_total   += 1

    total = sum(counter.values())
    print(f"\n=== SROIE {split.upper()} ({n_samples} samples) ===")
    for lbl in sorted(counter):
        print(f"  {lbl:20s}: {counter[lbl]:6d}  ({100*counter[lbl]/total:.1f}%)")
    print(f"  Samples with B-COMPANY : {n_with_company}/{n_samples} ({100*n_with_company/n_samples:.1f}%)")
    print(f"  Samples with B-ADDRESS : {n_with_address}/{n_samples} ({100*n_with_address/n_samples:.1f}%)")
    print(f"  Samples with B-DATE    : {n_with_date}/{n_samples}   ({100*n_with_date/n_samples:.1f}%)")
    print(f"  Samples with B-TOTAL   : {n_with_total}/{n_samples}  ({100*n_with_total/n_samples:.1f}%)")

# Show 3 example samples with all entities found
print("\n=== EXAMPLES WITH ALL ENTITIES ===")
shown = 0
with open(PROJECT_ROOT / "processed_data/sroie/train.jsonl", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        lbls = s["labels"]
        if all(f"B-{e}" in lbls for e in ("COMPANY","DATE","TOTAL")):
            pairs = [(w, l) for w, l in zip(s["words"], lbls) if l != "O"]
            print(f"  id={s['id']} | entities={s['entities']}")
            print(f"    labelled tokens: {pairs[:8]}")
            shown += 1
        if shown >= 3:
            break
