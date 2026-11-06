import json
from pathlib import Path

for split in ["test", "train"]:
    p = Path(f"processed_data/sroie/{split}.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    total_short_matches = 0
    total_samples = len(samples)
    
    for s in samples:
        gt_total = s["entities"].get("total", "").strip()
        matched_words = [w for w, l in zip(s["words"], s["labels"]) if l == "B-TOTAL"]
        if matched_words:
            w = matched_words[0]
            if w != gt_total and len(w) <= 2:
                total_short_matches += 1
                
    print(f"SROIE {split}: Total entities where matched B-TOTAL word has length <= 2 and != gt_total: {total_short_matches} / {total_samples}")
