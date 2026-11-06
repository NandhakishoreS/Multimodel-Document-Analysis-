import json
import re
import unicodedata
from collections import Counter

def normalize_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    # collapse punctuation/whitespace
    s = re.sub(r"[,\-\.:;/\\()'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

with open("processed_data/sroie/test.jsonl", "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f]

for ent_type in ["company", "date", "address"]:
    exact_count = 0
    multi_line_count = 0
    missing_count = 0
    
    for s in test_samples:
        gt_val = s["entities"].get(ent_type, "").strip()
        if not gt_val:
            continue
        norm_gt = normalize_text(gt_val)
        gt_words = set(norm_gt.split())
        
        words = s["words"]
        norm_words = [normalize_text(w) for w in words]
        
        # Check if any word equals norm_gt
        if norm_gt in norm_words:
            exact_count += 1
        else:
            # Check how many lines contain subsets of gt_words
            matching_lines = [i for i, nw in enumerate(norm_words) if nw and set(nw.split()).issubset(gt_words) and len(nw.split()) >= 1]
            if len(matching_lines) >= 1:
                multi_line_count += 1
            else:
                missing_count += 1
                
    print(f"--- {ent_type.upper()} in test (total {len(test_samples)}) ---")
    print(f"  Single exact line match: {exact_count}")
    print(f"  Multi-line / partial line matches: {multi_line_count}")
    print(f"  No match found: {missing_count}")
