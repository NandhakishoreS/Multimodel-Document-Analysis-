import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

def normalize_text(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def extract_numbers(s):
    # Find all float/currency numbers like 193.00, 193, 1,234.50
    return re.findall(r"\d+(?:[.,]\d+)?", s)

# Load test samples
with open("processed_data/sroie/test.jsonl", "r", encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f]

print(f"Loaded {len(test_samples)} test samples")

# Check TOTAL representation in words
total_exact_match = 0
total_in_word = 0
total_currency_prefix = 0
total_not_found = 0

for s in test_samples:
    gt_total = s["entities"].get("total", "").strip()
    norm_gt = normalize_text(gt_total)
    if not norm_gt:
        continue
    
    words = s["words"]
    norm_words = [normalize_text(w) for w in words]
    
    # Check if exact match
    if norm_gt in norm_words:
        total_exact_match += 1
    elif any(re.search(r'(?:^|[^\d.])' + re.escape(norm_gt) + r'(?:[^\d.]|$)', nw) for nw in norm_words):
        total_in_word += 1
    else:
        # Check clean numeric
        clean_gt = re.sub(r"[^\d.]", "", norm_gt)
        found = False
        for nw in norm_words:
            clean_nw = re.sub(r"[^\d.]", "", nw)
            if clean_gt and clean_gt == clean_nw:
                found = True
                break
        if found:
            total_currency_prefix += 1
        else:
            total_not_found += 1

print("TOTAL matching in test set:")
print(f"  Exact word match: {total_exact_match} / 361")
print(f"  Word boundary contains match (e.g. 'total: 193.00'): {total_in_word} / 361")
print(f"  Clean numeric match (e.g. 'rm193.00'): {total_currency_prefix} / 361")
print(f"  Not found by numeric rules: {total_not_found} / 361")
