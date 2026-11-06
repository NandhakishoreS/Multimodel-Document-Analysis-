import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
from scripts.test_corrected_alignment import split_line_into_words_with_bboxes, align_sroie_corrected

with open("processed_data/sroie/test.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

failed_addresses = []
for s in samples:
    expanded_words = []
    expanded_boxes = []
    for w, b in zip(s["words"], s["bboxes"]):
        w_list, b_list = split_line_into_words_with_bboxes(w, b)
        expanded_words.extend(w_list)
        expanded_boxes.extend(b_list)
        
    labels, info = align_sroie_corrected(expanded_words, s["entities"])
    if "address" in info["failed"]:
        failed_addresses.append({
            "id": s["id"],
            "raw_address": s["entities"]["address"],
            "words_sample": expanded_words[:30]
        })

print(f"Total failed addresses: {len(failed_addresses)}")
print("First 5 failed address samples:")
for item in failed_addresses[:5]:
    print(f"\nSample {item['id']}:")
    print(f"  GT Address: {item['raw_address']!r}")
    print(f"  First 25 words on receipt: {item['words_sample'][:25]}")
