import json
from pathlib import Path
from collections import Counter

# Inspect SROIE processed data
for split in ["train", "val", "test"]:
    p = Path(f"processed_data/sroie/{split}.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    multi_word_tokens = 0
    total_tokens = 0
    i_labels = 0
    b_labels = 0
    
    for s in samples:
        words = s["words"]
        labels = s["labels"]
        for w, l in zip(words, labels):
            total_tokens += 1
            if len(w.split()) > 1:
                multi_word_tokens += 1
            if l.startswith("I-"):
                i_labels += 1
            elif l.startswith("B-"):
                b_labels += 1
                
    print(f"SROIE {split}:")
    print(f"  Total words (entries): {total_tokens}")
    print(f"  Entries containing multiple whitespace-separated words: {multi_word_tokens} ({multi_word_tokens/total_tokens*100:.1f}%)")
    print(f"  B- labels: {b_labels}")
    print(f"  I- labels: {i_labels}")

# Look at 5 samples from SROIE test
with open("processed_data/sroie/test.jsonl", "r", encoding="utf-8") as f:
    for i in range(5):
        s = json.loads(f.readline())
        print(f"\nSample {s['id']}:")
        print("  Entities dict:", s["entities"])
        for idx, (w, l) in enumerate(zip(s["words"], s["labels"])):
            if l != "O":
                print(f"    [{idx}] {l:12s} | '{w}'")
