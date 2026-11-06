import json
from pathlib import Path

test_file = Path("processed_data/cord/test.jsonl")
found = 0
with open(test_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        words = data.get("words", [])
        labels = data.get("labels", [])
        bboxes = data.get("bboxes", [])
        
        # Look for instances of '1', '2', '3', '4'
        digit_indices = [idx for idx, w in enumerate(words) if w in ["1", "2", "3", "4"]]
        if len(digit_indices) >= 2:
            print(f"Sample {i} (id={data.get('id')}, image={data.get('image_path')}):")
            print(f"Total words: {len(words)}")
            for idx in digit_indices:
                print(f"  word[{idx}] = '{words[idx]}' (label={labels[idx]}, bbox={bboxes[idx]})")
            found += 1
            if found >= 5:
                break
