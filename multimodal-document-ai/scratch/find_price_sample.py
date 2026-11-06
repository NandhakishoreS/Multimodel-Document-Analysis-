import json
from pathlib import Path

with open("processed_data/cord/test.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "50.00" in line or "20.00" in line or "50,00" in line:
            data = json.loads(line)
            print(f"Found in sample {i} (id={data['id']}):")
            print("Words:", [w for w in data["words"] if any(c.isdigit() for c in w)])
            digits = [w for w in data["words"] if w in ["1", "2", "3", "4"]]
            print("Row digits in ground truth:", digits)
