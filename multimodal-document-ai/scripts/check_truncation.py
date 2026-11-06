import json
from pathlib import Path
from transformers import LayoutLMv3TokenizerFast

tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

for ds in ["sroie", "funsd", "cord"]:
    for split in ["train", "val", "test"]:
        p = Path(f"processed_data/{ds}/{split}.jsonl")
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        
        truncated_count = 0
        total = len(samples)
        for s in samples:
            words = s.get("words", [])
            bboxes = s.get("bboxes", [])
            # Fallback if empty
            if not words:
                continue
            enc = tokenizer(words, boxes=bboxes, is_split_into_words=True, add_special_tokens=True)
            if len(enc["input_ids"]) > 512:
                truncated_count += 1
                
        print(f"{ds:6s} {split:5s}: {truncated_count:3d} / {total:3d} samples ({truncated_count/total*100:4.1f}%) exceed 512 subwords")
