"""
Reprocess SROIE dataset into processed_data/sroie_corrected/ and generate audit report.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.sroie_corrected_processor import process_sroie_sample_corrected

def main():
    src_dir = Path("processed_data/sroie")
    dst_dir = Path("processed_data/sroie_corrected")
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reprocessing SROIE from {src_dir} to {dst_dir}...")
    
    audit_results = {}
    
    for split in ["train", "val", "test"]:
        src_file = src_dir / f"{split}.jsonl"
        dst_file = dst_dir / f"{split}.jsonl"
        
        with open(src_file, "r", encoding="utf-8") as f_in:
            orig_samples = [json.loads(line) for line in f_in]
            
        corrected_samples = []
        split_audit = {
            "total_samples": len(orig_samples),
            "orig_words": sum(len(s["words"]) for s in orig_samples),
            "orig_labels": Counter(l for s in orig_samples for l in s["labels"]),
            "corrected_labels": Counter(),
            "entity_coverage": defaultdict(lambda: {"success": 0, "failed": 0, "multi_token": 0}),
            "illegal_transitions": 0,
            "failed_details": [],
            "multi_token_examples": defaultdict(list),
        }
        
        with open(dst_file, "w", encoding="utf-8") as f_out:
            for s_idx, orig_s in enumerate(orig_samples):
                corr_s, info = process_sroie_sample_corrected(orig_s)
                f_out.write(json.dumps(corr_s, ensure_ascii=False) + "\n")
                corrected_samples.append(corr_s)
                
                # Labels
                for l in corr_s["labels"]:
                    split_audit["corrected_labels"][l] += 1
                    
                # Coverage
                for ent_type in ["company", "date", "address", "total"]:
                    gt_val = orig_s["entities"].get(ent_type, "").strip()
                    if gt_val:
                        if ent_type in info["aligned"]:
                            split_audit["entity_coverage"][ent_type]["success"] += 1
                            if ent_type in info["multi_token"]:
                                split_audit["entity_coverage"][ent_type]["multi_token"] += 1
                                if len(split_audit["multi_token_examples"][ent_type]) < 3:
                                    split_audit["multi_token_examples"][ent_type].append({
                                        "sample_id": orig_s["id"],
                                        "tokens": info["aligned"][ent_type]["tokens"],
                                        "target": gt_val
                                    })
                        else:
                            split_audit["entity_coverage"][ent_type]["failed"] += 1
                            if len(split_audit["failed_details"]) < 10:
                                split_audit["failed_details"].append({
                                    "sample_id": orig_s["id"],
                                    "entity_type": ent_type,
                                    "target": gt_val,
                                    "first_words": corr_s["words"][:15]
                                })
                                
                # Check BIO validity
                prev_tag = "O"
                prev_ent = None
                for l in corr_s["labels"]:
                    if l.startswith("B-"):
                        prev_tag = "B"
                        prev_ent = l[2:]
                    elif l.startswith("I-"):
                        ent = l[2:]
                        if prev_tag == "O" or prev_ent != ent:
                            split_audit["illegal_transitions"] += 1
                        prev_tag = "I"
                        prev_ent = ent
                    else:
                        prev_tag = "O"
                        prev_ent = None
                        
        split_audit["corrected_words"] = sum(len(s["words"]) for s in corrected_samples)
        audit_results[split] = split_audit
        print(f"Wrote {len(corrected_samples)} samples to {dst_file.name}")
        
    return audit_results

if __name__ == "__main__":
    results = main()
    print("\nProcessing complete. Key metrics:")
    for split, data in results.items():
        print(f"\n--- {split.upper()} ---")
        print(f"Words: {data['orig_words']} -> {data['corrected_words']}")
        print(f"Illegal BIO transitions: {data['illegal_transitions']}")
        print("Coverage:")
        for ent, c in data["entity_coverage"].items():
            tot = c["success"] + c["failed"]
            pct = (c["success"] / tot * 100) if tot > 0 else 0
            print(f"  {ent:10s}: {c['success']}/{tot} ({pct:.1f}%) | Multi-token: {c['multi_token']}")
        print("Labels:")
        for l, cnt in data["corrected_labels"].most_common():
            print(f"  {l:15s}: {cnt}")
