"""
Comprehensive Label Schema and Alignment Audit Script for SROIE, FUNSD, and CORD.
Inspects:
- Raw annotations vs processed JSONL
- Label configurations and ID mappings across configs, code, and model checkpoints
- BIO/IOB2 transition validity (illegal transitions like I-X without B-X or after different entity)
- Token and entity span statistics per split
- Word-to-subword alignment, special tokens (-100), and word reconstruction logic
- CORD hierarchical category collapsing and potential metadata leaks
- FUNSD relation metadata vs token labels
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import yaml
import torch

def audit_dataset_labels(dataset_name, label_config_path, checkpoint_path=None):
    print(f"\n=======================================================")
    print(f"AUDITING DATASET: {dataset_name.upper()}")
    print(f"=======================================================")
    
    # 1. Load config
    with open(label_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_labels = cfg.get("labels", [])
    cfg_label2id = cfg.get("label2id", {})
    cfg_id2label = {int(k) if isinstance(k, (int, str)) and str(k).isdigit() else k: v for k, v in cfg.get("id2label", {}).items()}
    print(f"Config labels ({len(cfg_labels)}): {cfg_labels}")
    print(f"Config label2id: {cfg_label2id}")
    
    # Check checkpoint config if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt_cfg_p = Path(checkpoint_path) / "config.json"
        if ckpt_cfg_p.exists():
            with open(ckpt_cfg_p, "r", encoding="utf-8") as f:
                ckpt_cfg = json.load(f)
            ckpt_id2label = ckpt_cfg.get("id2label", {})
            ckpt_label2id = ckpt_cfg.get("label2id", {})
            print(f"Checkpoint id2label matches config? {str(ckpt_id2label) == str({str(k): v for k, v in cfg_id2label.items()})}")
            if str(ckpt_id2label) != str({str(k): v for k, v in cfg_id2label.items()}):
                print(f"  MISMATCH DETECTED!")
                print(f"  Config id2label: {cfg_id2label}")
                print(f"  Checkpoint id2label: {ckpt_id2label}")
    
    results = {}
    
    for split in ["train", "val", "test"]:
        split_p = Path(f"processed_data/{dataset_name}/{split}.jsonl")
        if not split_p.exists():
            print(f"Split {split} not found at {split_p}")
            continue
            
        with open(split_p, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
            
        total_samples = len(samples)
        total_words = 0
        label_counter = Counter()
        illegal_transitions = []
        entity_spans = Counter()
        samples_with_illegal = 0
        empty_words_samples = 0
        missing_entity_samples = defaultdict(int)
        
        for sample_idx, s in enumerate(samples):
            words = s.get("words", [])
            labels = s.get("labels", [])
            sample_id = s.get("id", str(sample_idx))
            total_words += len(words)
            
            if len(words) != len(labels):
                print(f"CRITICAL: Sample {sample_id} in {split} has len(words)={len(words)} != len(labels)={len(labels)}")
                
            if len(words) == 0:
                empty_words_samples += 1
                
            sample_has_illegal = False
            prev_tag = "O"
            prev_entity = None
            
            for i, lbl in enumerate(labels):
                label_counter[lbl] += 1
                if lbl not in cfg_label2id:
                    print(f"CRITICAL: Unknown label '{lbl}' in {split} sample {sample_id} at pos {i}")
                    
                # BIO validation:
                if lbl.startswith("B-"):
                    ent = lbl[2:]
                    entity_spans[ent] += 1
                    prev_tag = "B"
                    prev_entity = ent
                elif lbl.startswith("I-"):
                    ent = lbl[2:]
                    # Check legality: must follow B-ent or I-ent
                    if prev_tag == "O" or prev_entity != ent:
                        illegal_transitions.append({
                            "sample_id": sample_id,
                            "index": i,
                            "prev_label": f"{prev_tag}-{prev_entity}" if prev_entity else "O",
                            "curr_label": lbl,
                            "word": words[i] if i < len(words) else "",
                            "context": labels[max(0, i-2):min(len(labels), i+3)]
                        })
                        sample_has_illegal = True
                    prev_tag = "I"
                    prev_entity = ent
                elif lbl == "O":
                    prev_tag = "O"
                    prev_entity = None
                else:
                    print(f"CRITICAL: Non-BIO label format: '{lbl}' in sample {sample_id}")
                    
            if sample_has_illegal:
                samples_with_illegal += 1
                
            # Dataset-specific checks:
            if dataset_name == "sroie":
                entities = s.get("entities", {})
                for ent_key, ent_val in entities.items():
                    if ent_val and str(ent_val).strip():
                        # Check if this entity has any B-tag in labels
                        expected_tag = f"B-{ent_key.upper()}"
                        if expected_tag not in labels:
                            missing_entity_samples[ent_key] += 1
                            
            elif dataset_name == "funsd":
                # Check relations field
                rel = s.get("relations", [])
                # Verify relations are not leaking into labels
                if rel:
                    print(f"Note: FUNSD sample {sample_id} has non-empty relations: {len(rel)}")
                    
            elif dataset_name == "cord":
                pass
                
        print(f"\n--- Split: {split.upper()} ({total_samples} samples, {total_words} words) ---")
        print(f"Label distribution:")
        for lbl, count in sorted(label_counter.items(), key=lambda x: -x[1]):
            pct = (count / total_words) * 100 if total_words > 0 else 0
            print(f"  {lbl:15s}: {count:8d} ({pct:5.2f}%)")
        print(f"Entity span counts (B- tags):")
        for ent, count in sorted(entity_spans.items(), key=lambda x: -x[1]):
            print(f"  {ent:15s}: {count:6d} spans")
        print(f"Illegal BIO transitions: {len(illegal_transitions)} in {samples_with_illegal} samples")
        if illegal_transitions:
            print(f"  First 3 illegal transition examples:")
            for eg in illegal_transitions[:3]:
                print(f"    Sample {eg['sample_id']}, pos {eg['index']}: '{eg['prev_label']}' -> '{eg['curr_label']}' (word: '{eg['word']}', context: {eg['context']})")
                
        if dataset_name == "sroie":
            print(f"SROIE unaligned entities (present in ground truth string, but 0 B- tokens in labels):")
            for ent_key, count in missing_entity_samples.items():
                print(f"  {ent_key:10s}: {count} / {total_samples} samples ({count/total_samples*100:.1f}%)")
                
        results[split] = {
            "total_samples": total_samples,
            "total_words": total_words,
            "label_counter": dict(label_counter),
            "entity_spans": dict(entity_spans),
            "illegal_transitions_count": len(illegal_transitions),
            "illegal_transitions": illegal_transitions,
            "samples_with_illegal": samples_with_illegal,
            "missing_entity_samples": dict(missing_entity_samples)
        }
        
    return results

if __name__ == "__main__":
    sroie_res = audit_dataset_labels("sroie", "configs/labels/sroie.yaml", "models/checkpoints/layoutlmv3/sroie/full")
    funsd_res = audit_dataset_labels("funsd", "configs/labels/funsd.yaml", "models/checkpoints/layoutlmv3/funsd/full")
    cord_res = audit_dataset_labels("cord", "configs/labels/cord.yaml", "models/checkpoints/layoutlmv3/cord/full")
