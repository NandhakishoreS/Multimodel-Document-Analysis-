"""
Phase 4 diagnostic script.
Checks: label distribution, token alignment, word_ids, model config, GPU, overfit test.
Run from project root:
    python scripts/diagnose_phase4.py
"""
import json
import sys
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ─────────────────────────────────────────────────────────────────────────────
# 1. SROIE label distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. SROIE LABEL DISTRIBUTION")
print("="*60)

counter = Counter()
samples_with_entity = 0
all_samples = []
with open(PROJECT_ROOT / "processed_data/sroie/train.jsonl", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        labels = s.get("labels", [])
        counter.update(labels)
        if any(l != "O" for l in labels):
            samples_with_entity += 1
        all_samples.append(s)

total = sum(counter.values())
print(f"Total tokens: {total}, samples: {len(all_samples)}, samples_with_entity: {samples_with_entity}")
for lbl, cnt in sorted(counter.items()):
    print(f"  {lbl:20s}: {cnt:6d}  ({100*cnt/total:.1f}%)")

# Show first 3 samples with entity labels
print("\nSample entity tokens:")
shown = 0
for s in all_samples:
    entity_toks = [(w, l) for w, l in zip(s["words"], s["labels"]) if l != "O"]
    if entity_toks:
        print(f"  id={s['id']} -> {entity_toks[:6]}")
        shown += 1
    if shown >= 3:
        break

# ─────────────────────────────────────────────────────────────────────────────
# 2. GPU check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. HARDWARE CHECK")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("WARNING: No CUDA device found")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Token alignment diagnostic
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. TOKEN ALIGNMENT DIAGNOSTIC")
print("="*60)

from models.layoutlmv3.config import load_dataset_config
from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper, align_labels_to_tokens

dataset_cfg = load_dataset_config("sroie")
print(f"num_labels: {dataset_cfg.num_labels}")
print(f"label2id: {dataset_cfg.label2id}")

tokenizer_wrapper = LayoutLMv3TokenizerWrapper.from_pretrained(
    "microsoft/layoutlmv3-base", max_seq_length=512
)

# Take a sample WITH entity labels
sample_with_entity = None
for s in all_samples:
    if any(l != "O" for l in s["labels"]):
        sample_with_entity = s
        break

print(f"\nDiagnosing sample: id={sample_with_entity['id']}")
print(f"Words[:10]: {sample_with_entity['words'][:10]}")
print(f"Labels[:10]: {sample_with_entity['labels'][:10]}")
entity_count_in_sample = sum(1 for l in sample_with_entity["labels"] if l != "O")
print(f"Non-O labels in sample: {entity_count_in_sample} / {len(sample_with_entity['labels'])}")

# Encode sample
encoding = tokenizer_wrapper.encode_sample(
    sample=sample_with_entity,
    label2id=dataset_cfg.label2id,
    project_root=PROJECT_ROOT,
)

word_ids = encoding["word_ids"]
label_tensor = encoding["labels"].squeeze(0).tolist()
input_ids = encoding["input_ids"].squeeze(0).tolist()
attn = encoding["attention_mask"].squeeze(0).tolist()

print(f"\nSequence length: {len(label_tensor)}")
print(f"Non-padding tokens: {sum(attn)}")
non_ignore = [l for l in label_tensor if l != -100]
print(f"Non-ignore labels in tensor: {len(non_ignore)}")
label_id_counter = Counter(label_tensor)
print(f"Label id distribution in tensor (first token rule):")
for lid, cnt in sorted(label_id_counter.items()):
    lbl = dataset_cfg.id2label.get(lid, f"ignore({lid})")
    print(f"  {lbl:20s} (id={lid}): {cnt}")

# Check: do word_ids produce the right word-level labels?
from training.train_layoutlmv3 import _reconstruct_word_ids_from_labels, _extract_true_word_labels
word_ids_approx = _reconstruct_word_ids_from_labels(label_tensor, attn)
true_word_labels = _extract_true_word_labels(label_tensor, word_ids_approx, dataset_cfg.id2label)

print(f"\nWord-level labels recovered: {len(true_word_labels)}")
print(f"Original word-level labels: {len(sample_with_entity['labels'])}")
non_o_original = sum(1 for l in sample_with_entity["labels"] if l != "O")
non_o_recovered = sum(1 for l in true_word_labels if l != "O")
print(f"Non-O original: {non_o_original}")
print(f"Non-O recovered: {non_o_recovered}")
print(f"Original labels: {sample_with_entity['labels'][:20]}")
print(f"Recovered labels: {true_word_labels[:20]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Identify the root cause of 0 F1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. ROOT CAUSE ANALYSIS")
print("="*60)

# The _reconstruct_word_ids_from_labels heuristic:
# - label_ids[i] == -100 AND attention_mask==1 → continuation subword OR special token
# - Problem: [CLS] token also has label=-100 and attention=1 → current_word stays -1,
#   gets None (correct). BUT [SEP] token has label=-100 and attention=1 AND
#   current_word >= 0, so it gets word_id=current_word (WRONG — it's a special token)

# Also: the first real token (after CLS) with label=-100 but attention=1
# will be treated as continuation of "previous word" when current_word==0

# Let's trace through the first 20 positions
print("\nDetailed trace of first 20 positions:")
print(f"{'pos':>4} {'attn':>5} {'label_id':>9} {'word_id_orig':>12} {'word_id_approx':>14} {'label_str':>15}")

# Get real word_ids for comparison
real_word_ids = word_ids  # from the actual processor output
for i in range(min(20, len(attn))):
    a = attn[i]
    l = label_tensor[i]
    real_wid = real_word_ids[i]
    approx_wid = word_ids_approx[i]
    lbl_str = dataset_cfg.id2label.get(l, f"-100") if l != -100 else "-100"
    mismatch = " ← MISMATCH" if real_wid != approx_wid else ""
    print(f"  {i:4d} {a:5d} {l:9d} {str(real_wid):>12} {str(approx_wid):>14} {lbl_str:>15}{mismatch}")

mismatches = sum(1 for r, a in zip(real_word_ids, word_ids_approx) if r != a and a is not None)
total_real_words = sum(1 for w in real_word_ids if w is not None)
print(f"\nTotal word token positions: {total_real_words}")
print(f"word_id mismatches (heuristic vs real): {mismatches}")

print("\n*** DIAGNOSIS ***")
if mismatches > 0:
    print("BUG FOUND: _reconstruct_word_ids_from_labels produces wrong word_ids.")
    print("The heuristic incorrectly assigns word_ids to SEP and padding tokens.")
    print("This causes _extract_true_word_labels to produce wrong label sequences,")
    print("which makes seqeval F1 = 0 even when the model makes correct predictions.")
else:
    print("word_ids appear to match. The bug may be elsewhere.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Test the FIX: store word_ids in the Dataset and pass them through
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. PROPOSED FIX VERIFICATION")
print("="*60)

# Use the REAL word_ids from the tokenizer to extract true labels
true_labels_with_real_wids = _extract_true_word_labels(
    label_tensor, real_word_ids, dataset_cfg.id2label
)
print(f"True word labels using REAL word_ids: {len(true_labels_with_real_wids)}")
print(f"Non-O with real word_ids: {sum(1 for l in true_labels_with_real_wids if l != 'O')}")
print(f"Labels: {true_labels_with_real_wids[:20]}")
print(f"Original: {sample_with_entity['labels'][:20]}")

print("\nDone.")
