"""
Diagnose the SROIE label assignment bug by inspecting a few raw samples
and tracing exactly what align_sroie_entity_labels produces.
Run from project root: python scripts/diagnose_labels.py
"""
import sys
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.text_utils import clean_text

# ── Load 5 SROIE train samples ────────────────────────────────────────────
samples = []
with open(PROJECT_ROOT / "processed_data/sroie/train.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        samples.append(json.loads(line))

print("="*70)
print("SROIE entity alignment diagnostic — first 5 samples")
print("="*70)

for s in samples:
    words = s["words"]
    labels = s["labels"]
    entities = s["entities"]
    print(f"\nSample: {s['id']}")
    print(f"  entities: {entities}")
    non_o = [(w, l) for w, l in zip(words, labels) if l != "O"]
    print(f"  non-O tokens ({len(non_o)}): {non_o[:10]}")
    if not non_o:
        print("  *** ALL LABELS ARE O — alignment failed ***")

# ── Diagnose alignment logic directly ────────────────────────────────────
print("\n" + "="*70)
print("ALIGNMENT TRACE for sample 0")
print("="*70)

s = samples[0]
words = s["words"]
entities = s["entities"]

print(f"words[:20] = {words[:20]}")
print(f"entities   = {entities}")
print()

cleaned_words = [clean_text(w).lower() for w in words]

for ent_type in ["company", "address", "date", "total"]:
    ent_val = str(entities.get(ent_type, "")).strip()
    print(f"\n--- matching '{ent_type}' = '{ent_val}' ---")
    ent_tokens = [clean_text(t).lower() for t in ent_val.split() if clean_text(t).strip()]
    print(f"  entity tokens: {ent_tokens}")
    k = len(ent_tokens)
    if k == 0:
        print("  SKIP: empty entity")
        continue
    found = False
    for i in range(len(cleaned_words) - k + 1):
        window = cleaned_words[i : i + k]
        if window == ent_tokens:
            print(f"  MATCH at words[{i}:{i+k}] = {words[i:i+k]}")
            found = True
            break
    if not found:
        print(f"  NO EXACT MATCH found")
        # Try single-word check
        for ent_tok in ent_tokens:
            matches = [i for i, w in enumerate(cleaned_words) if ent_tok in w or w in ent_tok]
            if matches:
                print(f"  Partial match for '{ent_tok}': positions {matches[:5]}, words: {[words[m] for m in matches[:5]]}")

# ── Overall label distribution check ────────────────────────────────────
print("\n" + "="*70)
print("FULL TRAIN LABEL DISTRIBUTION")
print("="*70)
counter = Counter()
with open(PROJECT_ROOT / "processed_data/sroie/train.jsonl", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        counter.update(s.get("labels", []))
total = sum(counter.values())
for lbl, cnt in sorted(counter.items()):
    print(f"  {lbl:20s}: {cnt:6d}  ({100*cnt/total:.1f}%)")

print("\nConclusion:")
if counter.get("B-COMPANY", 0) < 100:
    print("  BUG: B-COMPANY is severely underrepresented — entity matching is failing.")
    print("  Expected ~533 B-COMPANY tokens (one per sample), got", counter.get("B-COMPANY", 0))
if counter.get("B-ADDRESS", 0) < 100:
    print("  BUG: B-ADDRESS is 0 or near-zero — multi-word address matching fails.")

print("\nDone.")
