import sys
from pathlib import Path
import json
from collections import Counter

_proj = str(Path('.').resolve())
_removed = [p for p in list(sys.path) if p == _proj or p == '']
for _p in _removed:
    sys.path.remove(_p)
sys.modules.pop('datasets', None)
import datasets as hf_datasets

ds_funsd = hf_datasets.load_dataset('nielsr/funsd')
print("FUNSD HF splits:", ds_funsd.keys())
print("FUNSD features:", ds_funsd['train'].features)

raw_train_tags = Counter()
for s in ds_funsd['train']:
    raw_train_tags.update(s['ner_tags'])

print("Raw train ner_tags distribution:", raw_train_tags)

# Compare with our processed_data/funsd/train.jsonl
with open("processed_data/funsd/train.jsonl", "r", encoding="utf-8") as f:
    proc_samples = [json.loads(line) for line in f]

proc_tags = Counter()
for s in proc_samples:
    proc_tags.update(s['labels'])

print("Processed train labels distribution:", proc_tags)
