import sys
from pathlib import Path
import json

_proj = str(Path('.').resolve())
_removed = [p for p in list(sys.path) if p == _proj or p == '']
for _p in _removed:
    sys.path.remove(_p)
sys.modules.pop('datasets', None)
import datasets as hf_datasets

ds_cord = hf_datasets.load_dataset('naver-clova-ix/cord-v2', split='test')
print(f"Total test samples: {len(ds_cord)}")

all_categories = set()
lines_count = 0
repeating_symbols = set()

for idx, sample in enumerate(ds_cord):
    gt = sample['ground_truth']
    gt_json = json.loads(gt) if isinstance(gt, str) else gt
    for line in gt_json.get('valid_line', []):
        cat = line.get('category')
        all_categories.add(cat)
        lines_count += 1
    for k in gt_json.keys():
        if k != 'valid_line':
            repeating_symbols.add(k)

print(f"Total valid_lines in test: {lines_count}")
print(f"Other keys in gt_json: {repeating_symbols}")
print(f"All raw categories in test ({len(all_categories)}):")
for cat in sorted(all_categories):
    print(f"  {cat}")
