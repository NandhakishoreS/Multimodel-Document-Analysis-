import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

out_records = {}

for img_name in ["cord_test_6.png", "cord_test_4.png", "cord_test_2.png", "cord_test_8.png", "cord_test_0.png"]:
    img_path = Path("processed_data/cord/images") / img_name
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    rec = {"size": [w, h], "tests": {}}
    
    # 1. Baseline OCR
    res_base, _ = ocr(np.array(img))
    rec["tests"]["baseline"] = [
        {"poly": np.array(p).tolist(), "text": str(t), "conf": float(c)}
        for p, t, c in (res_base or [])
    ]
    
    # 2. Upscale 2.0x
    img_20 = img.resize((int(w * 2.0), int(h * 2.0)), Image.Resampling.LANCZOS)
    res_20, _ = ocr(np.array(img_20))
    rec["tests"]["upscale_2x"] = [
        {"poly": np.array(p).tolist(), "text": str(t), "conf": float(c)}
        for p, t, c in (res_20 or [])
    ]

    # 3. Upscale 3.0x
    img_30 = img.resize((int(w * 3.0), int(h * 3.0)), Image.Resampling.LANCZOS)
    res_30, _ = ocr(np.array(img_30))
    rec["tests"]["upscale_3x"] = [
        {"poly": np.array(p).tolist(), "text": str(t), "conf": float(c)}
        for p, t, c in (res_30 or [])
    ]

    # 4. RapidOCR with det_limit_side_len = max(w, h) or higher
    # In RapidOCR, can we pass det_limit_side_len or parameters?
    # Let's check if we can pass det options or tune RapidOCR
    out_records[img_name] = rec

with open("scratch/ocr_results.json", "w", encoding="utf-8") as f:
    json.dump(out_records, f, indent=2, ensure_ascii=False)

print("Saved scratch/ocr_results.json successfully!")
