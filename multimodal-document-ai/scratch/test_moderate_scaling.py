import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

for sample_id in ["cord_test_6", "cord_test_35", "cord_test_92", "cord_test_97", "cord_test_49"]:
    img = Image.open(f"processed_data/cord/images/{sample_id}.png").convert("RGB")
    w, h = img.size
    print(f"\n==================== {sample_id} ({w}x{h}) ====================")
    
    # Check 1.0x (original)
    res_10, _ = ocr(np.array(img))
    words_10 = []
    for p, t, c in (res_10 or []):
        words_10.extend(t.split())
    print("1.0x words with digits:", [w for w in words_10 if any(ch.isdigit() for ch in w) and len(w) <= 4])
    
    # Check 1.25x
    img_125 = img.resize((int(w * 1.25), int(h * 1.25)), Image.Resampling.BICUBIC)
    res_125, _ = ocr(np.array(img_125))
    words_125 = []
    for p, t, c in (res_125 or []):
        words_125.extend(t.split())
    print("1.25x words with digits:", [w for w in words_125 if any(ch.isdigit() for ch in w) and len(w) <= 4])

    # Check 1.5x
    img_15 = img.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.BICUBIC)
    res_15, _ = ocr(np.array(img_15))
    words_15 = []
    for p, t, c in (res_15 or []):
        words_15.extend(t.split())
    print("1.5x words with digits:", [w for w in words_15 if any(ch.isdigit() for ch in w) and len(w) <= 4])
