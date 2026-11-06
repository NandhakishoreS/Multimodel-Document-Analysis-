from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import numpy as np

ocr = RapidOCR()
for sample_id in ["cord_test_4", "cord_test_49", "cord_test_92", "cord_test_35", "cord_test_44", "cord_test_97"]:
    img_path = f"processed_data/cord/images/{sample_id}.png"
    img = Image.open(img_path).convert("RGB")
    res, _ = ocr(np.array(img))
    print(f"\n==================== {sample_id} ====================")
    for i, (poly, text, conf) in enumerate(res or []):
        print(f"  {i:2d}: conf={conf:.3f} | text=\"{text}\"")
