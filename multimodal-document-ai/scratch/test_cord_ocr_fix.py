import cv2
import numpy as np
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR

ocr_default = RapidOCR()
# Test with custom RapidOCR settings
# In RapidOCR, we can pass det_box_thresh, det_unclip_ratio, etc.?
# Let's inspect RapidOCR kwargs:
ocr_sensitive = RapidOCR(det_db_box_thresh=0.3, det_db_unclip_ratio=2.0)

img_path = "processed_data/cord/images/cord_test_6.png"
img_pil = Image.open(img_path).convert("RGB")
img_cv = np.array(img_pil)

# Experiment 1: default
res1, _ = ocr_default(img_cv)
print("1. Default:", len(res1))
for p, t, c in res1:
    if "REDBEAN" in t or "1" in t or "EGG" in t:
        print(f"   text='{t}' conf={c:.3f} poly={p}")

# Experiment 2: RapidOCR with det_db_box_thresh=0.3, det_db_unclip_ratio=2.0
res2, _ = ocr_sensitive(img_cv)
print("\n2. Sensitive OCR:", len(res2) if res2 else 0)
for p, t, c in (res2 or []):
    if "REDBEAN" in t or "1" in t or "EGG" in t:
        print(f"   text='{t}' conf={c:.3f} poly={p}")

# Experiment 3: Contrast Enhancement (1.5x)
enhancer = ImageEnhance.Contrast(img_pil)
img_enh = enhancer.enhance(1.5)
res3, _ = ocr_default(np.array(img_enh))
print("\n3. Contrast 1.5x:", len(res3) if res3 else 0)
for p, t, c in (res3 or []):
    if "REDBEAN" in t or "1" in t or "EGG" in t:
        print(f"   text='{t}' conf={c:.3f} poly={p}")

# Experiment 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl_img = clahe.apply(gray)
cl_rgb = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB)
res4, _ = ocr_default(cl_rgb)
print("\n4. CLAHE:", len(res4) if res4 else 0)
for p, t, c in (res4 or []):
    if "REDBEAN" in t or "1" in t or "EGG" in t:
        print(f"   text='{t}' conf={c:.3f} poly={p}")

# Experiment 5: Upscale 2x + Sharpen
w, h = img_pil.size
img_up = img_pil.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
# Sharpen kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
img_sharp = cv2.filter2D(np.array(img_up), -1, kernel)
res5, _ = ocr_default(img_sharp)
print("\n5. Upscale 2x + Sharpen:", len(res5) if res5 else 0)
for p, t, c in (res5 or []):
    if "REDBEAN" in t or "1" in t or "EGG" in t:
        print(f"   text='{t}' conf={c:.3f} poly={p}")
