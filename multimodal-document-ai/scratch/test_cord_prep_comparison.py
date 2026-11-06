import sys
import re
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

sys.path.insert(0, str(Path(".").resolve()))
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

def preprocess_cord_for_ocr(image: Image.Image, scale: float = 2.0, enhance_contrast: float = 1.2):
    w, h = image.size
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # High quality upsampling
    up_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if enhance_contrast != 1.0:
        enhancer = ImageEnhance.Contrast(up_img)
        up_img = enhancer.enhance(enhance_contrast)
    return up_img, scale

def split_ocr_text_to_words(text: str):
    """
    Split OCR text into words, ensuring standalone row/table numbers 
    glued to words (e.g., '1EGG' -> '1', 'EGG', '2CHOCO' -> '2', 'CHOCO')
    are properly separated into individual numeric tokens.
    """
    raw_words = text.split()
    refined_words = []
    for w in raw_words:
        # Check for leading digits glued to letters, e.g., '1EGG' or '2TWIST' or '4Vivamus'
        m = re.match(r"^(\d+)([A-Za-z].*)$", w)
        if m:
            refined_words.append(m.group(1))
            refined_words.append(m.group(2))
        else:
            refined_words.append(w)
    return refined_words

for sample_id in ["cord_test_6", "cord_test_35", "cord_test_97", "cord_test_92", "cord_test_49"]:
    img = Image.open(f"processed_data/cord/images/{sample_id}.png").convert("RGB")
    w, h = img.size
    
    # 1. Baseline OCR
    base_res, _ = ocr(np.array(img))
    base_words = []
    for p, t, c in (base_res or []):
        base_words.extend(t.split())
    
    # 2. Preprocessed OCR (upscale 2x + contrast 1.2x + word splitting)
    prep_img, scale = preprocess_cord_for_ocr(img, scale=2.0, enhance_contrast=1.2)
    prep_res, _ = ocr(np.array(prep_img))
    prep_words = []
    for p, t, c in (prep_res or []):
        prep_words.extend(split_ocr_text_to_words(t))
        
    print(f"\n==================== {sample_id} ({w}x{h}) ====================")
    base_digits = [x for x in base_words if x in ["1", "2", "3", "4"]]
    prep_digits = [x for x in prep_words if x in ["1", "2", "3", "4"]]
    print(f"Baseline digits:     {base_digits}")
    print(f"Preprocessed digits: {prep_digits}")
