import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

for font_size in [12, 14, 16, 18, 20]:
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "1  Lorem Ipsum Dolor   $50.00",
        "2  Pellentesque id neque ligula   $20.00",
        "3  Interdum et malesuada fames   $30.00",
        "4  Vivamus volutpat faucibus   $40.00",
    ]
    # Default font or load font
    y = 30
    for line in lines:
        draw.text((30, y), line, fill=(0, 0, 0))
        y += 35
    
    # If we resize the image to be smaller or simulate low resolution receipt
    # e.g. scale down by 0.7 or 0.5
    for scale in [1.0, 0.75, 0.5]:
        w = int(600 * scale)
        h = int(300 * scale)
        scaled_img = img.resize((w, h), Image.Resampling.BILINEAR)
        res, _ = ocr(np.array(scaled_img))
        detected_digits = []
        for p, t, c in (res or []):
            for w_tok in t.split():
                if w_tok in ["1", "2", "3", "4"]:
                    detected_digits.append(w_tok)
        print(f"Scale {scale:.2f} (size {w}x{h}): detected row digits: {detected_digits}")
    break
