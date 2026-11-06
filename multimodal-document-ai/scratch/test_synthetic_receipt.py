import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Create synthetic receipt table image matching user's exact example
img = Image.new("RGB", (600, 400), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Draw table text
# 1   Lorem Ipsum Dolor
# 2   Pellentesque id neque ligula
# 3   Interdum et malesuada fames
# 4   Vivamus volutpat faucibus
lines = [
    "1   Lorem Ipsum Dolor   $50.00",
    "2   Pellentesque id neque ligula   $20.00",
    "3   Interdum et malesuada fames   $30.00",
    "4   Vivamus volutpat faucibus   $40.00",
]

y = 50
for line in lines:
    draw.text((40, y), line, fill=(0, 0, 0))
    y += 50

ocr = RapidOCR()
res, _ = ocr(np.array(img))

print(f"Synthetic receipt results count: {len(res) if res else 0}")
for i, (p, t, c) in enumerate(res or []):
    print(f"  {i}: text='{t}' conf={c:.3f}")
    words = t.split()
    print(f"      words={words}")
