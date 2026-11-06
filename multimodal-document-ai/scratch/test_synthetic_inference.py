import sys
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline

img = Image.new("RGB", (600, 400), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

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

pipeline = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)
res = pipeline.predict_image(img, min_confidence=0.0)

print(f"Total tokens: {len(res.tokens)}")
for t in res.tokens:
    if t.word in ["1", "2", "3", "4"]:
        print(f"  Token: word='{t.word}', label='{t.label}', conf={t.confidence}, bbox={t.bbox}")

print(f"\nTotal entities: {len(res.entities)}")
for e in res.entities:
    print(f"  Entity: label='{e.label}', text='{e.text}', conf={e.confidence:.4f}")
