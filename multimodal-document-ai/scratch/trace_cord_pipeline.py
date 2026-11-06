import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline
from rapidocr_onnxruntime import RapidOCR

pipeline = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)
ocr = RapidOCR()

test_images = [
    "processed_data/cord/images/cord_test_2.png",
    "processed_data/cord/images/cord_test_4.png",
    "processed_data/cord/images/cord_test_6.png",
    "processed_data/cord/images/cord_test_8.png",
]

for img_path in test_images:
    print("=" * 60)
    print(f"Testing image: {img_path}")
    img = Image.open(img_path).convert("RGB")
    print(f"Image size: {img.size}")
    
    # Raw RapidOCR directly
    img_np = np.array(img)
    raw_ocr_res, _ = ocr(img_np)
    print(f"Raw RapidOCR results count: {len(raw_ocr_res) if raw_ocr_res else 0}")
    
    found_digits = []
    if raw_ocr_res:
        for idx, item in enumerate(raw_ocr_res):
            poly, text, conf = item
            text = str(text).strip()
            # print all tokens or tokens that are numeric / single char
            for token in text.split():
                if token in ["1", "2", "3", "4", "1.", "2.", "3.", "4."]:
                    found_digits.append((idx, token, conf, poly))
                    print(f"  [RAW OCR HIT] item {idx}: token='{token}' full_text='{text}' conf={conf:.3f}")
    print(f"Total digit hits in raw OCR: {len(found_digits)}")
    
    # Run full pipeline
    result = pipeline.predict_image(img, min_confidence=0.0)
    print(f"Pipeline words count: {result.ocr_word_count}")
    pipe_digits = [t for t in result.tokens if t.word in ["1", "2", "3", "4", "1.", "2.", "3.", "4."]]
    print(f"Pipeline digit tokens count: {len(pipe_digits)}")
    for t in pipe_digits:
        print(f"  Token: word='{t.word}', label='{t.label}', conf={t.confidence}, bbox={t.bbox}")
        
    print(f"Extracted entities count: {len(result.entities)}")
    for ent in result.entities:
        if any(d in ent.text.split() for d in ["1", "2", "3", "4"]):
            print(f"  Entity: label='{ent.label}', text='{ent.text}', conf={ent.confidence}")
