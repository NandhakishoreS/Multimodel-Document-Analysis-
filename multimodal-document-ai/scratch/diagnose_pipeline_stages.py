import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline

pipeline = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)

for sample_id in ["cord_test_97", "cord_test_92", "cord_test_35", "cord_test_6", "cord_test_2"]:
    img_path = f"processed_data/cord/images/{sample_id}.png"
    img = Image.open(img_path).convert("RGB")
    print("\n" + "=" * 80)
    print(f"SAMPLE: {sample_id}")
    
    # 1. OCR Words
    words, orig_boxes, norm_boxes, raw_text = pipeline.run_ocr(img)
    print(f"OCR Words ({len(words)}):")
    for i, (w, b) in enumerate(zip(words, orig_boxes)):
        if any(d in w for d in ["1", "2", "3", "4"]):
            print(f"   word[{i:2d}]: '{w}' bbox={b}")
            
    # 2. Predict image
    res = pipeline.predict_image(img, min_confidence=0.0)
    
    # 3. Check tokens
    print(f"\nPredicted Tokens with digits ({len(res.tokens)} total):")
    for t in res.tokens:
        if any(d in t.word for d in ["1", "2", "3", "4"]):
            print(f"   token[{t.word_index:2d}]: word='{t.word}', label='{t.label}', conf={t.confidence}")
            
    # 4. Check entities
    print(f"\nEntities containing digits ({len(res.entities)} total entities):")
    for e in res.entities:
        if any(d in e.text.split() for d in ["1", "2", "3", "4"]) or any(d in e.text for d in ["1", "2", "3", "4"]):
            print(f"   entity: label='{e.label}', text='{e.text}', conf={e.confidence:.4f}")
