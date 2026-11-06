import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline

print("=== FINAL INFERENCE VERIFICATION ===")

# 1. SROIE Inference Test
print("\n--- 1. SROIE Real Inference ---")
pipe_sroie = UnseenDocumentInferencePipeline(dataset_type="sroie", use_robust_model=True)
sroie_img_path = "processed_data/sroie/images/X00016469612.png"
if not Path(sroie_img_path).exists():
    sroie_img_path = list(Path("processed_data/sroie/images").glob("*.png"))[0]
res_sroie = pipe_sroie.predict_image(sroie_img_path, min_confidence=0.20)
print(f"SROIE Document: {sroie_img_path}")
print(f"OCR Words: {res_sroie.ocr_word_count}")
print(f"Entities: {len(res_sroie.entities)}")
for e in res_sroie.entities:
    print(f"  [{e.label}] '{e.text}' (conf={e.confidence:.2%}, bbox={e.bbox})")
print(f"Aggregated JSON fields: {res_sroie.raw_entities_dict}")
assert len(res_sroie.entities) > 0, "SROIE returned zero entities!"

# 2. FUNSD Inference Test
print("\n--- 2. FUNSD Real Inference ---")
pipe_funsd = UnseenDocumentInferencePipeline(dataset_type="funsd", use_robust_model=True)
funsd_img_path = "processed_data/funsd/images/0.png"
if not Path(funsd_img_path).exists():
    funsd_img_path = list(Path("processed_data/funsd/images").glob("*.png"))[0]
res_funsd = pipe_funsd.predict_image(funsd_img_path, min_confidence=0.20)
print(f"FUNSD Document: {funsd_img_path}")
print(f"OCR Words: {res_funsd.ocr_word_count}")
print(f"Entities: {len(res_funsd.entities)}")
for e in res_funsd.entities[:8]:
    print(f"  [{e.label}] '{e.text[:40]}' (conf={e.confidence:.2%}, bbox={e.bbox})")
assert len(res_funsd.entities) > 0, "FUNSD returned zero entities!"

# 3. CORD Inference Test (specifically testing row digits and decomposition)
print("\n--- 3. CORD Real Inference & Numeric Token Verification ---")
pipe_cord = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)
cord_img_path = "processed_data/cord/images/cord_test_35.png"
res_cord = pipe_cord.predict_image(cord_img_path, min_confidence=0.20)
print(f"CORD Document: {cord_img_path}")
print(f"OCR Words: {res_cord.ocr_word_count}")
print(f"Entities: {len(res_cord.entities)}")
cord_digits = [e for e in res_cord.entities if any(d in e.text.split() for d in ["1", "2", "3", "4"])]
print(f"Extracted numeric entities ({len(cord_digits)}):")
for e in cord_digits:
    print(f"  [{e.label}] '{e.text}' (conf={e.confidence:.2%}, bbox={e.bbox})")

extracted_values = [e.text for e in cord_digits]
assert "1" in extracted_values, "CORD failed to extract standalone digit '1'!"
assert "2" in extracted_values, "CORD failed to extract standalone digit '2'!"
print("\nCORD numeric token verification: SUCCESS! '1' and '2' properly extracted!")

# Also test cord_test_92 for digit '4'
res_cord_92 = pipe_cord.predict_image("processed_data/cord/images/cord_test_92.png", min_confidence=0.20)
cord_92_digits = [e for e in res_cord_92.entities if "4" in e.text.split()]
print(f"CORD cord_test_92 digit '4' entities: {[e.text for e in cord_92_digits]}")
assert any("4" in e.text.split() for e in res_cord_92.entities), "CORD failed to extract digit '4'!"

print("\nALL INFERENCE PIPELINE TESTS PASSED COMPLETELY!")
