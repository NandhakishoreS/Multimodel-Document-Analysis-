import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline
from visualization.draw_predictions import draw_predictions_on_image, draw_raw_ocr_boxes

out_dir = Path("reports/visualizations")
out_dir.mkdir(parents=True, exist_ok=True)

pipeline = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)

test_samples = [
    ("cord_test_6", "processed_data/cord/images/cord_test_6.png"),
    ("cord_test_35", "processed_data/cord/images/cord_test_35.png"),
    ("cord_test_97", "processed_data/cord/images/cord_test_97.png"),
]

for name, path_str in test_samples:
    print(f"\n{'='*30} {name} {'='*30}")
    img = Image.open(path_str).convert("RGB")
    
    # 1. Get raw OCR debug
    raw_debug = pipeline.get_raw_ocr_debug(img)
    print(f"Total raw lines: {raw_debug['total_raw_lines']}")
    print(f"Numeric search hits: {raw_debug['numeric_search']}")
    
    # 2. Draw raw OCR debug image
    ocr_results_list = [
        [item["polygon"], item["text"], item["confidence"]]
        for item in raw_debug["raw_ocr_items"]
    ]
    debug_raw_img = draw_raw_ocr_boxes(
        img,
        ocr_results_list,
        highlight_tokens=["1", "2", "3", "4"],
        line_width=2,
    )
    raw_vis_path = out_dir / f"{name}_raw_ocr_debug.png"
    debug_raw_img.save(raw_vis_path)
    print(f"Saved raw OCR debug visualization: {raw_vis_path}")
    
    # 3. Run full inference
    result = pipeline.predict_image(img, min_confidence=0.0)
    print(f"OCR words extracted: {result.ocr_word_count}")
    print(f"Entities extracted: {len(result.entities)}")
    
    numeric_entities = [e for e in result.entities if any(d in e.text.split() for d in ["1", "2", "3", "4"])]
    print(f"Extracted entities with numeric tokens ({len(numeric_entities)}):")
    for e in numeric_entities:
        print(f"   [{e.label}] '{e.text}' (conf={e.confidence:.4f}, bbox={e.bbox})")
        
    # 4. Draw predicted entities
    pred_vis_img = draw_predictions_on_image(
        img,
        result.entities,
        min_confidence=0.0,
        line_width=3,
    )
    pred_vis_path = out_dir / f"{name}_predictions_annotated.png"
    pred_vis_img.save(pred_vis_path)
    print(f"Saved prediction visualization: {pred_vis_path}")

print("\nVisual debug generation complete!")
