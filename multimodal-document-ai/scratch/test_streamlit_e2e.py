import urllib.request
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline
from visualization.draw_predictions import draw_predictions_on_image, draw_raw_ocr_boxes

print("=== STREAMLIT END-TO-END APPLICATION TEST ===")

# 1. HTTP Endpoint check
url = "http://localhost:8501"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        status_code = response.status
        content = response.read(10000).decode("utf-8", errors="ignore")
        print(f"1. HTTP GET {url}: STATUS {status_code}")
        assert status_code == 200, f"Expected status 200, got {status_code}"
        assert "Multimodal" in content or "Streamlit" in content or "DOCTYPE" in content, "Invalid HTML response"
        print("   [OK] Server responded with valid HTML application shell.")
except Exception as e:
    print(f"   [FAIL] Could not connect to Streamlit server: {e}")
    sys.exit(1)

# 2. Test Streamlit Pipeline Cache and Components Programmatically
print("\n2. Testing Application Components Under Streamlit Runtime Logic:")
for doc_type in ["sroie", "funsd", "cord"]:
    pipe = UnseenDocumentInferencePipeline(dataset_type=doc_type, use_robust_model=True)
    assert pipe.checkpoint_dir.exists(), f"Missing checkpoint for {doc_type}"
    print(f"   [OK] Pipeline loaded for {doc_type}: {pipe.checkpoint_dir.name}")

# 3. Test Debug Raw OCR and Visualization in Streamlit
blank = Image.new("RGB", (300, 200), color="white")
raw_debug = pipe.get_raw_ocr_debug(blank)
assert "raw_ocr_items" in raw_debug
assert "numeric_search" in raw_debug
print("   [OK] Raw OCR Debug capability functional.")

raw_vis = draw_raw_ocr_boxes(blank, [], highlight_tokens=["1", "2", "3", "4"])
assert raw_vis.size == (300, 200)
print("   [OK] Raw OCR visualization functional.")

print("\nSTREAMLIT END-TO-END VERIFICATION: SUCCESS!")
