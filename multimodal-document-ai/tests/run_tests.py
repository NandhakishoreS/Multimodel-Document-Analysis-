"""Direct test runner for fast test execution and reporting."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_bbox_utils import test_validate_bbox, test_normalize_and_denormalize_bbox, test_quad_to_bbox, test_compute_iou, test_extract_spatial_features
from tests.test_text_utils import test_clean_text, test_extract_lexical_features
from tests.test_label_mapping import test_label_encoder, test_dataset_encoders
from tests.test_metrics import test_token_classification_metrics, test_imperfect_metrics, test_cer_and_wer
from tests.test_baselines import get_dummy_samples, test_text_baseline_fit_predict, test_layout_baseline_fit_predict


def run_all_unit_tests():
    print("==================================================")
    print("RUNNING ALL MULTIMODAL DOCUMENT AI UNIT TESTS")
    print("==================================================")

    tests = [
        ("test_validate_bbox", test_validate_bbox),
        ("test_normalize_and_denormalize_bbox", test_normalize_and_denormalize_bbox),
        ("test_quad_to_bbox", test_quad_to_bbox),
        ("test_compute_iou", test_compute_iou),
        ("test_extract_spatial_features", test_extract_spatial_features),
        ("test_clean_text", test_clean_text),
        ("test_extract_lexical_features", test_extract_lexical_features),
        ("test_label_encoder", test_label_encoder),
        ("test_dataset_encoders", test_dataset_encoders),
        ("test_token_classification_metrics", test_token_classification_metrics),
        ("test_imperfect_metrics", test_imperfect_metrics),
        ("test_cer_and_wer", test_cer_and_wer),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    # Baselines fixture tests
    ds_fixture = get_dummy_samples()
    for name, fn in [
        ("test_text_baseline_fit_predict", lambda: test_text_baseline_fit_predict(ds_fixture)),
        ("test_layout_baseline_fit_predict", lambda: test_layout_baseline_fit_predict(ds_fixture)),
    ]:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("==================================================")
    print(f"RESULTS: {passed} PASSED, {failed} FAILED (TOTAL {passed + failed})")
    print("==================================================")
    return failed == 0


if __name__ == "__main__":
    success = run_all_unit_tests()
    sys.exit(0 if success else 1)
