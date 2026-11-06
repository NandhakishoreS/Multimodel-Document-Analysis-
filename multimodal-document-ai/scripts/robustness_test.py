"""
Phase 5 robustness testing: tests the inference pipeline on documents with
varying resolution, noise, rotation, and density.

Run AFTER full training:
    python scripts/robustness_test.py --dataset sroie --stage full

Results saved to reports/robustness_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robustness_test")


def load_model_and_tokenizer(dataset_name: str, stage: str):
    """Load the best checkpoint for a dataset."""
    from models.layoutlmv3.config import load_dataset_config, load_training_config
    from models.layoutlmv3.model import LayoutLMv3Classifier
    from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper

    cfg   = load_training_config(dataset_name=dataset_name)
    ds_cfg = load_dataset_config(dataset_name, labels_dir=cfg.labels_dir)

    ckpt_dir = cfg.resolved_checkpoint_dir() / dataset_name / stage
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_dir}")

    proc_dir = ckpt_dir / "processor"
    proc_src = str(proc_dir) if proc_dir.exists() else cfg.model_checkpoint

    tokenizer = LayoutLMv3TokenizerWrapper.from_pretrained(proc_src, cfg.max_seq_length)
    classifier = LayoutLMv3Classifier.load_checkpoint(ckpt_dir, ds_cfg)
    classifier.eval_mode()

    return classifier, tokenizer, ds_cfg


# ── Image perturbation functions ──────────────────────────────────────────────

def perturb_resolution(img: Image.Image, scale: float) -> Image.Image:
    """Downscale then upscale to simulate lower-resolution scan."""
    w, h = img.size
    small = img.resize((max(10, int(w * scale)), max(10, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def perturb_noise(img: Image.Image, noise_level: float = 0.05) -> Image.Image:
    """Add Gaussian noise to simulate a noisy scan."""
    arr  = np.array(img, dtype=np.float32)
    std  = noise_level * 255
    noisy = arr + np.random.normal(0, std, arr.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def perturb_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Slightly rotate image to simulate skewed scans."""
    return img.rotate(angle, expand=False, fillcolor=(255, 255, 255))


def perturb_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    """Apply Gaussian blur to simulate out-of-focus scans."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def predict_sample(
    sample: Dict[str, Any],
    perturbed_img: Image.Image,
    classifier,
    tokenizer,
    ds_cfg,
) -> Tuple[List[str], float]:
    """
    Run inference on a sample with a potentially perturbed image.
    Returns (predicted_word_labels, inference_time_ms).
    """
    import torch
    from training.train_layoutlmv3 import word_ids_tensor_to_list
    from inference.layoutlmv3_predict import token_preds_to_word_preds

    # Override sample image with perturbed version
    perturbed_sample = dict(sample)
    # Encode using the perturbed image directly (bypass image_path loading)
    encoding = tokenizer.processor(
        images=perturbed_img.convert("RGB"),
        text=sample.get("words", ["[EMPTY]"]),
        boxes=[tokenizer._load_image.__func__  # not used here
               ] if False else sample.get("bboxes", [[0, 0, 1, 1]] * len(sample.get("words", ["[EMPTY]"]))),
        max_length=tokenizer.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_ids, probs = classifier.predict_batch(
            input_ids      = encoding["input_ids"],
            attention_mask = encoding["attention_mask"],
            bbox           = encoding["bbox"],
            pixel_values   = encoding["pixel_values"],
        )
    inf_ms = (time.perf_counter() - t0) * 1000

    word_ids = encoding.word_ids(batch_index=0)
    word_ids_list = [None if wid is None else int(wid) for wid in word_ids]
    words = sample.get("words", [])

    pred_labels, _ = token_preds_to_word_preds(
        token_label_ids=pred_ids[0].cpu().tolist(),
        token_probs=probs[0].max(dim=-1).values.cpu().tolist(),
        word_ids=word_ids_list,
        id2label=ds_cfg.id2label,
        num_words=len(words),
    )
    return pred_labels, inf_ms


def entity_f1_score(true_labels: List[str], pred_labels: List[str]) -> float:
    """Simple exact-match span F1 via seqeval."""
    from seqeval.metrics import f1_score
    try:
        return f1_score([true_labels], [pred_labels], zero_division=0)
    except Exception:
        return 0.0


# ── Main robustness test ───────────────────────────────────────────────────────

PERTURBATIONS = [
    ("baseline",         lambda img: img),
    ("low_res_50%",      lambda img: perturb_resolution(img, 0.5)),
    ("low_res_25%",      lambda img: perturb_resolution(img, 0.25)),
    ("noise_5%",         lambda img: perturb_noise(img, 0.05)),
    ("noise_15%",        lambda img: perturb_noise(img, 0.15)),
    ("rotation_3deg",    lambda img: perturb_rotation(img, 3)),
    ("rotation_10deg",   lambda img: perturb_rotation(img, 10)),
    ("blur_radius2",     lambda img: perturb_blur(img, 2.0)),
    ("blur_radius5",     lambda img: perturb_blur(img, 5.0)),
]


def run_robustness_test(
    dataset_name: str,
    stage: str,
    n_samples: int = 20,
) -> Dict[str, Any]:
    import json as _json

    logger.info("Loading model for %s [%s]...", dataset_name, stage)
    classifier, tokenizer, ds_cfg = load_model_and_tokenizer(dataset_name, stage)

    # Load test samples
    from models.layoutlmv3.config import load_training_config
    cfg = load_training_config(dataset_name=dataset_name)
    test_jsonl = cfg.resolved_processed_dir() / dataset_name / "test.jsonl"
    samples = []
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(_json.loads(line))
    samples = samples[:n_samples]
    logger.info("Testing on %d samples with %d perturbation types", n_samples, len(PERTURBATIONS))

    results: Dict[str, Any] = {
        "dataset": dataset_name, "stage": stage,
        "n_samples": n_samples,
        "perturbation_results": {},
        "failures": [],
    }

    for pert_name, pert_fn in PERTURBATIONS:
        f1_scores = []
        inf_times = []
        failures  = []

        for i, sample in enumerate(samples):
            img_path = PROJECT_ROOT / sample.get("image_path", "")
            if not img_path.exists():
                failures.append({"sample_id": sample.get("id"), "reason": "image_not_found"})
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                perturbed = pert_fn(img)
                pred_labels, inf_ms = predict_sample(
                    sample, perturbed, classifier, tokenizer, ds_cfg
                )
                true_labels = sample.get("labels", [])
                min_len = min(len(true_labels), len(pred_labels))
                f1 = entity_f1_score(true_labels[:min_len], pred_labels[:min_len])
                f1_scores.append(f1)
                inf_times.append(inf_ms)
            except Exception as exc:
                failures.append({"sample_id": sample.get("id"), "error": str(exc)})

        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        mean_ms = float(np.mean(inf_times)) if inf_times else 0.0

        results["perturbation_results"][pert_name] = {
            "mean_f1": round(mean_f1, 4),
            "mean_inference_ms": round(mean_ms, 2),
            "n_evaluated": len(f1_scores),
            "n_failures": len(failures),
        }
        logger.info(
            "%-20s | mean_f1=%.4f | mean_ms=%.1f | failures=%d",
            pert_name, mean_f1, mean_ms, len(failures),
        )

    # Print summary table
    baseline_f1 = results["perturbation_results"].get("baseline", {}).get("mean_f1", 0.0)
    print(f"\n{'Perturbation':<22} | {'Mean F1':>8} | {'vs Baseline':>12} | {'Inference(ms)':>14}")
    print("-" * 65)
    for pert_name, r in results["perturbation_results"].items():
        delta = r["mean_f1"] - baseline_f1
        print(f"{pert_name:<22} | {r['mean_f1']:>8.4f} | {delta:>+12.4f} | {r['mean_inference_ms']:>14.1f}")

    # Save
    out_path = PROJECT_ROOT / "reports" / f"robustness_{dataset_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, indent=2)
    logger.info("Robustness results saved to %s", out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness test for LayoutLMv3")
    parser.add_argument("--dataset", default="sroie", choices=["sroie", "funsd", "cord"])
    parser.add_argument("--stage",   default="full",  choices=["smoke", "small", "full"])
    parser.add_argument("--n",       type=int, default=20, help="Number of test samples")
    args = parser.parse_args()

    run_robustness_test(args.dataset, args.stage, args.n)
