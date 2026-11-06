"""
SROIE Duplicate-Adjusted Entity F1 Evaluation & Span-Extraction Sanity Check
Calculates strict vs duplicate-adjusted entity F1 and investigates bare '1' predictions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from seqeval.metrics.sequence_labeling import get_entities

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    return re.sub(r"\s+", " ", s)

def normalize_amount(s: str) -> str:
    clean = re.sub(r"[^0-9.]", "", s.strip())
    try:
        val = float(clean)
        return f"{val:.2f}"
    except Exception:
        return s.strip().lower()

def run_duplicate_adjusted_eval():
    preds_file = _PROJECT_ROOT / "reports" / "layoutlmv3_sroie_test_predictions.json"
    with open(preds_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    test_jsonl = _PROJECT_ROOT / "processed_data" / "sroie" / "test.jsonl"
    with open(test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    classes = ["COMPANY", "ADDRESS", "DATE", "TOTAL"]

    strict_tp = {c: 0 for c in classes}
    strict_fp = {c: 0 for c in classes}
    strict_fn = {c: 0 for c in classes}

    recovered_fp = {c: 0 for c in classes}
    adj_tp = {c: 0 for c in classes}
    adj_fp = {c: 0 for c in classes}
    adj_fn = {c: 0 for c in classes}

    recovered_examples = []

    for doc_idx, (sample, pred_pair) in enumerate(zip(samples, preds)):
        sid = sample["id"]
        words = sample["words"]
        true_seq = pred_pair["true"]
        pred_seq = pred_pair["pred"]
        gt_entities_dict = sample.get("entities", {})

        gold_ents = get_entities(true_seq)
        pred_ents = get_entities(pred_seq)

        gold_set = set(gold_ents)
        pred_set = set(pred_ents)

        # All gold text values for this document by entity type
        gold_texts_by_type = {c: [] for c in classes}
        for c, s, e in gold_ents:
            t = " ".join(words[s: e + 1])
            gold_texts_by_type[c].append(t)

        # Also add canonical ground truth string from sample['entities']
        for c in classes:
            canonical = str(gt_entities_dict.get(c.lower(), "")).strip()
            if canonical and canonical not in gold_texts_by_type[c]:
                gold_texts_by_type[c].append(canonical)

        for c in classes:
            for ent in pred_ents:
                if ent[0] == c:
                    if ent in gold_set:
                        strict_tp[c] += 1
                        adj_tp[c] += 1
                    else:
                        strict_fp[c] += 1
                        # Check if predicted value matches any gold value in the document
                        p_text = " ".join(words[ent[1]: ent[2] + 1])
                        norm_p = normalize_text(p_text)

                        matched = False
                        # For TOTAL, check numeric equality
                        if c == "TOTAL":
                            p_amt = normalize_amount(p_text)
                            for g_t in gold_texts_by_type[c]:
                                if p_amt == normalize_amount(g_t) and p_amt != "":
                                    matched = True
                                    break
                        elif c == "DATE":
                            for g_t in gold_texts_by_type[c]:
                                norm_g = normalize_text(g_t)
                                if norm_p == norm_g or norm_g in norm_p or norm_p in norm_g:
                                    if any(char.isdigit() for char in norm_p) and len(norm_p) >= 6:
                                        matched = True
                                        break
                        else:
                            for g_t in gold_texts_by_type[c]:
                                norm_g = normalize_text(g_t)
                                if norm_p == norm_g:
                                    matched = True
                                    break

                        if matched:
                            recovered_fp[c] += 1
                            adj_tp[c] += 1
                            recovered_examples.append({
                                "sample_id": sid,
                                "class": c,
                                "span": ent[1:],
                                "predicted_text": p_text,
                                "gold_texts": gold_texts_by_type[c]
                            })
                        else:
                            adj_fp[c] += 1

            for ent in gold_ents:
                if ent[0] == c:
                    if ent not in pred_set:
                        strict_fn[c] += 1
                        adj_fn[c] += 1

    print("=== STRICT vs DUPLICATE-ADJUSTED ENTITY F1 ===")
    results = {}
    for c in classes:
        p_s = strict_tp[c] / max(1, (strict_tp[c] + strict_fp[c]))
        r_s = strict_tp[c] / max(1, (strict_tp[c] + strict_fn[c]))
        f1_s = 2 * p_s * r_s / max(1e-6, p_s + r_s)

        p_a = adj_tp[c] / max(1, (adj_tp[c] + adj_fp[c]))
        r_a = adj_tp[c] / max(1, (adj_tp[c] + adj_fn[c]))
        f1_a = 2 * p_a * r_a / max(1e-6, p_a + r_a)

        results[c] = {
            "strict_p": p_s, "strict_r": r_s, "strict_f1": f1_s,
            "adj_p": p_a, "adj_r": r_a, "adj_f1": f1_a,
            "recovered_fp": recovered_fp[c],
            "strict_tp": strict_tp[c], "strict_fp": strict_fp[c], "strict_fn": strict_fn[c],
            "adj_tp": adj_tp[c], "adj_fp": adj_fp[c], "adj_fn": adj_fn[c]
        }
        print(f"{c:<10}: Strict F1={f1_s:.4f} (P={p_s:.4f}, R={r_s:.4f}) | Adj F1={f1_a:.4f} (P={p_a:.4f}, R={r_a:.4f}) | Recovered FPs={recovered_fp[c]}")

    tot_s_tp = sum(strict_tp.values())
    tot_s_fp = sum(strict_fp.values())
    tot_s_fn = sum(strict_fn.values())
    mic_p_s = tot_s_tp / (tot_s_tp + tot_s_fp)
    mic_r_s = tot_s_tp / (tot_s_tp + tot_s_fn)
    mic_f1_s = 2 * mic_p_s * mic_r_s / (mic_p_s + mic_r_s)
    mac_f1_s = sum(results[c]["strict_f1"] for c in classes) / len(classes)

    tot_a_tp = sum(adj_tp.values())
    tot_a_fp = sum(adj_fp.values())
    tot_a_fn = sum(adj_fn.values())
    mic_p_a = tot_a_tp / (tot_a_tp + tot_a_fp)
    mic_r_a = tot_a_tp / (tot_a_tp + tot_a_fn)
    mic_f1_a = 2 * mic_p_a * mic_r_a / (mic_p_a + mic_r_a)
    mac_f1_a = sum(results[c]["adj_f1"] for c in classes) / len(classes)

    print(f"\nOVERALL STRICT : Micro F1={mic_f1_s:.4f} (P={mic_p_s:.4f}, R={mic_r_s:.4f}) | Macro F1={mac_f1_s:.4f}")
    print(f"OVERALL ADJUSTED: Micro F1={mic_f1_a:.4f} (P={mic_p_a:.4f}, R={mic_r_a:.4f}) | Macro F1={mac_f1_a:.4f}")
    print("Total recovered FPs:", sum(recovered_fp.values()))

    return results, recovered_examples

def run_sanity_check():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from models.layoutlmv3.config import load_dataset_config, load_training_config
    from models.layoutlmv3.model import LayoutLMv3Classifier
    from models.layoutlmv3.tokenizer import LayoutLMv3TokenizerWrapper

    cfg = load_training_config("configs/layoutlmv3.yaml", dataset_name="sroie")
    ds_cfg = load_dataset_config("sroie", labels_dir=cfg.labels_dir)
    ckpt_dir = _PROJECT_ROOT / "models" / "checkpoints" / "layoutlmv3" / "sroie" / "full"
    tok_wrap = LayoutLMv3TokenizerWrapper.from_pretrained(str(ckpt_dir / "processor"), max_seq_length=cfg.max_seq_length)
    classifier = LayoutLMv3Classifier.load_checkpoint(ckpt_dir, ds_cfg)
    classifier.eval_mode()

    targets = ["X51005337867", "X51005361906", "X51005433548", "X51005433556"]
    with open(_PROJECT_ROOT / "processed_data" / "sroie" / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["id"] in targets:
                print(f"\n=== SAMPLE: {s['id']} ===")
                encoding = tok_wrap.encode_sample(s, label2id=None, project_root=_PROJECT_ROOT)
                with torch.no_grad():
                    pred_ids, probs = classifier.predict_batch(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        bbox=encoding["bbox"],
                        pixel_values=encoding["pixel_values"],
                    )
                word_ids = encoding["word_ids"]
                pred_id_list = pred_ids[0].cpu().tolist()
                prob_list = probs[0].cpu().tolist()

                bboxes = s.get("bboxes", s.get("boxes", []))
                for w_idx, w in enumerate(s["words"]):
                    if w.strip() == "1":
                        box = bboxes[w_idx] if w_idx < len(bboxes) else [0, 0, 0, 0]
                        tokens = [
                            (t_idx, tok_wrap.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0][t_idx].item()),
                             pred_id_list[t_idx], ds_cfg.id2label[pred_id_list[t_idx]], round(max(prob_list[t_idx]), 4))
                            for t_idx, wid in enumerate(word_ids) if wid == w_idx
                        ]
                        prev_w = s["words"][w_idx-1] if w_idx > 0 else "START"
                        next_w = s["words"][w_idx+1] if w_idx+1 < len(s["words"]) else "END"
                        print(f"Word {w_idx}: '{w}' | Box: {box} | Context: '{prev_w}' -> ['{w}'] -> '{next_w}'")
                        for t in tokens:
                            print(f"  Subword {t[0]}: {repr(t[1])} -> Label: {t[3]} (id={t[2]}), Prob: {t[4]}")

if __name__ == "__main__":
    run_duplicate_adjusted_eval()
    run_sanity_check()
