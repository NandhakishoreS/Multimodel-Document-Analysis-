import sys
import re
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(".").resolve()))
from inference.final_inference import UnseenDocumentInferencePipeline
from preprocessing.bbox_utils import normalize_bbox

pipeline = UnseenDocumentInferencePipeline(dataset_type="cord", use_robust_model=True)

def split_token(w: str):
    # Split leading digit from letters: 1EGG -> 1, EGG; 2CHOCO -> 2, CHOCO; 1BANANA -> 1, BANANA
    # Split trailing period from row numbers: 1. -> 1, .
    m_lead = re.match(r"^(\d+)([A-Za-z].*)$", w)
    if m_lead:
        return [m_lead.group(1), m_lead.group(2)]
    m_dot = re.match(r"^(\d+)\.$", w)
    if m_dot:
        return [m_dot.group(1)]
    return [w]

for sample_id in ["cord_test_6", "cord_test_35", "cord_test_49"]:
    img = Image.open(f"processed_data/cord/images/{sample_id}.png").convert("RGB")
    width, height = img.size
    
    # Custom run_ocr with token splitting
    ocr = pipeline._get_ocr_engine()
    import numpy as np
    ocr_results, _ = ocr(np.array(img))
    
    words = []
    original_bboxes = []
    normalized_bboxes = []
    raw_lines = []
    
    for poly, text, conf in ocr_results:
        text = str(text).strip()
        if not text:
            continue
        raw_lines.append(text)
        poly_np = np.array(poly)
        x_min = max(0, min(int(np.min(poly_np[:, 0])), width - 1))
        y_min = max(0, min(int(np.min(poly_np[:, 1])), height - 1))
        x_max = max(x_min + 1, min(int(np.max(poly_np[:, 0])), width))
        y_max = max(y_min + 1, min(int(np.max(poly_np[:, 1])), height))
        
        raw_words = text.split()
        line_words = []
        for rw in raw_words:
            line_words.extend(split_token(rw))
            
        line_w = x_max - x_min
        total_chars = max(1, sum(len(w) for w in line_words) + len(line_words) - 1)
        char_w = line_w / total_chars
        
        cur_x = x_min
        for w in line_words:
            w_len = len(w)
            w_x0 = int(cur_x)
            w_x1 = int(min(x_max, cur_x + w_len * char_w))
            w_box = [w_x0, y_min, max(w_x0 + 1, w_x1), y_max]
            norm_box = normalize_bbox(w_box, width, height)
            words.append(w)
            original_bboxes.append(w_box)
            normalized_bboxes.append(norm_box)
            cur_x += (w_len + 1) * char_w

    # Run LayoutLMv3 forward pass using pipeline internals
    pipeline._load_model_components()
    import torch
    import torch.nn.functional as F
    from inference.postprocessing import collapse_token_preds_to_words, reconstruct_entity_spans
    
    encoding = pipeline._tokenizer.processor(
        images=img,
        text=words,
        boxes=normalized_bboxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(pipeline.device)
    attention_mask = encoding["attention_mask"].to(pipeline.device)
    bbox = encoding["bbox"].to(pipeline.device)
    pixel_vals = encoding["pixel_values"].to(pipeline.device)
    word_ids = encoding.word_ids(batch_index=0)

    outputs = pipeline._classifier.forward(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        pixel_values=pixel_vals,
    )
    logits = outputs["logits"].squeeze(0)
    probs = F.softmax(logits, dim=-1)
    max_probs, pred_ids = torch.max(probs, dim=-1)

    word_labels, word_probs = collapse_token_preds_to_words(
        token_label_ids=pred_ids.cpu().tolist(),
        token_probs=max_probs.cpu().tolist(),
        word_ids=word_ids,
        id2label=pipeline._dataset_config.id2label,
        num_words=len(words),
    )

    entities, tokens, flat_agg = reconstruct_entity_spans(
        words=words,
        word_labels=word_labels,
        word_probs=word_probs,
        original_bboxes=original_bboxes,
        normalized_bboxes=normalized_bboxes,
    )

    print(f"\n==================== {sample_id} ====================")
    for t in tokens:
        if t.word in ["1", "2", "3", "4"]:
            print(f"  Token: word='{t.word}', label='{t.label}', conf={t.confidence:.4f}")
    print("Entities:")
    for e in entities:
        if any(d in e.text.split() for d in ["1", "2", "3", "4"]):
            print(f"  Entity: label='{e.label}', text='{e.text}', conf={e.confidence:.4f}")
