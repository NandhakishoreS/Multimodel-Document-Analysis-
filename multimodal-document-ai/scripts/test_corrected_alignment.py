"""
Experimental development script for corrected SROIE label alignment.
Refined to support short address lines (e.g. '12 TAMPOI') when only partial address is printed.
"""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def normalize_str(s: str) -> str:
    """NFKC-normalize, lowercase, collapse whitespace."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_word_for_matching(w: str) -> str:
    """Strip surrounding punctuation for token comparison."""
    w = normalize_str(w)
    return re.sub(r"^[^\w]+|[^\w]+$", "", w)

def split_line_into_words_with_bboxes(text: str, bbox: list):
    """
    Split a line-level OCR text entry into word tokens with interpolated bounding boxes.
    Preserves exact geometry: union of word bboxes covers the line bbox.
    """
    raw_words = text.split()
    if len(raw_words) <= 1:
        return [text] if text.strip() else [], [bbox] if text.strip() else []
    
    L = max(1, len(text))
    x0, y0, x1, y1 = bbox
    w_width = x1 - x0
    
    out_words = []
    out_boxes = []
    curr = 0
    for w in raw_words:
        start = text.find(w, curr)
        if start == -1:
            start = curr
        end = start + len(w)
        curr = end
        
        bx0 = int(round(x0 + (start / L) * w_width))
        bx1 = int(round(x0 + (end / L) * w_width))
        bx0 = max(0, min(1000, bx0))
        bx1 = max(bx0, min(1000, bx1))
        
        out_words.append(w)
        out_boxes.append([bx0, y0, bx1, y1])
        
    return out_words, out_boxes

def parse_numeric_total(s: str):
    """Extract canonical float value from total string."""
    norm = normalize_str(s)
    norm = re.sub(r"^(?:rm|myr|\$)\s*", "", norm)
    norm = norm.replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", norm)
    if m:
        try:
            return float(m.group(1)), m.group(1)
        except ValueError:
            return None, None
    return None, None

def align_sroie_corrected(words: list, entities: dict):
    """
    Corrected deterministic label alignment for SROIE.
    """
    labels = ["O"] * len(words)
    norm_words = [normalize_str(w) for w in words]
    clean_words = [clean_word_for_matching(w) for w in words]
    
    audit_info = {
        "aligned": {},
        "failed": {},
        "multi_token": {},
    }
    
    # =========================================================================
    # 1. TOTAL ALIGNMENT
    # =========================================================================
    raw_total = entities.get("total", "").strip()
    if raw_total:
        target_val, target_str = parse_numeric_total(raw_total)
        matched_idx = None
        
        if target_val is not None:
            candidates = []
            for i, (w, nw, cw) in enumerate(zip(words, norm_words, clean_words)):
                if labels[i] != "O":
                    continue
                val, sval = parse_numeric_total(w)
                if val is not None and abs(val - target_val) < 1e-4:
                    if len(target_str) > 1 and len(sval) == 1:
                        continue
                    
                    score = 0
                    for offset in range(-5, 6):
                        neighbor_idx = i + offset
                        if 0 <= neighbor_idx < len(norm_words):
                            nw_neighbor = norm_words[neighbor_idx]
                            if any(k in nw_neighbor for k in ["total", "amount", "net", "grand", "cash", "rm"]):
                                score += 2
                    score += (i / len(words))
                    candidates.append((score, i))
            
            if candidates:
                candidates.sort(key=lambda x: -x[0])
                matched_idx = candidates[0][1]
                labels[matched_idx] = "B-TOTAL"
                audit_info["aligned"]["total"] = {
                    "matched_token": words[matched_idx],
                    "target": raw_total,
                    "span": [matched_idx, matched_idx]
                }
            else:
                audit_info["failed"]["total"] = raw_total
        else:
            audit_info["failed"]["total"] = raw_total

    # =========================================================================
    # 2. DATE ALIGNMENT
    # =========================================================================
    raw_date = entities.get("date", "").strip()
    if raw_date:
        norm_date = normalize_str(raw_date)
        date_tokens = norm_date.split()
        date_clean_tokens = [clean_word_for_matching(t) for t in date_tokens if clean_word_for_matching(t)]
        
        matched_span = None
        k = len(date_tokens)
        if k > 0:
            for i in range(len(norm_words) - k + 1):
                if all(labels[i + j] == "O" for j in range(k)):
                    if [clean_words[i + j] for j in range(k)] == date_clean_tokens:
                        matched_span = (i, i + k - 1)
                        break
        
        if not matched_span and date_clean_tokens:
            target_joined = "".join(date_clean_tokens)
            for i, cw in enumerate(clean_words):
                if labels[i] != "O":
                    continue
                if cw == target_joined or norm_date in norm_words[i]:
                    matched_span = (i, i)
                    break
                    
        if not matched_span:
            clean_date_str = re.sub(r"[^\d/.-]", "", norm_date)
            if len(clean_date_str) >= 6:
                for i, nw in enumerate(norm_words):
                    if labels[i] != "O":
                        continue
                    if clean_date_str in nw:
                        matched_span = (i, i)
                        break
                        
        if matched_span:
            s_start, s_end = matched_span
            labels[s_start] = "B-DATE"
            for j in range(s_start + 1, s_end + 1):
                labels[j] = "I-DATE"
            audit_info["aligned"]["date"] = {
                "span": matched_span,
                "tokens": words[s_start:s_end+1],
                "target": raw_date
            }
            if s_end > s_start:
                audit_info["multi_token"]["date"] = s_end - s_start + 1
        else:
            audit_info["failed"]["date"] = raw_date

    # =========================================================================
    # 3. COMPANY ALIGNMENT
    # =========================================================================
    raw_company = entities.get("company", "").strip()
    if raw_company:
        norm_comp = normalize_str(raw_company)
        comp_tokens = [clean_word_for_matching(t) for t in norm_comp.split() if clean_word_for_matching(t)]
        matched_span = None
        
        k = len(comp_tokens)
        if k > 0:
            search_limit = min(len(words), 40)
            for i in range(search_limit - k + 1):
                if all(labels[i + j] == "O" for j in range(k)):
                    if [clean_words[i + j] for j in range(k)] == comp_tokens:
                        matched_span = (i, i + k - 1)
                        break
                        
            if not matched_span:
                for i in range(len(words) - k + 1):
                    if all(labels[i + j] == "O" for j in range(k)):
                        if [clean_words[i + j] for j in range(k)] == comp_tokens:
                            matched_span = (i, i + k - 1)
                            break
                            
            if not matched_span and k >= 2:
                best_subspan = None
                best_len = 0
                for sub_len in range(k - 1, max(1, int(k * 0.6)) - 1, -1):
                    for start_sub in range(k - sub_len + 1):
                        target_sub = comp_tokens[start_sub:start_sub + sub_len]
                        for i in range(min(len(words), 40) - sub_len + 1):
                            if all(labels[i + j] == "O" for j in range(sub_len)):
                                if [clean_words[i + j] for j in range(sub_len)] == target_sub:
                                    best_subspan = (i, i + sub_len - 1)
                                    best_len = sub_len
                                    break
                        if best_subspan:
                            break
                    if best_subspan:
                        break
                if best_subspan:
                    matched_span = best_subspan

        if matched_span:
            s_start, s_end = matched_span
            labels[s_start] = "B-COMPANY"
            for j in range(s_start + 1, s_end + 1):
                labels[j] = "I-COMPANY"
            audit_info["aligned"]["company"] = {
                "span": matched_span,
                "tokens": words[s_start:s_end+1],
                "target": raw_company
            }
            if s_end > s_start:
                audit_info["multi_token"]["company"] = s_end - s_start + 1
        else:
            audit_info["failed"]["company"] = raw_company

    # =========================================================================
    # 4. ADDRESS ALIGNMENT (Complete Span, Multi-Line)
    # =========================================================================
    raw_addr = entities.get("address", "").strip()
    if raw_addr:
        norm_addr = normalize_str(raw_addr)
        addr_clean_tokens = [clean_word_for_matching(t) for t in norm_addr.split() if clean_word_for_matching(t)]
        addr_token_set = set(addr_clean_tokens)
        
        matched_span = None
        k = len(addr_clean_tokens)
        
        if k > 0:
            for i in range(len(words) - k + 1):
                if all(labels[i + j] == "O" for j in range(k)):
                    if [clean_words[i + j] for j in range(k)] == addr_clean_tokens:
                        matched_span = (i, i + k - 1)
                        break
                        
        if not matched_span and addr_token_set:
            best_window = None
            best_score = 0
            limit = min(len(words), 80)
            
            start_candidates = [i for i in range(limit) if labels[i] == "O" and clean_words[i] in addr_token_set]
            
            for start_idx in start_candidates:
                consecutive_non_matches = 0
                matching_count = 0
                end_idx = start_idx
                
                for curr_idx in range(start_idx, limit):
                    if labels[curr_idx] != "O":
                        break
                    cw = clean_words[curr_idx]
                    if cw in addr_token_set:
                        matching_count += 1
                        consecutive_non_matches = 0
                        end_idx = curr_idx
                    else:
                        consecutive_non_matches += 1
                        if consecutive_non_matches > 2:
                            break
                            
                span_len = end_idx - start_idx + 1
                density = matching_count / max(1, span_len)
                
                # Accept if:
                # 1) At least 3 matching words and density >= 0.60
                # 2) Or at least 2 matching words and density >= 0.80 (for short addresses like '12 TAMPOI')
                is_valid = False
                if matching_count >= 3 and density >= 0.60:
                    is_valid = True
                elif matching_count >= 2 and density >= 0.80:
                    is_valid = True
                    
                if is_valid:
                    score = matching_count * 10 + span_len
                    if score > best_score:
                        best_score = score
                        best_window = (start_idx, end_idx)
                            
            if best_window:
                matched_span = best_window

        if matched_span:
            s_start, s_end = matched_span
            labels[s_start] = "B-ADDRESS"
            for j in range(s_start + 1, s_end + 1):
                labels[j] = "I-ADDRESS"
            audit_info["aligned"]["address"] = {
                "span": matched_span,
                "tokens": words[s_start:s_end+1],
                "target": raw_addr
            }
            if s_end > s_start:
                audit_info["multi_token"]["address"] = s_end - s_start + 1
        else:
            audit_info["failed"]["address"] = raw_addr

    return labels, audit_info

if __name__ == "__main__":
    with open("processed_data/sroie/test.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        
    total_aligned = Counter()
    total_failed = Counter()
    multi_token_counts = Counter()
    label_counts = Counter()
    
    for idx, s in enumerate(samples):
        expanded_words = []
        expanded_boxes = []
        for w, b in zip(s["words"], s["bboxes"]):
            w_list, b_list = split_line_into_words_with_bboxes(w, b)
            expanded_words.extend(w_list)
            expanded_boxes.extend(b_list)
            
        labels, info = align_sroie_corrected(expanded_words, s["entities"])
        
        for l in labels:
            label_counts[l] += 1
        for k in info["aligned"]:
            total_aligned[k] += 1
        for k in info["failed"]:
            total_failed[k] += 1
        for k in info["multi_token"]:
            multi_token_counts[k] += 1
            
    print("\n--- Test Set Alignment Results ---")
    print("Alignment success rate:")
    for ent in ["company", "date", "address", "total"]:
        succ = total_aligned[ent]
        fail = total_failed[ent]
        tot = succ + fail
        print(f"  {ent:10s}: {succ}/{tot} ({succ/tot*100:.1f}%) | Multi-token spans: {multi_token_counts[ent]}")
        
    print("\nCorrected Label Counts (Test):")
    for l, c in label_counts.most_common():
        print(f"  {l:15s}: {c}")
