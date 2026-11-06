# Phase 10: Confidence Estimation Methodology

**Project**: Multimodal Transformer-Based Intelligent Document Understanding and Information Extraction  
**Date**: 2026-09-09  
**Status**: APPROVED STANDARD METHODOLOGY  

---

## 1. Overview & Problem Formulation

In token-classification multimodal transformers such as LayoutLMv3, the classification head produces a raw logit vector $\mathbf{z}_t \in \mathbb{R}^K$ for each token position $t$ across the $K$ label classes (including `O`, `B-<ENTITY>`, and `I-<ENTITY>`).

Applying the softmax function yields a normalized probability distribution over classes:
$$P(y_t = k \mid \mathbf{x}) = \frac{\exp(z_{t,k})}{\sum_{j=1}^K \exp(z_{t,j})}$$

During token prediction, the predicted class is:
$$\hat{y}_t = \arg\max_k P(y_t = k \mid \mathbf{x})$$
and the single-token confidence is the maximum posterior probability:
$$c_t = \max_k P(y_t = k \mid \mathbf{x})$$

However, document information extraction evaluates **multi-token entity spans** (e.g., multi-word company names, multi-line addresses, or item descriptions) rather than isolated subwords. A rigorous mathematical aggregation method is required to estimate the overall confidence of reconstructed entities.

---

## 2. Token-to-Word Alignment & First-Subword Convention

During preprocessing, words are tokenized into subwords using Byte-Pair Encoding (BPE). In accordance with the project's training protocol:
1. Only the **first subword token** of each word is supervised with the ground-truth entity label.
2. Subsequent continuation tokens are assigned the PyTorch cross-entropy ignore index (`-100`).

At inference time, we maintain complete consistency with this alignment protocol:
- For each word $w_i$ comprising subwords $\{t_{i,1}, t_{i,2}, \dots, t_{i,m}\}$, the word-level prediction $\hat{y}(w_i)$ and word-level confidence $c(w_i)$ are derived from the **first subword token** $t_{i,1}$:
$$\hat{y}(w_i) = \hat{y}(t_{i,1}), \quad c(w_i) = \max_k P(y_{t_{i,1}} = k \mid \mathbf{x})$$

---

## 3. Entity Span Confidence Aggregation: Mean Token Probability

An extracted entity span $E$ consists of an ordered sequence of $N$ constituent words:
$$E = (w_1, w_2, \dots, w_N), \quad N \ge 1$$
where $w_1$ is labeled `B-<TYPE>` and $w_2, \dots, w_N$ are labeled `I-<TYPE>`.

### Chosen Method: Arithmetic Mean Word Probability
The confidence score of entity span $E$ is defined as the arithmetic mean of its constituent word-level confidence probabilities:
$$\text{Conf}_{\text{mean}}(E) = \frac{1}{N} \sum_{i=1}^N c(w_i)$$

### Comparison with Alternative Methods

| Method | Mathematical Definition | Advantages | Disadvantages | Selection Decision |
|---|---|---|---|---|
| **Arithmetic Mean** | $\frac{1}{N} \sum_{i=1}^N c(w_i)$ | • Smooth and interpretable.<br>• Balanced across long spans (e.g., addresses).<br>• Robust to single punctuation ambiguity. | Can mask a single very low-confidence subtoken if others are high. | **SELECTED PRIMARY METHOD** |
| **Minimum Probability** | $\min_{i \in \{1,\dots,N\}} c(w_i)$ | • Ultra-conservative worst-case guarantee.<br>• Flags any uncertain subword. | Severely penalizes long entities; one rare word drives entire span confidence down. | Available as optional metric in schema. |
| **Geometric Mean (Joint)** | $\left(\prod_{i=1}^N c(w_i)\right)^{1/N}$ | Approximates joint likelihood under naive independence assumption. | Drops rapidly as span length $N$ increases; non-intuitive scaling for users. | Rejected for end-user reporting. |

### Justification for Arithmetic Mean:
For document understanding, entities such as `ADDRESS` in SROIE often span 10–25 words. In long strings containing commas, postal codes, and rare street names, a single subword may receive confidence $0.65$ while the remaining 15 words receive $>0.98$. Using minimum probability would artificially label the entire verified address as $0.65$ confidence, distorting downstream thresholding. The arithmetic mean reflects the aggregate certainty of the full semantic span.

---

## 4. Multi-Span Aggregation (Same Entity Type)

In document receipts, the same entity type may appear across multiple disconnected lines (e.g., a multi-line address where each line starts with `B-ADDRESS`).

When multiple spans $E_1, E_2, \dots, E_M$ of the same entity type are merged into a single document-level entity field:
1. The concatenated text is:
$$\text{Text}_{\text{merged}} = \text{Text}(E_1) \mathbin{\Vert} \text{Text}(E_2) \mathbin{\Vert} \dots \mathbin{\Vert} \text{Text}(E_M)$$
2. The merged confidence is the **length-weighted mean** across all spans:
$$\text{Conf}_{\text{merged}} = \frac{\sum_{j=1}^M N_j \cdot \text{Conf}(E_j)}{\sum_{j=1}^M N_j}$$
where $N_j$ is the word count of span $E_j$.

---

## 5. Implementation Guarantees & Bounds

1. **Probability Bounds**: For every extracted entity, $0.0 \le \text{Conf}(E) \le 1.0$.
2. **Rounding**: For readability, JSON schemas and UI displays round confidence values to 4 decimal places ($0.XXXX$).
3. **No Arbitrary Numbers**: Every confidence score originates directly from the Softmax tensor output of the PyTorch forward pass. No synthetic, default, or hardcoded values are substituted.
