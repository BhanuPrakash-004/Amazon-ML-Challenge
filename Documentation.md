# ML Challenge 2026: Business Entity Resolution — Methodology

**Pipeline:** deterministic 12-channel blocking + multilingual ANN → pre-rank top-K=22 →
36-feature LightGBM → calibration → F0.5 + per-source thresholds → singleton gate →
family support. Kaggle T4×2, <3h, offline business data.

---

## 1. Executive Summary

Country-partitioned recall-first blocking (exact/core/translit/compact name,
house/numeric/postal, rare name/address tokens) unioned with
`minishlab/potion-multilingual-128M` ANN (S2 top-12 + S3 top-12), pre-ranked to
K=22 per S1, then scored by a LightGBM pairwise classifier tuned directly for
macro F0.5 with a singleton gate. Precision-first decisions with zero/one/many
support and open-set country handling (France generalizes with no training labels).

---

## 2. Methodology

### 2.1 Problem Analysis

S1 is clean/Latin; S2/S3 multilingual with ~3.3% missing addresses; ~89–95%
house-number patterns; only ~21.85% positives share exact normalized names and
~8.27% exact addresses; all observed positives share country; ~3.46 targets/S1
mean; ~5.58% true singletons; S2/S3 contain duplicate name+address groups. All
assumptions re-verified from training data before use; nothing hard-codes US/India.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + entity decision.
**Core Innovation:** 12 capped deterministic channels (never intersected, always
unioned) plus a lightweight multilingual ANN recall channel, fused by a cheap
pre-score into exactly the K=22 candidates the model scores — so blocking sets a
~99% recall ceiling and LightGBM + F0.5 thresholds spend the precision budget.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys:** (1) country+name_norm (2) +translit (3) +core (4) +compact
  (5) +core+house (6) name-token+postal (7) name-token+house (8) rare name token
  (9) rare address token+house (10) numeric signature (11) postal+house
  (12) exact name+address. Rarity cutoff ~100–300; postings capped at 20000.
- **Dense channel:** name_core+translit+address_norm+translit through
  potion-multilingual-128M (no fine-tune), FP16 batches, FAISS
  (nlist=4096, nprobe=8, cosine/IP), GPU0=S2 / GPU1=S3 lanes, CPU fallback.
- **Union + dedup** with per-candidate metadata (exact/core/translit/numeric/
  address/ANN hits, block counts/ranks) → cheap pre-score → **top-22/S1** written
  to `candidate_pairs.tsv` (the exact inference set; matches ⊆ candidates).
- **Mandatory experiment (Sec 36):** A (name) → B (+house/numeric) → C (+rare) →
  D (+ANN), reporting recall, avg/p95 counts, runtime; smallest set at ~≥99%
  recall wins, then K ∈ {18,20,22,24,26} validated (recall@10/15/20/22/24/26 logged).
- **True matches not lost:** union-never-intersect, candidate recall prioritized
  over minimizing K, per-phase recall/candidate stats decide blocking fixes
  before any classifier complexification.

---

## 4. Matching Model

**Features (36, Sec 18 groups):**
- Name: exact/core/translit/compact exact, fuzz_ratio, WRatio, token_sort/set,
  jaccard, length ratio.
- Address: exact, ratio, token sort/set, jaccard, length ratio.
- Numeric: house exact/normalized, shared count, numeric jaccard, postal
  exact/conflict, house/numeric conflict.
- Retrieval: dense cosine, ANN rank/hit, shared block count.
- Rarity: rare name/address overlap. Structural: S2/S3 indicator, family size,
  duplicate count. Missingness: source/target/both-present.
- RapidFuzz only on final ~22/S1; never all-pairs.

**Model:** LightGBM binary (lr≈0.06, ~300 trees, leaves 31–63, depth 7–9,
subsample/colsample 0.8), early stopping on S1-entity 80/20 split
(200–300k S1 sampled, country×singleton×match-count stratified), all positives
kept, 2–4× mined hard negatives (same name/core/house/postal/rare/ANN/address,
wrong target).
**Calibration:** isotonic/Platt on held-out S1 val, kept only on F0.5 gain.
**Thresholds:** coarse 0.30–0.95 + refine, separate S2/S3 thresholds; singleton
gate on top score/margin/evidence; family-sibling support only on val gain;
BGE reranker (`BAAI/bge-reranker-v2-m3`) disabled by default, max top 2–3
ambiguous only.

---

## 5. Results & Error Analysis

- **F_0.5 (macro):** tuned directly on S1-entity validation (threshold + S2/S3 +
  singleton operating points); per-phase runtime, recall@K grid, avg/p50/p90/
  p95/p99/max candidates, pair P/R, singleton accuracy logged.
- **False positives:** same-core-name + same-house collisions across distinct
  businesses; mitigated by conflict features, hard negatives, singleton gate.
- **False negatives:** transliteration/multilingual pairs with no token overlap;
  recovered by translit blocks + ANN channel (primary D-lift).

---

## 6. Conclusion

Recall-first union blocking with a strict K=22 pre-rank gives the classifier a
high-ceiling, precision-tunable candidate set; direct F0.5/singleton/family
decisions convert pair scores into zero/one/many entity outputs that validate
cleanly, including unseen France, inside the 3-hour Kaggle budget.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
  src/config.py src/main.py src/experiment_blocking.py kaggle_run.py
  src/io/ src/preprocessing/ src/blocking/ src/embeddings/
  src/features/ src/training/ src/inference/ src/pipeline/ src/evaluation/
  README.md requirements.txt models/
kaggle/kaggle_pipeline.ipynb
```

Reproduce: `pip install -r requirements.txt`;
`python -m src.main --experiment --K 22`;
`python -m src.main --mode full --K 22` (or `python kaggle_run.py --K 22` on Kaggle);
validator: `python utils/validate_submission.py --matching output/matching_results.tsv
--candidate output/candidate_pairs.tsv --test-dir dataset/test`.

### B. Additional Results

Blocking A→D recall/count/runtime table, recall@K sweep, threshold/F0.5 curves,
and singleton/family ablations are printed to stdout and saved alongside
`models/selected_pipeline.json` on every run.
