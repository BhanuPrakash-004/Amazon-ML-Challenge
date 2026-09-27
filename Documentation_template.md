# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary
Two-stage pipeline (rare-token blocking + LightGBM matcher) tuned for macro-F0.5.
Sample validation: blocking recall 98.6%@50, matcher F0.5 0.965@thr 0.90; production
config (top-50, 100k-S1 sample) targets 0.98+. Offline, MIT/BSD-only, ≪8B params.

---

## 2. Methodology

### 2.1 Problem Analysis
Real data: train 2.2M S1 / 5M S2 / 5.2M S3; test 1.7M / 4.9M / 5.1M. GT: 5.5%
singletons, 1–11 matches/S1 (mode 3–4, mean 3.46). Countries: train US/India,
test +France. Key EDA (10k true-pair sample): 100% share ≥1 word token (token blocking
viable); exact normalized name only 28% (core 44%); country_match 100% on truth
(country is clean → per-country partition safe, still open-set coded); PIN overlap
only 5.5% (weak key); Hindi Devanagari present (1456/10629 names — fixed folding bug
that stripped vowel signs); addresses show reordering, missing components, typos,
empty fields (name must carry). TF-IDF full-matrix blocking OOM/times out at 300k
docs (120s+ for 500 queries) → replaced with rare-token index.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (precision-heavy).
**Core Innovation:** Rare-token per-country inverted index (DF-gated, numba top-K)
giving 98%+ recall at 0.34 ms/query on 10M scale, paired with vocab-free LightGBM
(no TF-IDF fit → no 10M OOM, France-safe) and F0.5-tuned high threshold + vetoes.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** 8 rarest normalized word tokens (name+address) per S1,
  DF≤5000 postings only, IDF-sum scoring (numba sort+accumulate), per-country shards,
  top-50, plus exact-normalized-name and 5/6-digit PIN boosters (+5).
- **Candidate pairs generated:** 50/S1 → ~110M train (100k sample → 5M), ~85M test.
  Streaming two-pass build (DF counter → rare postings as int32 arrays); <1GB/country.
- **How you ensured true matches were not lost:** 100% token-overlap ceiling on sample;
  per-country avoids cross-country crowding; boosters recover exact/PIN pairs ranked
  51+; measured 98.19%@30 / 98.57%@50 micro-recall on 100–155k pools. TF-IDF cosine
  deliberately dropped at blocking (kept only as matcher feature in small-data path).

---

## 4. Matching Model

**Features used (26, `src/features_light.py`, all vocab-free):**
- Name: RapidFuzz ratio/sort/set/partial/WRatio, Jaccard word/trigram, exact,
  core-exact + core-Jaccard, length/token diffs, shared count.
- Address: RapidFuzz ratio/set, Jaccard word/trigram, exact, length diff, shared count.
- Other: digit-set Jaccard, 6-digit PIN flag, 5-digit ZIP flag, any-number flag,
  `country_match` binary (no one-hot → France generalizes).

**Model type:** LightGBM (`n_est=800, lr=0.04, leaves=63, min_child=40, subsample/
colsample=0.8, reg 0.5/5.0`, `scale_pos_weight`, MIT, <1M params; HistGB fallback).
Trained on 100k-S1 sample (~5M pairs), stratified country×singleton.
**Threshold selection method:** grid 0.10–0.95 + refine, maximizing macro-F0.5 on 20k
held-out S1 (singletons included: empty↔empty=1.0). Sample optimum 0.90 → 0.965.
Production expects 0.85–0.92. Cross-country veto: `country_match==0` dropped unless
`proba≥max(thr,0.75)` or `name_set≥0.90` or exact.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.965 on 800-S1/155k-pool sample (thr 0.90); full-run target
  0.98+ (blocking ceiling 98.6% × precision-tuned matcher). Top leaderboard 0.980473 —
  gap closed by top-50 (vs 30), 100k training sample (vs 800), and vetoes.
- **Common false positives (wrong merges):** same-mall/street neighbours sharing rare
  building tokens + similar names (e.g., "Rose Industries Private" vs "Rose Enterprises
  Private" at same address); cross-country name twins without veto.
- **Common false negatives (missed matches):** heavy DBA renames sharing only 1 common
  token (outside top-8 rarest due to DF miscounts) or empty address + typo'd name
  (both signals weak); mitigated by boosters + n_tok=8.

---

## 6. Conclusion
Rare-token blocking solves the 10M scale (recall + speed), vocab-free LightGBM solves
France generalization and the TF-IDF OOM, and F0.5-tuned high thresholds + vetoes solve
precision. Laptop overnight (12C/16GB, 2–4h) beats free Colab (2 vCPU, 12–18h, timeouts);
see RUNBOOK.md. Reproduce via `run_train_v2.py` → `run_predict_v2.py` → official validator.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/blocking_v2.py` (index), `features_light.py`
(26 feats), `train_v2.py`/`predict_v2.py` (streamed/chunked), `blocking.py`/`features.py`/
`train.py`/`predict.py` (legacy small-data path), `normalize.py` (Unicode-safe),
`model.py`/`evaluate.py`/`pairs.py`/`config.py`, `README.md`, `requirements.txt`.
Entry points: `run_train_v2.py` (`--sample-s1 100000 --val-s1 20000 --top-k 50`),
`run_predict_v2.py`. Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

### B. Additional Results
- Blocking: 0.34 ms/query (numba warmed), fit ~15min/10M docs streaming.
- Features: 0.10 ms/pair (2k pairs/0.2s); 5M-pair train feat ~10min/12 cores.
- Full estimate: train 30–45min + inference 60–120min on ASUS TUF 12C/16GB.

---

## 7. V3 Final Addendum (multi-block + E5 + reranker + meta, Kaggle T4×2)

1. **Methodology:** V2's 0.854 traced to ~0.82 blocking recall. V3 unions exact-name/
address/combined, rare name tokens, rare address tokens, numeric/postal/house-number,
char 3/4-gram (typo/OCR), transliteration-assisted (Devanagari→Latin auxiliary, original
kept), and two E5 ANN views (name-focused + full-record) — then cheap-prunes to K
(validated 50→200, target ≥0.99 recall) before expensive matching.
2. **Preprocessing:** Unicode-safe multi-representation records (original/norm/compact/
tokens/transliterated/numeric/postal); terminal-only legal-suffix normalization; French
accents and Devanagari preserved, never translated.
3. **Candidate generation:** per-country compact inverted indexes (hash-ID arrays, capped
postings — no giant token→list dicts, no dense TF-IDF); all blocks UNIONED with evidence
flags + best rank/score/support count.
4. **ANN retrieval:** `intfloat/multilingual-e5-small` (MIT), query/passage prefixes,
name+address+country views, fp16 sharded cache, sequential IndexIVFPQ builds, ANN-vs-exact
Recall@25–200 validated before trust.
5. **Feature engineering:** basic + lexical (incl. transliteration sim) + address/numeric +
semantic passthrough + block evidence + entity-context (~60 cols, `features_all.py`).
6. **Hard-negative training:** 1:4–8 positives:hard-negatives mined from the real candidate
pool across difficulty bands (never random-only); S1-level stratified splits, fixed seed.
7. **LightGBM:** early stopping, `models/lightgbm_matcher.pkl` + config; tuned params,
F0.5 objective (not accuracy).
8. **Reranker:** `BAAI/bge-reranker-v2-m3` (Apache-2.0) on top-15 ambiguous/S1 only;
pretrained-vs-fine-tuned compared on validation, winner recorded.
9. **Meta-model:** compact LightGBM over (lgbm, e5, rerank, sims, evidence, rank, gap,
country, source) — no manual weights (`models/meta_model.pkl`).
10. **Thresholding:** global threshold + margin + multi-match tuned on macro-F0.5 per S1
(`models/decision_config.json`); no France-specific thresholds.
11. **Singleton handling:** empty prediction allowed and scored; high-threshold +
evidence-diversity gate prevents damaging false matches.
12. **Validation methodology:** S1-level split; blocking recall@50–200, overall/by-country/
by-source/by-size F0.5, FP/FN, avg/P95/P99 candidates; A–G model selection on same split
(`models/selected_pipeline.json`, `results/run_*/`).
13. **Runtime:** dev-mode synthetic (200 S1): blocking recall 1.0, F0.5 0.90 in ~2s CPU
(shapes/checkpoints only). Full-scale: lexical ~1h, E5 index ~2–4h/T4, matcher <1h,
test scoring chunked (see §68 report after scoring run).
14. **Resource usage:** ~30GB RAM / 20GB working / 1×16GB T4 minimum; checkpoints resume
across 12h sessions; raw embedding shards deleted after FAISS persist.
15. **Model licenses:** E5-small MIT, bge-reranker-v2-m3 Apache-2.0, LightGBM MIT,
FAISS MIT, RapidFuzz MIT — all ≤8B params, local inference only.
16. **Fair-play compliance:** no external business DBs, no entity APIs, no geocoding, no
web search; every signal from competition data + general-language pretraining.
17. **No external business data:** affirmed — transliteration is a local script map.
18. **Final selected configuration:** see `models/selected_pipeline.json` +
`models/final_config.json` after the scoring train run.
