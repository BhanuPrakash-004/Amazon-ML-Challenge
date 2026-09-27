# Business Entity Resolution — Production Pipeline (Kaggle T4×2, 3h)

Spec implementation: macro-F0.5 per S1, K=22 candidates, 12 deterministic
blocks + `minishlab/potion-multilingual-128M` ANN (S2 top-12 + S3 top-12),
cheap pre-rank, 36-feature LightGBM (all Sec-18 groups), isotonic/Platt
calibration, F0.5 + per-source S2/S3 thresholds, singleton gate,
duplicate-family support, BGE reranker disabled by default, open-set country
(France generalizes via Unicode normalization + multilingual embeddings).

## Layout (Sec 32)

```
src/config.py  src/main.py  src/experiment_blocking.py
src/io/loader.py src/io/writer.py
src/preprocessing/normalize.py transliterate.py numbers.py tokens.py rarity.py
src/blocking/exact_blocks.py numeric_blocks.py rare_blocks.py ann_retrieval.py candidate_union.py rerank.py
src/embeddings/encoder.py faiss_index.py
src/features/name_features.py address_features.py numeric_features.py semantic_features.py structural_features.py
src/training/make_training_pairs.py hard_negatives.py train_lgbm.py calibrate.py tune_threshold.py
src/inference/score_candidates.py singleton_gate.py family_support.py final_decision.py
src/pipeline/train.py src/pipeline/predict.py
src/evaluation/f05.py candidate_recall.py
kaggle_run.py  kaggle/kaggle_pipeline.ipynb
```

Legacy flat `src/*.py` (V2/V3) kept for compat (`run_train_v2.py`, laptop/*).

## Run

```bash
pip install -r requirements.txt
# mandatory first: Blocking A/B/C/D experiment (Sec 36)
python -m src.main --experiment --K 22
# full: train + test inference + validator (K=22; try 18/20/24/26 on validation)
python -m src.main --mode full --K 22 --batch-size 150000 --workers 4 --gpu-ids 0,1
# train / predict separately:
python -m src.main --mode train --train-path <train_dir> --model-path ./models
python -m src.main --mode predict --test-path <test_dir> --model-path ./models --output-path ../../output
python ../../student_resource/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../student_resource/dataset/test
```

Kaggle: open `kaggle/kaggle_pipeline.ipynb`, attach public dataset
`satwiksps/amazon-ml-challenge-2026` as Input, Run All. Or
`python kaggle_run.py --K 22`. Data resolves automatically
(`/kaggle/input/...` → `$DATA_ROOT` → `student_resource/dataset` → `dataset`).

## Design notes

* Country is an open-set partition (never hard-coded US/India); France flows
  through normalization + generic token/numeric + multilingual ANN.
* All 12 blocks capped (`MAX_POSTING_LEN=20000`); generic tokens never indexed.
* Embeddings cached once per record; S1 chunked 100–250k; float32/int32 +
  Arrow/Parquet discipline; GPU0=S2 / GPU1=S3 with CPU-FAISS fallback.
* `candidate_pairs.tsv` = exact top-K fed to the scorer; every match ⊆ candidates;
  one row per S1, S2/S3 ids only, no dupes. `matching_results.tsv` scored.
* BGE reranker (`BAAI/bge-reranker-v2-m3`) is implemented as opt-in only
  (`USE_RERANKER=False`); enable solely for top 2–3 ambiguous candidates when
  profiling proves budget remains.
* No internet business lookup, geocoding, or external augmentation anywhere.

## Acceptance (Sec 35)

Validator PASS; every test S1 exactly once; matches ⊆ candidates; no S1-as-target;
no dupes; France present in outputs; recall/count/F0.5/singleton stats logged
per phase including recall@K=10/15/20/22/24/26.
