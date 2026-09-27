# Business Entity Resolution — V3 Final Pipeline (Kaggle T4×2)

V2 diagnosis: full-train blocking recall ≈ 0.81–0.82 → val macro-F0.5 ≈ 0.854.
Bottleneck is candidate generation, not LightGBM. V3 adds multi-block UNION
(exact + rare name/address + numeric/postal + char 3-gram + transliteration +
multilingual E5 ANN), cheap prune, hard negatives, enriched LightGBM, selective
cross-encoder rerank, meta-model, entity-level F0.5 decision. V2 files kept and reused.

## Problem / data
`dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv`,
`dataset/test/{test_source1,test_source2,test_source3}.tsv` (or `student_resource/dataset/...`
locally — auto-detected). S1 zero/one/many matches in S2/S3. Metric: macro F0.5 per S1,
singletons included. No external business data, no geocoding, no web lookup — all
signals local. France works with no France training labels (open-set country handling).

## Architecture
normalization (multi-representation, Unicode-safe) → exact / rare-name / rare-address /
numeric-postal / char-3gram / transliteration / E5-ANN(name+full) → UNION (never intersect)
→ cheap prune → rich pair features → LightGBM → selective reranker (top-N ambiguous)
→ meta-model → entity decision (zero/one/many) → TSVs + validator.

## Kaggle setup (T4×2, 1×T4 minimum, ~30GB RAM, 20GB /kaggle/working, 12h sessions)
Stages are resumable via `cache/checkpoints/`. Run across sessions:
1. preprocess + lexical indexes → 2. E5 index A → 3. E5 index B → 4. candidates/mining →
5. matcher/selection → 6. test candidates → 7. scoring/rerank → 8. outputs/validator/ZIP.
GPU only for E5/reranker; LightGBM/lexical/output on CPU. fp16 sharded embeddings;
raw shards deleted after compressed FAISS (IndexIVFPQ) persisted. One index in GPU
memory at a time; OOM auto-halves batch (min 8).

## Commands (run from `code/business_entity_resolution/`)
```bash
pip install -r requirements.txt
# full train (lexical + E5 + reranker pipeline)
python -m src.run_train --data-root dataset --output-root output --model-root models --cache-root cache --candidate-k 150 --rerank-top-n 15
# lexical-only (no GPU needed)
python -m src.run_train --data-root dataset --no-e5 --no-reranker --candidate-k 150
# dev-mode: correctness/shapes only, NOT for quality decisions
python -m src.run_train --dev-mode --no-e5 --no-reranker --candidate-k 30
# predict (resumable) + validator
python -m src.run_predict --data-root dataset --model-root models --cache-root cache --output-root output
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
# analysis
python -m src.analyze_blocking_errors
```
Legacy V2 entry points (repo root): `python run_train_v2.py`, `python run_predict_v2.py`.

## Memory/GPU requirements
Lexical path: <8GB RAM. Full E5: ~30GB RAM, 1×16GB T4 (batch auto-tuned, seq 256).
Never: giant pandas loads, giant token→list dicts, dense 10M TF-IDF, full feature
DataFrames, dual raw embedding matrices. Chunked S1 (50k), sharded features/ mmap cache.

## Licenses / fair play
`intfloat/multilingual-e5-small` (MIT), `BAAI/bge-reranker-v2-m3` (Apache-2.0),
LightGBM (MIT), FAISS (MIT), RapidFuzz (MIT). Pretrained general-language models only;
no external business databases/APIs/geocoding. All matching signals from competition data.

## Outputs / reproducibility
`output/matching_results.tsv` (source1_entity_id, matched_entity_ids),
`output/candidate_pairs.tsv` (source1_entity_id, candidate_entity_ids) — every test S1
exactly once, matches ⊆ candidates, S2/S3 ids only. `results/run_*/` holds config,
metrics, blocking report, thresholds, runtime. `models/selected_pipeline.json` +
`final_config.json` record the validation winner. Seed 42, deterministic S1-level splits.
Final ZIP via `python make_package.py --team <name>` (excludes cache/embeddings/indexes).
