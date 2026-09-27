# RUNBOOK — Sequential Laptop + Kaggle Hybrid (9h budget, adaptive 8–15)

> Baseline preserved: V2 CPU rare-token + 26-feat LightGBM, full-train val
> macro-F0.5 **0.8541** (`run_train_v2.py` / `run_predict_v2.py`).
> V3 adds multi-block UNION + E5 ANN + selective BGE + meta.
> Every command below exists in this repo. Run each step on the stated device.

```text
LAPTOP = i5-12500H, 12-16 threads, 16GB RAM, SSD (low-memory: chunked 50k S1,
  200k TSV chunks, int32 postings, Parquet+ZSTD; never 20M rows / all embeddings /
  dense matrices in RAM).
KAGGLE = 2x T4 15-16GB, ~30GB RAM, ~4 CPU, 12h session max
  (E5 ANN + ambiguous-only BGE; checkpointed/resumable per shard).
```

Pipeline: TSV → normalize → 7 blockers → UNION → raw 100–300 (internal) →
cheap prune → internal 20–30 → adaptive final 8–15 avg → candidate_pairs.tsv →
~60 feats → Stage-1 LightGBM → ambiguous top 5–15 → BGE → meta → singleton-safe
per-S1 F0.5 decision → matching_results.tsv → validator.
`--adaptive` (default) enforces avg ≤ 15 (`CANDIDATE BUDGET EXCEEDED` if over);
`--no-adaptive` = legacy K=150 SAFE BASELINE only.

---

## STEP 0 — Setup | DEVICE: BOTH (run once per machine)

| # | Device | Workdir | Command |
|---|--------|---------|---------|
| 0.1 | LAPTOP + KAGGLE | repo root | `pip install -r code/business_entity_resolution/requirements.txt` |
| 0.2 | LAPTOP + KAGGLE | repo root | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` |

Expect: laptop `False 0`, Kaggle `True 2`.

---

## STEP 1 — Laptop SAFE BASELINE train (V2, CPU-only) | DEVICE: LAPTOP

| # | Workdir | Command | Time | Output |
|---|---------|---------|------|--------|
| 1.1 | repo root | `python run_train_v2.py --sample-s1 100000 --val-s1 20000` | ~30–45min laptop | models + `results.txt` (recall + thr/F0.5) |

Smoke test only (≈1min, recall ~0 by design):

```bash
# DEVICE: LAPTOP, repo root
python run_train_v2.py --sample-s1 500 --val-s1 100 --top-k 30 --s23-head 50000
```

## STEP 2 — Laptop SAFE BASELINE inference | DEVICE: LAPTOP

| # | Workdir | Command | Time | Output |
|---|---------|---------|------|--------|
| 2.1 | repo root | `python run_predict_v2.py` | ~60–120min laptop | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

## STEP 3 — Validate baseline | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

## STEP 4 — V3 lexical train (adaptive, no GPU) | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP
cd code/business_entity_resolution
# 4.1 dev check — shapes only, NOT quality; must use temp model-root
python -m src.run_train --dev-mode --no-e5 --no-reranker --candidate-k 30 --model-root /tmp/dev_models
# 4.2 full lexical + adaptive 8-15
python -m src.run_train --data-root ../../student_resource/dataset \
  --output-root ../../output --model-root ./models --cache-root ./cache \
  --no-e5 --no-reranker
cd ../..
```

Flags: `--adaptive` (default) / `--no-adaptive` (legacy K=150),
`--internal-k 25 --final-max 15 --max-avg 15 --candidate-k 150` (internal K).
Stats: `results/run_*/candidate_stats.json`, `blocking_metrics.json`;
winner: `models/selected_pipeline.json`, `final_config.json`.

## STEP 5 — V3 predict + validate + package | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP
cd code/business_entity_resolution
python -m src.run_predict --data-root ../../student_resource/dataset \
  --model-root ./models --cache-root ./cache --output-root ../../output
python ../../student_resource/utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../student_resource/dataset/test
python ../make_package.py --team <TEAM>
cd ../..
```

Predict is resumable (skips S1 already in `matching_results.tsv`).
Checkpoints: `cache/checkpoints/*.json`; progress: `output/run_state.json`.

---

## HYBRID 9H PATH — follow Steps 6→13 in order

Budget: blocking 1.5–2h | transfer 10–20min | E5 1.5–2h | union 20–40min |
feats+Stage-1 1–1.5h | BGE 20–45min | meta+decision 30–60min | buffer 30–45min.
Fallback: FULL → FAST (drop BGE) → SAFE BASELINE (Step 1–3 outputs).

## STEP 6 — Prepare E5 package | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python scripts/create_transfer_e5.py --data-root student_resource/dataset --out transfer_e5 --split test
python scripts/verify_artifacts.py --dir transfer_e5
# wrapper (same output):
python laptop/prepare_e5.py --data-root student_resource/dataset --out transfer_e5 --split test
```

Produces `transfer_e5/{s1_queries.parquet,targets.parquet,manifest.json,
checksums.sha256}` (only `id,normalized_name,normalized_address,country,
source`). **Upload `transfer_e5/` as a Kaggle Dataset.**

## STEP 7 — E5 ANN retrieval | DEVICE: KAGGLE

| # | Device | Action |
|---|--------|--------|
| 7.1 | KAGGLE | New notebook → attach `transfer_e5` dataset |
| 7.2 | KAGGLE | Open `kaggle/e5_retrieval.ipynb` → Run All (prints GPU×2/VRAM, batch 128→64→32→16 on OOM, rows/s + ETA; `multilingual-e5-small` FP16 + FAISS IVF-PQ, name view → save → release → full view, per-shard checkpoint) |
| 7.3 | KAGGLE | Download `/kaggle/working/e5_out/e5_recovery.parquet` (+ manifest/checksums) to laptop as `downloaded_e5/` |

## STEP 8 — Import E5 | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python scripts/import_e5_results.py --in downloaded_e5 --out artifacts/e5_recovery
# optional offline merge inspection only (production union is inside pipelines):
python laptop/merge_candidates.py --candidates output/candidate_pairs.tsv \
  --e5 artifacts/e5_recovery/e5_recovery.parquet --out output/candidate_pairs_v3.tsv
```

Verifies checksum/schema/rows/dups → `artifacts/e5_recovery/e5_recovery.parquet`
`(s1_id,target_id,country,source,channel,rank,similarity)`.
Write `candidate_pairs_v3.tsv` first; copy to `candidate_pairs.tsv` only
after stats pass; keep `candidate_pairs_baseline.tsv`.

## STEP 9 — Stage-1 train | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python laptop/train_stage1.py --no-e5 --no-reranker
# identical to: cd code/business_entity_resolution && python -m src.run_train --no-e5 --no-reranker
```

Hard negatives 1:4–8 (`src/hard_negative_mining.py`), S1 split, macro-F0.5
tune (`src/threshold_tuning.py`), singleton-safe (`src/decision.py`).

## STEP 10 — Export ambiguous BGE workload | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python scripts/create_transfer_bge.py --scored output/stage1_scored.parquet --out transfer_bge --top-n 10
# wrapper:
python laptop/export_bge.py --scored output/stage1_scored.parquet --out transfer_bge --top-n 10
```

Schema `s1_id,target_id,name,address,country,source,stage1_probability`
(top 5–15 ambiguous/S1 only, never 20M). **Upload `transfer_bge/` as Kaggle Dataset.**

## STEP 11 — Selective BGE rerank | DEVICE: KAGGLE

| # | Device | Action |
|---|--------|--------|
| 11.1 | KAGGLE | New notebook → attach `transfer_bge` dataset |
| 11.2 | KAGGLE | Open `kaggle/bge_rerank.ipynb` → Run All (`BAAI/bge-reranker-v2-m3` FP16, batch auto-halve on OOM, per-shard checkpoint) |
| 11.3 | KAGGLE | Download `bge_scores.parquet (s1_id,target_id,bge_score)` to laptop as `downloaded_bge/` |

No fine-tune in 9h; pretrained only.

## STEP 12 — Import BGE + finalize | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python scripts/import_bge_results.py --in downloaded_bge --out artifacts/bge
python laptop/finalize.py --data-root student_resource/dataset \
  --model-root code/business_entity_resolution/models --output-root output
# finalize = src.run_predict (meta blend -> decision) + run_manifest.json + validator
```

Meta (`src/meta_model.py` + `src/ensemble.py`, validation-chosen blend) →
calibration → per-S1 F0.5 decision (0/1/many, empty allowed) → local conflicts.

## STEP 13 — Final validate | DEVICE: LAPTOP

```bash
# DEVICE: LAPTOP, repo root
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

Required `output/`: `candidate_pairs.tsv`, `matching_results.tsv`,
`candidate_stats.json`, `run_manifest.json` (+ `run_state.json`,
`results/run_*/`, `config/final.yaml` reference).
Compare BASELINE vs NEW (recall, avg, F0.5/P/R, singleton, runtime) in
`experiment_results.csv`; order: prune → +E5 → +hard-neg → +BGE → +meta.
Report: pairs YES/NO+avg+recall; matches YES/NO+validator; best vs baseline
F0.5; runtime + peak RAM/GPU; mode FULL/FAST/SAFE_BASELINE.

## Do NOT

Rebuild; swap LightGBM for a large deep model; fine-tune transformers; BGE all
candidates; ship K=150 final; dense matrices; 20M-row frames; full embeddings
in RAM; duplicate datasets; recompute done shards; random-only negatives; force
matches; hard-code US/India; destroy originals; change TSV schemas; delete baseline.

## File map

```text
DEVICE LAPTOP: src/normalize.py, transliterate.py,
  blocking_{exact,rare,numeric,char}.py, candidate_union.py, candidate_prune.py,
  features_*.py, features_all.py, train_matcher.py, hard_negative_mining.py,
  meta_model.py, ensemble.py, decision.py, threshold_tuning.py,
  train_pipeline.py, predict_pipeline.py, run_train.py, run_predict.py,
  run_state.py, checkpoint.py, run_train_v2.py, run_predict_v2.py
DEVICE KAGGLE: src/blocking_e5.py, embeddings.py, faiss_index.py, reranker.py,
  kaggle/e5_retrieval.ipynb, kaggle/bge_rerank.ipynb
BOTH: laptop/*.py, scripts/*.py, config/final.yaml
```
