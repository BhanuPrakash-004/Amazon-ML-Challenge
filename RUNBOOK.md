# RUNBOOK — Laptop + Kaggle (clone + public dataset)

> Baseline V2: CPU rare-token + 26-feat LightGBM, val macro-F0.5 **0.8541**.
> V3: multi-block UNION + E5 ANN + selective BGE + meta. `--adaptive` (default) caps avg ≤ 15.
> Repo: `https://github.com/BhanuPrakash-004/Amazon-ML-Challenge.git`

Rule: LAPTOP = `python ...`. KAGGLE = `!python ...`. `%cd` persists, `!cd` does not.

---

## 0. SYNC — do this first (once)

Laptop has uncommitted fixes (requirements + notebook auto-IN). Kaggle has the
old commit. Sync before anything else. While S1/S7 are running, DO NOT sync —
wait for both to finish (they don't need the new code).

```bash
# LAPTOP, repo root — publish fixes
git add RUNBOOK.md code/business_entity_resolution/requirements.txt kaggle/e5_retrieval.ipynb kaggle/bge_rerank.ipynb
git commit -m "fix kaggle paths and requirements"
git push origin main
```

```python
# KAGGLE — after laptop push, and only when no job is running
%cd /kaggle/working/Amazon-ML-Challenge
!git pull
!git log --oneline -3
```

Alternate — no `git pull` (use when pull fails or is blocked; input-only safe,
no repo wipe, running jobs unaffected):

```python
# 0a. LAPTOP: upload these 4 files as ONE Kaggle Dataset (site → Datasets →
# New Dataset): RUNBOOK.md, code/business_entity_resolution/requirements.txt,
# kaggle/e5_retrieval.ipynb, kaggle/bge_rerank.ipynb. Name it e.g. `patch`.
# 0b. KAGGLE notebook: Add-ons → Datasets → Add `patch`. Then:
%cd /kaggle/working/Amazon-ML-Challenge
!ls /kaggle/input/
!find /kaggle/input -maxdepth 3 -name "e5_retrieval.ipynb" -o -maxdepth 3 -name "requirements.txt" | head
!P=$(dirname $(find /kaggle/input -name "e5_retrieval.ipynb" | head -n 1)); echo $P; ls $P
!P=$(dirname $(find /kaggle/input -name "e5_retrieval.ipynb" | head -n 1)); cp "$P/e5_retrieval.ipynb" kaggle/e5_retrieval.ipynb; cp "$P/bge_rerank.ipynb" kaggle/bge_rerank.ipynb; cp "$P/requirements.txt" code/business_entity_resolution/requirements.txt; cp "$P/RUNBOOK.md" RUNBOOK.md
!pip install -r code/business_entity_resolution/requirements.txt
```

---

## 1. WHERE EVERYTHING LIVES

| Place | Path | Read / Write |
|---|---|---|
| Laptop repo | `C:\Users\sanga\OneDrive\Desktop\ML-Challenge` (repo root) | read + write |
| Laptop data | `student_resource/dataset/train/*.tsv`, `test/*.tsv` | read only (never commit, gitignored) |
| Laptop outputs | `output/`, `models→code/business_entity_resolution/models/`, `artifacts/`, `downloaded_e5/`, `downloaded_bge/` | write |
| Kaggle repo | `/kaggle/working/Amazon-ML-Challenge` (git clone) | read + write (`/kaggle/working` only) |
| Kaggle public data (input only) | `/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/train/*.tsv`, `test/*.tsv` | read only — you can only ADD datasets via Input, never write there |
| Kaggle working outputs | `/kaggle/working/e5_transfer/`, `/kaggle/working/e5_out/`, `/kaggle/working/transfer_bge`, `/kaggle/working/bge_out/` | write + download |
| Kaggle uploaded input (after Step 10) | `/kaggle/input/transfer-bge/bge_pairs.parquet` (slug varies) | read only |

Shortcut used below:

```text
KAGGLE_DATA = /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset
```

No file edits needed on Kaggle. Notebooks auto-resolve:
E5 `IN`: `$E5_IN` → `/kaggle/working/e5_transfer` → `/kaggle/input/transfer-e5`.
BGE `IN`: `$BGE_IN` → `/kaggle/working/transfer_bge` → `/kaggle/input/transfer-bge`.
V2 scripts need the symlink from Step 0K (they have no `--data-root` flag).

---

## 2. WHAT RUNS WHERE + WHAT RUNS IN PARALLEL

| Group | Laptop (sequential) | Kaggle (sequential) | Parallel? |
|---|---|---|---|
| G1 now | S1 V2 train | S7-transfer build + S7 E5 notebook | YES — independent (models/ vs e5_transfer/) |
| G2 | S2 predict → S3 validate → S4 V3 train → S5 V3 predict → S9 Stage-1 | E5 keeps running | YES |
| SYNC-1 | — | download `e5_recovery.parquet` | Kaggle→laptop, after G1 Kaggle done |
| G3 | S8 import E5 → S10 export BGE | idle (or G2 leftover) | — |
| SYNC-2 | upload `transfer_bge/` as Kaggle Dataset + Attach | attach it as Input | Laptop→Kaggle, after S10 |
| G4 | wait / analyze | S11 BGE notebook | — |
| SYNC-3 | download `bge_scores.parquet` | — | Kaggle→laptop, after S11 |
| G5 | S12 finalize → S13 validate → package | done | — |

Never overlap: S2 after S1 (needs models). S8 after SYNC-1. S11 after SYNC-2. S12 after SYNC-3.

---

## 3. TRANSFER TABLE — exact folders, exact method

| # | Direction | Laptop path | Kaggle path | How |
|---|---|---|---|---|
| T1 | — (none after S1) | `models/` stays | — | S1 models stay on laptop for S2. Do NOT upload. |
| T2 | Kaggle→laptop (after S7) | `downloaded_e5/e5_recovery.parquet` | `/kaggle/working/e5_out/e5_recovery.parquet` | Kaggle notebook output → Download → place in `downloaded_e5/` → Step 8 |
| T3 | Laptop→Kaggle (after S10) | `transfer_bge/bge_pairs.parquet` + manifest | `/kaggle/input/transfer-bge/bge_pairs.parquet` (slug may be `transfer_bge`) | Kaggle site → Create Dataset → upload `transfer_bge/` contents → notebook Add-ons → Add → attach |
| T4 | Kaggle→laptop (after S11) | `downloaded_bge/bge_scores.parquet` | `/kaggle/working/bge_out/bge_scores.parquet` | Download → place in `downloaded_bge/` → Step 12 |
| T5 | Final out | `output/*.tsv` + `*_submission.zip` | `/kaggle/working/...` (if 11K used) | Download before session ends |

Kaggle rule: new data can ONLY enter via Input (read-only). Everything you
compute lands in `/kaggle/working` (downloadable). Never write to `/kaggle/input`.

---

## 4. SETUP

### STEP 0 — Laptop | repo root

```bash
pip install -r code/business_entity_resolution/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `False 0`.

### STEP 0K — Kaggle | one notebook, top to bottom

Attach Input first: Add-ons → Datasets → Add `satwiksps/amazon-ml-challenge-2026`.
After Step 10, also attach your uploaded `transfer-bge` dataset.

```python
!git clone https://github.com/BhanuPrakash-004/Amazon-ML-Challenge.git
%cd Amazon-ML-Challenge
!git log --oneline -3
!pip install -r code/business_entity_resolution/requirements.txt
!python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expect `True 2`.

```python
!ls /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/train | head
!ls /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/test | head
!mkdir -p student_resource
!ln -sfn /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset student_resource/dataset
!ls student_resource/dataset/train | head
```

Optional overrides (no edit needed — only if auto-resolve picks wrong path):

```python
%env E5_IN=/kaggle/working/e5_transfer
%env BGE_IN=/kaggle/input/transfer-bge
```

---

## 5. LAPTOP STEPS (repo root unless noted)

```bash
# S1 — V2 train. Smoke first only for env check (toy model), then full (real).
python run_train_v2.py --sample-s1 500 --val-s1 100 --top-k 30 --s23-head 50000
python run_train_v2.py --sample-s1 100000 --val-s1 20000
```

```bash
# S2 — V2 predict (needs S1 models)
python run_predict_v2.py
```

```bash
# S3 — validate V2
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

```bash
# S4 — V3 lexical train (needs S3 done; can overlap Kaggle E5)
cd code/business_entity_resolution
python -m src.run_train --dev-mode --no-e5 --no-reranker --candidate-k 30 --model-root ./cache/dev_check
python -m src.run_train --data-root ../../student_resource/dataset --output-root ../../output --model-root ./models --cache-root ./cache --no-e5 --no-reranker
cd ../..
```

```bash
# S5 — V3 lexical predict + package
cd code/business_entity_resolution
python -m src.run_predict --data-root ../../student_resource/dataset --model-root ./models --cache-root ./cache --output-root ../../output --no-e5 --no-reranker
python ../../student_resource/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../student_resource/dataset/test
python make_package.py --team <TEAM>
cd ../..
```

```bash
# S8 — after SYNC-1 (downloaded_e5 present)
python scripts/import_e5_results.py --in downloaded_e5 --out artifacts/e5_recovery
```

```bash
# S9 — Stage-1 (produces output/stage1_scored.parquet)
python laptop/train_stage1.py --no-e5 --no-reranker
```

```bash
# S10 — export BGE workload, then do SYNC-2 upload
python scripts/create_transfer_bge.py --scored output/stage1_scored.parquet --out transfer_bge --top-n 10
```

```bash
# S12 — after SYNC-3 (downloaded_bge present)
python scripts/import_bge_results.py --in downloaded_bge --out artifacts/bge
python laptop/finalize.py --data-root student_resource/dataset --model-root code/business_entity_resolution/models --output-root output
```

```bash
# S13 — final gate
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

---

## 6. KAGGLE STEPS

```python
# S7-transfer — build from public data (no laptop upload). After 0K.
%cd /kaggle/working/Amazon-ML-Challenge
!nvidia-smi
!mkdir -p /kaggle/working/e5_transfer /kaggle/working/e5_out
!python scripts/create_transfer_e5.py --data-root /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset --out /kaggle/working/e5_transfer --split test
!python scripts/verify_artifacts.py --dir /kaggle/working/e5_transfer
!ls -lh /kaggle/working/e5_transfer/
```

```python
# S7-E5 — Run All, no edits. Then SYNC-1 download.
# Open kaggle/e5_retrieval.ipynb → Run All → /kaggle/working/e5_out/e5_recovery.parquet
!ls -lh /kaggle/working/e5_out/
```

```python
# S11 — after SYNC-2 (transfer-bge attached as Input). Run All, no edits.
%cd /kaggle/working/Amazon-ML-Challenge
!nvidia-smi
!ls /kaggle/input/transfer-bge
!mkdir -p /kaggle/working/bge_out
# Open kaggle/bge_rerank.ipynb → Run All → /kaggle/working/bge_out/bge_scores.parquet
!ls -lh /kaggle/working/bge_out/
!python -c "import pandas as pd; df=pd.read_parquet('/kaggle/working/bge_out/bge_scores.parquet'); print(len(df)); print(df.head(3).to_string())"
```

Optional full-Kaggle (replaces laptop S4/S5 when laptop is busy):

```python
%cd /kaggle/working/Amazon-ML-Challenge/code/business_entity_resolution
!python -m src.run_train --data-root /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset --output-root ../../output --model-root ./models --cache-root ./cache --no-e5 --no-reranker
!python -m src.run_predict --data-root /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset --model-root ./models --cache-root ./cache --output-root ../../output --no-e5 --no-reranker
!python ../../student_resource/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset/test
```

---

## 7. SYNC POINTS — where to add / where it lands

SYNC-1 (Kaggle→laptop): in Kaggle, right-click
`/kaggle/working/e5_out/e5_recovery.parquet` → Download. On laptop create
`downloaded_e5/` and put the file there. Then S8. Verify:
`python scripts/verify_artifacts.py --dir downloaded_e5` (if manifest present).

SYNC-2 (laptop→Kaggle): on laptop, `transfer_bge/` contains
`bge_pairs.parquet` + `manifest.json` + `checksums.sha256`. On kaggle.com →
Datasets → New Dataset → upload those files → Create → notebook Add-ons →
Add `transfer-bge`. It lands at `/kaggle/input/transfer-bge/` (read-only).
Notebook finds it automatically; if slug differs (`transfer_bge`), set
`%env BGE_IN=/kaggle/input/transfer_bge`.

SYNC-3 (Kaggle→laptop): download `/kaggle/working/bge_out/bge_scores.parquet`
into laptop `downloaded_bge/`. Then S12.

## Do NOT

Push datasets/models/logs; run S2 before S1; run S8 before SYNC-1; run S11
before SYNC-2; run S12 before SYNC-3; `git pull` on Kaggle mid-job; write to
`/kaggle/input`; BGE all candidates; ship K=150; force matches.
