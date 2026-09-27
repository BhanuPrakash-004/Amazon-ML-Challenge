"""Spec-compliant central config (Sec 4, 8, 12, 14-16, 20, 22, 28, 33).

Keeps legacy V2/V3 constants for backward compat (run_train_v2 etc.)
and adds the Sec-2..36 production pipeline defaults:
  K=22 final candidates, 12 deterministic blocks, potion-multilingual-128M,
  FAISS nlist=4096/nprobe=8, LightGBM 250-350 estimators, F0.5 tuning.

All paths auto-resolve for: laptop repo, Kaggle clone, Kaggle public dataset.
"""
import os

# ---------------------------------------------------------------- paths ---
def _find_repo_root():
    here = os.path.abspath(os.getcwd())
    cur = here
    cands = []
    for _ in range(7):
        for sub in ("student_resource/dataset", "dataset"):
            if os.path.isdir(os.path.join(cur, sub)):
                cands.append(cur)
                break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Kaggle public dataset mount (input-only)
    for k in ("/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset",
              "/kaggle/input/amazon-ml-challenge-2026/dataset"):
        if os.path.isdir(k):
            return os.path.dirname(k), k
    if cands:
        for c in [here] + cands:
            if os.path.isdir(os.path.join(c, "student_resource", "dataset")):
                return c, os.path.join(c, "student_resource", "dataset")
        return cands[0], os.path.join(cands[0], "dataset")
    try:
        fdir = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
        if os.path.isdir(os.path.join(cand, "dataset")):
            return cand, os.path.join(cand, "dataset")
    except Exception:
        pass
    return here, os.path.join(here, "dataset")


ROOT, _DATA = _find_repo_root()

# env overrides (Kaggle notebooks set these; CLI flags win over env)
DATA_ROOT = os.environ.get("DATA_ROOT", _DATA)
if os.path.isdir(os.path.join(DATA_ROOT, "train")):
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
elif os.path.isdir(os.path.join(ROOT, "student_resource", "dataset", "train")):
    TRAIN_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")
    TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")
else:
    TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
    TEST_DIR = os.path.join(ROOT, "dataset", "test")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "output"))
MODEL_DIR = os.environ.get("MODEL_DIR",
                           os.path.join(ROOT, "code", "business_entity_resolution", "models"))
CACHE_DIR = os.environ.get("CACHE_DIR",
                           os.path.join(ROOT, "code", "business_entity_resolution", "cache"))
RESULTS_DIR = os.path.join(ROOT, "code", "business_entity_resolution", "results")

TRAIN_FILES = {
    "s1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "s2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "s3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "gt": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
}
TEST_FILES = {
    "s1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "s2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "s3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

# ------------------------------------------------- pipeline hyperparams ---
SEED = 42
RANDOM_STATE = 42
VAL_FRACTION = 0.2

# Sec 2/14/15: final candidates per S1
K = int(os.environ.get("CAND_K", "22"))
K_CANDIDATES = [18, 20, 22, 24, 26]
RECALL_K_GRID = [10, 15, 20, 22, 24, 26]
S1_CHUNK = int(os.environ.get("S1_CHUNK", "150000"))  # 100k-250k per Sec 4

# Sec 7: rarity cutoffs
RARE_DF_CAP = 300
RARE_DF_CAP_MIN, RARE_DF_CAP_MAX = 100, 300
MAX_POSTING_LEN = 20000
RARE_MAX_QUERY_TOKENS = 12

# Sec 10/12: dense retrieval
ENCODER_MODEL = "minishlab/potion-multilingual-128M"
ENCODER_MAX_LEN = 128
ENCODER_BATCH = int(os.environ.get("ENCODER_BATCH", "512"))
# Chars per semantic text (names first). Lower = faster ANN encoding at some
# recall cost. 2000 default (quality); 512 recommended for the 3h budget.
ENCODER_MAX_CHARS = int(os.environ.get("ENCODER_MAX_CHARS", "2000"))
ANN_TOPK_S2 = 12
ANN_TOPK_S3 = 12
FAISS_NLIST = 4096
FAISS_NPROBE = 8
FAISS_METRIC = "cosine"
GPU_IDS = [0, 1]

# Sec 16/17: training sample + hard negatives
TRAIN_S1_SAMPLE = int(os.environ.get("TRAIN_S1_SAMPLE", "250000"))
VAL_S1_SAMPLE = 50000
NEG_PER_POS = 3  # 2-4x hard negatives (spec value; legacy V3 below keeps its own)
SPEC_NEG_PER_POS = 3
SPEC_RERANK_TOP_N = 3  # Sec 26: max 2-3 ambiguous candidates per S1

# Sec 20: LightGBM starting config
LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.06,
    n_estimators=300,
    num_leaves=47,
    max_depth=8,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=5.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# Sec 22: threshold search
THRESHOLD_GRID = [round(x, 2) for x in
                  [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                   0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
THRESHOLD_REFINE_STEP = 0.02

# Sec 26: BGE reranker disabled by default
USE_RERANKER = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 3

# Sec 3: runtime budget (seconds)
TIME_BUDGET_S = 3 * 3600 - 300  # 3h minus 5min safety

# ------------------------------------------------- legacy compat ---------
# (kept so run_train_v2.py / run_predict_v2.py / laptop/* keep working.
#  WARNING: names below intentionally shadow the spec defaults above for
#  legacy modules only. New pipeline code (src/io, preprocessing, blocking,
#  embeddings, features, training, inference, pipeline, evaluation) must use
#  the SPEC_* constants or its own explicit values, never these.)
BLOCK_TOP_K = 100
BLOCK_DF_CAP = 2000
BLOCK_N_TOK = 12
BLOCK_CHUNK_S1 = 50000
BLOCK_MIN_K = 10
TFIDF_CHAR_NGRAM = (3, 5)
TFIDF_WORD_NGRAM = (1, 2)
TFIDF_MAX_FEATURES = 50000
TRAIN_SAMPLE_S1 = 100000
VAL_SAMPLE_S1 = 20000
CANDIDATE_K = 150
CANDIDATE_KS = [50, 75, 100, 125, 150, 175, 200]
ADAPTIVE_ENABLED_DEFAULT = True
INTERNAL_KEEP_K = 25
FINAL_MAX_PER_S1 = 15
FINAL_TARGET_AVG = 12
FINAL_MAX_AVG = 15
FINAL_MIN_KEEP = 3
RARE_MAX_DF = 2000
RARE_MIN_LEN = 2
RARE_MAX_QUERY_TOKENS = 12
RARE_TOP_K_NAME = 120
RARE_TOP_K_ADDR = 80
CHAR_TOP_K = 40
NUM_TOP_K = 40
E5_TOP_K_NAME = 75
E5_TOP_K_FULL = 75
E5_MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N = 15
MAX_SEQ_LEN = 256
E5_BATCH = 96
RERANK_BATCH = 16
NEG_PER_POS = 6
USE_E5_DEFAULT = True
USE_RERANKER_DEFAULT = True
PRUNE_KEEP_K = 150
CHUNK_S1 = 50000
EMB_DTYPE = "float16"
FAISS_M = 32
FAISS_NBITS = 8
