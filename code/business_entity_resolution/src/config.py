"""Shared config + paths. V2 names preserved; V3 additions appended."""
import os


def find_root():
    here = os.path.abspath(os.getcwd())
    cands = []
    cur = here
    for _ in range(6):
        for sub in ("student_resource/dataset", "dataset"):
            p = os.path.join(cur, sub)
            if os.path.isdir(p):
                cands.append(cur)
                break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if cands:
        for c in [here] + cands:
            if os.path.isdir(os.path.join(c, "student_resource", "dataset")):
                return c
        return cands[0]
    try:
        fdir = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
        if os.path.isdir(os.path.join(cand, "dataset")):
            return cand
    except Exception:
        pass
    return here


ROOT = find_root()
if os.path.isdir(os.path.join(ROOT, "student_resource", "dataset", "train")):
    TRAIN_DIR = os.path.join(ROOT, "student_resource", "dataset", "train")
    TEST_DIR = os.path.join(ROOT, "student_resource", "dataset", "test")
else:
    TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
    TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "code", "business_entity_resolution", "models")
CACHE_DIR = os.path.join(ROOT, "code", "business_entity_resolution", "cache")
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

# ---- V2 blocking (kept for compat) ----
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

RANDOM_STATE = 42
VAL_FRACTION = 0.2
THRESHOLD_GRID = [round(x, 2) for x in
                  [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
LGBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=5.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# ---- V3 additions ----
# INTERNAL retrieval pool (never written directly to candidate_pairs.tsv).
CANDIDATE_K = 150            # internal K after union+prune (validated 50..200)
CANDIDATE_KS = [50, 75, 100, 125, 150, 175, 200]
# ADAPTIVE final candidate set (written to candidate_pairs.tsv).
# Objective: maximize recall subject to avg 8-15. Raw 100-300 -> internal
# 20-30 -> adaptive final ~8-12 preferred, ~15 hard ceiling.
ADAPTIVE_ENABLED_DEFAULT = True
INTERNAL_KEEP_K = 25         # cheap-prune internal pool before features
FINAL_MAX_PER_S1 = 15        # hard per-S1 cap for final file
FINAL_TARGET_AVG = 12        # preferred average
FINAL_MAX_AVG = 15           # fail loudly above this unless overridden
FINAL_MIN_KEEP = 3           # never force-fill weak candidates to reach K
RARE_MAX_DF = 2000
RARE_MIN_LEN = 2
RARE_MAX_QUERY_TOKENS = 12
RARE_TOP_K_NAME = 120
RARE_TOP_K_ADDR = 80
MAX_POSTING_LEN = 20000
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
FAISS_NLIST = 1024
FAISS_NPROBE = 16
FAISS_M = 32
FAISS_NBITS = 8
