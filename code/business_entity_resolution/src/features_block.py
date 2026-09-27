"""Block-evidence features passthrough."""
import pandas as pd

from .candidate_union import EVIDENCE_COLS

BLOCK_COLS = EVIDENCE_COLS + ["block_support", "best_rank", "best_score", "bscore"]


def block_features(pairs: pd.DataFrame) -> pd.DataFrame:
    n = len(pairs)
    out = {}
    for c in EVIDENCE_COLS:
        out[c] = pairs[c].fillna(0).astype(int).tolist() if c in pairs.columns else [0] * n
    out["block_support"] = pairs["block_support"].fillna(1).astype(float).tolist() if "block_support" in pairs.columns else [1.0] * n
    out["best_rank"] = pairs["best_rank"].fillna(999).astype(float).tolist() if "best_rank" in pairs.columns else [999.0] * n
    out["best_score"] = pairs["best_score"].fillna(0).astype(float).tolist() if "best_score" in pairs.columns else [0.0] * n
    out["bscore"] = pairs["bscore"].fillna(0).astype(float).tolist() if "bscore" in pairs.columns else out["best_score"]
    return pd.DataFrame(out, columns=BLOCK_COLS)
