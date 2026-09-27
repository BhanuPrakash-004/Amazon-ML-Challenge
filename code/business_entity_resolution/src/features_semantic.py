"""Semantic pair features: E5 cosine passthrough + rank/gap (computed upstream).

To keep training CPU-light, this module passes through precomputed columns
(e5_name, e5_full, e5_rank, e5_gap) and adds normalized variants. If columns
are absent, zeros are emitted (lexical-only mode).
"""
import pandas as pd

SEM_COLS = ["e5_name", "e5_full", "e5_rank", "e5_gap", "e5_max"]


def semantic_features(pairs: pd.DataFrame) -> pd.DataFrame:
    n = len(pairs)
    cols = {}
    for c in SEM_COLS:
        if c in pairs.columns:
            cols[c] = pairs[c].fillna(0.0).astype(float).tolist()
        else:
            cols[c] = [0.0] * n
    return pd.DataFrame(cols, columns=SEM_COLS)
