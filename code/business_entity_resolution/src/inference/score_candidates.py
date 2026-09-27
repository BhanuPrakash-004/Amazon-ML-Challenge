"""Candidate scoring (Sec 19). RapidFuzz only on final ~22/S1."""
import numpy as np

from ..features.semantic_features import row_vector, FEATURE_ORDER


def score_pairs(pairs, model, fam_counts=None, idfn=None, idfa=None):
    """pairs: list of (s1id, tid, s1rec, trec, meta). Returns probs array."""
    X = []
    for s1id, tid, a, b, meta in pairs:
        fc = (fam_counts or {}).get(tid, 1)
        X.append(row_vector(a, b, tid, meta, fc, idfn, idfa))
    X = np.asarray(X, dtype=np.float32)
    try:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    except Exception:
        return np.asarray(model.predict(X), dtype=float)
