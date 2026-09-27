"""Validation-trained ensemble (meta-model preferred; fixed weights only if proven)."""
import numpy as np


def blend(meta_clf, meta_X, lgbm_probs, e5=None, rerank=None, fixed=None):
    if meta_clf is not None:
        try:
            return np.asarray(meta_clf.predict_proba(meta_X)[:, 1], dtype=np.float32)
        except Exception:
            pass
    if fixed:
        w = fixed
        s = np.asarray(lgbm_probs, dtype=np.float32) * w.get("lgbm", 1.0)
        if e5 is not None:
            s += np.asarray(e5, dtype=np.float32) * w.get("e5", 0.0)
        if rerank is not None:
            s += np.asarray(rerank, dtype=np.float32) * w.get("rerank", 0.0)
        return s
    return np.asarray(lgbm_probs, dtype=np.float32)
