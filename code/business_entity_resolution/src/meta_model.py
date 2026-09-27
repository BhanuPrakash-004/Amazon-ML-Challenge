"""Meta-model: compact LightGBM over (lgbm, e5, reranker, sims, evidence).

No hard-coded weights. Trained on validation-safe features.
"""
import os

import joblib
import numpy as np

META_COLS = ["lgbm", "e5_name", "e5_full", "rerank", "name_set", "addr_set",
             "char_jacc", "num_jacc", "block_support", "cand_rank", "score_gap",
             "country_match", "exact_any", "is_s2"]


def build_meta_frame(base: dict) -> np.ndarray:
    import pandas as pd
    df = pd.DataFrame({c: base.get(c, 0.0) for c in META_COLS})
    return df[META_COLS].to_numpy(dtype=np.float32)


def train_meta(X, y):
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=60, subsample=0.9, colsample_bytree=0.9,
                                 reg_lambda=10.0, verbose=-1, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31)
    clf.fit(X, y)
    return clf


def save_meta(model_root, clf):
    os.makedirs(model_root, exist_ok=True)
    joblib.dump(clf, os.path.join(model_root, "meta_model.pkl"))
