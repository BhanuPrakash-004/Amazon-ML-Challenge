"""LightGBM matcher training (V3): sharded/float32, early stopping, F0.5 objective.

Uses features_all columns. Trains on positives + mined hard negatives.
Saves models/lightgbm_matcher.pkl + lightgbm_config.json.
"""
import json
import os

import joblib
import numpy as np

from . import config as C
from .features_all import ALL_COLS


def make_matcher(params=None):
    try:
        import lightgbm as lgb
        p = dict(C.LGBM_PARAMS)
        if params:
            p.update(params)
        return lgb.LGBMClassifier(**p)
    except Exception as e:
        print("LightGBM unavailable (%s); HistGradientBoosting fallback" % e)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=63, min_samples_leaf=40,
                                              l2_regularization=5.0, random_state=C.RANDOM_STATE)


def to_matrix(feat_df):
    return feat_df[ALL_COLS].to_numpy(dtype=np.float32)


def train_matcher(Xtr, ytr, Xva=None, yva=None, params=None):
    clf = make_matcher(params)
    try:
        import lightgbm as lgb
        if isinstance(clf, lgb.LGBMClassifier):
            n_pos = max(int(np.sum(ytr)), 1)
            clf.set_params(scale_pos_weight=max(len(ytr) - n_pos, 1) / n_pos)
    except Exception:
        pass
    if Xva is not None:
        try:
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        except TypeError:
            clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr)
    return clf


def save_matcher(model_root, clf, extra=None):
    os.makedirs(model_root, exist_ok=True)
    joblib.dump(clf, os.path.join(model_root, "lightgbm_matcher.pkl"))
    # legacy name for V2 predict compat
    joblib.dump(clf, os.path.join(model_root, "model.pkl"))
    with open(os.path.join(model_root, "lightgbm_config.json"), "w", encoding="utf-8") as f:
        json.dump({"cols": ALL_COLS, "params": C.LGBM_PARAMS, "extra": extra or {}}, f, indent=1)
