"""Matching model: LightGBM (fallback: HistGradientBoosting) + F0.5 threshold tuning."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import macro_f05

MODEL_PKL = "model.pkl"
THR_JSON = "threshold.json"
FEAT_PKL = "feature_builder.pkl"


def _make_classifier():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(**C.LGBM_PARAMS)
    except Exception as e:
        print("LightGBM unavailable (%s); falling back to sklearn HistGradientBoosting" % e)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=5.0,
            random_state=C.RANDOM_STATE,
        )


def train_classifier(X_train, y_train, X_val=None, y_val=None):
    clf = _make_classifier()
    # class imbalance: scale_pos_weight for lgbm
    try:
        import lightgbm as lgb
        if isinstance(clf, lgb.LGBMClassifier):
            n_pos = max(int(np.sum(y_train)), 1)
            n_neg = max(len(y_train) - n_pos, 1)
            clf.set_params(scale_pos_weight=n_neg / n_pos)
    except Exception:
        pass
    if X_val is not None:
        try:
            import lightgbm as lgb
            if isinstance(clf, lgb.LGBMClassifier):
                clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                clf.fit(X_train, y_train)
        except TypeError:
            clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)
    return clf


def tune_threshold(df_val_pairs, y_val, proba_val, truth_val: dict):
    """Grid-search threshold maximizing macro F0.5 on validation S1s.

    df_val_pairs must contain source1_entity_id + candidate id column.
    Returns (best_thr, best_f05, history).
    """
    cand_col = "candidate_id" if "candidate_id" in df_val_pairs.columns else "entity_id_2"
    s1s = df_val_pairs["source1_entity_id"].tolist()
    cands = df_val_pairs[cand_col].tolist()
    best_thr, best_f, hist = 0.5, -1.0, []
    for thr in C.THRESHOLD_GRID:
        pred = {}
        for s, c, p in zip(s1s, cands, proba_val):
            if p >= thr:
                pred.setdefault(s, set()).add(c)
        # ensure all val S1 present (missing -> empty)
        for s in truth_val.keys():
            pred.setdefault(s, set())
        f = macro_f05(truth_val, pred)
        hist.append((thr, f))
        if f > best_f:
            best_f, best_thr = f, thr
    # refine around best by +-0.04 step 0.01
    for d in (-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04):
        thr = round(best_thr + d, 2)
        if thr <= 0 or thr >= 1 or thr in [h[0] for h in hist]:
            continue
        pred = {}
        for s, c, p in zip(s1s, cands, proba_val):
            if p >= thr:
                pred.setdefault(s, set()).add(c)
        for s in truth_val.keys():
            pred.setdefault(s, set())
        f = macro_f05(truth_val, pred)
        hist.append((thr, f))
        if f > best_f:
            best_f, best_thr = f, thr
    hist.sort()
    return best_thr, best_f, hist


def apply_veto_rules(pairs_df: pd.DataFrame, proba: np.ndarray, threshold: float):
    """Precision vetoes (rule-based, applied after threshold):
    - cross-country pairs require strong evidence: proba >= max(thr, 0.75)
      OR name_token_set >= 0.9 OR name_exact == 1. Otherwise drop.
    This curbs false merges (F0.5 is precision-heavy) while allowing
    legitimate cross-country links when names match strongly.
    """
    keep = proba >= threshold
    if "country_match" in pairs_df.columns:
        cm = pairs_df["country_match"].to_numpy()
        nts = pairs_df["name_token_set"].to_numpy() if "name_token_set" in pairs_df.columns else np.ones_like(cm)
        nex = pairs_df["name_exact"].to_numpy() if "name_exact" in pairs_df.columns else np.zeros_like(cm)
        for i in range(len(keep)):
            if keep[i] and cm[i] == 0:
                if not (proba[i] >= max(threshold, 0.75) or nts[i] >= 0.90 or nex[i] == 1.0):
                    keep[i] = False
    return keep


def save_artifacts(model_dir, clf, feat_builder, threshold):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, MODEL_PKL))
    joblib.dump(feat_builder, os.path.join(model_dir, FEAT_PKL))
    with open(os.path.join(model_dir, THR_JSON), "w", encoding="utf-8") as f:
        json.dump({"threshold": float(threshold)}, f)


def load_artifacts(model_dir):
    import joblib, json
    clf = joblib.load(os.path.join(model_dir, MODEL_PKL))
    fb = joblib.load(os.path.join(model_dir, FEAT_PKL))
    with open(os.path.join(model_dir, THR_JSON), encoding="utf-8") as f:
        thr = float(json.load(f)["threshold"])
    return clf, fb, thr
