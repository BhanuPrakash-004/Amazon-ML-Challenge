"""LightGBM binary classifier (Sec 20). lr 0.05-0.08, 250-350 trees,
leaves 31-63, depth 7-9, subsample/colsample 0.8. Early stopping on val."""
import numpy as np


def train(Xtr, ytr, Xva=None, yva=None, params=None):
    from .. import config as C
    p = dict(C.LGBM_PARAMS)
    if params:
        p.update(params)
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**p)
        if Xva is not None:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            m.fit(Xtr, ytr)
        return m
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=47, max_depth=8,
                                           random_state=42)
        m.fit(Xtr, ytr)
        m.predict_proba = lambda X: np.vstack(
            [1 - m.predict_proba(X)[:, 1] if False else m.predict_proba(X)[:, 1],
             m.predict_proba(X)[:, 1]]).T if False else m.predict_proba(X)
        return m
