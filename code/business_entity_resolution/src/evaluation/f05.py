"""Macro F0.5 per S1 (primary metric). Precision > recall (beta=0.5)."""


def f05_single(pred, true):
    pred, true = set(pred), set(true)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    p = tp / max(len(pred), 1)
    r = tp / max(len(true), 1)
    if p == 0 and r == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(truth, pred):
    keys = set(truth) | set(pred)
    if not keys:
        return 0.0
    return sum(f05_single(pred.get(k, set()), truth.get(k, set())) for k in keys) / len(keys)


def precision_recall_macro(truth, pred):
    ps, rs = [], []
    for k in set(truth) | set(pred):
        p_, t_ = set(pred.get(k, set())), set(truth.get(k, set()))
        tp = len(p_ & t_)
        ps.append(tp / max(len(p_), 1) if p_ or t_ else 1.0)
        rs.append(tp / max(len(t_), 1) if t_ else (1.0 if not p_ else 0.0))
    import numpy as np
    return float(np.mean(ps)), float(np.mean(rs))
