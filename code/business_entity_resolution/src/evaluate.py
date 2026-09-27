"""Macro F0.5 scorer (per-S1 average, singletons included) + V3 validation report."""
import numpy as np


def f05_per_entity(y_true_set: set, y_pred_set: set) -> float:
    if len(y_true_set) == 0 and len(y_pred_set) == 0:
        return 1.0
    if len(y_true_set) == 0 or len(y_pred_set) == 0:
        return 0.0
    tp = len(y_true_set & y_pred_set)
    if tp == 0:
        return 0.0
    prec = tp / len(y_pred_set)
    rec = tp / len(y_true_set)
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def macro_f05(truth: dict, pred: dict) -> float:
    keys = set(truth.keys()) | set(pred.keys())
    if not keys:
        return 0.0
    return float(np.mean([f05_per_entity(truth.get(k, set()), pred.get(k, set())) for k in keys]))


def precision_recall_macro(truth: dict, pred: dict):
    ps, rs = [], []
    for k in set(truth.keys()) | set(pred.keys()):
        t, p = truth.get(k, set()), pred.get(k, set())
        if len(p) == 0 and len(t) == 0:
            ps.append(1.0); rs.append(1.0)
        elif len(p) == 0 or len(t) == 0:
            ps.append(0.0); rs.append(0.0)
        else:
            tp = len(t & p)
            ps.append(tp / len(p) if p else 0.0)
            rs.append(tp / len(t) if t else 0.0)
    return float(np.mean(ps)), float(np.mean(rs))


def validation_report(truth, pred, cand_dict=None, meta=None):
    """Overall F0.5/P/R + by-entity-size + candidate stats."""
    meta = meta or {}
    f = macro_f05(truth, pred)
    p, r = precision_recall_macro(truth, pred)
    by_size = {}
    for k in ("singleton", "one", "two", "three_plus"):
        by_size[k] = {}
    for size, label in ((0, "singleton"), (1, "one"), (2, "two")):
        keys = [k for k, v in truth.items() if len(v) == size]
        if keys:
            t2 = {k: truth[k] for k in keys}
            p2 = {k: pred.get(k, set()) for k in keys}
            fp, fr = precision_recall_macro(t2, p2)
            by_size[label] = {"n": len(keys), "f05": macro_f05(t2, p2), "P": fp, "R": fr}
    keys = [k for k, v in truth.items() if len(v) >= 3]
    if keys:
        t2 = {k: truth[k] for k in keys}
        p2 = {k: pred.get(k, set()) for k in keys}
        fp, fr = precision_recall_macro(t2, p2)
        by_size["three_plus"] = {"n": len(keys), "f05": macro_f05(t2, p2), "P": fp, "R": fr}
    rep = {"macro_f05": f, "macro_P": p, "macro_R": r, "by_size": by_size}
    if cand_dict:
        counts = sorted(len(v) for v in cand_dict.values())
        import numpy as _np
        rep["avg_cands"] = float(_np.mean(counts)) if counts else 0.0
        rep["p95_cands"] = float(_np.percentile(counts, 95)) if counts else 0.0
        rep["p99_cands"] = float(_np.percentile(counts, 99)) if counts else 0.0
        rep["max_cands"] = int(max(counts)) if counts else 0
    rep.update(meta)
    return rep
