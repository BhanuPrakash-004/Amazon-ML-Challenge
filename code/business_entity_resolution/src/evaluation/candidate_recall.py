"""Candidate recall + count stats (Sec 28). Report avg/p50/p90/p95/p99/max
and recall at K=10/15/20/22/24/26 when affordable."""
import numpy as np


def recall(truth, cands):
    tp = fp = fn = 0
    per = []
    for s1, t in truth.items():
        c = set(cands.get(s1, set()))
        tp += len(t & c)
        fn += len(t - c)
        per.append(len(t & c) / max(len(t), 1) if t else 1.0)
    micro = tp / max(tp + fn, 1)
    return {"micro": micro, "macro": float(np.mean(per)) if per else 0.0}


def count_stats(cands):
    n = np.array([len(set(v)) for v in cands.values()], dtype=float)
    if not len(n):
        return {}
    return {"avg": float(n.mean()), "p50": float(np.percentile(n, 50)),
            "p90": float(np.percentile(n, 90)), "p95": float(np.percentile(n, 95)),
            "p99": float(np.percentile(n, 99)), "max": int(n.max())}
