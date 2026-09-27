"""Singleton / confidence gate (Sec 23-24). ~5.58% true singletons.

Evidence per S1: top score, second score, score margin, number of strong
candidates, name/address/numeric evidence. If no candidate has sufficient
evidence -> predict []. Threshold `sing_thr` tuned on validation (Phase 9).
"""
import numpy as np


def gate(tids, probs, s1rec=None, trecs=None, thr=0.5, sing_thr=0.6):
    if not tids:
        return set()
    p = np.asarray(list(probs), dtype=float)
    order = np.argsort(-p)
    top = float(p[order[0]])
    second = float(p[order[1]]) if len(order) > 1 else 0.0
    margin = top - second
    n_strong = int((p >= thr).sum())
    if top < max(thr, sing_thr):
        return set()
    keep = {t for t, v in zip(tids, p) if v >= thr}
    # numeric-conflict veto for lonely weak predictions: single keeper with a
    # small margin and a house/postal conflict, and no exact-name evidence,
    # is demoted to singleton unless the top score is decisive.
    if len(keep) == 1 and margin < 0.10 and top < 0.85 and s1rec is not None \
            and trecs is not None:
        t = next(iter(keep))
        b = trecs.get(t, {}) if isinstance(trecs, dict) else {}
        hn_a, hn_b = s1rec.get("house_number", ""), b.get("house_number", "")
        pc_a, pc_b = s1rec.get("postal_candidate", ""), b.get("postal_candidate", "")
        conflict = (hn_a and hn_b and hn_a != hn_b) or \
                   (pc_a and pc_b and pc_a != pc_b)
        exact = s1rec.get("name_norm", "") == b.get("name_norm", "") and \
            bool(s1rec.get("name_norm", ""))
        if conflict and not exact:
            return set()
    return keep
