"""Entity-level decision logic: zero/one/multiple matches per S1.

Uses score distribution (top, gap, strong counts, evidence diversity).
Never forces a match (singleton-safe). Multi-match allowed on independent
strong evidence. No France-specific hacks; globally calibrated.
"""
import numpy as np


def decide_for_s1(cids, scores, evidence=None, threshold=0.5, margin=0.15,
                  multi_min=0.65, max_matches=11):
    """Returns accepted subset of cids."""
    if not cids:
        return []
    order = np.argsort([-s for s in scores])
    cids = [cids[i] for i in order]
    scores = [float(scores[i]) for i in order]
    if scores[0] < threshold:
        return []
    accepted = [cids[0]]
    for c, s in zip(cids[1:], scores[1:]):
        if s < threshold:
            continue
        gap = scores[0] - s
        strong = s >= multi_min
        ev = (evidence or {}).get(c, {})
        multi_ev = sum(1 for k in ("exact_name", "exact_address", "e5_name", "rare_name") if ev.get(k))
        if strong and (gap <= margin or multi_ev >= 2):
            accepted.append(c)
        if len(accepted) >= max_matches:
            break
    return accepted
