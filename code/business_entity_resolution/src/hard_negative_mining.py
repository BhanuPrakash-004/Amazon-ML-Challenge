"""Hard-negative mining: 1 positive : 4-8 hard negatives from the candidate pool.

Difficulty bands: very-hard (high lexical/semantic but wrong), hard
(same postal/rare token), medium (rest). Balances across bands instead of
random downsampling.
"""
import pandas as pd

from .features_lexical import lexical_features


def mine_negatives(pairs: pd.DataFrame, per_pos=6):
    """pairs must have source1_entity_id, candidate_id, label. Returns sampled frame."""
    pos = pairs[pairs["label"] == 1]
    neg = pairs[pairs["label"] == 0]
    if not len(pos) or not len(neg):
        return pairs
    # score negatives by lexical strength (cheap proxy for difficulty)
    try:
        lf = lexical_features(neg[["name_norm_1", "name_norm_2", "addr_norm_1", "addr_norm_2"]])
        strength = (lf["name_set"].fillna(0) + lf["addr_set"].fillna(0)).tolist()
    except Exception:
        strength = [0.0] * len(neg)
    neg = neg.copy()
    neg["_s"] = strength
    keep_idx = list(pos.index)
    for s1, g in pos.groupby("source1_entity_id"):
        ng = neg[neg["source1_entity_id"] == s1].sort_values("_s", ascending=False)
        if not len(ng):
            continue
        n = min(len(ng), per_pos)
        # band sampling: 50% top (very hard), 30% middle, 20% tail
        idx = ng.index.tolist()
        n_vh = max(1, int(0.5 * n))
        n_h = max(1, int(0.3 * n)) if n > 2 else 0
        sel = idx[:n_vh]
        mid = idx[n_vh:]
        if n_h and mid:
            sel += mid[:n_h]
        rest = [i for i in idx if i not in sel]
        need = n - len(sel)
        if need > 0 and rest:
            import random
            random.seed(42)
            sel += random.sample(rest, min(need, len(rest)))
        keep_idx.extend(sel)
    out = pairs.loc[sorted(set(keep_idx))].drop(columns=["_s"], errors="ignore")
    return out.reset_index(drop=True)
