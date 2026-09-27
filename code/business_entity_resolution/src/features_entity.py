"""Entity-context features: rank/gap/counts within each S1 candidate list."""
import numpy as np
import pandas as pd

ENT_COLS = ["cand_rank", "score_gap", "n_cands", "n_strong", "ev_diversity"]


def entity_features(pairs: pd.DataFrame, score_col="bscore") -> pd.DataFrame:
    scores = pairs[score_col].fillna(0).astype(float).tolist() if score_col in pairs.columns else [0.0] * len(pairs)
    s1s = pairs["source1_entity_id"].tolist()
    order = sorted(range(len(pairs)), key=lambda i: (s1s[i], -scores[i]))
    rank = [0] * len(pairs)
    gap = [0.0] * len(pairs)
    cnt = {}
    top = {}
    for i in order:
        s = s1s[i]
        cnt[s] = cnt.get(s, 0) + 1
    for i in order:
        s = s1s[i]
        if s not in top:
            top[s] = scores[i]
            rank[i] = 1
            gap[i] = 0.0
        else:
            # rank within group
            rank[i] = sum(1 for j in order if s1s[j] == s and scores[j] > scores[i]) + 1
            gap[i] = top[s] - scores[i]
    # per-group counts
    grp_n = {s: cnt[s] for s in cnt}
    out = []
    for i in range(len(pairs)):
        s = s1s[i]
        n = grp_n[s]
        n_strong = sum(1 for j in order if s1s[j] == s and scores[j] >= 0.5 * (top[s] or 1.0))
        ev = 0
        for c in ("exact_name", "rare_name", "e5_name"):
            if c in pairs.columns and pairs[c].iloc[i]:
                ev += 1
        out.append([float(rank[i]), float(gap[i]), float(n), float(n_strong), float(ev)])
    return pd.DataFrame(out, columns=ENT_COLS)
