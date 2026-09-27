"""Build labelled pair table from blocking candidates + ground truth.

For training: each (S1, candidate) pair gets label 1 if candidate in GT set else 0.
Also computes recall ceiling of blocking on the given split.
"""
import pandas as pd


def pairs_from_candidates(s1_df, s23_df, cand_df, truth: dict | None = None):
    """Returns pair-level DataFrame with normalized + raw fields for featurization.

    Columns: source1_entity_id, candidate_id,
             name_norm_1/2, addr_norm_1/2, country_norm_1/2, addr_raw_1/2,
             label (if truth given).
    """
    from .blocking_v2 import prepare_frame as _pf
    s1p = _pf(s1_df)
    s23p = _pf(s23_df)
    s1_map = {r.entity_id: r for r in s1p.itertuples()}
    s23_map = {r.entity_id: r for r in s23p.itertuples()}
    rows = []
    for _, r in cand_df.iterrows():
        s1id = r["source1_entity_id"]
        cands = r["candidate_entity_ids"]
        if isinstance(cands, str):
            cands = [c for c in cands.split(",") if c] if cands else []
        a = s1_map.get(s1id)
        if a is None:
            continue
        for cid in cands:
            b = s23_map.get(cid)
            if b is None:
                continue
            label = None
            if truth is not None:
                label = 1 if cid in truth.get(s1id, set()) else 0
            rows.append({
                "source1_entity_id": s1id,
                "candidate_id": cid,
                "name_norm_1": a.name_norm, "name_norm_2": b.name_norm,
                "addr_norm_1": a.addr_norm, "addr_norm_2": b.addr_norm,
                "country_norm_1": a.country_norm, "country_norm_2": b.country_norm,
                "addr_raw_1": str(a.business_address), "addr_raw_2": str(b.business_address),
                **({"label": label} if label is not None else {}),
            })
    return pd.DataFrame(rows)


def blocking_recall(cand_df, truth: dict):
    """Fraction of true matches covered by candidates (micro) + per-entity recall."""
    tot_true, tot_hit = 0, 0
    per = []
    for _, r in cand_df.iterrows():
        s1 = r["source1_entity_id"]
        cands = r["candidate_entity_ids"]
        if isinstance(cands, str):
            cands = set(c for c in cands.split(",") if c) if cands else set()
        else:
            cands = set(cands or [])
        t = truth.get(s1, set())
        tot_true += len(t)
        tot_hit += len(t & cands)
        per.append(1.0 if len(t) == 0 else len(t & cands) / len(t))
    micro = tot_hit / tot_true if tot_true else 1.0
    macro = sum(per) / len(per) if per else 1.0
    return micro, macro
