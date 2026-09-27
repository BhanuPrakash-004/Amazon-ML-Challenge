"""Candidate union + dedup (Sec 13). Merges deterministic + ANN hits.

Country is a hard partition (Sec 9): ANN hits from a different country than
the query are dropped — training shows no positive cross-country matches.
Deterministic blocks are country-keyed by construction.
"""
from collections import defaultdict


def merge_per_s1(s1_id, det_hits, ann_s2=None, ann_s3=None,
                 s2_ids=None, s3_ids=None, s1_country=None, country_of=None):
    """det_hits: {tid: meta}; ann lists of (pos, score). Returns {tid: meta}.

    s1_country + country_of(tid)->country enable the hard country partition
    for ANN hits (exact-equivalence for country-partitioned inference).
    """
    merged = dict(det_hits or {})

    def _ok(tid):
        return s1_country is None or country_of is None or \
            country_of.get(tid) == s1_country

    if ann_s2:
        for rank, (pos, score) in enumerate(ann_s2):
            tid = s2_ids[int(pos)]
            if not _ok(tid):
                continue
            d = merged.get(tid)
            if d is None:
                merged[tid] = {"blocks": set(), "hit_ann": True,
                               "ann_rank": rank, "ann_score": float(score)}
            else:
                d["hit_ann"] = True
                d.setdefault("ann_rank", rank)
                d["ann_score"] = max(d.get("ann_score", 0.0), float(score))
    if ann_s3:
        for rank, (pos, score) in enumerate(ann_s3):
            tid = s3_ids[int(pos)]
            if not _ok(tid):
                continue
            d = merged.get(tid)
            if d is None:
                merged[tid] = {"blocks": set(), "hit_ann": True,
                               "ann_rank": rank, "ann_score": float(score)}
            else:
                d["hit_ann"] = True
                d.setdefault("ann_rank", rank)
                d["ann_score"] = max(d.get("ann_score", 0.0), float(score))
    for tid, d in merged.items():
        d["block_count"] = len(d.get("blocks", set()))
    return merged
