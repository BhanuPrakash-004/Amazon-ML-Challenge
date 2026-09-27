"""Multi-block UNION: exact/rare/address/numeric/char/translit/E5 -> one pool.

Never intersects major blockers. Stores per-candidate evidence flags,
best rank/score, supporting-block count.
"""
from collections import defaultdict

EVIDENCE_COLS = ["exact_name", "exact_address", "rare_name", "rare_address",
                 "numeric", "char", "transliterated", "e5_name", "e5_address"]


def union_for_s1(s1id, lists_by_block):
    """lists_by_block: dict block -> list[(cid, rank, score)] (+ exact ids).

    Returns (ordered_cids, evidence_dict[cid -> flags], meta[cid -> (best_rank, best_score, n_blocks)]).
    """
    per = defaultdict(lambda: {"flags": defaultdict(int), "best_rank": 10 ** 9,
                               "best_score": 0.0, "n": 0})
    for block, items in (lists_by_block or {}).items():
        seen = set()
        for cid, rank, score in items:
            if cid in seen:
                continue
            seen.add(cid)
            e = per[cid]
            e["flags"][block] = 1
            e["best_rank"] = min(e["best_rank"], int(rank))
            e["best_score"] = max(e["best_score"], float(score))
    for cid in per:
        per[cid]["n"] = sum(1 for v in per[cid]["flags"].values() if v)
    # order: multi-support first, then score
    cids = sorted(per.keys(), key=lambda c: (-per[c]["n"], -per[c]["best_score"], per[c]["best_rank"]))
    ev = {c: {k: int(per[c]["flags"].get(k, 0)) for k in EVIDENCE_COLS} for c in cids}
    meta = {c: (per[c]["best_rank"], per[c]["best_score"], per[c]["n"]) for c in cids}
    return cids, ev, meta


def normalize_block_alias(tag):
    m = {"rare_name": "rare_name", "rare_address": "rare_address", "numeric": "numeric",
         "char_name": "char", "char_addr": "char", "char": "char",
         "translit_name": "transliterated", "transliterated": "transliterated",
         "e5_name": "e5_name", "e5_address": "e5_address",
         "exact": "exact_name"}
    return m.get(tag, tag)
