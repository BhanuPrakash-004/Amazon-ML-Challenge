"""Cheap pre-ranking 50-200 -> top-K (Sec 14). Vectorized, RapidFuzz only after cut.

pre_score = exact/core/translit name + house/numeric + rare overlap
            + ANN similarity + address token Jaccard.
"""
import numpy as np

W = {"exact": 5.0, "core": 3.0, "translit": 2.5, "house": 2.0,
     "numeric": 1.5, "rare": 1.0, "ann": 2.0, "addr": 1.0}


def _addr_jacc(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def prescore(s1rec, tid, trec, meta):
    s = 0.0
    if s1rec.get("name_norm") and s1rec["name_norm"] == trec.get("name_norm"):
        s += W["exact"]
    if s1rec.get("name_core") and s1rec["name_core"] == trec.get("name_core"):
        s += W["core"]
    if s1rec.get("name_translit") and s1rec["name_translit"] == trec.get("name_translit"):
        s += W["translit"]
    if s1rec.get("house_number") and s1rec["house_number"] == trec.get("house_number"):
        s += W["house"]
    sn = set(s1rec.get("numeric_tokens", []))
    tn = set(trec.get("numeric_tokens", []))
    if sn and tn:
        inter = len(sn & tn)
        if inter:
            s += W["numeric"] * inter / max(len(sn | tn), 1)
    s += W["rare"] * min(meta.get("rare_w", 0.0) / 5.0, 2.0)
    if meta.get("hit_ann"):
        s += W["ann"] * float(meta.get("ann_score", 0.0))
    s += W["addr"] * _addr_jacc(s1rec.get("address_norm", ""), trec.get("address_norm", ""))
    s += 0.3 * meta.get("block_count", 0)
    return s


def topk(s1rec, merged, trec_by_id, K=22):
    scored = [(prescore(s1rec, tid, trec_by_id[tid], meta), tid)
              for tid, meta in merged.items() if tid in trec_by_id]
    scored.sort(reverse=True)
    out = []
    for rank, (sc, tid) in enumerate(scored[:K]):
        merged[tid]["block_rank"] = rank  # Sec 13 per-candidate metadata
        out.append((tid, sc))
    return out
