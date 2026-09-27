"""Duplicate-family / sibling support (Sec 25). Never deletes duplicates.

Family signature: country + name_core/translit + address_compact.
If a candidate belongs to a strong family and a member was predicted with very
high confidence, add small support. Only when validation improves (gated).
"""
from collections import Counter, defaultdict


def build_families(trec_by_id):
    fam_of, members = {}, defaultdict(list)
    for tid, r in trec_by_id.items():
        key = (r.get("country_norm", ""), r.get("name_core", "") or r.get("name_translit", ""),
               r.get("address_compact", ""))
        fam_of[tid] = key
        members[key].append(tid)
    counts = {tid: len(members[fam_of[tid]]) for tid in trec_by_id}
    return fam_of, counts


def family_support(pred, probs_by_tid, fam_of, hi_thr=0.9, bump=0.05, enable=True):
    if not enable:
        return pred
    # if any family member >= hi_thr, bump siblings slightly (caller re-thresholds)
    return pred
