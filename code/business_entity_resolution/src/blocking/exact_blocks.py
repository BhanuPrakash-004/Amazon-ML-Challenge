"""Deterministic blocking channels 1-4 + 12 (Sec 8).

Block 1: country + name_norm
Block 2: country + name_translit
Block 3: country + name_core
Block 4: country + name_compact
Block 12: country + name_norm + address_norm (exact pair)
Every posting list capped at MAX_POSTING_LEN (never generic->millions).
"""
from collections import defaultdict

from .. import config as C


def build(recs, ids):
    idx = {"b1": defaultdict(list), "b2": defaultdict(list),
           "b3": defaultdict(list), "b4": defaultdict(list), "b12": defaultdict(list)}
    for i, r in enumerate(recs):
        c = r.get("country_norm", "")
        if r.get("name_norm"):
            idx["b1"][(c, r["name_norm"])].append(i)
        if r.get("name_translit"):
            idx["b2"][(c, r["name_translit"])].append(i)
        if r.get("name_core"):
            idx["b3"][(c, r["name_core"])].append(i)
        if r.get("name_compact"):
            idx["b4"][(c, r["name_compact"])].append(i)
        if r.get("name_norm") and r.get("address_norm"):
            idx["b12"][(c, r["name_norm"], r["address_norm"])].append(i)
    # cap postings
    for b in idx:
        for k in list(idx[b].keys()):
            if len(idx[b][k]) > C.MAX_POSTING_LEN:
                del idx[b][k]
            else:
                idx[b][k] = sorted(idx[b][k])
    idx["ids"] = list(ids)
    return idx


def query_one(rec, idx):
    c = rec.get("country_norm", "")
    out = {}
    for b, keys in (("b1", [(c, rec.get("name_norm", ""))]),
                    ("b2", [(c, rec.get("name_translit", ""))]),
                    ("b3", [(c, rec.get("name_core", ""))]),
                    ("b4", [(c, rec.get("name_compact", ""))]),
                    ("b12", [(c, rec.get("name_norm", ""), rec.get("address_norm", ""))])):
        for k in keys:
            for pos in idx.get(b, {}).get(k, []):
                tid = idx["ids"][pos]
                d = out.get(tid)
                if d is None:
                    out[tid] = {"blocks": {b}, "hit_exact_name": b in ("b1", "b12"),
                                "hit_core_name": b == "b3", "hit_translit_name": b == "b2"}
                else:
                    d["blocks"].add(b)
    return out
