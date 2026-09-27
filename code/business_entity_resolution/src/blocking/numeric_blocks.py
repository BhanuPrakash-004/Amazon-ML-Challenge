"""Numeric blocking channels 5-7 + 10-11 (Sec 6, 8).

Block 5:  country + name_core + house_number
Block 6:  country + name token + postal
Block 7:  country + name token + house_number
Block 10: country + numeric_signature
Block 11: country + postal + house_number
"""
from collections import defaultdict
from .. import config as C
from ..preprocessing.normalize import name_toks


def build(recs, ids):
    idx = {"b5": defaultdict(list), "b6": defaultdict(list), "b7": defaultdict(list),
           "b10": defaultdict(list), "b11": defaultdict(list)}
    for i, r in enumerate(recs):
        c = r.get("country_norm", "")
        hn = r.get("house_number", "")
        pc = r.get("postal_candidate", "")
        sig = r.get("numeric_signature", "")
        core = r.get("name_core", "")
        if core and hn:
            idx["b5"][(c, core, hn)].append(i)
        for t in (name_toks(r) or [])[:8]:
            if pc:
                idx["b6"][(c, t, pc)].append(i)
            if hn:
                idx["b7"][(c, t, hn)].append(i)
        if sig:
            idx["b10"][(c, sig)].append(i)
        if pc and hn:
            idx["b11"][(c, pc, hn)].append(i)
    for b in idx:
        for k in list(idx[b].keys()):
            if len(idx[b][k]) > C.MAX_POSTING_LEN:
                del idx[b][k]
    idx["ids"] = list(ids)
    return idx


def query_one(rec, idx):
    c = rec.get("country_norm", "")
    hn = rec.get("house_number", "")
    pc = rec.get("postal_candidate", "")
    out = {}
    keys = []
    if rec.get("name_core") and hn:
        keys.append(("b5", (c, rec["name_core"], hn)))
    for t in (name_toks(rec) or [])[:8]:
        if pc:
            keys.append(("b6", (c, t, pc)))
        if hn:
            keys.append(("b7", (c, t, hn)))
    if rec.get("numeric_signature"):
        keys.append(("b10", (c, rec["numeric_signature"],)))
    if pc and hn:
        keys.append(("b11", (c, pc, hn)))
    for b, k in keys:
        for pos in idx.get(b, {}).get(k, []):
            tid = idx["ids"][pos]
            d = out.get(tid)
            if d is None:
                out[tid] = {"blocks": {b}, "hit_numeric": True}
            else:
                d["blocks"].add(b)
    return out
