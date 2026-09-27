"""Rare-token blocking channels 8-9 (Sec 7, 8).

Block 8: country + rare_name_token
Block 9: country + rare_address_token + house_number
Key-frequency cutoffs configurable (default 300, range 100-300 per Sec 7).
Capped postings; common tokens never indexed.
"""
import math
from collections import Counter, defaultdict
import numpy as np

from .. import config as C
from ..preprocessing.normalize import name_toks, addr_toks


def build(recs, ids, df_cap=None, dfn=None, dfa=None, N=None):
    """External dfn/dfa (global tables, Sec 7) may be passed to avoid recounting."""
    df_cap = C.RARE_DF_CAP if df_cap is None else df_cap
    if dfn is None:
        dfn = Counter()
        for r in recs:
            dfn.update(set(name_toks(r)))
    if dfa is None:
        dfa = Counter()
        for r in recs:
            dfa.update(set(addr_toks(r)))
    N = N or max(len(recs), 1)
    idfn = {t: math.log(N / (1 + d)) for t, d in dfn.items()}
    idfa = {t: math.log(N / (1 + d)) for t, d in dfa.items()}
    post8, post9 = defaultdict(list), defaultdict(list)
    for i, r in enumerate(recs):
        c = r.get("country_norm", "")
        for t in set(name_toks(r)):
            if dfn.get(t, 0) <= df_cap and len(post8[(c, t)]) < C.MAX_POSTING_LEN:
                post8[(c, t)].append(i)
        hn = r.get("house_number", "")
        for t in set(addr_toks(r)):
            if dfa.get(t, 0) <= df_cap and len(post9[(c, t, hn)]) < C.MAX_POSTING_LEN:
                post9[(c, t, hn)].append(i)
    return {"b8": {k: np.asarray(v, dtype=np.int32) for k, v in post8.items()},
            "b9": {k: np.asarray(v, dtype=np.int32) for k, v in post9.items()},
            "dfn": dfn, "dfa": dfa, "idfn": idfn, "idfa": idfa, "ids": list(ids),
            "df_cap": df_cap}


def query_one(rec, idx, max_toks=12):
    out = {}
    c = rec.get("country_norm", "")
    dfn, idfn = idx["dfn"], idx["idfn"]
    cand8 = sorted(set(name_toks(rec)), key=lambda t: dfn.get(t, 10**9))[:max_toks]
    for t in cand8:
        if dfn.get(t, 10**9) > idx["df_cap"]:
            continue
        for pos in idx["b8"].get((c, t), [])[:C.MAX_POSTING_LEN]:
            tid = idx["ids"][int(pos)]
            d = out.get(tid)
            if d is None:
                out[tid] = {"blocks": {"b8"}, "rare_w": float(idfn.get(t, 0.0))}
            else:
                d["blocks"].add("b8")
                d["rare_w"] = d.get("rare_w", 0) + float(idfn.get(t, 0.0))
    hn = rec.get("house_number", "")
    for t in sorted(set(addr_toks(rec)),
                    key=lambda t: idx["dfa"].get(t, 10**9))[:max_toks]:
        if idx["dfa"].get(t, 10**9) > idx["df_cap"]:
            continue
        for pos in idx["b9"].get((c, t, hn), [])[:C.MAX_POSTING_LEN]:
            tid = idx["ids"][int(pos)]
            d = out.get(tid)
            if d is None:
                out[tid] = {"blocks": {"b9"}, "hit_address": True}
            else:
                d["blocks"].add("b9")
    return out
