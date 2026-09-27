"""Token rarity index helpers (Sec 7). Capped inverted indexes.

Separate name/address frequency tables; generic tokens (company, limited,
road, ...) are never indexed. attach_rare() fills the per-record
name_rare_tokens / address_rare_tokens fields post-hoc once global
frequencies are known.
"""
import math
from collections import Counter, defaultdict
import numpy as np


def build(df_cap=300):
    return {"df_cap": df_cap}


def fit_rare(s23_recs, df_cap=300, field="name_tokens"):
    df = Counter()
    for r in s23_recs:
        df.update(set(r.get(field, [])))
    post = defaultdict(list)
    for i, r in enumerate(s23_recs):
        for t in set(r.get(field, [])):
            if df.get(t, 0) <= df_cap and len(post[t]) < 20000:
                post[t].append(i)
    N = max(len(s23_recs), 1)
    idf = {t: math.log(N / (1 + d)) for t, d in df.items()}
    return {"df": df, "idf": idf,
            "post": {t: np.asarray(v, dtype=np.int32) for t, v in post.items()}}


def attach_rare(recs, dfn, dfa, df_cap=300, max_toks=12):
    """Fill name_rare_tokens / address_rare_tokens in place (Sec 5 + 7)."""
    from .normalize import name_toks, addr_toks
    for r in recs:
        rn = sorted(set(name_toks(r)), key=lambda t: dfn.get(t, 10 ** 9))
        ra = sorted(set(addr_toks(r)), key=lambda t: dfa.get(t, 10 ** 9))
        r["name_rare_tokens"] = [t for t in rn if dfn.get(t, 10 ** 9) <= df_cap][:max_toks]
        r["address_rare_tokens"] = [t for t in ra if dfa.get(t, 10 ** 9) <= df_cap][:max_toks]
    return recs
