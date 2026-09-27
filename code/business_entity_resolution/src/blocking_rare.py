"""Rare-token blocking (V3): compact per-country index.

Memory-safe design: postings stored as numpy int32 arrays per rare token
(never a giant dense matrix). DF counted streaming; only tokens with
DF<=MAX_DF indexed. Query returns (candidate_id, block_type, rank, score).

Wraps/extends the proven V2 RareTokenBlocker logic for name and address
as SEPARATE indexes.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .normalize import normalize_name, normalize_address, normalize_country


def _toks_name(nm):
    return normalize_name(nm).split()


def _toks_addr(ad):
    return normalize_address(ad).split()


class CompactRareIndex:
    """One field (name or address) per-country rare-token index."""

    def __init__(self, field="name", max_df=2000, min_len=2, top_k=120,
                 max_query_tokens=12, max_posting_len=20000):
        self.field = field
        self.max_df = max_df
        self.min_len = min_len
        self.top_k = top_k
        self.max_query_tokens = max_query_tokens
        self.max_posting_len = max_posting_len
        self.per_country = {}

    def _toks(self, nm, ad):
        return _toks_name(nm) if self.field == "name" else _toks_addr(ad)

    def fit(self, rows):
        """rows: iterable (eid, name, addr, country). Streaming-friendly (call once)."""
        buf = defaultdict(list)  # cc -> list of (eid, toks)
        for eid, nm, ad, cc_raw in rows:
            cc = normalize_country(cc_raw)
            toks = [t for t in self._toks(nm, ad) if len(t) >= self.min_len]
            buf[cc].append((eid, toks))
        for cc, items in buf.items():
            ids = [e for e, _ in items]
            toks = [t for _, t in items]
            N = len(items)
            df = Counter()
            for t in toks:
                df.update(set(t))
            idf = {t: math.log(N / (1 + d)) for t, d in df.items() if d <= self.max_df and len(t) >= self.min_len}
            post = {}
            for pos, t in enumerate(toks):
                for tok in set(t):
                    if tok in idf:
                        post.setdefault(tok, []).append(pos)
            post_np = {}
            for t, lst in post.items():
                if len(lst) > self.max_posting_len:
                    lst = lst[:self.max_posting_len]
                post_np[t] = np.asarray(lst, dtype=np.int32)
            self.per_country[cc] = {"ids": ids, "idf": idf, "post": post_np, "df": df, "N": N}
        return self

    def query_one(self, nm, ad, cc_raw):
        cc = normalize_country(cc_raw)
        blk = self.per_country.get(cc)
        if not blk:
            return []
        qt = [t for t in self._toks(nm, ad) if len(t) >= self.min_len]
        if not qt:
            return []
        idf, post = blk["idf"], blk["post"]
        ranked = sorted(set(qt), key=lambda t: idf.get(t, -1), reverse=True)
        cands = [t for t in ranked if t in post][:self.max_query_tokens]
        if not cands:
            cands = [t for t in ranked if t in post][:3]
            if not cands:
                return []
        scores = defaultdict(float)
        for t in cands:
            a = post.get(t)
            if a is None:
                continue
            w = idf.get(t, 0.0)
            for p in a.tolist():
                scores[int(p)] += w
        ranked_p = sorted(scores.items(), key=lambda x: -x[1])[:self.top_k]
        ids = blk["ids"]
        return [(ids[p], ("rare_name" if self.field == "name" else "rare_address"), r + 1, float(s))
                for r, (p, s) in enumerate(ranked_p)]
