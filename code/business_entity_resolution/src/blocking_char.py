"""Character 3/4-gram blocking via compact inverted index.

Covers typos/OCR/transposition/abbreviation. Common n-grams ignored,
long posting lists capped, weighted-overlap ranked. Name and address
are separate signals.
"""
from collections import Counter, defaultdict

import numpy as np

from .normalize import normalize_name, normalize_address, normalize_country


def _ngrams(s, n):
    s = "#" + s + "#"
    if len(s) < n:
        return [s] if s.strip("#") else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


class CharBlocker:
    def __init__(self, n=3, top_k=40, max_df_frac=0.02, max_posting_len=20000):
        self.n = n
        self.top_k = top_k
        self.max_df_frac = max_df_frac
        self.max_posting_len = max_posting_len
        self.per_country = {}

    def fit(self, rows, field="name"):
        buf = defaultdict(list)
        for eid, nm, ad, cc_raw in rows:
            cc = normalize_country(cc_raw)
            txt = normalize_name(nm) if field == "name" else normalize_address(ad)
            buf[(cc, field)].append((eid, _ngrams(txt, self.n)))
        for (cc, fld), items in buf.items():
            key = (cc, fld)
            ids = [e for e, _ in items]
            grams = [g for _, g in items]
            N = len(items)
            df = Counter()
            for g in grams:
                df.update(set(g))
            cap = max(50, int(N * self.max_df_frac))
            keep = {g for g, d in df.items() if d <= cap}
            idf = {g: float(np.log(N / (1 + df[g]))) for g in keep}
            post = defaultdict(list)
            for pos, g in enumerate(grams):
                for x in set(g):
                    if x in keep:
                        post[x].append(pos)
            post_np = {g: np.asarray(lst[:self.max_posting_len], dtype=np.int32) for g, lst in post.items()}
            self.per_country[key] = {"ids": ids, "idf": idf, "post": post_np}
        return self

    def query_one(self, nm, ad, cc_raw, field="name"):
        cc = normalize_country(cc_raw)
        blk = self.per_country.get((cc, field))
        if not blk:
            return []
        txt = normalize_name(nm) if field == "name" else normalize_address(ad)
        q = _ngrams(txt, self.n)
        if not q:
            return []
        idf, post = blk["idf"], blk["post"]
        scores = defaultdict(float)
        for g in set(q):
            a = post.get(g)
            if a is None:
                continue
            w = idf.get(g, 0.0)
            for p in a.tolist():
                scores[int(p)] += w
        top = sorted(scores.items(), key=lambda x: -x[1])[:self.top_k]
        tag = "char_name" if field == "name" else "char_addr"
        return [(blk["ids"][p], tag, r + 1, float(s)) for r, (p, s) in enumerate(top)]
