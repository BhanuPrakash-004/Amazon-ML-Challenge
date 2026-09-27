"""Numeric / postal blocking (local only, no geocoding).

Indexes: postal candidates (5-6 digit), house numbers, numeric signatures.
"""
from collections import defaultdict

from .normalize import (normalize_country, extract_postal_candidates,
                        extract_house_numbers, extract_numeric_signature)


class NumericBlocker:
    def __init__(self, top_k=40):
        self.top_k = top_k
        self.per_country = {}

    def fit(self, rows):
        per = {}
        for eid, nm, ad, cc_raw in rows:
            cc = normalize_country(cc_raw)
            b = per.setdefault(cc, {"ids": [], "postal": defaultdict(list),
                                    "house": defaultdict(list), "sig": defaultdict(list)})
            pos = len(b["ids"])
            b["ids"].append(eid)
            for p in extract_postal_candidates(ad):
                b["postal"][p].append(pos)
            for h in extract_house_numbers(ad):
                b["house"][h.lower()].append(pos)
            sig = extract_numeric_signature("%s %s" % (nm, ad))
            if sig:
                b["sig"][sig].append(pos)
        for c in per:
            for k in ("postal", "house", "sig"):
                per[c][k] = dict(per[c][k])
        self.per_country = per
        return self

    def query_one(self, nm, ad, cc_raw):
        cc = normalize_country(cc_raw)
        b = self.per_country.get(cc)
        if not b:
            return []
        scored = defaultdict(float)
        for p in extract_postal_candidates(ad):
            for pos in b["postal"].get(p, [])[:5000]:
                scored[pos] += 3.0
        for h in extract_house_numbers(ad):
            for pos in b["house"].get(h.lower(), [])[:5000]:
                scored[pos] += 1.5
        sig = extract_numeric_signature("%s %s" % (nm, ad))
        if sig:
            for pos in b["sig"].get(sig, [])[:5000]:
                scored[pos] += 2.0
        top = sorted(scored.items(), key=lambda x: -x[1])[:self.top_k]
        return [(b["ids"][p], "numeric", r + 1, float(s)) for r, (p, s) in enumerate(top)]
