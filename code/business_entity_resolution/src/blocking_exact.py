"""Exact blocking: normalized name / address / combined signatures.

Streaming build per country (open-set). Query returns candidate ids +
evidence flags. Exact matches are evidence, not auto-accepts.
"""
from collections import defaultdict

from .normalize import normalize_name, normalize_address, normalize_country


class ExactBlocker:
    def __init__(self):
        self.per_country = {}

    def fit_stream(self, rows):
        """rows: iterable of (entity_id, name, addr, country)."""
        per = {}
        for eid, nm, ad, cc_raw in rows:
            cc = normalize_country(cc_raw)
            nn = normalize_name(nm)
            aa = normalize_address(ad)
            b = per.setdefault(cc, {"ids": [], "name": defaultdict(list),
                                    "addr": defaultdict(list), "comb": defaultdict(list)})
            pos = len(b["ids"])
            b["ids"].append(eid)
            if nn:
                b["name"][nn].append(pos)
            if aa:
                b["addr"][aa].append(pos)
            if nn or aa:
                b["comb"][(nn, aa)].append(pos)
        # freeze
        for c in per:
            per[c]["name"] = dict(per[c]["name"])
            per[c]["addr"] = dict(per[c]["addr"])
            per[c]["comb"] = dict(per[c]["comb"])
        self.per_country = per
        return self

    def query(self, s1_rows):
        """s1_rows: list of (s1id, name, addr, country) -> dict s1id -> (ids, evidence)."""
        out = {}
        for s1id, nm, ad, cc_raw in s1_rows:
            cc = normalize_country(cc_raw)
            b = self.per_country.get(cc)
            if not b:
                out[s1id] = ([], {"exact_name": 0, "exact_address": 0, "exact_combined": 0})
                continue
            nn = normalize_name(nm)
            aa = normalize_address(ad)
            ids, ev = [], {"exact_name": 0, "exact_address": 0, "exact_combined": 0}
            for p in b["comb"].get((nn, aa), []):
                ids.append(b["ids"][p])
            if b["comb"].get((nn, aa)):
                ev["exact_combined"] = 1
            for p in b["name"].get(nn, []) if nn else []:
                if b["ids"][p] not in ids:
                    ids.append(b["ids"][p])
            if nn and b["name"].get(nn):
                ev["exact_name"] = 1
            for p in b["addr"].get(aa, []) if aa else []:
                if b["ids"][p] not in ids:
                    ids.append(b["ids"][p])
            if aa and b["addr"].get(aa):
                ev["exact_address"] = 1
            out[s1id] = (ids[:50], ev)
        return out
