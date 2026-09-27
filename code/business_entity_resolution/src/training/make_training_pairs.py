"""S1-entity validation split + training pair assembly (Sec 16).

- Split by S1 entity (never by pair). 80/20, stratified by country x singleton x
  match-count bucket. Samples 200-300k S1 for the 3h budget.
- All positive candidate pairs for sampled S1 are kept.
"""
import random
from collections import defaultdict


def bucket(n):
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def split_s1(s1_countries, truth, n_train=250000, n_val=50000, seed=42):
    rnd = random.Random(seed)
    strata = defaultdict(list)
    for s1, c in s1_countries.items():
        m = len(truth.get(s1, set()))
        strata[(c, "sing" if m == 0 else "nonsing", bucket(m))].append(s1)
    tr, va = [], []
    for k, ids in strata.items():
        rnd.shuffle(ids)
        n = len(ids)
        nt = max(1, int(round(n_train * n / max(len(s1_countries), 1)))) if n > 1 else 0
        nv = max(1, int(round(n_val * n / max(len(s1_countries), 1)))) if n > 1 else 0
        tr.extend(ids[:nt])
        va.extend(ids[nt:nt + nv])
    rnd.shuffle(tr)
    rnd.shuffle(va)
    return tr[:n_train], va[:n_val]
