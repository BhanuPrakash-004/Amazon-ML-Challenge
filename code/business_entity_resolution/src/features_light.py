"""Lightweight vocab-free features (no TF-IDF fit) for 10M scale.

All features computable per-pair in Python without any global fit:
- RapidFuzz ratio/token_sort/token_set/partial/WRatio on name + address
- Jaccard word / trigram, exact / core-exact, length diffs, shared tokens
- digit/PIN/ZIP overlap, country_match, blocking IDF score passthrough
Handles France/Hindi automatically (no vocab).
"""
import re
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAS_RF = True
except Exception:
    HAS_RF = False
    fuzz = None

from .normalize import core_name

LIGHT_COLS = [
    "name_ratio", "name_sort", "name_set", "name_partial", "name_wratio",
    "name_jw", "name_tri", "name_exact", "name_core_exact", "name_core_jacc",
    "name_lendiff", "name_tokdiff", "name_shared",
    "addr_ratio", "addr_set", "addr_jw", "addr_tri", "addr_exact",
    "addr_lendiff", "addr_shared",
    "num_jacc", "pin_match", "zip_match", "any_num",
    "country_match", "bscore",
]

_DIG = re.compile(r"\d+")


def _jacc(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tri(s: str) -> set:
    s = "#" + s + "#"
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def light_features(pairs: pd.DataFrame) -> pd.DataFrame:
    n1 = pairs["name_norm_1"].fillna("").astype(str).tolist()
    n2 = pairs["name_norm_2"].fillna("").astype(str).tolist()
    a1 = pairs["addr_norm_1"].fillna("").astype(str).tolist()
    a2 = pairs["addr_norm_2"].fillna("").astype(str).tolist()
    c1 = pairs["country_norm_1"].fillna("").astype(str).tolist()
    c2 = pairs["country_norm_2"].fillna("").astype(str).tolist()
    r1 = pairs.get("addr_raw_1", pairs["addr_norm_1"]).fillna("").astype(str).tolist()
    r2 = pairs.get("addr_raw_2", pairs["addr_norm_2"]).fillna("").astype(str).tolist()
    bs = pairs.get("bscore", pd.Series([0.0] * len(pairs))).fillna(0.0).tolist()
    rows = []
    for i in range(len(pairs)):
        x1, x2, y1, y2 = n1[i], n2[i], a1[i], a2[i]
        if HAS_RF:
            nr = fuzz.ratio(x1, x2) / 100.0
            ns = fuzz.token_sort_ratio(x1, x2) / 100.0
            nt = fuzz.token_set_ratio(x1, x2) / 100.0
            npr = fuzz.partial_ratio(x1, x2) / 100.0
            nwr = fuzz.WRatio(x1, x2) / 100.0
            ar = fuzz.ratio(y1, y2) / 100.0
            at = fuzz.token_set_ratio(y1, y2) / 100.0 if (y1 and y2) else 0.0
        else:
            nr = ns = nt = npr = nwr = _jacc(set(x1.split()), set(x2.split()))
            ar = at = _jacc(set(y1.split()), set(y2.split()))
        njw = _jacc(set(x1.split()), set(x2.split()))
        ntri = _jacc(_tri(x1), _tri(x2))
        cx1, cx2 = core_name(x1), core_name(x2)
        cj = _jacc(set(cx1.split()), set(cx2.split())) if (cx1 or cx2) else 1.0
        ajw = _jacc(set(y1.split()), set(y2.split()))
        atri = _jacc(_tri(y1), _tri(y2))
        d1, d2 = set(_DIG.findall(r1[i])), set(_DIG.findall(r2[i]))
        nj = _jacc(d1, d2) if (d1 or d2) else 0.0
        pinm = 1.0 if any(len(d) == 6 for d in (d1 & d2)) else 0.0
        zipm = 1.0 if any(len(d) == 5 for d in (d1 & d2)) else 0.0
        anym = 1.0 if len(d1 & d2) > 0 else 0.0
        rows.append([
            nr, ns, nt, npr, nwr, njw, ntri,
            1.0 if x1 == x2 and x1 else 0.0,
            1.0 if cx1 == cx2 and cx1 else 0.0, cj,
            abs(len(x1) - len(x2)), abs(len(x1.split()) - len(x2.split())),
            float(len(set(x1.split()) & set(x2.split()))),
            ar, at, ajw, atri,
            1.0 if y1 == y2 and y1 else 0.0,
            abs(len(y1) - len(y2)),
            float(len(set(y1.split()) & set(y2.split()))),
            nj, pinm, zipm, anym,
            1.0 if c1[i] == c2[i] and c1[i] else 0.0,
            float(bs[i]),
        ])
    return pd.DataFrame(rows, columns=LIGHT_COLS)
