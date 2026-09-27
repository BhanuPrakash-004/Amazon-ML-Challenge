"""Lexical pair features: RapidFuzz ratios, Jaccard, char overlap, transliteration sim."""
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAS_RF = True
except Exception:
    HAS_RF = False
    fuzz = None

LEX_COLS = ["name_ratio", "name_sort", "name_set", "name_partial", "name_wratio",
            "name_jacc", "name_tri", "name_exact", "name_core_exact", "name_core_jacc",
            "name_shared", "name_translit_ratio", "addr_ratio", "addr_set",
            "addr_jacc", "addr_tri", "addr_exact", "addr_shared"]


def _jacc(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tri(s):
    s = "#" + s + "#"
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def lexical_features(pairs: pd.DataFrame) -> pd.DataFrame:
    from .normalize import core_name
    from .transliterate import transliterate_local
    n1 = pairs["name_norm_1"].fillna("").astype(str).tolist()
    n2 = pairs["name_norm_2"].fillna("").astype(str).tolist()
    a1 = pairs["addr_norm_1"].fillna("").astype(str).tolist()
    a2 = pairs["addr_norm_2"].fillna("").astype(str).tolist()
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
            t1, t2 = transliterate_local(x1), transliterate_local(x2)
            tr = fuzz.ratio(t1, t2) / 100.0 if (t1 and t2) else 0.0
        else:
            nr = ns = nt = npr = nwr = _jacc(set(x1.split()), set(x2.split()))
            ar = at = _jacc(set(y1.split()), set(y2.split()))
            tr = _jacc(set(transliterate_local(x1).split()), set(transliterate_local(x2).split()))
        cx1, cx2 = core_name(x1), core_name(x2)
        rows.append([nr, ns, nt, npr, nwr, _jacc(set(x1.split()), set(x2.split())),
                     _jacc(_tri(x1), _tri(x2)), 1.0 if x1 == x2 and x1 else 0.0,
                     1.0 if cx1 == cx2 and cx1 else 0.0,
                     _jacc(set(cx1.split()), set(cx2.split())) if (cx1 or cx2) else 1.0,
                     float(len(set(x1.split()) & set(x2.split()))), tr,
                     ar, at, _jacc(set(y1.split()), set(y2.split())),
                     _jacc(_tri(y1), _tri(y2)), 1.0 if y1 == y2 and y1 else 0.0,
                     float(len(set(y1.split()) & set(y2.split())))])
    return pd.DataFrame(rows, columns=LEX_COLS)
