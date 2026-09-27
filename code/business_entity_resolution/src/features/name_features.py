"""Name features (Sec 18)."""
try:
    from rapidfuzz import fuzz
    HAS_RF = True
except Exception:
    HAS_RF = False
    fuzz = None


def _jacc(a, b):
    a, b = set(a.split()), set(b.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def name_feats(a, b):
    an, bn = a.get("name_norm", ""), b.get("name_norm", "")
    ac, bc = a.get("name_core", ""), b.get("name_core", "")
    at, bt = a.get("name_translit", ""), b.get("name_translit", "")
    ax, bx = a.get("name_compact", ""), b.get("name_compact", "")
    if HAS_RF and an and bn:
        r = fuzz.ratio(an, bn) / 100.0
        w = fuzz.WRatio(an, bn) / 100.0
        ts = fuzz.token_sort_ratio(an, bn) / 100.0
        te = fuzz.token_set_ratio(an, bn) / 100.0
    else:
        r = w = ts = te = _jacc(an, bn)
    la = len(an) + 1
    return {
        "name_exact": 1.0 if an and an == bn else 0.0,
        "name_core_exact": 1.0 if ac and ac == bc else 0.0,
        "name_translit_exact": 1.0 if at and at == bt else 0.0,
        "name_compact_exact": 1.0 if ax and ax == bx else 0.0,
        "fuzz_ratio": r, "wratio": w,
        "token_sort_ratio": ts, "token_set_ratio": te,
        "name_token_jaccard": _jacc(an, bn),
        "name_length_ratio": min(len(an), len(bn)) / max(la, len(bn) + 1),
    }
