"""Address features (Sec 18)."""
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


def addr_feats(a, b):
    x, y = a.get("address_norm", ""), b.get("address_norm", "")
    if HAS_RF and x and y:
        r = fuzz.ratio(x, y) / 100.0
        ts = fuzz.token_sort_ratio(x, y) / 100.0
        te = fuzz.token_set_ratio(x, y) / 100.0
    else:
        r = ts = te = _jacc(x, y)
    return {
        "address_exact": 1.0 if x and x == y else 0.0,
        "address_ratio": r,
        "address_token_sort_ratio": ts,
        "address_token_set_ratio": te,
        "address_token_jaccard": _jacc(x, y),
        "address_length_ratio": min(len(x), len(y)) / max(len(x) + 1, len(y) + 1),
    }
