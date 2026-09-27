"""Numeric + missingness features (Sec 6, 18)."""
import re


def num_feats(a, b):
    an, bn = set(a.get("numeric_tokens", [])), set(b.get("numeric_tokens", []))
    inter = len(an & bn)
    union = len(an | bn)
    ah, bh = a.get("house_number", ""), b.get("house_number", "")
    ap, bp = a.get("postal_candidate", ""), b.get("postal_candidate", "")
    am, bm = a.get("address_missing", 0), b.get("address_missing", 0)
    return {
        "house_exact": 1.0 if ah and ah == bh else 0.0,
        "house_normalized_exact": 1.0 if ah and ah == bh else 0.0,
        "shared_numeric_count": float(inter),
        "numeric_jaccard": (inter / union) if union else 0.0,
        "postal_exact": 1.0 if ap and ap == bp else 0.0,
        "postal_conflict": 1.0 if ap and bp and ap != bp else 0.0,
        "house_conflict": 1.0 if ah and bh and ah != bh else 0.0,
        "numeric_conflict": 1.0 if (an and bn and inter == 0) else 0.0,
        "source_address_missing": float(am),
        "target_address_missing": float(bm),
        "both_address_present": 1.0 if (not am and not bm) else 0.0,
    }
