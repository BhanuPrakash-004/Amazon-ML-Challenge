"""Address/numeric pair features: token/char/numeric overlap, postal/house equality."""
import re

import pandas as pd

ADDR_COLS = ["addr_tok_jacc", "addr_char_jacc", "num_jacc", "pin_match", "zip_match",
             "any_num", "house_match", "addr_num_shared"]


def _jacc(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_DIG = re.compile(r"\d+")


def address_features(pairs: pd.DataFrame) -> pd.DataFrame:
    from .normalize import extract_house_numbers
    a1 = pairs["addr_norm_1"].fillna("").astype(str).tolist()
    a2 = pairs["addr_norm_2"].fillna("").astype(str).tolist()
    r1 = pairs.get("addr_raw_1", pairs["addr_norm_1"]).fillna("").astype(str).tolist()
    r2 = pairs.get("addr_raw_2", pairs["addr_norm_2"]).fillna("").astype(str).tolist()
    rows = []
    for i in range(len(pairs)):
        y1, y2 = a1[i], a2[i]
        d1, d2 = set(_DIG.findall(r1[i])), set(_DIG.findall(r2[i]))
        h1, h2 = set(x.lower() for x in extract_house_numbers(r1[i])), set(x.lower() for x in extract_house_numbers(r2[i]))
        rows.append([_jacc(set(y1.split()), set(y2.split())),
                     _jacc(set(y1), set(y2)),
                     _jacc(d1, d2) if (d1 or d2) else 0.0,
                     1.0 if any(len(d) == 6 for d in (d1 & d2)) else 0.0,
                     1.0 if any(len(d) == 5 for d in (d1 & d2)) else 0.0,
                     1.0 if len(d1 & d2) > 0 else 0.0,
                     1.0 if (h1 & h2) else 0.0,
                     float(len(d1 & d2))])
    return pd.DataFrame(rows, columns=ADDR_COLS)
