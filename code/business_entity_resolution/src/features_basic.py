"""Basic pair features: lengths, digit counts, missing flags, source pair."""
import pandas as pd

BASIC_COLS = ["name_len1", "name_len2", "addr_len1", "addr_len2", "name_lendiff",
              "addr_lendiff", "tokdiff", "digit1", "digit2", "miss_name", "miss_addr",
              "both_present", "is_s2", "country_match"]


def basic_features(pairs: pd.DataFrame) -> pd.DataFrame:
    n1 = pairs["name_norm_1"].fillna("").astype(str)
    n2 = pairs["name_norm_2"].fillna("").astype(str)
    a1 = pairs["addr_norm_1"].fillna("").astype(str)
    a2 = pairs["addr_norm_2"].fillna("").astype(str)
    c1 = pairs["country_norm_1"].fillna("").astype(str)
    c2 = pairs["country_norm_2"].fillna("").astype(str)
    cid = pairs["candidate_id"].fillna("").astype(str) if "candidate_id" in pairs.columns else ""
    import re
    rows = []
    for i in range(len(pairs)):
        x1, x2 = n1.iloc[i], n2.iloc[i]
        y1, y2 = a1.iloc[i], a2.iloc[i]
        d1 = len(re.findall(r"\d", x1 + " " + y1))
        d2 = len(re.findall(r"\d", x2 + " " + y2))
        mn = 1.0 if not x1 or not x2 else 0.0
        ma = 1.0 if not y1 or not y2 else 0.0
        rows.append([len(x1), len(x2), len(y1), len(y2), abs(len(x1) - len(x2)),
                     abs(len(y1) - len(y2)), abs(len(x1.split()) - len(x2.split())),
                     d1, d2, mn, ma, 1.0 if (x1 and y1 and x2 and y2) else 0.0,
                     1.0 if str(cid.iloc[i]).startswith("S2-") else 0.0,
                     1.0 if c1.iloc[i] == c2.iloc[i] and c1.iloc[i] else 0.0])
    return pd.DataFrame(rows, columns=BASIC_COLS)
