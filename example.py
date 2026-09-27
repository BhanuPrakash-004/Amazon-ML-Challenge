import pandas as pd
import numpy as np
import re
from collections import Counter

BASE = "student_resource"

FILES = {
    "s1_train": f"{BASE}/dataset/train/train_source1.tsv",
    "s2_train": f"{BASE}/dataset/train/train_source2.tsv",
    "s3_train": f"{BASE}/dataset/train/train_source3.tsv",
    "gt":      f"{BASE}/dataset/train/train_ground_truth.tsv",
    "s1_test": f"{BASE}/dataset/test/test_source1.tsv",
    "s2_test": f"{BASE}/dataset/test/test_source2.tsv",
    "s3_test": f"{BASE}/dataset/test/test_source3.tsv",
}

dfs = {}
for name, path in FILES.items():
    dfs[name] = pd.read_csv(path, sep="\t", dtype=str).fillna("")

# ------------------------------------------------------------
# 1. BASIC STRUCTURE
# ------------------------------------------------------------
print("=" * 100)
print("BASIC DATASET STRUCTURE")
print("=" * 100)

for name, df in dfs.items():
    print(f"\n{name}: shape={df.shape}")
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes.to_string())

# ------------------------------------------------------------
# 2. SAMPLE ROWS
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 250)
pd.set_option("display.width", 220)

for name in ["s1_train", "s2_train", "s3_train", "gt"]:
    print("\n" + "=" * 100)
    print(f"SAMPLE: {name}")
    print("=" * 100)
    print(dfs[name].head(12).to_string(index=False))

# ------------------------------------------------------------
# 3. MISSINGNESS / UNIQUENESS
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("MISSINGNESS + UNIQUE COUNTS")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    df = dfs[name]

    print(f"\n--- {name} ---")
    for c in df.columns:
        x = df[c].astype(str)
        missing = ((x == "") | x.isna()).mean() * 100
        unique = x.nunique()
        print(
            f"{c:20s} "
            f"unique={unique:10d} "
            f"unique_pct={100*unique/len(df):7.2f}% "
            f"missing={missing:7.2f}%"
        )

# ------------------------------------------------------------
# 4. COUNTRY DISTRIBUTION
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("COUNTRY DISTRIBUTION")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train", "s1_test", "s2_test", "s3_test"]:
    print(f"\n{name}")
    print(dfs[name]["country"].value_counts(dropna=False).head(30).to_string())

# ------------------------------------------------------------
# 5. STRING LENGTH / DIGIT / CHARACTER ANALYSIS
# ------------------------------------------------------------
def stats_for_series(s):
    s = s.astype(str)

    return {
        "count": len(s),
        "mean_len": s.str.len().mean(),
        "median_len": s.str.len().median(),
        "p10_len": s.str.len().quantile(0.10),
        "p90_len": s.str.len().quantile(0.90),
        "empty_pct": (s == "").mean() * 100,
        "digit_pct": s.str.contains(r"\d", regex=True).mean() * 100,
        "latin_pct": s.str.contains(r"[A-Za-z]", regex=True).mean() * 100,
        "non_ascii_pct": s.map(lambda x: any(ord(ch) > 127 for ch in x)).mean() * 100,
    }

print("\n" + "=" * 100)
print("NAME / ADDRESS CHARACTERISTICS")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    df = dfs[name]

    for c in ["business_name", "business_address"]:
        st = stats_for_series(df[c])
        print(f"\n{name} - {c}")
        for k, v in st.items():
            print(f"{k:20s}: {v}")

# ------------------------------------------------------------
# 6. DETECT IMPORTANT ADDRESS COMPONENTS
# ------------------------------------------------------------
patterns = {
    "6_digit_number": r"\b\d{6}\b",
    "5_digit_number": r"\b\d{5}\b",
    "4_digit_number": r"\b\d{4}\b",
    "phone_like": r"\b(?:\+?\d[\d\s\-()]{7,}\d)\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "url": r"(?:https?://|www\.)\S+",
    "house_no": r"\b(?:no|number|#)?\s*\d+[A-Za-z]?(?:/\d+[A-Za-z]?)?\b",
}

print("\n" + "=" * 100)
print("ADDRESS PATTERN ANALYSIS")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    print(f"\n--- {name} ---")
    s = dfs[name]["business_address"].astype(str).str.lower()

    for pname, pat in patterns.items():
        cnt = s.str.contains(pat, regex=True).sum()
        print(f"{pname:20s}: {cnt:10d} / {len(s)} = {100*cnt/len(s):7.2f}%")

# ------------------------------------------------------------
# 7. NAME STRUCTURE
# ------------------------------------------------------------
LEGAL = [
    "limited", "ltd", "private", "pvt", "llc", "llp",
    "inc", "incorporated", "corporation", "corp",
    "company", "co", "plc"
]

print("\n" + "=" * 100)
print("NAME STRUCTURE")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    s = dfs[name]["business_name"].astype(str).str.lower()

    suffix_count = 0
    amp_count = 0
    digit_count = 0

    for x in s:
        toks = re.findall(r"\w+", x)
        if any(t in LEGAL for t in toks):
            suffix_count += 1
        if "&" in x or " and " in f" {x} ":
            amp_count += 1
        if re.search(r"\d", x):
            digit_count += 1

    print(f"\n{name}")
    print(f"legal suffix present : {suffix_count} ({100*suffix_count/len(s):.2f}%)")
    print(f"& / and present      : {amp_count} ({100*amp_count/len(s):.2f}%)")
    print(f"digits in name       : {digit_count} ({100*digit_count/len(s):.2f}%)")

# ------------------------------------------------------------
# 8. CHARACTER SET / LANGUAGE CLUES
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("NON-LATIN SCRIPT EXAMPLES")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    df = dfs[name]

    mask = (
        df["business_name"].str.contains(r"[^\x00-\x7F]", regex=True)
        |
        df["business_address"].str.contains(r"[^\x00-\x7F]", regex=True)
    )

    print(f"\n{name}: {mask.sum()} rows with non-ASCII characters")
    print(
        df.loc[mask, ["entity_id", "business_name", "business_address", "country"]]
        .head(15)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 9. DUPLICATES / NEAR-DUPLICATE STRUCTURE
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("DUPLICATE ANALYSIS")
print("=" * 100)

for name in ["s1_train", "s2_train", "s3_train"]:
    df = dfs[name]

    print(f"\n{name}")

    for c in ["business_name", "business_address"]:
        vc = df[c].value_counts()
        dup_rows = (vc > 1).sum()
        print(f"{c:20s} repeated values: {dup_rows}")

    both = (
        df.groupby(["business_name", "business_address"])
          .size()
          .sort_values(ascending=False)
    )

    print("exact name+address duplicate groups:", (both > 1).sum())
    print("largest duplicate group:", both.iloc[0] if len(both) else 0)

# ------------------------------------------------------------
# 10. GROUND TRUTH MATCH COUNT DISTRIBUTION
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("GROUND TRUTH MATCH COUNTS")
print("=" * 100)

gt = dfs["gt"].copy()

def split_matches(x):
    x = str(x).strip()
    return [] if x == "" else [z.strip() for z in x.split(",") if z.strip()]

gt["matches"] = gt["matched_entity_ids"].map(split_matches)
gt["match_count"] = gt["matches"].str.len()

print(gt["match_count"].describe().to_string())

print("\nmatch count distribution:")
print(gt["match_count"].value_counts().sort_index().to_string())

print(f"\nsingletons = {(gt['match_count']==0).sum()}")
print(f"singleton % = {100*(gt['match_count']==0).mean():.2f}%")

# ------------------------------------------------------------
# 11. POSITIVE PAIR EXAMPLES
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("REAL POSITIVE MATCH EXAMPLES")
print("=" * 100)

s1 = dfs["s1_train"].set_index("entity_id")
s2 = dfs["s2_train"].set_index("entity_id")
s3 = dfs["s3_train"].set_index("entity_id")

shown = 0

for _, row in gt.iterrows():
    s1id = row["source1_entity_id"]

    for mid in row["matches"]:
        if mid.startswith("S2-"):
            target = s2
        elif mid.startswith("S3-"):
            target = s3
        else:
            continue

        if s1id not in s1.index or mid not in target.index:
            continue

        a = s1.loc[s1id]
        b = target.loc[mid]

        print("\nS1:", s1id)
        print("NAME :", a["business_name"])
        print("ADDR :", a["business_address"])
        print("COUNTRY:", a["country"])

        print("MATCH:", mid)
        print("NAME :", b["business_name"])
        print("ADDR :", b["business_address"])
        print("COUNTRY:", b["country"])

        print("-" * 80)

        shown += 1
        if shown >= 20:
            break

    if shown >= 20:
        break

# ------------------------------------------------------------
# 12. STRONG CROSS-SOURCE EXACT OVERLAPS
# ------------------------------------------------------------
def norm_basic(x):
    x = str(x).lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

print("\n" + "=" * 100)
print("CROSS-SOURCE EXACT OVERLAPS AFTER BASIC NORMALIZATION")
print("=" * 100)

for field in ["business_name", "business_address"]:
    print(f"\nFIELD: {field}")

    sets = {}
    for name in ["s1_train", "s2_train", "s3_train"]:
        sets[name] = set(
            dfs[name][field].map(norm_basic)
            .loc[lambda x: x != ""]
        )

    print("S1 ∩ S2:", len(sets["s1_train"] & sets["s2_train"]))
    print("S1 ∩ S3:", len(sets["s1_train"] & sets["s3_train"]))
    print("S2 ∩ S3:", len(sets["s2_train"] & sets["s3_train"]))

# ------------------------------------------------------------
# 13. POSITIVE PAIR BASIC EXACTNESS
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("HOW OFTEN REAL MATCHES SHARE EXACT / PARTIAL FIELDS")
print("=" * 100)

records = []

for _, row in gt.iterrows():
    s1id = row["source1_entity_id"]

    if s1id not in s1.index:
        continue

    a = s1.loc[s1id]

    for mid in row["matches"]:
        if mid.startswith("S2-"):
            target = s2
        elif mid.startswith("S3-"):
            target = s3
        else:
            continue

        if mid not in target.index:
            continue

        b = target.loc[mid]

        an = norm_basic(a["business_name"])
        bn = norm_basic(b["business_name"])

        aa = norm_basic(a["business_address"])
        ba = norm_basic(b["business_address"])

        records.append({
            "name_exact": an == bn and an != "",
            "address_exact": aa == ba and aa != "",
            "country_exact": norm_basic(a["country"]) == norm_basic(b["country"]),
            "name_nonempty": bool(an),
            "address_nonempty": bool(aa),
            "name_len_a": len(an),
            "name_len_b": len(bn),
            "addr_len_a": len(aa),
            "addr_len_b": len(ba),
        })

pf = pd.DataFrame(records)

print("positive pairs:", len(pf))

for c in ["name_exact", "address_exact", "country_exact"]:
    print(f"{c:20s}: {100*pf[c].mean():7.2f}%")

# ------------------------------------------------------------
# 14. SAMPLE OF MULTI-MATCH ENTITIES
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("MULTI-MATCH S1 EXAMPLES")
print("=" * 100)

multi = gt[gt["match_count"] >= 2].head(20)

for _, row in multi.iterrows():
    print("\nS1:", row["source1_entity_id"])
    print("matches:", ", ".join(row["matches"]))

    s1row = s1.loc[row["source1_entity_id"]]
    print("name:", s1row["business_name"])
    print("addr:", s1row["business_address"])
    print("country:", s1row["country"])

print("\nDONE")