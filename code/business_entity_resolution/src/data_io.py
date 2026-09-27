"""I/O helpers: load TSVs with explicit tab separator."""
import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # normalize header whitespace/BOM
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    for c in SOURCE_COLS:
        if c not in df.columns:
            raise ValueError("file %s missing column %r (got %s). Did you use sep='\\t'?" % (path, c, list(df.columns)))
    # keep only expected cols, strip ids
    df = df[SOURCE_COLS].copy()
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    df = df[df["entity_id"] != ""].reset_index(drop=True)
    return df


def read_ground_truth(path) -> dict:
    """Returns dict S1 -> set(S2/S3 ids). Empty list -> empty set."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "source1_entity_id" not in df.columns or "matched_entity_ids" not in df.columns:
        raise ValueError("ground truth must have source1_entity_id, matched_entity_ids (got %s)" % list(df.columns))
    out = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m = str(row["matched_entity_ids"]).strip() if row["matched_entity_ids"] is not None else ""
        if m == "" or m.lower() in ("nan", "none"):
            out[s1] = set()
        else:
            ids = [x.strip() for x in m.split(",") if x.strip() != ""]
            # dedupe
            out[s1] = set(ids)
    return out


def write_matching(path, s1_ids_in_order, pred_dict):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1 in s1_ids_in_order:
            ids = sorted(pred_dict.get(s1, set()))
            # sort S2 before S3, then lexicographically for determinism
            ids = sorted(ids, key=lambda x: (0 if x.startswith("S2-") else 1, x))
            w.writerow([s1, ",".join(ids)])


def write_candidates(path, s1_ids_in_order, cand_dict):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1 in s1_ids_in_order:
            ids = cand_dict.get(s1, [])
            # dedupe preserve sorted determinism
            ids = sorted(set(ids), key=lambda x: (0 if x.startswith("S2-") else 1, x))
            w.writerow([s1, ",".join(ids)])
