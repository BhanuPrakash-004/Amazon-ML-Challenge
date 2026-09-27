"""Polars/chunked TSV loading (Sec 2, 4, 33).

- Prefers polars (scan_csv, float32/int32 discipline); falls back to pandas.
- Explicit tab separator (Sec file-format requirement).
- Chunked S1 inference (100k-250k), streaming S2/S3 for index builds.
- Open-set country: never filters or one-hots countries.
"""
import csv
import os

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False


def _check_cols(cols):
    for c in SOURCE_COLS:
        if c not in cols:
            raise ValueError("missing column %r (got %s). Did you use sep='\\t'?" % (c, cols))


def read_source(path, columns=None):
    """Load one source TSV fully (use streaming variants for 5M-row files)."""
    cols = columns or SOURCE_COLS
    if HAS_POLARS:
        df = pl.read_csv(path, separator="\t", dtypes={"entity_id": pl.Utf8,
                         "business_name": pl.Utf8, "business_address": pl.Utf8,
                         "country": pl.Utf8}, null_values=[""], ignore_errors=True)
        df = df.with_columns([pl.col(c).fill_null("") for c in cols if c in df.columns])
        _check_cols(df.columns)
        return df.select(cols)
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    _check_cols(list(df.columns))
    df = df[SOURCE_COLS].copy()
    df["entity_id"] = df["entity_id"].astype(str).str.strip()
    return df[df["entity_id"] != ""].reset_index(drop=True)


def scan_source_batched(path, batch_rows=200000):
    """Yield DataFrame chunks (pandas or polars). Never loads 5M rows at once."""
    if HAS_POLARS:
        # polars has no native chunked read_csv iterator; stream via pandas fallback
        # when file is huge, else single scan. Use lazy + slice loop.
        import pandas as pd
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              chunksize=batch_rows):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            yield ch
    else:
        import pandas as pd
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              chunksize=batch_rows):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            yield ch


def read_ground_truth(path):
    """Return dict S1 -> set(S2/S3 ids). Streaming, low-memory."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        if "source1_entity_id" not in (r.fieldnames or []) or \
           "matched_entity_ids" not in (r.fieldnames or []):
            raise ValueError("ground truth must have source1_entity_id, matched_entity_ids")
        for row in r:
            s1 = (row.get("source1_entity_id") or "").strip()
            m = (row.get("matched_entity_ids") or "").strip()
            if not s1:
                continue
            out[s1] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
    return out


def s1_id_order(path):
    """Ordered S1 ids (for output completeness). Streaming."""
    ids = []
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            e = (row.get("entity_id") or "").strip()
            if e:
                ids.append(e)
    return ids


def count_rows(path):
    n = 0
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            n += blk.count(b"\n")
    return max(0, n - 1)
