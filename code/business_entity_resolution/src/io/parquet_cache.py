"""On-disk normalized caches (Sec 4, 33): TSV -> lean parquet, once.

Normalizing 10M rows costs ~100us/row (~17min). Caching lean normalized
columns to parquet means every later pass (DF counting, index builds,
candidate generation, feature fetching) is a fast columnar scan with no
re-normalization. Embeddings are cached separately by the encoder caller.
"""
import os

PQ_COLS = ["entity_id", "country_norm", "name_norm", "name_translit",
           "name_core", "name_compact", "address_norm", "address_translit",
           "address_compact", "house_number", "numeric_tokens",
           "numeric_signature", "postal_candidate", "address_missing"]


def rec_to_row(eid, rec):
    return (eid, rec.get("country_norm", ""), rec.get("name_norm", ""),
            rec.get("name_translit", ""), rec.get("name_core", ""),
            rec.get("name_compact", ""), rec.get("address_norm", ""),
            rec.get("address_translit", ""), rec.get("address_compact", ""),
            rec.get("house_number", ""), " ".join(rec.get("numeric_tokens", [])),
            rec.get("numeric_signature", ""), rec.get("postal_candidate", ""),
            int(rec.get("address_missing", 0)))


def row_to_rec(row):
    return {"country_norm": row[1], "name_norm": row[2], "name_translit": row[3],
            "name_core": row[4], "name_compact": row[5], "address_norm": row[6],
            "address_translit": row[7], "address_compact": row[8],
            "house_number": row[9],
            "numeric_tokens": row[10].split() if row[10] else [],
            "numeric_signature": row[11], "postal_candidate": row[12],
            "address_missing": int(row[13])}


def build_cache(tsv_path, pq_path, chunk=200000):
    """Stream TSV -> normalize (lean) -> parquet. Returns row count."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pqp
    from ..preprocessing.normalize import normalize_record
    os.makedirs(os.path.dirname(os.path.abspath(pq_path)), exist_ok=True)
    tmp = pq_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    writer = None
    n = 0
    try:
        for ch in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False,
                              chunksize=chunk):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            rows = []
            for r in ch.itertuples():
                rec = normalize_record(r.business_name, r.business_address,
                                       r.country, lean=True)
                rows.append(rec_to_row(str(r.entity_id).strip(), rec))
            table = pa.Table.from_pylist(
                [dict(zip(PQ_COLS, row)) for row in rows])
            if writer is None:
                writer = pqp.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            n += len(rows)
            del rows, table, ch
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, pq_path)
    return n


def ensure_cache(tsv_path, pq_path, chunk=200000):
    import pandas as pd
    if os.path.isfile(pq_path):
        try:
            return pd.read_parquet(pq_path, columns=["entity_id"]).shape[0]
        except Exception:
            pass
    return build_cache(tsv_path, pq_path, chunk)


def scan_parquet(pq_path, columns=None, batch_rows=500000):
    """Yield (ids, recs) batches from a lean parquet cache (true streaming)."""
    import pyarrow.parquet as pq
    cols = columns or PQ_COLS
    pf = pq.ParquetFile(pq_path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        d = batch.to_pydict()
        ids = [str(x) for x in d["entity_id"]]
        n = len(ids)
        col = {c: d[c] for c in cols if c != "entity_id"}
        recs = []
        for i in range(n):
            nt = col["numeric_tokens"][i]
            recs.append({
                "country_norm": col["country_norm"][i] or "",
                "name_norm": col["name_norm"][i] or "",
                "name_translit": col["name_translit"][i] or "",
                "name_core": col["name_core"][i] or "",
                "name_compact": col["name_compact"][i] or "",
                "address_norm": col["address_norm"][i] or "",
                "address_translit": col["address_translit"][i] or "",
                "address_compact": col["address_compact"][i] or "",
                "house_number": col["house_number"][i] or "",
                "numeric_tokens": nt.split() if nt else [],
                "numeric_signature": col["numeric_signature"][i] or "",
                "postal_candidate": col["postal_candidate"][i] or "",
                "address_missing": int(col["address_missing"][i] or 0)})
        yield ids, recs
        del d, col, recs


def parquet_count(pq_path):
    import pandas as pd
    return pd.read_parquet(pq_path, columns=["entity_id"]).shape[0]
