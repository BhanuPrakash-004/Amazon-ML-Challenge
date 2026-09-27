"""Create laptop -> Kaggle E5 transfer package (spec section 12/32).

Reads S1 queries + S2/S3 targets from TSVs (streamed, never full 20M in
one DataFrame), normalizes with src.normalize, writes Parquet+ZSTD +
manifest.json + checksums.sha256. Only E5-needed fields are transferred.

Usage (repo root):
  python scripts/create_transfer_e5.py --data-root student_resource/dataset \
      --out transfer_e5 --s1-shard 200000
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stream_normalized_tsv(path, source, chunk=200000):
    import pandas as pd
    from src.normalize import normalize_name, normalize_address, normalize_country
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
        out = {
            "id": ch["entity_id"].astype(str).tolist(),
            "normalized_name": [normalize_name(x) for x in ch["business_name"].astype(str).tolist()],
            "normalized_address": [normalize_address(x) for x in ch["business_address"].astype(str).tolist()],
            "country": [normalize_country(x) for x in ch["country"].astype(str).tolist()],
            "source": [source] * len(ch),
        }
        yield out


def write_parquet_zstd(rows_iter, out_path):
    import pandas as pd
    parts = []
    for part in rows_iter:
        parts.append(pd.DataFrame(part))
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    return len(df)


def main(cmd=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "student_resource", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "transfer_e5"))
    ap.add_argument("--s1-shard", type=int, default=200000)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    a = ap.parse_args(cmd)
    os.makedirs(a.out, exist_ok=True)
    if a.split == "test":
        s1 = os.path.join(a.out, "..", a.data_root, "test", "test_source1.tsv") if False else os.path.join(a.data_root, "test", "test_source1.tsv")
        s2 = os.path.join(a.data_root, "test", "test_source2.tsv")
        s3 = os.path.join(a.data_root, "test", "test_source3.tsv")
        for alt in (os.path.join(a.data_root, "test_source1.tsv"),):
            if not os.path.exists(s1) and os.path.exists(alt):
                s1 = alt
    else:
        s1 = os.path.join(a.data_root, "train", "train_source1.tsv")
        s2 = os.path.join(a.data_root, "train", "train_source2.tsv")
        s3 = os.path.join(a.data_root, "train", "train_source3.tsv")
    for p in (s1, s2, s3):
        if not os.path.exists(p):
            # flat layout fallback
            base = os.path.basename(p)
            alt = os.path.join(a.data_root, base)
            if os.path.exists(alt):
                if "source1" in base:
                    s1 = alt
                elif "source2" in base:
                    s2 = alt
                else:
                    s3 = alt
    for p in (s1, s2, s3):
        if not os.path.exists(p):
            raise FileNotFoundError("missing input TSV: %s" % p)
    q_path = os.path.join(a.out, "s1_queries.parquet")
    t_path = os.path.join(a.out, "targets.parquet")
    nq = write_parquet_zstd(stream_normalized_tsv(s1, "S1", a.s1_shard), q_path)

    def _tgt():
        yield from stream_normalized_tsv(s2, "S2", a.s1_shard)
        yield from stream_normalized_tsv(s3, "S3", a.s1_shard)
    # stream both without holding 10M rows: write in two appends
    import pandas as pd
    first = True
    nt = 0
    for part in _tgt():
        df = pd.DataFrame(part)
        nt += len(df)
        if first:
            df.to_parquet(t_path, compression="zstd", index=False)
            first = False
        else:
            old = pd.read_parquet(t_path)
            pd.concat([old, df], ignore_index=True).to_parquet(t_path, compression="zstd", index=False)
            del old
        del df
    manifest = {"s1_queries": q_path, "targets": t_path, "n_queries": nq, "n_targets": nt,
                "format": "parquet+zstd", "fields": ["id", "normalized_name", "normalized_address", "country", "source"]}
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(a.out, "checksums.sha256"), "w", encoding="utf-8") as f:
        for fn in ("s1_queries.parquet", "targets.parquet", "manifest.json"):
            f.write("%s  %s\n" % (sha256_of(os.path.join(a.out, fn)), fn))
    print("wrote %s queries=%d targets=%d" % (a.out, nq, nt), flush=True)
    print("upload %s as a Kaggle Dataset, attach to kaggle/e5_retrieval.ipynb" % a.out, flush=True)


if __name__ == "__main__":
    main()
