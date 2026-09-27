"""Export ONLY ambiguous pairs for BGE (spec section 35).

Reads Stage-1 scored pairs (output/stage1_scored.parquet or a CSV with
s1_id,target_id,stage1_probability) plus raw names/addresses, selects top
5-15 ambiguous per S1 (small top1-top2 gap / low confidence), writes
Parquet+ZSTD + manifest + checksums. Never exports all 20M candidates.

Usage:
  python scripts/create_transfer_bge.py --scored output/stage1_scored.parquet \
      --out transfer_bge --top-n 10
"""
import argparse
import hashlib
import json
import os


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(cmd=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True,
                    help="parquet with s1_id,target_id,name,address,country,source,stage1_probability")
    ap.add_argument("--out", default="transfer_bge")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.25)
    a = ap.parse_args(cmd)
    import pandas as pd
    df = pd.read_parquet(a.scored) if a.scored.endswith(".parquet") else pd.read_csv(a.scored, sep="\t")
    need = {"s1_id", "target_id", "stage1_probability"}
    if not need.issubset(df.columns):
        raise ValueError("scored file must contain %s (have %s)" % (sorted(need), list(df.columns)))
    out_rows = []
    for s, g in df.groupby("s1_id", sort=False):
        g = g.sort_values("stage1_probability", ascending=False).head(max(a.top_n, 15))
        if len(g) < 2:
            out_rows.append(g.head(a.top_n))
            continue
        top = g["stage1_probability"].to_numpy()
        gap = float(top[0] - top[1])
        if gap <= a.margin or float(top[0]) < 0.7:
            out_rows.append(g.head(a.top_n))
        else:
            out_rows.append(g.head(3))  # high-confidence: keep tiny probe only
    sel = pd.concat(out_rows, ignore_index=True) if out_rows else df.iloc[0:0]
    keep = [c for c in ("s1_id", "target_id", "name", "address", "country", "source", "stage1_probability")
            if c in sel.columns]
    sel = sel[keep]
    os.makedirs(a.out, exist_ok=True)
    out_p = os.path.join(a.out, "bge_pairs.parquet")
    sel.to_parquet(out_p, compression="zstd", index=False)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(sel), "top_n": a.top_n, "margin": a.margin}, f, indent=1)
    with open(os.path.join(a.out, "checksums.sha256"), "w", encoding="utf-8") as f:
        for fn in ("bge_pairs.parquet", "manifest.json"):
            f.write("%s  %s\n" % (sha256_of(os.path.join(a.out, fn)), fn))
    print("bge export rows=%d (from %d scored) -> %s" % (len(sel), len(df), out_p), flush=True)


if __name__ == "__main__":
    main()
