"""Import Kaggle BGE output (spec section 23).

Expects bge_scores.parquet (s1_id,target_id,bge_score). Verifies checksum +
schema, copies to artifacts/bge/ for meta-model blending on the laptop.

Usage:
  python scripts/import_bge_results.py --in downloaded_bge --out artifacts/bge
"""
import argparse
import glob
import json
import os
import shutil


def main(cmd=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=os.path.join("artifacts", "bge"))
    a = ap.parse_args(cmd)
    from scripts.verify_artifacts import sha256_of
    sum_p = os.path.join(a.inp, "checksums.sha256")
    if os.path.exists(sum_p):
        with open(sum_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exp, fn = line.split(None, 1)
                    p = os.path.join(a.inp, os.path.basename(fn.strip()))
                    if sha256_of(p) != exp:
                        raise ValueError("checksum mismatch: %s" % fn)
                    print("checksum OK: %s" % fn, flush=True)
    cands = sorted(glob.glob(os.path.join(a.inp, "*.parquet")))
    if not cands:
        raise FileNotFoundError("no parquet in %s" % a.inp)
    import pandas as pd
    src = [p for p in cands if "bge_score" in os.path.basename(p)] or cands
    df = pd.read_parquet(src[0])
    for c in ("s1_id", "target_id", "bge_score"):
        if c not in df.columns:
            raise ValueError("schema mismatch: missing %s" % c)
    print("rows=%d" % len(df), flush=True)
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "bge_scores.parquet")
    shutil.copyfile(src[0], dst)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(df)}, f, indent=1)
    print("imported -> %s" % dst, flush=True)


if __name__ == "__main__":
    main()
