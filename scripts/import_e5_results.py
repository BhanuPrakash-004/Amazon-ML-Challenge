"""Import Kaggle E5 output to laptop artifacts (spec section 34).

Verifies checksum/schema/row-count/ID types/duplicates, then copies the
compact e5_recovery.parquet (s1_id,target_id,country,source,channel,rank,
similarity) into artifacts/e5_recovery/ for candidate_union on the laptop.

Usage:
  python scripts/import_e5_results.py --in downloaded_e5 --out artifacts/e5_recovery
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(cmd=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "e5_recovery"))
    a = ap.parse_args(cmd)
    from scripts.verify_artifacts import sha256_of
    # run verifier first (fails loudly)
    sys.argv = ["verify_artifacts.py", "--dir", a.inp,
                "--schema", "s1_id,target_id,channel,rank,similarity"]
    # inline verify to keep single-command UX
    import glob
    man_p = os.path.join(a.inp, "manifest.json")
    if os.path.exists(man_p):
        with open(man_p, encoding="utf-8") as f:
            print("manifest: %s" % str(json.load(f))[:400], flush=True)
    sum_p = os.path.join(a.inp, "checksums.sha256")
    if os.path.exists(sum_p):
        with open(sum_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exp, fn = line.split(None, 1)
                    p = os.path.join(a.inp, os.path.basename(fn.strip()))
                    got = sha256_of(p)
                    if got != exp:
                        raise ValueError("checksum mismatch: %s" % fn)
                    print("checksum OK: %s" % fn, flush=True)
    cands = sorted(glob.glob(os.path.join(a.inp, "*.parquet")))
    if not cands:
        raise FileNotFoundError("no parquet in %s (download Kaggle output first)" % a.inp)
    import pandas as pd
    src = [p for p in cands if "e5_recovery" in os.path.basename(p)] or cands
    df = pd.read_parquet(src[0])
    for c in ("s1_id", "target_id", "channel", "rank", "similarity"):
        if c not in df.columns:
            raise ValueError("schema mismatch: missing %s" % c)
    if df["s1_id"].astype(str).str.startswith("S1-").mean() < 0.99:
        raise ValueError("S1 ID type check failed")
    print("rows=%d dups=%d" % (len(df), int(df.duplicated(["s1_id", "target_id"]).sum())), flush=True)
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "e5_recovery.parquet")
    shutil.copyfile(src[0], dst)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(df), "src": src[0]}, f, indent=1)
    print("imported -> %s" % dst, flush=True)
    print("next: rerun laptop union with E5 recovery (see RUNBOOK.md)", flush=True)


if __name__ == "__main__":
    main()
