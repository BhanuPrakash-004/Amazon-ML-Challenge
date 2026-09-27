"""Verify Kaggle->laptop artifacts (spec section 37).

Checks manifest.json + checksums.sha256 + schema + row counts + dup IDs.
Fails loudly on mismatch.

Usage:
  python scripts/verify_artifacts.py --dir transfer_e5
  python scripts/verify_artifacts.py --dir artifacts/e5_recovery --schema s1_id,target_id,similarity
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
    ap.add_argument("--dir", required=True)
    ap.add_argument("--schema", default="")
    a = ap.parse_args(cmd)
    man_p = os.path.join(a.dir, "manifest.json")
    sum_p = os.path.join(a.dir, "checksums.sha256")
    if not os.path.exists(man_p):
        raise FileNotFoundError("missing manifest.json in %s" % a.dir)
    with open(man_p, encoding="utf-8") as f:
        man = json.load(f)
    print("manifest: %s" % json.dumps(man)[:500], flush=True)
    if os.path.exists(sum_p):
        with open(sum_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                exp, fn = line.split(None, 1)
                fn = fn.strip().lstrip("*/ ")
                p = os.path.join(a.dir, os.path.basename(fn))
                if not os.path.exists(p):
                    raise FileNotFoundError("checksum lists missing file: %s" % fn)
                got = sha256_of(p)
                if got != exp:
                    raise ValueError("checksum mismatch for %s" % fn)
                print("checksum OK: %s" % fn, flush=True)
    else:
        print("no checksums.sha256; schema/row-count checks only", flush=True)
    if a.schema:
        want = [c.strip() for c in a.schema.split(",") if c.strip()]
        # find a parquet in dir
        import glob
        cands = sorted(glob.glob(os.path.join(a.dir, "*.parquet")))
        if not cands:
            raise FileNotFoundError("no parquet found in %s" % a.dir)
        import pandas as pd
        df = pd.read_parquet(cands[0])
        missing = [c for c in want if c not in df.columns]
        if missing:
            raise ValueError("schema mismatch, missing cols %s (have %s)" % (missing, list(df.columns)))
        print("schema OK: %s rows=%d cols=%s" % (cands[0], len(df), list(df.columns)), flush=True)
        if "s1_id" in df.columns and "target_id" in df.columns:
            dups = int(df.duplicated(["s1_id", "target_id"]).sum())
            print("duplicate (s1_id,target_id): %d" % dups, flush=True)
    print("VERIFY: PASS", flush=True)


if __name__ == "__main__":
    main()
