"""Laptop: merge CPU candidates + Kaggle E5 recovery into adaptive final pool.

Union is done inside train/predict pipelines; this helper merges an
imported artifacts/e5_recovery/e5_recovery.parquet into an existing
candidate map for offline inspection. For production runs prefer the
pipeline flags (--adaptive) which already enforce 8-15 avg.

Usage:
  python laptop/merge_candidates.py --candidates output/candidate_pairs.tsv \
      --e5 artifacts/e5_recovery/e5_recovery.parquet --out output/candidate_pairs_v3.tsv
"""
import argparse
import csv
import os


def read_cands(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            m = (row.get("candidate_entity_ids") or "").strip()
            d[row["source1_entity_id"]] = [x for x in m.split(",") if x] if m else []
    return d


def main(cmd=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--e5", default="artifacts/e5_recovery/e5_recovery.parquet")
    ap.add_argument("--out", default="output/candidate_pairs_v3.tsv")
    ap.add_argument("--final-max", type=int, default=15)
    a = ap.parse_args(cmd)
    base = read_cands(a.candidates)
    extra = {}
    if os.path.exists(a.e5):
        import pandas as pd
        df = pd.read_parquet(a.e5)
        for s, g in df.groupby("s1_id", sort=False):
            extra.setdefault(str(s), []).extend([str(x) for x in g["target_id"].tolist()])
    n0 = sum(len(v) for v in base.values()) / max(len(base), 1)
    for s, lst in extra.items():
        cur = base.setdefault(s, [])
        for x in lst:
            if x not in cur:
                cur.append(x)
        # adaptive cap: keep insertion order (CPU multi-evidence first, E5 recovery tail)
        base[s] = cur[:a.final_max]
    n1 = sum(len(v) for v in base.values()) / max(len(base), 1)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s in sorted(base):
            lst = sorted(set(base[s]), key=lambda x: (0 if x.startswith("S2-") else 1, x))
            w.writerow([s, ",".join(lst)])
    print("avg %.2f -> %.2f wrote %s (copy to candidate_pairs.tsv only after validation)" % (n0, n1, a.out), flush=True)


if __name__ == "__main__":
    main()
