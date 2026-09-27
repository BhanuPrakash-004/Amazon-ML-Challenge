"""Full-scale inference (v2): chunked, streaming, laptop-friendly.

- Builds rare-token index by streaming test S23 (no 10M DataFrame in RAM).
- Streams test S1 in chunks (BLOCK_CHUNK_S1), queries candidates, fetches S23
  raw for chunk candidates only, featurizes (light), predicts, appends to outputs.
- Writes BOTH output/matching_results.tsv and output/candidate_pairs.tsv
  in test-S1 order (via temp per-chunk files merged at end).
- matched ⊆ candidates by construction.

Usage:
  python run_predict_v2.py
Time on laptop 12C/16GB (1.7M S1, 10M S23, top-50): ~30-60min.
"""
import csv
import os
import time

import joblib
import pandas as pd

from . import config as C
from .train_v2 import build_index_streaming, fetch_s23_for_ids
from .features_light import light_features, LIGHT_COLS
from .model import apply_veto_rules


def stream_s1_chunks(s1_path, chunk):
    for ch in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
        yield ch[["entity_id", "business_name", "business_address", "country"]]


def main():
    import json
    t0 = time.time()
    print("PREDICT_V2 root=%s" % C.ROOT, flush=True)
    model_pkl = os.path.join(C.MODEL_DIR, "model.pkl")
    thr_json = os.path.join(C.MODEL_DIR, "threshold.json")
    if not os.path.exists(model_pkl):
        raise FileNotFoundError("train first: python run_train_v2.py (missing %s)" % model_pkl)
    clf = joblib.load(model_pkl)
    with open(thr_json, encoding="utf-8") as f:
        thr = float(json.load(f)["threshold"])
    # threshold + top_k/df_cap used in training (for consistent blocking)
    try:
        fb = joblib.load(os.path.join(C.MODEL_DIR, "feature_builder.pkl"))
        top_k = (fb.get("top_k") if isinstance(fb, dict) else None) or C.BLOCK_TOP_K
        df_cap = (fb.get("df_cap") if isinstance(fb, dict) else None) or C.BLOCK_DF_CAP
    except Exception:
        top_k, df_cap = C.BLOCK_TOP_K, C.BLOCK_DF_CAP
    print("model thr=%.3f top_k=%d df_cap=%d" % (thr, top_k, df_cap), flush=True)

    print("building test S23 index (streaming)...", flush=True)
    blk = build_index_streaming([C.TEST_FILES["s2"], C.TEST_FILES["s3"]], df_cap)
    blk.top_k = top_k

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    match_path = os.path.join(C.OUTPUT_DIR, "matching_results.tsv")
    cand_path = os.path.join(C.OUTPUT_DIR, "candidate_pairs.tsv")
    # write headers
    with open(match_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])

    from .pairs import pairs_from_candidates
    from .train_v2 import _count_data_rows, _fmt_hms
    total_s1 = _count_data_rows(C.TEST_FILES["s1"])
    print("test S1 total: %d (chunk %d)" % (total_s1, C.BLOCK_CHUNK_S1), flush=True)
    n_s1 = n_match = n_single = 0
    for ci, s1ch in enumerate(stream_s1_chunks(C.TEST_FILES["s1"], C.BLOCK_CHUNK_S1)):
        t1 = time.time()
        cand = blk.query(s1ch)
        # fetch S23 raw for this chunk's candidates
        need = set()
        for lst in cand["candidate_entity_ids"]:
            need.update(lst)
        s23need = fetch_s23_for_ids([C.TEST_FILES["s2"], C.TEST_FILES["s3"]], need)
        pairs = pairs_from_candidates(s1ch, s23need, cand, truth=None)
        if len(pairs):
            feat = light_features(pairs)
            X = feat[LIGHT_COLS].to_numpy(dtype=float)
            proba = clf.predict_proba(X)[:, 1]
            keep = apply_veto_rules(feat, proba, thr)
        else:
            import numpy as np
            keep = []
            proba = []
        pred = {}
        for s, c, k in zip(pairs["source1_entity_id"] if len(pairs) else [],
                           pairs["candidate_id"] if len(pairs) else [], keep):
            if k:
                pred.setdefault(s, set()).add(c)
        # append in chunk order
        with open(match_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            for s1id in s1ch["entity_id"].tolist():
                ids = sorted(pred.get(s1id, set()), key=lambda x: (0 if x.startswith("S2-") else 1, x))
                w.writerow([s1id, ",".join(ids)])
                n_match += len(ids)
                if not ids:
                    n_single += 1
                n_s1 += 1
        with open(cand_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            cmap = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
            for s1id in s1ch["entity_id"].tolist():
                lst = sorted(set(cmap.get(s1id, [])), key=lambda x: (0 if x.startswith("S2-") else 1, x))
                w.writerow([s1id, ",".join(lst)])
        el_min = (time.time() - t0) / 60
        rate = n_s1 / (time.time() - t0) if time.time() - t0 > 0 else 0
        eta_min = (total_s1 - n_s1) / rate / 60 if rate > 0 and total_s1 else -1
        print("chunk %d: %d S1 in %.1fs | total %d/%d (%.1f%%) remaining %d | %.0f S1/s | elapsed %s | ETA %s"
              % (ci, len(s1ch), time.time() - t1, n_s1, total_s1,
                 100.0 * n_s1 / total_s1 if total_s1 else 0, max(total_s1 - n_s1, 0), rate,
                 _fmt_hms((time.time() - t0)), _fmt_hms(eta_min * 60) if eta_min >= 0 else "--"), flush=True)
    print("done: %d S1, %d matches, %d singletons in %.1fm" % (n_s1, n_match, n_single, (time.time() - t0) / 60), flush=True)
    print("validate: python student_resource/utils/validate_submission.py --matching %s --candidate %s --test-dir %s [--check-ids]" % (match_path, cand_path, C.TEST_DIR), flush=True)
    try:
        with open(os.path.join(C.OUTPUT_DIR, "..", "results.txt"), "a", encoding="utf-8") as _f:
            _f.write("=== predict %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            _f.write("test S1=%d total_matches=%d singletons=%d thr=%.3f top_k=%d df_cap=%d elapsed=%.1fm\n\n"
                     % (n_s1, n_match, n_single, thr, top_k, df_cap, (time.time() - t0) / 60))
    except Exception as _e:
        print("results.txt logging skipped: %s" % _e, flush=True)


if __name__ == "__main__":
    main()
