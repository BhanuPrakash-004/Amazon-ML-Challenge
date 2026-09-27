"""Full-scale training (v2): sampled, streaming, laptop-friendly.

Steps:
 1. Stream S1+GT to sample TRAIN_SAMPLE_S1 + VAL_SAMPLE_S1 S1 ids
    (stratified by country x singleton, no full GT dict in memory).
 2. Stream S23 (chunks) to build per-country rare-token index (DF cap).
 3. Query sampled S1 in chunks -> candidates (top-K).
 4. Stream S23 again to fetch raw records for candidate IDs -> featurize
    (vocab-free light features, chunked, multiprocessing optional).
 5. Train LightGBM, tune threshold for macro F0.5, save artifacts.

Memory: <6GB (never holds 10M rows at once). Time on laptop 12C/16GB:
  ~20-40min for 100k+20k sample. Full test inference is separate (predict_v2).

Usage:
  python run_train_v2.py [--sample-s1 100000 --val-s1 20000 --top-k 50]
"""
import argparse
import csv
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _fmt_hms(sec):
    sec = max(0, int(sec))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def _log_progress(done, total, t0, label):
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    if total:
        pct = 100.0 * done / total
        eta = (total - done) / rate if rate > 0 else -1
        print("[%s] %d/%d (%.1f%%) | %.0f rows/s | elapsed %s | ETA %s"
              % (label, done, total, pct, rate, _fmt_hms(el), _fmt_hms(eta) if eta >= 0 else "--"),
              flush=True)
    else:
        print("[%s] %d rows | %.0f rows/s | elapsed %s" % (label, done, rate, _fmt_hms(el)), flush=True)


def _count_data_rows(path):
    """Fast data-row count (excl header) via binary newline count."""
    try:
        n = 0
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                n += blk.count(b"\n")
        return max(0, n - 1)
    except Exception:
        return 0

from . import config as C
from .normalize import normalize_name, normalize_address, normalize_country
from .blocking_v2 import RareTokenBlocker
from .features_light import light_features, LIGHT_COLS
from .model import train_classifier, tune_threshold, save_artifacts
from .evaluate import macro_f05


def stream_s1_with_gt(s1_path, gt_path, want_ids=None):
    """Yield (entity_id, name, addr, country, matched_set or None).
    If want_ids given, only yields those (GT looked up via streaming)."""
    # Load GT for want_ids only (or all if None -- caller must sample first)
    gt = {}
    if want_ids is not None:
        want = set(want_ids)
        with open(gt_path, encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                s = row["source1_entity_id"]
                if s in want:
                    m = (row["matched_entity_ids"] or "").strip()
                    gt[s] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
                    if len(gt) == len(want):
                        # can't break safely (GT order may differ from S1 order);
                        # continue to be safe? break only if GT covers all want
                        pass
    with open(s1_path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            eid = row["entity_id"]
            if want_ids is not None and eid not in want:
                continue
            yield eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), gt.get(eid)


def sample_s1_ids(s1_path, gt_path, n_train, n_val, seed=42):
    """Stratified sample by country x singleton. Streams twice, holds only ID lists.
    Returns (train_ids, val_ids, country_of, is_singleton)."""
    import random
    random.seed(seed)
    # Pass 1: read GT singleton flags (streaming, store per-S1 flag only for sampled? need country too)
    # To avoid holding 2.2M flags, reservoir-sample per stratum in one pass over JOIN of S1+GT?
    # Simpler: load S1 countries (2.2M rows: id->country, ~100MB) + stream GT for singleton flags,
    # then stratified sample. S1 country dict of 2.2M entries ~200MB -- okay on 16GB.
    print("pass 1: loading S1 countries...", flush=True)
    id2country = {}
    _t = time.time()
    _n = 0
    with open(s1_path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            id2country[row["entity_id"]] = row.get("country", "")
            _n += 1
            if _n % 500000 == 0:
                _log_progress(_n, 0, _t, "S1 countries")
    print("S1 loaded: %d" % len(id2country), flush=True)
    print("pass 2: streaming GT for singleton flags + stratification...", flush=True)
    from collections import defaultdict as dd
    strata = dd(list)
    n_gt = 0
    _t2 = time.time()
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row["source1_entity_id"]
            m = (row["matched_entity_ids"] or "").strip()
            sing = 1 if not m else 0
            c = id2country.get(s, "")
            strata[(c, sing)].append(s)
            n_gt += 1
            if n_gt % 500000 == 0:
                _log_progress(n_gt, 0, _t2, "GT scan")
    print("GT rows: %d, strata: %s" % (n_gt, {k: len(v) for k, v in strata.items()}), flush=True)
    train_ids, val_ids = [], []
    for key, ids in strata.items():
        random.shuffle(ids)
        n = len(ids)
        frac = n / max(n_gt, 1)
        nt = max(1, int(round(n_train * frac))) if n > 1 else 0
        nv = max(1, int(round(n_val * frac))) if n > 1 else 0
        train_ids.extend(ids[:nt])
        val_ids.extend(ids[nt:nt + nv])
    # trim to exact budget (random drop)
    random.shuffle(train_ids)
    random.shuffle(val_ids)
    train_ids, val_ids = train_ids[:n_train], val_ids[:n_val]
    print("sampled train=%d val=%d" % (len(train_ids), len(val_ids)), flush=True)
    return train_ids, val_ids


def build_index_streaming(s23_paths, df_cap, chunk=200000, max_rows=None):
    """Build RareTokenBlocker index by streaming S23 files in chunks.
    max_rows caps rows per file (quick-test only)."""
    from collections import Counter
    # Pass 1: DF (with totals for remaining/ETA)
    print("index pass 1/2: DF counting...", flush=True)
    totals = {p: (_count_data_rows(p) if max_rows is None else min(max_rows, _count_data_rows(p))) for p in s23_paths}
    grand = sum(totals.values())
    print("S23 rows to scan (pass 1): %d %s" % (grand, {os.path.basename(p): totals[p] for p in s23_paths}), flush=True)
    df_per_c = defaultdict(Counter)
    n_per_c = Counter()
    done_all = 0
    t_pass = time.time()
    for path in s23_paths:
        n_read = 0
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            if max_rows is not None:
                ch = ch.iloc[: max(0, max_rows - n_read)]
                if len(ch) == 0:
                    break
            names = ch["business_name"].fillna("").astype(str)
            addrs = ch["business_address"].fillna("").astype(str)
            coun = ch["country"].fillna("").astype(str)
            nns = names.map(normalize_name)
            aas = addrs.map(normalize_address)
            ccs = coun.map(normalize_country)
            toks = (nns + " " + aas).str.split()
            for cc, tk in zip(ccs.tolist(), toks.tolist()):
                n_per_c[cc] += 1
                df_per_c[cc].update(set(tk))
            n_read += len(ch)
            done_all += len(ch)
            _log_progress(done_all, grand, t_pass, "index pass 1/2 DF")
            if max_rows is not None and n_read >= max_rows:
                break
    print("DF done. countries: %s" % {k: n_per_c[k] for k in n_per_c}, flush=True)
    idf_per_c = {c: {t: math.log(n_per_c[c] / (1 + d)) for t, d in dfc.items()} for c, dfc in df_per_c.items()}
    # Pass 2: postings for rare tokens only + id lists + names + pins
    print("index pass 2/2: postings (rare DF<=%d)..." % df_cap, flush=True)
    blk = RareTokenBlocker(df_cap=df_cap, top_k=C.BLOCK_TOP_K, n_tok=C.BLOCK_N_TOK)
    # init empty structures
    per = {}
    for c in n_per_c:
        per[c] = {"ids": [], "nns": [], "df": df_per_c[c], "idf": idf_per_c[c],
                  "post": defaultdict(list), "name_to_pos": defaultdict(list),
                  "core_to_pos": defaultdict(list), "pin_to_pos": defaultdict(list)}
    pat = re.compile(r"\d{5,6}")
    done2 = 0
    t_pass2 = time.time()
    for path in s23_paths:
        n_read = 0
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            if max_rows is not None:
                ch = ch.iloc[: max(0, max_rows - n_read)]
                if len(ch) == 0:
                    break
            eids = ch["entity_id"].tolist()
            bns = ch["business_name"].fillna("").astype(str).tolist()
            bas = ch["business_address"].fillna("").astype(str).tolist()
            ccs = [normalize_country(x) for x in ch["country"].fillna("").astype(str).tolist()]
            nns = [normalize_name(x) for x in bns]
            aas = [normalize_address(x) for x in bas]
            for eid, cc, nn, aa, ba in zip(eids, ccs, nns, aas, bas):
                if cc not in per:
                    continue
                P = per[cc]
                pos = len(P["ids"])
                P["ids"].append(eid)
                P["nns"].append(nn)
                for t in set((nn + " " + aa).split()):
                    if df_per_c[cc].get(t, 0) <= df_cap:
                        P["post"][t].append(pos)
                if nn:
                    P["name_to_pos"][nn].append(pos)
                    from .normalize import core_name as _cn
                    _c = _cn(nn)
                    if _c:
                        P["core_to_pos"][_c].append(pos)
                for p in set(pat.findall(ba)):
                    P["pin_to_pos"][p].append(pos)
            n_read += len(ch)
            done2 += len(ch)
            _log_progress(done2, grand, t_pass2, "index pass 2/2 postings")
            if max_rows is not None and n_read >= max_rows:
                break
    # convert postings to np arrays
    for c, P in per.items():
        P["post"] = {t: np.asarray(lst, dtype=np.int32) for t, lst in P["post"].items()}
        P["name_to_pos"] = dict(P["name_to_pos"])
        P["core_to_pos"] = dict(P["core_to_pos"])
        P["pin_to_pos"] = dict(P["pin_to_pos"])
    blk.per_country = per
    print("index built.", flush=True)
    return blk


def fetch_s23_for_ids(s23_paths, need_ids, chunk=200000):
    """Stream S23 files, return DataFrame of rows whose entity_id in need_ids."""
    need = set(need_ids)
    total_need = len(need)
    out = []
    t_f = time.time()
    scanned = 0
    for path in s23_paths:
        for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
            ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
            scanned += len(ch)
            hit = ch[ch["entity_id"].isin(need)]
            if len(hit):
                out.append(hit[["entity_id", "business_name", "business_address", "country"]])
                need -= set(hit["entity_id"].tolist())
            _log_progress(total_need - len(need), total_need, t_f, "fetch S23 (found/scanned %d)" % scanned)
            if not need:
                break
        if not need:
            break
    if out:
        return pd.concat(out, ignore_index=True).drop_duplicates("entity_id")
    return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])


def load_s1_df(s1_path, ids):
    want = set(ids)
    out = []
    for ch in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        ch.columns = [str(c).strip().lstrip("\ufeff") for c in ch.columns]
        hit = ch[ch["entity_id"].isin(want)]
        if len(hit):
            out.append(hit[["entity_id", "business_name", "business_address", "country"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])


def pairs_with_truth(s1_df, s23_df, cand_df, truth):
    from .pairs import pairs_from_candidates
    return pairs_from_candidates(s1_df, s23_df, cand_df, truth)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-s1", type=int, default=C.TRAIN_SAMPLE_S1)
    ap.add_argument("--val-s1", type=int, default=C.VAL_SAMPLE_S1)
    ap.add_argument("--top-k", type=int, default=C.BLOCK_TOP_K)
    ap.add_argument("--df-cap", type=int, default=C.BLOCK_DF_CAP)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s23-head", type=int, default=None,
                    help="cap S23 rows per file (quick-test only; omit for full run)")
    a = ap.parse_args(args)
    t0 = time.time()
    print("TRAIN_V2 root=%s" % C.ROOT, flush=True)
    print("train files: %s" % C.TRAIN_FILES, flush=True)
    C.BLOCK_TOP_K = a.top_k
    train_ids, val_ids = sample_s1_ids(C.TRAIN_FILES["s1"], C.TRAIN_FILES["gt"], a.sample_s1, a.val_s1, a.seed)
    blk = build_index_streaming([C.TRAIN_FILES["s2"], C.TRAIN_FILES["s3"]], a.df_cap, max_rows=a.s23_head)
    blk.top_k = a.top_k
    # load sampled S1
    s1tr = load_s1_df(C.TRAIN_FILES["s1"], train_ids)
    s1va = load_s1_df(C.TRAIN_FILES["s1"], val_ids)
    print("querying candidates train/val...", flush=True)
    cand_tr = blk.query(s1tr)
    cand_va = blk.query(s1va)
    # load truth for sampled ids (streaming)
    def load_truth(ids):
        want = set(ids)
        d = {}
        with open(C.TRAIN_FILES["gt"], encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                s = row["source1_entity_id"]
                if s in want:
                    m = (row["matched_entity_ids"] or "").strip()
                    d[s] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
                    if len(d) == len(want):
                        pass
        for s in ids:
            d.setdefault(s, set())
        return d
    truth_tr, truth_va = load_truth(train_ids), load_truth(val_ids)
    from .pairs import blocking_recall
    rec_tr = blocking_recall(cand_tr, truth_tr)
    rec_va = blocking_recall(cand_va, truth_va)
    print("blocking recall train:", rec_tr, flush=True)
    print("blocking recall val:", rec_va, flush=True)
    # fetch S23 raw for candidate ids (union)
    need = set()
    for lst in list(cand_tr["candidate_entity_ids"]) + list(cand_va["candidate_entity_ids"]):
        if isinstance(lst, list):
            need.update(lst)
    print("fetching %d S23 records..." % len(need), flush=True)
    s23need = fetch_s23_for_ids([C.TRAIN_FILES["s2"], C.TRAIN_FILES["s3"]], need)
    print("fetched %d" % len(s23need), flush=True)
    pairs_tr = pairs_with_truth(s1tr, s23need, cand_tr, truth_tr)
    pairs_va = pairs_with_truth(s1va, s23need, cand_va, truth_va)
    print("pairs tr=%d va=%d pos=%.4f/%.4f" % (len(pairs_tr), len(pairs_va), pairs_tr["label"].mean() if len(pairs_tr) else 0, pairs_va["label"].mean() if len(pairs_va) else 0), flush=True)
    print("featurizing (light, vocab-free) with progress...", flush=True)
    def _feat_progress(df, tag):
        _t = time.time()
        _n = len(df)
        _parts = []
        _step = 500000
        for _i in range(0, _n, _step):
            _parts.append(light_features(df.iloc[_i:_i + _step]))
            _log_progress(min(_i + _step, _n), _n, _t, "featurize %s" % tag)
        import pandas as _pd
        return _pd.concat(_parts, ignore_index=True) if _parts else df.iloc[0:0]
    Xtr = _feat_progress(pairs_tr, "train")[LIGHT_COLS].to_numpy(dtype=float)
    ytr = pairs_tr["label"].to_numpy()
    Xva = _feat_progress(pairs_va, "val")[LIGHT_COLS].to_numpy(dtype=float)
    yva = pairs_va["label"].to_numpy()
    print("training LightGBM (n=%d, %d features)..." % (len(Xtr), Xtr.shape[1]), flush=True)
    clf = train_classifier(Xtr, ytr, Xva, yva)
    print("tuning threshold grid %s..." % C.THRESHOLD_GRID, flush=True)
    proba = clf.predict_proba(Xva)[:, 1]
    best, bf, hist = tune_threshold(pairs_va, yva, proba, truth_va)
    print("tuned threshold=%.2f val F0.5=%.4f" % (best, bf), flush=True)
    # refit on all sampled pairs
    import pandas as pd
    pall = pd.concat([pairs_tr, pairs_va], ignore_index=True)
    print("refitting on all %d pairs..." % len(pall), flush=True)
    Xall = _feat_progress(pall, "refit")[LIGHT_COLS].to_numpy(dtype=float)
    yall = pall["label"].to_numpy()
    clf_full = train_classifier(Xall, yall)
    # save with light-cols marker + blocker df_cap/top_k for inference
    save_artifacts(C.MODEL_DIR, clf_full, {"light_cols": LIGHT_COLS, "threshold": best,
                                           "df_cap": a.df_cap, "top_k": a.top_k}, best)
    elapsed = time.time() - t0
    print("saved to %s in %.1fs" % (C.MODEL_DIR, elapsed), flush=True)
    # append scoring summary to results.txt (repo root) for tracking
    try:
        from .evaluate import precision_recall_macro
        from .model import apply_veto_rules as _veto
        _keep = _veto(light_features(pairs_va), proba, best)
        _pred = {}
        for _s, _c, _k in zip(pairs_va["source1_entity_id"], pairs_va["candidate_id"], _keep):
            if _k:
                _pred.setdefault(_s, set()).add(_c)
        for _s in truth_va:
            _pred.setdefault(_s, set())
        _p, _r = precision_recall_macro(truth_va, _pred)
        with open(os.path.join(C.ROOT, "results.txt"), "a", encoding="utf-8") as _f:
            _f.write("=== train %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            _f.write("args: sample_s1=%d val_s1=%d top_k=%d df_cap=%d n_tok=%d seed=%d\n"
                     % (a.sample_s1, a.val_s1, a.top_k, a.df_cap, C.BLOCK_N_TOK, a.seed))
            _f.write("blocking recall train micro=%.4f macro=%.4f | val micro=%.4f macro=%.4f\n"
                     % (rec_tr[0], rec_tr[1], rec_va[0], rec_va[1]))
            _f.write("pairs tr=%d va=%d pos=%.4f/%.4f\n"
                     % (len(pairs_tr), len(pairs_va),
                        float(pairs_tr["label"].mean()) if len(pairs_tr) else 0,
                        float(pairs_va["label"].mean()) if len(pairs_va) else 0))
            _f.write("tuned threshold=%.2f val macro-F0.5=%.4f P=%.4f R=%.4f elapsed=%.1fs\n\n"
                     % (best, bf, _p, _r, elapsed))
    except Exception as _e:
        print("results.txt logging skipped: %s" % _e, flush=True)


if __name__ == "__main__":
    main()
