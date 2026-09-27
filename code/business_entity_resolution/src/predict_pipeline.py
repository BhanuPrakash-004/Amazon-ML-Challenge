"""V3 test prediction pipeline: chunked, streaming, checkpointed.

Per S1 chunk: normalize -> lexical candidates -> (optional) E5 -> union ->
cheap prune -> full features -> LightGBM -> selective rerank -> meta/blend ->
entity decision -> append outputs -> release memory. Runs official validator.
"""
import argparse
import csv
import json
import os
import time

from . import config as C
from . import hardware as HW
from . import checkpoint as CK
from .logging_utils import StageLog
from .utils import ensure_dir, count_data_rows


def _test_paths(data_root):
    for cand in (os.path.join(data_root, "test"), data_root):
        s1 = os.path.join(cand, "test_source1.tsv")
        if os.path.exists(s1):
            base = cand
            break
    else:
        base = os.path.join(data_root, "test")
    return {"s1": os.path.join(base, "test_source1.tsv"),
            "s2": os.path.join(base, "test_source2.tsv"),
            "s3": os.path.join(base, "test_source3.tsv")}


def run(args):
    t_all = time.time()
    HW.print_info()
    paths = _test_paths(args.data_root)
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError("missing %s: %s" % (k, p))
    import joblib
    import pandas as pd
    clf = joblib.load(os.path.join(args.model_root, "lightgbm_matcher.pkl"))
    with open(os.path.join(args.model_root, "decision_config.json"), encoding="utf-8") as f:
        dec = json.load(f)
    thr, margin = float(dec["threshold"]), float(dec.get("margin", 0.15))
    try:
        meta = joblib.load(os.path.join(args.model_root, "meta_model.pkl"))
    except Exception:
        meta = None
    print("[PREDICT] thr=%.3f margin=%.2f k=%d" % (thr, margin, args.candidate_k), flush=True)

    from .train_pipeline import _iter_rows, _s1_tuples
    from .blocking_exact import ExactBlocker
    from .blocking_rare import CompactRareIndex
    from .blocking_numeric import NumericBlocker
    from .blocking_char import CharBlocker
    from .candidate_union import union_for_s1, normalize_block_alias
    from .candidate_prune import prune, adaptive_prune_for_s1, candidate_stats, check_candidate_budget
    from .transliterate import transliterate_local

    st = StageLog("TEST-IDX")
    rare_n = CompactRareIndex("name", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_NAME).fit(
        _iter_rows([paths["s2"], paths["s3"]]))
    rare_a = CompactRareIndex("address", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_ADDR).fit(
        _iter_rows([paths["s2"], paths["s3"]]))
    num_b = NumericBlocker(C.NUM_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]]))
    ch_n = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]]), "name")
    ch_a = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]]), "address")
    ex_b = ExactBlocker().fit_stream(_iter_rows([paths["s2"], paths["s3"]]))
    st.close()

    try:
        from .reranker import Reranker, select_ambiguous
        rer = Reranker(device="cpu", batch=C.RERANK_BATCH, max_len=args.max_seq_length) if args.use_reranker else None
        if rer is not None and not rer.available():
            rer = None
    except Exception:
        rer = None
    from .decision import decide_for_s1
    from .features_all import features_all, ALL_COLS
    from .ensemble import blend
    from .train_v2 import fetch_s23_for_ids

    ensure_dir(args.output_root)
    mp = os.path.join(args.output_root, "matching_results.tsv")
    cp = os.path.join(args.output_root, "candidate_pairs.tsv")
    if not args.resume or not os.path.exists(mp):
        for p, h in ((mp, ["source1_entity_id", "matched_entity_ids"]),
                     (cp, ["source1_entity_id", "candidate_entity_ids"])):
            with open(p, "w", encoding="utf-8", newline="") as f:
                csv.writer(f, delimiter="\t", lineterminator="\n").writerow(h)
        done_ids = set()
    else:
        done_ids = set()
        with open(mp, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                done_ids.add(line.split("\t", 1)[0])
        print("[PREDICT] resume: %d S1 already done." % len(done_ids), flush=True)

    total_s1 = count_data_rows(paths["s1"])
    st = StageLog("PREDICT", total_s1)
    n_done = len(done_ids)
    n_match = 0
    chunk = args.chunk_size
    for ch in pd.read_csv(paths["s1"], sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        ch.columns = [str(c).strip().lstrip("﻿") for c in ch.columns]
        ch = ch[~ch["entity_id"].isin(done_ids)]
        if not len(ch):
            continue
        # candidates
        cand_map, ev_map = {}, {}
        for s1id, nm, ad, cc in _s1_tuples(ch):
            per = {}
            rn = [(c, r, s) for c, t, r, s in rare_n.query_one(nm, ad, cc)]
            if rn:
                per["rare_name"] = rn
            ra = [(c, r, s) for c, t, r, s in rare_a.query_one(nm, ad, cc)]
            if ra:
                per["rare_address"] = ra
            nb = [(c, r, s) for c, t, r, s in num_b.query_one(nm, ad, cc)]
            if nb:
                per["numeric"] = nb
            cn = [(c, r, s) for c, t, r, s in ch_n.query_one(nm, ad, cc, "name")]
            ca = [(c, r, s) for c, t, r, s in ch_a.query_one(nm, ad, cc, "address")]
            if cn or ca:
                per["char"] = (cn + ca)[:C.CHAR_TOP_K]
            t_nm = transliterate_local(nm)
            if t_nm and t_nm != nm:
                tr = [(c, r, s * 0.8) for c, t, r, s in rare_n.query_one(t_nm, ad, cc)]
                if tr:
                    per["transliterated"] = tr
            cids, ev, m = union_for_s1(s1id, {normalize_block_alias(k): v for k, v in per.items()})
            ex_ids, ex_ev = ex_b.query([(s1id, nm, ad, cc)])[s1id]
            for x in ex_ids:
                if x not in ev:
                    ev[x] = {}
                    cids.append(x)
                ev[x]["exact_name"] = max(ev[x].get("exact_name", 0), ex_ev.get("exact_name", 0))
                ev[x]["exact_address"] = max(ev[x].get("exact_address", 0), ex_ev.get("exact_address", 0))
            cand_map[s1id] = prune(cids, ev, m, keep_k=args.candidate_k)
            if getattr(args, "adaptive", True):
                cand_map[s1id] = adaptive_prune_for_s1(
                    cand_map[s1id], ev, m,
                    internal_k=min(len(cand_map[s1id]), getattr(args, "internal_k", C.INTERNAL_KEEP_K)),
                    final_max=getattr(args, "final_max", C.FINAL_MAX_PER_S1))
            ev_map[s1id] = ev
        cdf = pd.DataFrame([(k, v) for k, v in cand_map.items()],
                           columns=["source1_entity_id", "candidate_entity_ids"])
        need = set()
        for v in cand_map.values():
            need.update(v)
        s23need = fetch_s23_for_ids([paths["s2"], paths["s3"]], need)
        from .pairs import pairs_from_candidates
        pairs = pairs_from_candidates(ch, s23need, cdf, truth=None)
        if len(pairs):
            for s, c in zip(pairs["source1_entity_id"], pairs["candidate_id"]):
                e = ev_map.get(s, {}).get(c, {})
                for col in ["exact_name", "exact_address", "rare_name", "rare_address", "numeric", "char",
                            "transliterated", "e5_name", "e5_address"]:
                    pairs.loc[(pairs.source1_entity_id == s) & (pairs.candidate_id == c), col] = e.get(col, 0)
            feat = features_all(pairs)
            proba = clf.predict_proba(feat[ALL_COLS].to_numpy(dtype=float))[:, 1]
            final = blend(meta, None, proba) if meta is None else blend(meta, None, proba)
            # selective rerank on ambiguous
            if rer is not None:
                idx = select_ambiguous(pairs["source1_entity_id"].tolist(), final, top_n=args.rerank_top_n)
                if idx:
                    Lt = (pairs.iloc[idx]["name_norm_1"] + " " + pairs.iloc[idx]["addr_norm_1"]).tolist()
                    Rt = (pairs.iloc[idx]["name_norm_2"] + " " + pairs.iloc[idx]["addr_norm_2"]).tolist()
                    rs = rer.score(Lt, Rt)
                    w = dict(zip(idx, rs))
                    final = [0.7 * float(p) + 0.3 * float(w.get(i, p)) for i, p in enumerate(final)]
        else:
            final = []
        # entity decision + append
        per_s = {}
        per_c = {}
        if len(pairs):
            for s, c, p in zip(pairs["source1_entity_id"], pairs["candidate_id"], final):
                per_s.setdefault(s, []).append(float(p))
                per_c.setdefault(s, []).append(c)
        with open(mp, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            for s1id in ch["entity_id"].tolist():
                acc = decide_for_s1(per_c.get(s1id, []), per_s.get(s1id, []), ev_map.get(s1id), thr, margin)
                acc = sorted(set(acc), key=lambda x: (0 if x.startswith("S2-") else 1, x))
                w.writerow([s1id, ",".join(acc)])
                n_match += len(acc)
        with open(cp, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            for s1id in ch["entity_id"].tolist():
                lst = sorted(set(cand_map.get(s1id, [])), key=lambda x: (0 if x.startswith("S2-") else 1, x))
                w.writerow([s1id, ",".join(lst)])
        n_done += len(ch)
        st.update(n_done, extra="matches=%d" % n_match)
        del pairs, cdf, s23need
    st.close()
    # candidate stats (spec 17): recompute avg from written file, fail loudly if > max.
    try:
        import pandas as pd
        _cdf = pd.read_csv(cp, sep="\t", dtype=str, keep_default_na=False)
        _d = {}
        for r in _cdf.itertuples():
            m = (r.candidate_entity_ids or "").strip()
            _d[r.source1_entity_id] = [x for x in m.split(",") if x] if m else []
        _stats = candidate_stats(_d)
        with open(os.path.join(args.output_root, "candidate_stats.json"), "w", encoding="utf-8") as f:
            json.dump(_stats, f, indent=1)
        print("[CANDIDATES] avg=%.2f med=%.1f p95=%.0f max=%d coverage=%.3f" % (
            _stats["average_candidates"], _stats["median_candidates"],
            _stats["p95"], _stats["max"], _stats["s1_coverage"]), flush=True)
        check_candidate_budget(_stats, max_avg=getattr(args, "max_avg", C.FINAL_MAX_AVG),
                               strict=bool(getattr(args, "adaptive", True)))
    except ValueError:
        raise
    except Exception as e:
        print("[CANDIDATES] stats skipped: %s" % e, flush=True)
    # validator
    print("[VALIDATE] running official validator...", flush=True)
    import sys
    sys.path.insert(0, os.path.join(C.ROOT, "student_resource", "utils"))
    try:
        from validate_submission import validate
        errs, warns = validate(mp, cp, os.path.dirname(paths["s1"]), check_ids=False)
        for x in warns:
            print("WARNING: %s" % x, flush=True)
        if errs:
            for i, e in enumerate(errs, 1):
                print("ERROR %d: %s" % (i, e), flush=True)
            print("VALIDATOR: FAIL", flush=True)
        else:
            print("VALIDATOR: PASS", flush=True)
    except Exception as e:
        print("[VALIDATE] skipped: %s" % e, flush=True)
    print("TEST MATCHING TIME: %.1fs OUTPUT: %s %s" % (time.time() - t_all, mp, cp), flush=True)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--chunk-size", type=int, default=C.CHUNK_S1)
    ap.add_argument("--candidate-k", type=int, default=C.CANDIDATE_K)
    ap.add_argument("--adaptive", action="store_true", default=C.ADAPTIVE_ENABLED_DEFAULT)
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    ap.add_argument("--internal-k", type=int, default=C.INTERNAL_KEEP_K)
    ap.add_argument("--final-max", type=int, default=C.FINAL_MAX_PER_S1)
    ap.add_argument("--max-avg", type=float, default=C.FINAL_MAX_AVG)
    ap.add_argument("--rerank-top-n", type=int, default=C.RERANK_TOP_N)
    ap.add_argument("--max-seq-length", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--use-reranker", action="store_true", default=C.USE_RERANKER_DEFAULT)
    ap.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    return ap


def main(cmd=None):
    ap = build_parser()
    a = ap.parse_args(cmd)
    from . import config as C2
    a.data_root = a.data_root or os.path.join(C2.ROOT, "student_resource", "dataset")
    if not os.path.isdir(a.data_root):
        a.data_root = os.path.join(C2.ROOT, "dataset")
    a.model_root = a.model_root or C2.MODEL_DIR
    a.cache_root = a.cache_root or C2.CACHE_DIR
    a.output_root = a.output_root or C2.OUTPUT_DIR
    return run(a)


if __name__ == "__main__":
    main()
