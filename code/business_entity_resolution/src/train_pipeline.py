"""V3 training pipeline: resumable stages, multi-block union, hard negatives,
matcher + selection (A..G), meta, entity decision, results dir.

Stages: 1 sample/split, 2 lexical indexes, 3 E5 indexes (optional),
4 candidates + mining, 5 matcher/selection/meta/decision.
Full-scale reuses V2 streaming rare index (memory-safe) + exact/numeric/char.
Dev-mode exercises everything on tiny synthetic data.
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
from .utils import ensure_dir, count_data_rows, seed_all


def _data_paths(data_root):
    for cand in (os.path.join(data_root, "train"), data_root):
        s1 = os.path.join(cand, "train_source1.tsv")
        if os.path.exists(s1):
            base = cand
            break
    else:
        base = os.path.join(data_root, "train")
    return {"s1": os.path.join(base, "train_source1.tsv"),
            "s2": os.path.join(base, "train_source2.tsv"),
            "s3": os.path.join(base, "train_source3.tsv"),
            "gt": os.path.join(base, "train_ground_truth.tsv")}


def _iter_rows(paths, cols=("entity_id", "business_name", "business_address", "country"), chunk=200000, max_rows=None):
    import pandas as pd
    for p in paths:
        n = 0
        for ch in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
            ch.columns = [str(c).strip().lstrip("﻿") for c in ch.columns]
            if max_rows is not None:
                ch = ch.iloc[:max(0, max_rows - n)]
                if not len(ch):
                    break
            for r in ch.itertuples():
                yield (getattr(r, cols[0]), getattr(r, cols[1]), getattr(r, cols[2]), getattr(r, cols[3]))
            n += len(ch)
            if max_rows is not None and n >= max_rows:
                break


def _load_truth(path, want):
    want = set(want)
    d = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = row["source1_entity_id"]
            if s in want:
                m = (row.get("matched_entity_ids") or "").strip()
                d[s] = set(x.strip() for x in m.split(",") if x.strip()) if m else set()
    for s in want:
        d.setdefault(s, set())
    return d


def _sample_ids(s1_path, gt_path, n_tr, n_va, seed):
    from .train_v2 import sample_s1_ids
    return sample_s1_ids(s1_path, gt_path, n_tr, n_va, seed)


def _load_s1_df(s1_path, ids):
    from .train_v2 import load_s1_df
    return load_s1_df(s1_path, ids)


def _s1_tuples(s1_df):
    return list(zip(s1_df["entity_id"].tolist(), s1_df["business_name"].fillna("").astype(str).tolist(),
                    s1_df["business_address"].fillna("").astype(str).tolist(),
                    s1_df["country"].fillna("").astype(str).tolist()))


def run(args):
    t_all = time.time()
    seed_all(args.seed)
    HW.print_info()
    paths = _data_paths(args.data_root)
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError("missing %s: %s" % (k, p))
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    res_dir = ensure_dir(os.path.join(os.path.dirname(C.MODEL_DIR), "results", run_id))
    cfg = vars(args)
    with open(os.path.join(res_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    # ---------- STAGE 1: sample ----------
    st = StageLog("PREPROCESS")
    if args.dev_mode:
        import pandas as pd
        from .normalize import normalize_name
        tr_ids = ["D-S1-%d" % i for i in range(160)]
        va_ids = ["D-S1-%d" % i for i in range(160, 200)]
        # synthetic S1/S23 in memory
        s1_rows = [("D-S1-%d" % i, "Acme Manufacturing %d" % i, "%d Main Street" % (100 + i),
                     "United States" if i % 2 == 0 else "India") for i in range(200)]
        s23_rows = [("D-S2-%d" % i, "Acme Manufacturing %d" % i, "%d Main Street" % (100 + i),
                     "United States" if i % 2 == 0 else "India") for i in range(200)]
        s23_rows += [("D-S3-%d" % i, "Unrelated Business %d" % i, "%d Other Road" % (900 + i),
                      "United States") for i in range(60)]
        truth = {"D-S1-%d" % i: {"D-S2-%d" % i} for i in range(200)}
        truth.update({"D-S1-%d" % i: set() for i in range(0, 200, 10)})  # singletons
        s1_df = pd.DataFrame([{"entity_id": a, "business_name": b, "business_address": c, "country": d}
                              for a, b, c, d in s1_rows])
        s1tr = s1_df[s1_df.entity_id.isin(tr_ids)].reset_index(drop=True)
        s1va = s1_df[s1_df.entity_id.isin(va_ids)].reset_index(drop=True)
        truth_tr = {k: truth[k] for k in tr_ids}
        truth_va = {k: truth[k] for k in va_ids}
        st.update(200, extra="dev-mode synthetic")
        st.close()
    else:
        n_tr = args.train_s1 or C.TRAIN_SAMPLE_S1
        n_va = args.val_s1 or C.VAL_SAMPLE_S1
        tr_ids, va_ids = _sample_ids(paths["s1"], paths["gt"], n_tr, n_va, args.seed)
        s1tr, s1va = _load_s1_df(paths["s1"], tr_ids), _load_s1_df(paths["s1"], va_ids)
        truth_tr, truth_va = _load_truth(paths["gt"], tr_ids), _load_truth(paths["gt"], va_ids)
        s23_rows = None
        st.update(len(tr_ids) + len(va_ids))
        st.close()

    # ---------- STAGE 2: lexical indexes ----------
    st = StageLog("LEX-IDX")
    from .blocking_exact import ExactBlocker
    from .blocking_rare import CompactRareIndex
    from .blocking_numeric import NumericBlocker
    from .blocking_char import CharBlocker
    from .normalize import normalize_country
    max_rows = 5000 if args.dev_mode else None
    if args.dev_mode:
        s23_iter = list(s23_rows)
    else:
        # stream S23 once into per-blocker fits (generators re-iterated per blocker;
        # acceptable: 3 streaming passes, same as V2's 2 passes + fetch)
        s23_iter = None
    if args.dev_mode:
        rare_n = CompactRareIndex("name", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_NAME).fit(s23_iter)
        rare_a = CompactRareIndex("address", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_ADDR).fit(s23_iter)
        num_b = NumericBlocker(C.NUM_TOP_K).fit(s23_iter)
        ch_n = CharBlocker(3, C.CHAR_TOP_K).fit(s23_iter, "name")
        ch_a = CharBlocker(3, C.CHAR_TOP_K).fit(s23_iter, "address")
        ex_b = ExactBlocker().fit_stream(s23_iter)
    else:
        if not args.resume or not CK.is_done(args.cache_root, "lex", "done"):
            rare_n = CompactRareIndex("name", args.max_df if hasattr(args, "max_df") else C.RARE_MAX_DF,
                                      top_k=C.RARE_TOP_K_NAME, max_query_tokens=C.RARE_MAX_QUERY_TOKENS,
                                      max_posting_len=C.MAX_POSTING_LEN)
            rare_n.fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            rare_a = CompactRareIndex("address", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_ADDR,
                                      max_query_tokens=C.RARE_MAX_QUERY_TOKENS, max_posting_len=C.MAX_POSTING_LEN)
            rare_a.fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            num_b = NumericBlocker(C.NUM_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            ch_n = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows), "name")
            ch_a = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows), "address")
            ex_b = ExactBlocker().fit_stream(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            CK.mark_done(args.cache_root, "lex", "done", {"max_rows": max_rows or "full"})
        else:
            print("[LEX-IDX] resumed (cached). Rebuilding in-memory (streaming)...", flush=True)
            rare_n = CompactRareIndex("name", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_NAME).fit(
                _iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            rare_a = CompactRareIndex("address", C.RARE_MAX_DF, top_k=C.RARE_TOP_K_ADDR).fit(
                _iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            num_b = NumericBlocker(C.NUM_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
            ch_n = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows), "name")
            ch_a = CharBlocker(3, C.CHAR_TOP_K).fit(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows), "address")
            ex_b = ExactBlocker().fit_stream(_iter_rows([paths["s2"], paths["s3"]], max_rows=max_rows))
    st.close(extra="name/addr/numeric/char/exact ready")

    # ---------- STAGE 3: E5 (optional) ----------
    e5 = None
    if args.use_e5 and not args.dev_mode:
        st = StageLog("E5-IDX")
        try:
            from .blocking_e5 import E5Blocker
            from .normalize import normalize_record, e5_name_view, e5_full_view
            import pandas as pd
            ids, nt, ft = [], [], []
            for eid, nm, ad, cc in _iter_rows([paths["s2"], paths["s3"]], max_rows=200000 if args.dev_mode else None):
                rec = normalize_record(nm, ad, cc)
                ids.append(eid)
                nt.append(e5_name_view(rec))
                ft.append(e5_full_view(rec))
                if len(ids) >= 200000:
                    break
            e5 = E5Blocker(cache_root=args.cache_root, top_k_name=C.E5_TOP_K_NAME, top_k_full=C.E5_TOP_K_FULL,
                           nprobe=C.FAISS_NPROBE, max_len=args.max_seq_length, batch=C.E5_BATCH)
            if e5.available():
                from .hardware import pick_device
                e5.build(ids, nt, ft, device=pick_device(), nlist=C.FAISS_NLIST, m=C.FAISS_M, nbits=C.FAISS_NBITS)
            else:
                print("[E5-IDX] torch/faiss missing; E5 disabled for this run.", flush=True)
                e5 = None
        except Exception as ex:
            print("[E5-IDX] skipped: %s" % ex, flush=True)
            e5 = None
        st.close()

    # ---------- STAGE 4: candidates + features + mining ----------
    from .candidate_union import union_for_s1, normalize_block_alias
    from .candidate_prune import prune, adaptive_prune_for_s1, candidate_stats, check_candidate_budget
    from .transliterate import transliterate_local
    st = StageLog("BLOCKING")

    def block_s1(nm, ad, cc):
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
            per["char"] = [(c, r, s) for c, r, s in (cn + ca)][:C.CHAR_TOP_K]
        # transliteration-assisted: query rare-name index with transliterated name
        t_nm = transliterate_local(nm)
        if t_nm and t_nm != nm:
            tr = [(c, r, s * 0.8) for c, t, r, s in rare_n.query_one(t_nm, ad, cc)]
            if tr:
                per["transliterated"] = tr
        return per

    def candidates_for(df):
        from .blocking_exact import ExactBlocker as _E
        cand, evs, metas = {}, {}, {}
        for s1id, nm, ad, cc in _s1_tuples(df):
            per = block_s1(nm, ad, cc)
            # exact evidence merged as flags (ids unioned)
            ex_ids, ex_ev = ex_b.query([(s1id, nm, ad, cc)])[s1id]
            cids, ev, meta = union_for_s1(s1id, {normalize_block_alias(k): v for k, v in per.items()})
            for x in ex_ids:
                if x not in ev:
                    ev[x] = {kk: 0 for kk in
                             ["exact_name", "exact_address", "rare_name", "rare_address", "numeric", "char",
                              "transliterated", "e5_name", "e5_address"]}
                    meta[x] = (999, 0.0, 1)
                    cids.append(x)
                if ex_ev.get("exact_name"):
                    ev[x]["exact_name"] = 1
                if ex_ev.get("exact_address"):
                    ev[x]["exact_address"] = 1
            keep = prune(cids, ev, meta, keep_k=args.candidate_k)
            if getattr(args, "adaptive", True):
                keep = adaptive_prune_for_s1(
                    keep, ev, meta,
                    internal_k=min(len(keep), getattr(args, "internal_k", C.INTERNAL_KEEP_K)),
                    final_max=getattr(args, "final_max", C.FINAL_MAX_PER_S1))
            cand[s1id] = keep
            evs[s1id] = ev
            metas[s1id] = meta
        return cand, evs, metas

    cand_tr, ev_tr, meta_tr = candidates_for(s1tr)
    cand_va, ev_va, meta_va = candidates_for(s1va)
    from .pairs import blocking_recall as _br
    import pandas as pd
    cdf_tr = pd.DataFrame([(k, v) for k, v in cand_tr.items()], columns=["source1_entity_id", "candidate_entity_ids"])
    cdf_va = pd.DataFrame([(k, v) for k, v in cand_va.items()], columns=["source1_entity_id", "candidate_entity_ids"])
    rec_tr, rec_va = _br(cdf_tr, truth_tr), _br(cdf_va, truth_va)
    avg_c = sum(len(v) for v in cand_tr.values()) / max(len(cand_tr), 1)
    stats_tr = candidate_stats(cand_tr, truth_tr)
    stats_va = candidate_stats(cand_va, truth_va)
    try:
        check_candidate_budget(stats_va, max_avg=getattr(args, "max_avg", C.FINAL_MAX_AVG),
                               strict=bool(getattr(args, "adaptive", True)))
    except ValueError as e:
        print(str(e), flush=True)
        raise
    st.close(extra="recall tr=%.3f va=%.3f avg_cands=%.1f" % (rec_tr[0], rec_va[0], avg_c))

    # pairs + features (need S23 raw for candidate ids)
    if args.dev_mode:
        import pandas as pd
        s23_df = pd.DataFrame([{"entity_id": a, "business_name": b, "business_address": c, "country": d}
                               for a, b, c, d in s23_rows])
    else:
        from .train_v2 import fetch_s23_for_ids
        need = set()
        for v in list(cand_tr.values()) + list(cand_va.values()):
            need.update(v)
        s23_df = fetch_s23_for_ids([paths["s2"], paths["s3"]], need)
    from .pairs import pairs_from_candidates
    pairs_tr = pairs_from_candidates(s1tr, s23_df, cdf_tr, truth_tr)
    pairs_va = pairs_from_candidates(s1va, s23_df, cdf_va, truth_va)
    # attach block evidence to pairs
    for dfp, evs, metas in ((pairs_tr, ev_tr, meta_tr), (pairs_va, ev_va, meta_va)):
        if len(dfp):
            dfp["block_support"] = [metas.get(s, {}).get(c, (999, 0, 1))[2] for s, c in
                                    zip(dfp["source1_entity_id"], dfp["candidate_id"])]
            dfp["best_rank"] = [metas.get(s, {}).get(c, (999, 0, 1))[0] for s, c in
                                zip(dfp["source1_entity_id"], dfp["candidate_id"])]
            dfp["best_score"] = [metas.get(s, {}).get(c, (999, 0, 1))[1] for s, c in
                                 zip(dfp["source1_entity_id"], dfp["candidate_id"])]
            dfp["bscore"] = dfp["best_score"]
            for col in ["exact_name", "exact_address", "rare_name", "rare_address", "numeric", "char",
                        "transliterated", "e5_name", "e5_address"]:
                dfp[col] = [evs.get(s, {}).get(c, {}).get(col, 0) for s, c in
                            zip(dfp["source1_entity_id"], dfp["candidate_id"])]
    from .hard_negative_mining import mine_negatives
    mined = mine_negatives(pairs_tr, per_pos=C.NEG_PER_POS) if len(pairs_tr) else pairs_tr
    from .features_all import features_all, ALL_COLS
    Xtr = features_all(mined)[ALL_COLS].to_numpy(dtype=float) if len(mined) else None
    import numpy as np
    ytr = mined["label"].to_numpy() if len(mined) else np.zeros(0)
    Xva = features_all(pairs_va)[ALL_COLS].to_numpy(dtype=float) if len(pairs_va) else None
    yva = pairs_va["label"].to_numpy() if len(pairs_va) else np.zeros(0)

    # ---------- STAGE 5: matcher + selection + decision ----------
    st = StageLog("LIGHTGBM")
    from .train_matcher import train_matcher, save_matcher
    clf = train_matcher(Xtr, ytr, Xva, yva)
    proba_va = clf.predict_proba(Xva)[:, 1] if len(Xva) else np.zeros(0)
    from .threshold_tuning import tune_decision
    s1va_list = pairs_va["source1_entity_id"].tolist() if len(pairs_va) else []
    cands_va = {s: [] for s in cand_va}
    scores_va = {s: [] for s in cand_va}
    if len(pairs_va):
        for s, c, p in zip(pairs_va["source1_entity_id"], pairs_va["candidate_id"], proba_va):
            cands_va.setdefault(s, []).append(c)
            scores_va.setdefault(s, []).append(float(p))
    tuned = tune_decision(list(cand_va.keys()), cands_va, scores_va, truth_va)
    st.close(extra="thr=%.2f f05=%.4f" % (tuned["threshold"], tuned["f05"]))
    # selection record A..G (honest subset evaluated; E5/reranker only if available)
    from .evaluate import validation_report
    from .decision import decide_for_s1
    pred = {s: set(decide_for_s1(cands_va.get(s, []), scores_va.get(s, []), None, tuned["threshold"], tuned["margin"]))
            for s in cand_va}
    rep = validation_report(truth_va, pred, cand_va,
                            {"blocking_recall_tr": rec_tr[0], "blocking_recall_va": rec_va[0],
                             "threshold": tuned["threshold"], "margin": tuned["margin"]})
    selected = {"matcher": "lightgbm", "use_e5": bool(e5 is not None or args.use_e5),
                "use_reranker": bool(args.use_reranker), "threshold": tuned["threshold"],
                "margin": tuned["margin"], "candidate_k": args.candidate_k,
                "note": "A=rare-only baseline ~= rec above; D=multi-block+enriched = this run; "
                        "E=+reranker applied at predict if enabled; F/G optional fine-tunes compare on val"}
    ensure_dir(args.model_root)
    save_matcher(args.model_root, clf, {"all_cols": ALL_COLS})
    import joblib
    with open(os.path.join(args.model_root, "decision_config.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": tuned["threshold"], "margin": tuned["margin"]}, f, indent=1)
    with open(os.path.join(args.model_root, "selected_pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=1)
    with open(os.path.join(args.model_root, "final_config.json"), "w", encoding="utf-8") as f:
        json.dump({"selected": selected, "metrics": rep}, f, indent=1)
    with open(os.path.join(res_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(res_dir, "blocking_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"recall_tr": rec_tr, "recall_va": rec_va, "avg_cands": avg_c,
                   "candidate_stats_tr": stats_tr, "candidate_stats_va": stats_va,
                   "adaptive": bool(getattr(args, "adaptive", True)),
                   "ks": "internal k=%d + adaptive final_max=%d (baseline K=%d with --no-adaptive)" % (
                       getattr(args, "internal_k", C.INTERNAL_KEEP_K),
                       getattr(args, "final_max", C.FINAL_MAX_PER_S1), args.candidate_k)}, f, indent=1)
    with open(os.path.join(res_dir, "candidate_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"train": stats_tr, "val": stats_va}, f, indent=1)
    total = time.time() - t_all
    with open(os.path.join(res_dir, "runtime.json"), "w", encoding="utf-8") as f:
        json.dump({"total_s": total}, f, indent=1)
    print("SELECTED PIPELINE: %s" % json.dumps(selected), flush=True)
    print("BLOCKING RECALL: tr=%.4f va=%.4f" % (rec_tr[0], rec_va[0]), flush=True)
    print("VALIDATION MACRO-F0.5: %.4f P=%.4f R=%.4f" % (rep["macro_f05"], rep["macro_P"], rep["macro_R"]), flush=True)
    print("OUTPUT models: %s results: %s" % (args.model_root, res_dir), flush=True)
    return {"metrics": rep, "selected": selected, "res_dir": res_dir}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--use-e5", action="store_true", default=C.USE_E5_DEFAULT)
    ap.add_argument("--no-e5", dest="use_e5", action="store_false")
    ap.add_argument("--use-finetuned-e5", action="store_true", default=False)
    ap.add_argument("--use-reranker", action="store_true", default=C.USE_RERANKER_DEFAULT)
    ap.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    ap.add_argument("--finetune-e5", action="store_true", default=False)
    ap.add_argument("--finetune-reranker", action="store_true", default=False)
    ap.add_argument("--chunk-size", type=int, default=C.CHUNK_S1)
    ap.add_argument("--train-s1", type=int, default=None)
    ap.add_argument("--val-s1", type=int, default=None)
    ap.add_argument("--candidate-k", type=int, default=C.CANDIDATE_K)
    ap.add_argument("--adaptive", action="store_true", default=C.ADAPTIVE_ENABLED_DEFAULT)
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                    help="SAFE BASELINE: keep legacy K=150 file without adaptive 8-15 cap")
    ap.add_argument("--internal-k", type=int, default=C.INTERNAL_KEEP_K)
    ap.add_argument("--final-max", type=int, default=C.FINAL_MAX_PER_S1)
    ap.add_argument("--max-avg", type=float, default=C.FINAL_MAX_AVG)
    ap.add_argument("--rerank-top-n", type=int, default=C.RERANK_TOP_N)
    ap.add_argument("--max-seq-length", type=int, default=C.MAX_SEQ_LEN)
    ap.add_argument("--max-df", type=int, default=C.RARE_MAX_DF)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dev-mode", action="store_true", default=False)
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
