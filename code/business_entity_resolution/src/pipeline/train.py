"""Training pipeline (Sec 2, 16-22, 28-29, 34 Phases 1-10).

Order: normalize+indexes -> deterministic blocking -> recall eval -> ANN ->
recall eval -> features -> LightGBM+hard negatives -> F0.5 thresholds ->
singleton gate -> family support. Logs runtime/recall/avg-candidates/F0.5
at every phase. Saves models/*.pkl + selected thresholds.
"""
import csv
import json
import os
import pickle
import time

import numpy as np

from .. import config as C
from ..io.loader import scan_source_batched, read_ground_truth, count_rows
from ..preprocessing.normalize import normalize_record
from ..blocking import exact_blocks as EB
from ..blocking import numeric_blocks as NB
from ..blocking import rare_blocks as RB
from ..blocking.candidate_union import merge_per_s1
from ..blocking.rerank import topk as prerank_topk
from ..embeddings.encoder import Encoder, semantic_text
from ..embeddings.faiss_index import FaissIndex
from ..features.semantic_features import row_vector, FEATURE_ORDER
from ..training.make_training_pairs import split_s1
from ..training.hard_negatives import mine
from ..training.train_lgbm import train as train_clf
from ..training.calibrate import fit_calibrator, apply as apply_cal
from ..training.tune_threshold import tune as tune_thr
from ..evaluation.f05 import macro_f05
from ..evaluation.candidate_recall import recall as rec_fn, count_stats


def log(stage, t0, extra=""):
    el = time.time() - t0
    print("[train] %-28s %6.1fs %s" % (stage, el, extra), flush=True)


def _rec_to_norm(row):
    return normalize_record(row.get("business_name", ""), row.get("business_address", ""),
                            row.get("country", ""))


def load_sampled_recs(s1_path, s23_paths, train_ids, val_ids, chunk=200000):
    want_tr, want_va = set(train_ids), set(val_ids)
    s1rec = {}
    for ch in scan_source_batched(s1_path, chunk):
        for _, r in ch.iterrows():
            e = str(r["entity_id"]).strip()
            if e in want_tr or e in want_va:
                s1rec[e] = _rec_to_norm(r)
                if len(s1rec) >= len(want_tr) + len(want_va):
                    pass
    return s1rec


def build_det_indexes(s23_paths, df_cap=None, chunk=200000, max_rows=None):
    """Stream S2+S3 once; collect recs+ids per country partition (open-set)."""
    recs, ids = [], []
    for p in s23_paths:
        n = 0
        for ch in scan_source_batched(p, chunk):
            for _, r in ch.iterrows():
                recs.append(_rec_to_norm(r))
                ids.append(str(r["entity_id"]).strip())
                n += 1
                if max_rows and n >= max_rows:
                    break
            if max_rows and n >= max_rows:
                break
    eb = EB.build(recs, ids)
    nb = NB.build(recs, ids)
    rb = RB.build(recs, ids, df_cap or C.RARE_DF_CAP)
    trec = dict(zip(ids, recs))
    return {"eb": eb, "nb": nb, "rb": rb, "trec": trec, "ids": ids, "recs": recs}


def det_query(s1rec, idx):
    out = {}
    eb_h = EB.query_one(s1rec, idx["eb"])
    nb_h = NB.query_one(s1rec, idx["nb"])
    rb_h = RB.query_one(s1rec, idx["rb"])
    for src in (eb_h, nb_h, rb_h):
        for tid, m in src.items():
            d = out.get(tid)
            if d is None:
                out[tid] = {"blocks": set(m.get("blocks", set())),
                            **{k: v for k, v in m.items() if k != "blocks"}}
            else:
                d["blocks"] |= m.get("blocks", set())
                for k, v in m.items():
                    if k != "blocks":
                        d[k] = v
    for tid, d in out.items():
        d["block_count"] = len(d["blocks"])
    return out


def run(train_dir=None, model_dir=None, K=None, s1_sample=None, no_ann=False,
        max_rows=None):
    t_all = time.time()
    train_dir = train_dir or C.TRAIN_DIR
    model_dir = model_dir or C.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    K = K or C.K
    s1p = os.path.join(train_dir, "train_source1.tsv")
    s2p = os.path.join(train_dir, "train_source2.tsv")
    s3p = os.path.join(train_dir, "train_source3.tsv")
    gtp = os.path.join(train_dir, "train_ground_truth.tsv")

    # Phase 1: sample S1 ids + truth (stratified by S1 entity)
    t0 = time.time()
    truth_all = read_ground_truth(gtp)
    s1c = {}
    for ch in scan_source_batched(s1p):
        for _, r in ch.iterrows():
            from ..preprocessing.normalize import normalize_country
            s1c[str(r["entity_id"]).strip()] = normalize_country(r.get("country", ""))
    n_tr = s1_sample or C.TRAIN_S1_SAMPLE
    tr_ids, va_ids = split_s1(s1c, truth_all, n_train=n_tr, n_val=C.VAL_S1_SAMPLE)
    log("phase1 sample+truth", t0, "tr=%d va=%d" % (len(tr_ids), len(va_ids)))

    # Phase 1b: normalize sampled S1 + build deterministic indexes
    t0 = time.time()
    s1rec = {}
    want = set(tr_ids) | set(va_ids)
    for ch in scan_source_batched(s1p):
        for _, r in ch.iterrows():
            e = str(r["entity_id"]).strip()
            if e in want:
                s1rec[e] = _rec_to_norm(r)
    idx = build_det_indexes([s2p, s3p], max_rows=max_rows)
    # Sec 5/7: fill per-record rare-token fields from global frequencies
    from ..preprocessing.rarity import attach_rare
    attach_rare(idx["recs"], idx["rb"]["dfn"], idx["rb"]["dfa"],
                C.RARE_DF_CAP, C.RARE_MAX_QUERY_TOKENS)
    log("phase1 normalize+det-index", t0, "targets=%d" % len(idx["ids"]))

    # Phase 2/3: deterministic candidates + recall
    t0 = time.time()
    det_cands = {s: det_query(s1rec[s], idx) for s in s1rec}
    # prerank to raw pool 150 then we keep K sweep for reporting
    from ..blocking.rerank import topk as _topk
    cand_lists = {}
    for s, m in det_cands.items():
        cand_lists[s] = [t for t, _ in _topk(s1rec[s], m, idx["trec"], K=150)]
    r = rec_fn({s: truth_all.get(s, set()) for s in s1rec},
               {s: set(v) for s, v in cand_lists.items()})
    log("phase2/3 det-blocking recall", t0, "micro=%.4f macro=%.4f %s"
        % (r["micro"], r["macro"], count_stats(cand_lists)))

    # Phase 4/5: ANN (potion-multilingual-128M) union, S2 top12 + S3 top12.
    # Sec 11 lanes: S2 encoded/searched on GPU0, S3 on GPU1 (device selection
    # per lane; static model2vec backend is CPU-shared by design, ST uses GPUs).
    ann_info = {}
    if not no_ann:
        try:
            t0 = time.time()
            from ..blocking.ann_retrieval import lane_device
            enc_q = Encoder(device=lane_device(0) if _has_cuda() else "cpu")
            s1_texts = [semantic_text(s1rec[s]) for s in s1rec]
            s1ordered = list(s1rec.keys())
            V1 = enc_q.encode(s1_texts)
            # split targets by source for GPU0/GPU1 lanes
            s2_idx = [i for i, t in enumerate(idx["ids"]) if t.startswith("S2-")]
            s3_idx = [i for i, t in enumerate(idx["ids"]) if t.startswith("S3-")]
            T2 = [semantic_text(idx["recs"][i]) for i in s2_idx]
            T3 = [semantic_text(idx["recs"][i]) for i in s3_idx]
            enc2 = Encoder(device=lane_device(0))
            enc3 = Encoder(device=lane_device(1))
            V2 = enc2.encode(T2) if T2 else np.zeros((0, V1.shape[1]), np.float32)
            V3 = enc3.encode(T3) if T3 else np.zeros((0, V1.shape[1]), np.float32)
            I2 = FaissIndex(V1.shape[1]); I2.add(V2)
            I3 = FaissIndex(V1.shape[1]); I3.add(V3)
            D2, P2 = I2.search(V1, C.ANN_TOPK_S2)
            D3, P3 = I3.search(V1, C.ANN_TOPK_S3)
            s2ids = [idx["ids"][i] for i in s2_idx]
            s3ids = [idx["ids"][i] for i in s3_idx]
            for r_i, s in enumerate(s1ordered):
                m = det_cands[s]
                m = merge_per_s1(s, m,
                                 [(int(p), float(d)) for p, d in zip(P2[r_i], D2[r_i])],
                                 [(int(p), float(d)) for p, d in zip(P3[r_i], D3[r_i])],
                                 s2ids, s3ids)
                det_cands[s] = m
                cand_lists[s] = [t for t, _ in _topk(s1rec[s], m, idx["trec"], K=150)]
            r2 = rec_fn({s: truth_all.get(s, set()) for s in s1rec},
                        {s: set(v) for s, v in cand_lists.items()})
            log("phase4/5 ANN union recall", t0, "micro=%.4f macro=%.4f backend=%s"
                % (r2["micro"], r2["macro"], enc_q.backend))
        except Exception as e:
            log("phase4/5 ANN skipped", t0, "err=%s" % str(e)[:200])

    # K sweep report (Sec 28)
    for kk in C.RECALL_K_GRID:
        tmp = {}
        for s, m in det_cands.items():
            tmp[s] = set(t for t, _ in prerank_topk(s1rec[s], m, idx["trec"], K=kk))
        rr = rec_fn({s: truth_all.get(s, set()) for s in s1rec}, tmp)
        print("[train] recall@K=%d micro=%.4f macro=%.4f" % (kk, rr["micro"], rr["macro"]),
              flush=True)

    # Phase 6/7: features + hard negatives + LightGBM
    t0 = time.time()
    fam_counts = {}
    Xtr, ytr, Xva, yva = [], [], [], []
    v_tr, v_va = [], []  # (s1,[tids]) + prob placeholders
    tr_set = set(tr_ids)
    for s in tr_ids + va_ids:
        if s not in s1rec:
            continue
        m = det_cands.get(s, {})
        top = prerank_topk(s1rec[s], m, idx["trec"], K=K)
        tids = [t for t, _ in top]
        pos = set(truth_all.get(s, set())) & set(idx["trec"])
        # ensure positives present even if missed by blocking (recall accounting)
        for t in pos:
            if t not in tids:
                tids.append(t)
        is_tr = s in tr_set
        negs = mine(s1rec[s], {t: m.get(t, {}) for t in tids}, idx["trec"],
                    pos, neg_per_pos=C.SPEC_NEG_PER_POS)
        keep = [t for t in tids if t in pos] + [t for t in negs if t in idx["trec"]]
        for t in keep:
            v = row_vector(s1rec[s], idx["trec"][t], t, m.get(t, {}), 1)
            (Xtr if is_tr else Xva).append(v)
            (ytr if is_tr else yva).append(1 if t in pos else 0)
        (v_tr if is_tr else v_va).append((s, [t for t in keep if t in idx["trec"]]))
    Xtr = np.asarray(Xtr, np.float32)
    ytr = np.asarray(ytr)
    Xva = np.asarray(Xva, np.float32)
    yva = np.asarray(yva)
    clf = train_clf(Xtr, ytr, Xva, yva)
    log("phase6/7 features+lgbm", t0, "pairs tr=%d va=%d" % (len(ytr), len(yva)))

    # Phase 8: calibration + thresholds (val only)
    t0 = time.time()
    from ..inference.score_candidates import score_pairs
    # rebuild val pairs in order for tuning
    vpairs, vitems = [], []
    for s, tids in v_va:
        for t in tids:
            vpairs.append((s, t, s1rec[s], idx["trec"][t], det_cands.get(s, {}).get(t, {})))
        vitems.append((s, tids))
    pva = score_pairs(vpairs, clf) if vpairs else np.array([])
    cal = fit_calibrator(yva, pva) if len(pva) else ("none", None)
    pva_c = apply_cal(cal, pva) if len(pva) else pva
    # tune on raw and calibrated; keep winner (validated choice, Sec 21)
    vtruth = {s: truth_all.get(s, set()) for s, _ in vitems}
    res_raw = tune_thr(vitems, _chunk_probs(vitems, pva), vtruth)
    res_cal = tune_thr(vitems, _chunk_probs(vitems, pva_c), vtruth)
    best = res_cal if res_cal["f05"] >= res_raw["f05"] else res_raw
    use_cal = best is res_cal
    log("phase8 calibrate+tune", t0, "f05=%.4f thr=%s cal=%s"
        % (best["f05"], best["threshold"], use_cal))

    # Phase 9: singleton gate tuning (Sec 24). Grid over singleton_score_threshold
    # using top score + margin + name/address/numeric evidence via gate().
    t0 = time.time()
    from ..inference.singleton_gate import gate as sing_gate
    from ..evaluation.f05 import macro_f05 as _mf05
    best_sing, best_sf = max(best["threshold"], 0.6), best["f05"]
    pva_best = pva_c if use_cal else pva
    for cand in [round(x, 2) for x in
                 [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]]:
        pred = {}
        for (s, tids), ps in zip(vitems, _chunk_probs(vitems, pva_best)):
            keep = {t for t, p in zip(tids, ps)
                    if p >= (best["s2_threshold"] if t.startswith("S2-")
                             else best["s3_threshold"])}
            if keep:
                keep &= sing_gate(tids, np.asarray(ps), s1rec.get(s),
                                  {t: idx["trec"][t] for t in tids if t in idx["trec"]},
                                  thr=best["threshold"], sing_thr=cand)
            pred[s] = keep
        f = _mf05(vtruth, pred)
        if f > best_sf:
            best_sf, best_sing = f, cand
    log("phase9 singleton gate", t0, "sing_thr=%.2f f05=%.4f" % (best_sing, best_sf))

    with open(os.path.join(model_dir, "lgbm.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, "selected_pipeline.json"), "w") as f:
        json.dump({"K": K, "threshold": best["threshold"],
                   "s2_threshold": best["s2_threshold"], "s3_threshold": best["s3_threshold"],
                   "singleton_threshold": best_sing,
                   "calibrated": bool(use_cal), "cal": cal[0],
                   "features": FEATURE_ORDER, "val_f05": best_sf}, f, indent=2)
    with open(os.path.join(model_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(cal, f)
    print("[train] DONE val_f05=%.4f total=%.1fs" % (best_sf, time.time() - t_all),
          flush=True)
    return {"threshold": best["threshold"], "s2_threshold": best["s2_threshold"],
            "s3_threshold": best["s3_threshold"],
            "singleton_threshold": best_sing, "f05": best_sf,
            "calibrated": bool(use_cal)}


def _chunk_probs(vitems, pva):
    out, i = [], 0
    for _, tids in vitems:
        out.append(np.asarray(pva[i:i + len(tids)]))
        i += len(tids)
    return out


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
