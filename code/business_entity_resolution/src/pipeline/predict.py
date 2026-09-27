"""Test inference pipeline (Sec 2-3, 11, 15, 23-30).

Full test: load+normalize (0-30min) -> ANN (20-60min, GPU0=S2/GPU1=S3,
multiprocessed, CPU fallback) -> test candidates+scoring (80-160min, chunked
100-250k S1) -> post (150-175min) -> validate (175-180min).
Writes output/matching_results.tsv + output/candidate_pairs.tsv.
"""
import json
import os
import pickle
import time

import numpy as np

from .. import config as C
from ..io.loader import scan_source_batched, s1_id_order
from ..io.writer import write_matching, write_candidates
from ..preprocessing.normalize import normalize_record
from ..blocking import exact_blocks as EB
from ..blocking import numeric_blocks as NB
from ..blocking import rare_blocks as RB
from ..blocking.candidate_union import merge_per_s1
from ..blocking.rerank import topk as prerank_topk
from ..embeddings.encoder import Encoder, semantic_text
from ..embeddings.faiss_index import FaissIndex
from ..inference.score_candidates import score_pairs
from ..inference.final_decision import decide
from ..inference.family_support import build_families
from ..training.calibrate import apply as apply_cal
from ..evaluation.candidate_recall import count_stats


def log(stage, t0, extra=""):
    print("[predict] %-28s %7.1fs %s" % (stage, time.time() - t0, extra), flush=True)


def _rec(row):
    return normalize_record(row.get("business_name", ""), row.get("business_address", ""),
                            row.get("country", ""))


def run(test_dir=None, model_dir=None, output_dir=None, K=None, batch=None,
        workers=None, gpu_ids=None, no_ann=False):
    t_all = time.time()
    test_dir = test_dir or C.TEST_DIR
    model_dir = model_dir or C.MODEL_DIR
    output_dir = output_dir or C.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    K = K or C.K
    batch = batch or C.S1_CHUNK
    s1p = os.path.join(test_dir, "test_source1.tsv")
    s2p = os.path.join(test_dir, "test_source2.tsv")
    s3p = os.path.join(test_dir, "test_source3.tsv")

    with open(os.path.join(model_dir, "lgbm.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, "selected_pipeline.json")) as f:
        sel = json.load(f)
    try:
        with open(os.path.join(model_dir, "calibrator.pkl"), "rb") as f:
            cal = pickle.load(f)
    except Exception:
        cal = ("none", None)
    thr, t2, t3 = sel.get("threshold", 0.5), sel.get("s2_threshold", 0.5), \
        sel.get("s3_threshold", 0.5)
    use_cal = sel.get("calibrated", False)
    sing_thr = sel.get("singleton_threshold", max(thr, 0.6))

    # 0-30min: load targets + build deterministic indexes (streamed)
    t0 = time.time()
    trec, tids, trecs_list = {}, [], []
    for p in (s2p, s3p):
        for ch in scan_source_batched(p):
            for _, r in ch.iterrows():
                e = str(r["entity_id"]).strip()
                nr = _rec(r)
                trec[e] = nr
                tids.append(e)
                trecs_list.append(nr)
    eb, nb, rb = EB.build(trecs_list, tids), NB.build(trecs_list, tids), \
        RB.build(trecs_list, tids)
    from ..preprocessing.rarity import attach_rare
    attach_rare(trecs_list, rb["dfn"], rb["dfa"], C.RARE_DF_CAP,
                C.RARE_MAX_QUERY_TOKENS)
    fam_of, fam_counts = build_families(trec)
    log("load+normalize+det-index", t0, "targets=%d" % len(tids))

    # 20-60min: ANN embeddings. GPU0=S2 lane, GPU1=S3 lane (processes); CPU fallback.
    ann = None
    if not no_ann:
        t0 = time.time()
        try:
            ann = _build_ann(tids, trecs_list, gpu_ids=gpu_ids)
            log("dense embeddings+ANN", t0, "backend=%s" % ann.get("backend", "?"))
        except Exception as e:
            ann = None
            log("ANN fallback (lexical only)", t0, "err=%s" % str(e)[:200])

    # 80-160min: chunked S1 -> candidates -> scoring -> decisions
    t0 = time.time()
    order = s1_id_order(s1p)
    cand_out, pred_out = {}, {}
    s1rec_buf, s1ids_buf = {}, []
    n_done = 0
    # query encoder created ONCE per run (Sec 33.4: never re-embed/reload);
    # reuse the S2-lane encoder so queries live in the same space.
    qenc = (ann.get("qenc") if ann is not None else None) or \
        (Encoder(device="cpu") if ann is not None else None)

    def flush():
        nonlocal n_done
        if not s1ids_buf:
            return
        _flush_chunk(s1ids_buf, s1rec_buf, eb, nb, rb, trec, ann, tids,
                     clf, cal, use_cal, thr, t2, t3, sing_thr, K,
                     fam_counts, cand_out, pred_out, qenc=qenc)
        n_done += len(s1ids_buf)
        el = time.time() - t0
        from .resources import ram_gb, gpu_text
        print("[predict] scored %d/%d (%.1f%%) %.0f/s elapsed %.0fs ram=%.1fGB %s" %
              (n_done, len(order), 100.0 * n_done / len(order),
               n_done / max(el, 1), el, ram_gb(), gpu_text()), flush=True)

    # stream S1 in file order, accumulate `batch` then flush
    pending = {}
    s1_countries = {}
    for ch in scan_source_batched(s1p, batch_rows=100000):
        for _, r in ch.iterrows():
            e = str(r["entity_id"]).strip()
            nr = _rec(r)
            pending[e] = nr
            s1_countries[e] = nr.get("country_norm", "")
            if len(pending) >= batch:
                s1rec_buf.update(pending)
                s1ids_buf.extend(list(pending.keys()))
                pending = {}
                flush()
                s1rec_buf, s1ids_buf = {}, []
    if pending:
        s1rec_buf.update(pending)
        s1ids_buf.extend(list(pending.keys()))
        flush()
    # S1 with zero rows safety
    for s in order:
        cand_out.setdefault(s, [])
        pred_out.setdefault(s, set())
    log("test candidates+scoring", t0, "%s" % count_stats(cand_out))

    # 150-175min: outputs + France check (Sec 27/35)
    t0 = time.time()
    write_matching(os.path.join(output_dir, "matching_results.tsv"), order, pred_out)
    write_candidates(os.path.join(output_dir, "candidate_pairs.tsv"), order, cand_out)
    # Sec 35: France must appear in test output (open-set country check)
    from collections import Counter
    cc = Counter(s1_countries.values())
    n_fr = sum(1 for s in order if s1_countries.get(s) == "france")
    log("outputs", t0, "rows=%d matched=%d countries=%s france_rows=%d" %
        (len(order), sum(1 for v in pred_out.values() if v),
         dict(cc.most_common(8)), n_fr))
    total = time.time() - t_all
    print("[predict] DONE total=%.0fs budget=%ds" % (total, C.TIME_BUDGET_S), flush=True)
    return {"n": len(order), "total_s": total}


def _build_ann(tids, trecs_list, gpu_ids=None):
    # Sec 11 lanes: S2 lane on GPU0, S3 lane on GPU1 (independent encoders).
    from ..blocking.ann_retrieval import lane_device
    enc2 = Encoder(device=lane_device(0))
    enc3 = Encoder(device=lane_device(1))
    texts = [semantic_text(r) for r in trecs_list]
    s2pos = [i for i, t in enumerate(tids) if t.startswith("S2-")]
    s3pos = [i for i, t in enumerate(tids) if t.startswith("S3-")]
    T2 = [texts[i] for i in s2pos]
    T3 = [texts[i] for i in s3pos]
    V2 = enc2.encode(T2) if T2 else np.zeros((0, 256), np.float32)
    V3 = enc3.encode(T3) if T3 else np.zeros((0, V2.shape[1]), np.float32)
    dim = V2.shape[1]
    I2, I3 = FaissIndex(dim), FaissIndex(dim)
    I2.add(V2)
    I3.add(V3)
    return {"backend": enc2.backend + "+" + enc3.backend, "I2": I2, "I3": I3,
            "s2pos": s2pos, "s3pos": s3pos,
            "s2ids": [tids[i] for i in s2pos], "s3ids": [tids[i] for i in s3pos],
            "qenc": enc2}


def _flush_chunk(s1ids, s1rec, eb, nb, rb, trec, ann, tids_all,
                 clf, cal, use_cal, thr, t2, t3, sing_thr, K,
                 fam_counts, cand_out, pred_out, qenc=None):
    # ANN query vectors for chunk (encoder created once per run, Sec 33.4)
    Q = None
    if ann is not None:
        try:
            enc = qenc or Encoder(device="cpu")
            Q = enc.encode([semantic_text(s1rec[s]) for s in s1ids])
        except Exception:
            Q = None
            ann = None
    for qi, s in enumerate(s1ids):
        a = s1rec[s]
        m = {}
        for src in (EB.query_one(a, eb), NB.query_one(a, nb), RB.query_one(a, rb)):
            for tid, mm in src.items():
                d = m.get(tid)
                if d is None:
                    m[tid] = {"blocks": set(mm.get("blocks", set())),
                              **{k: v for k, v in mm.items() if k != "blocks"}}
                else:
                    d["blocks"] |= mm.get("blocks", set())
        if ann is not None and Q is not None:
            D2, P2 = ann["I2"].search(Q[qi:qi + 1], C.ANN_TOPK_S2)
            D3, P3 = ann["I3"].search(Q[qi:qi + 1], C.ANN_TOPK_S3)
            # map ANN positions (within S2/S3 sub-index) back to global ids
            a2 = [(int(p), float(d)) for p, d in zip(P2[0], D2[0]) if 0 <= p < len(ann["s2ids"])]
            a3 = [(int(p), float(d)) for p, d in zip(P3[0], D3[0]) if 0 <= p < len(ann["s3ids"])]
            m = merge_per_s1(s, m, a2, a3, ann["s2ids"], ann["s3ids"])
        top = prerank_topk(a, m, trec, K=K)
        cands = [t for t, _ in top]
        cand_out[s] = cands
        if not cands:
            pred_out[s] = set()
            continue
        pairs = [(s, t, a, trec[t], m.get(t, {})) for t in cands]
        probs = score_pairs(pairs, clf, fam_counts)
        if use_cal:
            probs = apply_cal(cal, probs)
        pred_out[s] = decide(cands, probs, thr=thr, s2_thr=t2, s3_thr=t3,
                             sing_thr=sing_thr)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
