"""Mandatory blocking experiment (Sec 36):
  A: exact/core/translit name
  B: A + house/numeric
  C: B + rare address/name tokens
  D: C + multilingual ANN
Report per variant: candidate recall, avg/p95 candidates, runtime.
Choose smallest set with >=99% recall; default final K=22 pipeline follows.
"""
import time

from . import config as C
from .io.loader import scan_source_batched, read_ground_truth
from .preprocessing.normalize import normalize_record
from .blocking import exact_blocks as EB
from .blocking import numeric_blocks as NB
from .blocking import rare_blocks as RB
from .blocking.candidate_union import merge_per_s1
from .blocking.rerank import topk as prerank_topk
from .evaluation.candidate_recall import recall as rec_fn
from .evaluation.candidate_recall import count_stats as cs


def _rec(r):
    return normalize_record(r.get("business_name", ""), r.get("business_address", ""),
                            r.get("country", ""))


def run(train_dir=None, K=22, no_ann=False, max_s1=20000):
    train_dir = train_dir or C.TRAIN_DIR
    import os
    s1p = os.path.join(train_dir, "train_source1.tsv")
    s2p = os.path.join(train_dir, "train_source2.tsv")
    s3p = os.path.join(train_dir, "train_source3.tsv")
    truth = read_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    # sample S1 for speed
    s1rec, s1ids = {}, []
    for ch in scan_source_batched(s1p, 100000):
        for _, r in ch.iterrows():
            e = str(r["entity_id"]).strip()
            if e in truth and len(s1ids) < max_s1:
                s1rec[e] = _rec(r)
                s1ids.append(e)
        if len(s1ids) >= max_s1:
            break
    trec, tids, tl = {}, [], []
    for p in (s2p, s3p):
        for ch in scan_source_batched(s1p if False else p, 200000):
            for _, r in ch.iterrows():
                e = str(r["entity_id"]).strip()
                nr = _rec(r)
                trec[e] = nr
                tids.append(e)
                tl.append(nr)
            break  # one chunk is enough for the experiment
    eb, nb, rb = EB.build(tl, tids), NB.build(tl, tids), RB.build(tl, tids)
    small_truth = {s: truth[s] & set(trec) for s in s1ids}
    results = {}
    # A
    t0 = time.time()
    A = {s: set(t for t, _ in prerank_topk(
        s1rec[s], EB.query_one(s1rec[s], eb), trec, K=K)) for s in s1ids}
    results["A"] = (rec_fn(small_truth, A), cs(A), time.time() - t0)
    # B
    t0 = time.time()
    B = {}
    for s in s1ids:
        m = dict(EB.query_one(s1rec[s], eb))
        for k, v in NB.query_one(s1rec[s], nb).items():
            m.setdefault(k, v)
        B[s] = set(t for t, _ in prerank_topk(s1rec[s], m, trec, K=K))
    results["B"] = (rec_fn(small_truth, B), cs(B), time.time() - t0)
    # C
    t0 = time.time()
    Cc = {}
    for s in s1ids:
        m = dict(EB.query_one(s1rec[s], eb))
        for src in (NB.query_one(s1rec[s], nb), RB.query_one(s1rec[s], rb)):
            for k, v in src.items():
                m.setdefault(k, v)
        Cc[s] = set(t for t, _ in prerank_topk(s1rec[s], m, trec, K=K))
    results["C"] = (rec_fn(small_truth, Cc), cs(Cc), time.time() - t0)
    # D
    if not no_ann:
        t0 = time.time()
        try:
            from .embeddings.encoder import Encoder, semantic_text
            from .embeddings.faiss_index import FaissIndex
            enc = Encoder(device="cpu")
            V = enc.encode([semantic_text(trec[t]) for t in tids])
            Q = enc.encode([semantic_text(s1rec[s]) for s in s1ids])
            IX = FaissIndex(V.shape[1])
            IX.add(V)
            Dd, Pp = IX.search(Q, 12)
            Dd_ = {}
            for i, s in enumerate(s1ids):
                m = dict(EB.query_one(s1rec[s], eb))
                for src in (NB.query_one(s1rec[s], nb), RB.query_one(s1rec[s], rb)):
                    for k, v in src.items():
                        m.setdefault(k, v)
                m = merge_per_s1(s, m, [(int(p), float(d)) for p, d in
                                        zip(Pp[i], Dd[i])], None, tids, [])
                Dd_[s] = set(t for t, _ in prerank_topk(s1rec[s], m, trec, K=K))
            results["D"] = (rec_fn(small_truth, Dd_), cs(Dd_), time.time() - t0)
        except Exception as e:
            results["D"] = ({"micro": -1, "macro": -1}, {}, 0)
            print("D skipped:", str(e)[:200])
    print("Blocking experiment (K=%d, n=%d):" % (K, len(s1ids)))
    for k in ("A", "B", "C", "D"):
        if k not in results:
            print("  %s skipped (no_ann=True)" % k)
            continue
        r, c, t = results[k]
        print("  %s recall micro=%.4f macro=%.4f avg=%.1f p95=%.0f time=%.1fs" %
              (k, r.get("micro", -1), r.get("macro", -1), c.get("avg", -1),
               c.get("p95", -1), t))
    print("Choose smallest set with ~>=99% recall; default final K=22.")
    return results
