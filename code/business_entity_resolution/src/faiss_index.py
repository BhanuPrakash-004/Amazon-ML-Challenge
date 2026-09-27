"""FAISS large-scale index helpers (IVFPQ default, cosine via IP on L2-normed vecs).

Includes ANN-vs-exact recall validation (Recall@25..200) on a subset.
Lazy faiss import so CPU-lexical path works without faiss installed.
"""
import numpy as np


def build_index(vecs, nlist=1024, m=32, nbits=8):
    import faiss
    d = vecs.shape[1]
    nlist = max(1, min(nlist, max(1, len(vecs) // 256)))
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(quant, d, nlist, m, nbits)
    if len(vecs) < 50000:
        idx.train(vecs.astype(np.float32))
    else:
        rng = np.random.RandomState(42)
        sample = vecs[rng.choice(len(vecs), 50000, replace=False)].astype(np.float32)
        idx.train(sample)
    idx.add(vecs.astype(np.float32))
    return idx


def query_index(idx, q, top_k=75, nprobe=16):
    try:
        idx.nprobe = nprobe
    except Exception:
        pass
    D, I = idx.search(q.astype(np.float32), top_k)
    return D, I


def ann_recall_report(exact_fn, ann_fn, q, k_list=(25, 50, 75, 100, 150, 200)):
    """Compare exact vs ANN id-sets. Returns {k: recall}."""
    rep = {}
    for k in k_list:
        e = exact_fn(q, k)
        a = ann_fn(q, k)
        rs = []
        for es, ac in zip(e, a):
            es = set(es)
            rs.append(len(es & set(ac)) / len(es) if es else 1.0)
        rep[k] = float(np.mean(rs)) if rs else 1.0
    return rep
