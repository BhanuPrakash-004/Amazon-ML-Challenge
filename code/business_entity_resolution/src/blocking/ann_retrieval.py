"""ANN retrieval orchestration (Sec 10-12). S2 on GPU0, S3 on GPU1.

Falls back to CPU FAISS, then to brute-force numpy, rather than burning
runtime troubleshooting GPU libs (Sec 11).
"""
import os

from .. import config as C


def lane_device(lane):
    """GPU id for a retrieval lane (0 -> S2, 1 -> S3); 'cpu' when no CUDA."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            ids = C.GPU_IDS or [0]
            return "cuda:%d" % ids[lane % len(ids)]
    except Exception:
        pass
    return "cpu"


def build_and_search(s1_emb, s2_emb, s3_emb, k_s2=None, k_s3=None,
                     nlist=None, nprobe=None):
    k_s2 = C.ANN_TOPK_S2 if k_s2 is None else k_s2
    k_s3 = C.ANN_TOPK_S3 if k_s3 is None else k_s3
    nlist = C.FAISS_NLIST if nlist is None else nlist
    nprobe = C.FAISS_NPROBE if nprobe is None else nprobe
    from ..embeddings.faiss_index import FaissIndex
    out = {}
    if s2_emb is not None and len(s2_emb):
        ix = FaissIndex(s2_emb.shape[1], nlist=nlist, nprobe=nprobe)
        ix.add(s2_emb)
        D, I = ix.search(s1_emb, k_s2)
        out["s2"] = (D, I)
    if s3_emb is not None and len(s3_emb):
        ix = FaissIndex(s3_emb.shape[1], nlist=nlist, nprobe=nprobe)
        ix.add(s3_emb)
        D, I = ix.search(s1_emb, k_s3)
        out["s3"] = (D, I)
    return out
