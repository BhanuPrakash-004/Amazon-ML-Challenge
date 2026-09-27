"""FAISS ANN index (Sec 12). cosine/IP after normalization.

nlist=4096, nprobe=8. GPU attempt -> CPU fallback (never burn runtime).
If faiss missing, numpy brute-force fallback (correct, slower).
"""
import numpy as np

from .. import config as C


class FaissIndex:
    def __init__(self, dim, nlist=None, nprobe=None, use_gpu=False):
        self.dim = dim
        self.nlist = nlist or C.FAISS_NLIST
        self.nprobe = nprobe or C.FAISS_NPROBE
        self.use_gpu = use_gpu
        self.index = None
        self.vecs = None
        try:
            import faiss
            self.faiss = faiss
        except Exception:
            self.faiss = None

    def add(self, vecs):
        V = np.ascontiguousarray(vecs.astype(np.float32))
        n = len(V)
        if self.faiss is None:
            self.vecs = V
            return
        faiss = self.faiss
        try:
            if n < 5000 or self.nlist > n // 10:
                index = faiss.IndexFlatIP(self.dim)
            else:
                quant = faiss.IndexFlatIP(self.dim)
                index = faiss.IndexIVFFlat(quant, self.dim, min(self.nlist, n // 10))
                index.train(V)
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except Exception:
                    pass
            index.add(V)
            try:
                index.nprobe = self.nprobe
            except Exception:
                pass
            self.index = index
        except Exception:
            self.index = None
            self.vecs = V

    def search(self, queries, k):
        Q = np.ascontiguousarray(queries.astype(np.float32))
        if self.index is not None:
            try:
                D, I = self.index.search(Q, k)
                return D.astype(np.float32), I.astype(np.int64)
            except Exception:
                pass
        # brute force
        V = self.vecs
        D = Q @ V.T
        I = np.argsort(-D, axis=1)[:, :k]
        D = np.take_along_axis(D, I, axis=1)
        return D.astype(np.float32), I.astype(np.int64)
