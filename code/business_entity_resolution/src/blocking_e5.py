"""E5 ANN blocking wrapper: two views (name / full), sequential index use.

Storage-safe: build/query one view at a time; raw shards deleted after
compressed index persisted. Without torch/faiss installed, query returns []
so the lexical pipeline still runs (dev-mode / CPU-only safe).
"""
import os

import numpy as np


class E5Blocker:
    def __init__(self, cache_root=None, model_name=None, top_k_name=75, top_k_full=75,
                 nprobe=16, max_len=256, batch=96):
        self.cache_root = cache_root
        self.model_name = model_name
        self.top_k_name = top_k_name
        self.top_k_full = top_k_full
        self.nprobe = nprobe
        self.max_len = max_len
        self.batch = batch
        self.idx_name = None
        self.idx_full = None
        self.ids = []

    def available(self):
        try:
            import torch, transformers, faiss  # noqa
            return True
        except Exception:
            return False

    def build(self, id_list, name_texts, full_texts, device="cpu", nlist=1024, m=32, nbits=8):
        """Build view A then view B sequentially; keep only compact indexes."""
        from .embeddings import embed_texts
        from . import faiss_index as FI
        self.ids = list(id_list)
        # VIEW A (name)
        va = embed_texts(name_texts, self.model_name, device, self.batch, self.max_len, is_query=False)
        self.idx_name = FI.build_index(va, nlist, m, nbits)
        del va
        # VIEW B (full)
        vb = embed_texts(full_texts, self.model_name, device, self.batch, self.max_len, is_query=False)
        self.idx_full = FI.build_index(vb, nlist, m, nbits)
        del vb
        return self

    def query(self, name_q, full_q, device="cpu"):
        from .embeddings import embed_texts
        from . import faiss_index as FI
        out = {}
        if self.idx_name is None:
            return out
        qn = embed_texts(name_q, self.model_name, device, self.batch, self.max_len, is_query=True)
        Dn, In = FI.query_index(self.idx_name, qn, self.top_k_name, self.nprobe)
        qf = embed_texts(full_q, self.model_name, device, self.batch, self.max_len, is_query=True)
        Df, If = FI.query_index(self.idx_full, qf, self.top_k_full, self.nprobe)
        return (Dn, In, Df, If)
