"""Multilingual dense encoder (Sec 10). potion-multilingual-128M, no fine-tune.

Text view: [name_core] [SEP] [name_translit] [SEP] [address_norm] [SEP]
[address_translit], truncated to ENCODER_MAX_LEN tokens/chars.
Backend priority: model2vec (static, fastest on T4/CPU) -> sentence-transformers
-> hashing TF-IDF fallback. Batched, FP16-safe, L2-normalized float32 output.
Embed each record exactly once; cache to disk (Sec 33).
"""
import os
import numpy as np

from .. import config as C

_SEP = " [SEP] "


def semantic_text(rec):
    parts = [rec.get("name_core", ""), rec.get("name_translit", ""),
             rec.get("address_norm", ""), rec.get("address_translit", "")]
    t = _SEP.join(p for p in parts if p)
    return t[:2000]


class Encoder:
    def __init__(self, model_name=None, batch=None, device="cpu"):
        self.model_name = model_name or C.ENCODER_MODEL
        self.batch = batch or C.ENCODER_BATCH
        self.device = device
        self.backend = None
        self.model = None
        self._init_backend()

    def _init_backend(self):
        # Fast offline path first: cached model only (no download stall).
        try:
            from model2vec import StaticModel
            try:
                self.model = StaticModel.from_pretrained(self.model_name,
                                                         local_files_only=True)
                self.backend = "model2vec"
                return
            except Exception:
                pass
            # Not cached: try download (Kaggle has internet for models).
            # Business data is never fetched; only the pretrained encoder.
            self.model = StaticModel.from_pretrained(self.model_name)
            self.backend = "model2vec"
            return
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.backend = "st"
            return
        except Exception:
            pass
        self.backend = "hash"

    def encode(self, texts):
        if self.backend == "model2vec":
            vecs = self.model.encode(texts, batch_size=self.batch,
                                     show_progress_bar=False)
            V = np.asarray(vecs, dtype=np.float32)
        elif self.backend == "st":
            import torch
            V = np.asarray(self.model.encode(texts, batch_size=self.batch,
                                             show_progress_bar=False,
                                             convert_to_numpy=True,
                                             normalize_embeddings=False),
                           dtype=np.float32)
        else:
            V = _hash_encode(texts, dim=256)
        n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return (V / n).astype(np.float32)


def _hash_encode(texts, dim=256):
    import hashlib
    V = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in str(t).lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            V[i, h] += 1.0
    return V
