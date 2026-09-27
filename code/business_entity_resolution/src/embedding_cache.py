"""Sharded fp16 embedding cache (never one giant float32 matrix)."""
import os

import numpy as np


class EmbeddingCache:
    def __init__(self, root, view, dim, dtype="float16", shard_size=200000):
        self.root = os.path.join(root, "emb_%s" % view)
        os.makedirs(self.root, exist_ok=True)
        self.view = view
        self.dim = dim
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.shard_size = shard_size

    def shard_path(self, shard):
        return os.path.join(self.root, "shard_%05d.npy" % shard)

    def write_shard(self, shard, arr):
        np.save(self.shard_path(shard), np.asarray(arr, dtype=self.dtype))

    def read_shard(self, shard):
        return np.load(self.shard_path(shard), mmap_mode="r")

    def list_shards(self):
        out = []
        for f in sorted(os.listdir(self.root)):
            if f.startswith("shard_") and f.endswith(".npy"):
                try:
                    out.append(int(f[6:11]))
                except Exception:
                    pass
        return out
