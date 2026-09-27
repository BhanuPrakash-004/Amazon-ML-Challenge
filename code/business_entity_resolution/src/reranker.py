"""Selective multilingual reranker (bge-reranker-v2-m3, Apache-2.0, local only).

Reranks ONLY ambiguous top-N candidates per S1 (never all 150).
Lazy imports: returns base scores unchanged when transformers/torch missing.
"""
import numpy as np


def select_ambiguous(s1ids, probs, top_n=15, margin=0.25):
    """Indices (into pair arrays) selected for reranking."""
    from collections import defaultdict
    by = defaultdict(list)
    for i, (s, p) in enumerate(zip(s1ids, probs)):
        by[s].append((p, i))
    sel = []
    for s, lst in by.items():
        lst.sort(reverse=True)
        top = lst[:top_n]
        if not top:
            continue
        p0 = top[0][0]
        for p, i in top:
            if p0 - p <= margin or p >= 0.3:
                sel.append(i)
    return sorted(sel)


class Reranker:
    def __init__(self, model_name=None, device="cpu", batch=16, max_len=256):
        from . import config as C
        self.model_name = model_name or C.RERANKER_MODEL_NAME
        self.device = device
        self.batch = batch
        self.max_len = max_len
        self._tok = None
        self._mdl = None

    def available(self):
        try:
            import torch, transformers  # noqa
            return True
        except Exception:
            return False

    def _load(self):
        if self._mdl is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=False)
        self._mdl.eval().to(self.device)

    def score(self, left_texts, right_texts):
        """Cross-encoder scores for aligned pairs. Falls back to zeros."""
        if not self.available():
            return np.zeros(len(left_texts), dtype=np.float32)
        import torch
        self._load()
        out = []
        bs = max(1, int(self.batch))
        for i in range(0, len(left_texts), bs):
            L = left_texts[i:i + bs]
            R = right_texts[i:i + bs]
            try:
                enc = self._tok(list(L), list(R), padding=True, truncation=True,
                                max_length=self.max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = self._mdl(**enc).logits.float().cpu().numpy().ravel()
                out.extend([float(x) for x in logits])
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    bs = max(1, bs // 2)
                    # retry same slice with smaller batch
                    enc = self._tok(list(L)[:bs], list(R)[:bs], padding=True, truncation=True,
                                    max_length=self.max_len, return_tensors="pt")
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = self._mdl(**enc).logits.float().cpu().numpy().ravel()
                    out.extend([float(x) for x in logits])
                    # remaining handled next loop by stepping bs
                    rest_L, rest_R = L[bs:], R[bs:]
                    for j in range(0, len(rest_L), bs):
                        out.extend(self.score(rest_L[j:j + bs], rest_R[j:j + bs]).tolist())
                    break
                raise
        import math
        return np.asarray([1.0 / (1.0 + math.exp(-x)) for x in out], dtype=np.float32)
