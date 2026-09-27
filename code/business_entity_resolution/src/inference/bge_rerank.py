"""Optional BGE reranker stage (Sec 26). DISABLED BY DEFAULT.

Model: BAAI/bge-reranker-v2-m3 (Apache-2.0, local cross-encoder — no business
data ever leaves the machine). Applied only to ambiguous candidates:
  LightGBM prob in uncertainty band [0.3, 0.7], OR top1-top2 margin tiny,
  OR name/address evidence disagree. Maximum top 2-3 per ambiguous S1
  (C.SPEC_RERANK_TOP_N). Never over the full candidate matrix. If runtime
  exceeds budget, callers skip this stage entirely.
"""
import numpy as np

from .. import config as C


def select_ambiguous(tids, probs, s1rec=None, trecs=None, top_n=None, margin=0.08):
    top_n = C.SPEC_RERANK_TOP_N if top_n is None else top_n
    p = np.asarray(list(probs), dtype=float)
    if not len(p):
        return []
    order = np.argsort(-p)
    cands = [tids[i] for i in order[:top_n]]
    top, second = float(p[order[0]]), float(p[order[1]]) if len(order) > 1 else 0.0
    uncertain = any(0.3 <= float(p[order[i]]) <= 0.7 for i in range(min(len(order), top_n)))
    tight = (top - second) <= margin
    disagree = False
    if s1rec is not None and trecs is not None:
        b = trecs.get(tids[order[0]], {})
        name_ok = s1rec.get("name_core", "") == b.get("name_core", "") and \
            bool(s1rec.get("name_core", ""))
        addr_ok = s1rec.get("address_norm", "") == b.get("address_norm", "") and \
            bool(s1rec.get("address_norm", ""))
        disagree = name_ok != addr_ok
    if uncertain or tight or disagree:
        return cands
    return []


def rerank_scores(tids, probs, s1rec, trecs, device="cpu"):
    """Return {tid: blended_score} for ambiguous tids only. Empty when disabled
    or when the reranker backend is unavailable (lazy import, CPU fallback)."""
    if not C.USE_RERANKER:
        return {}
    amb = select_ambiguous(tids, probs, s1rec, trecs)
    if not amb:
        return {}
    try:
        from ..reranker import Reranker
        rr = Reranker(device=device)
        if not rr.available():
            return {}
        from ..embeddings.encoder import semantic_text
        L = [semantic_text(s1rec)] * len(amb)
        R = [semantic_text(trecs[t]) for t in amb]
        s = rr.score(L, R)
        return {t: float(v) for t, v in zip(amb, s)}
    except Exception:
        return {}
