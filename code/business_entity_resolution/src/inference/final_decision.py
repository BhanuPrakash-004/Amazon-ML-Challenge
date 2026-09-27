"""Final per-S1 decision (Sec 23). Zero/one/many (never forced one-to-one).

Per-source S2/S3 thresholds on (optionally BGE-blended) calibrated scores,
then the singleton/confidence gate (top score, margin, numeric conflicts).
Optional BGE rerank hook (disabled by default, top 2-3 ambiguous only).
"""
import numpy as np

from .singleton_gate import gate


def decide(tids, probs, s1rec=None, trecs=None, thr=0.5, s2_thr=None,
           s3_thr=None, sing_thr=0.6, bge_scores=None):
    s2_thr = thr if s2_thr is None else s2_thr
    s3_thr = thr if s3_thr is None else s3_thr
    if not tids:
        return set()
    blended = []
    for t, p in zip(tids, probs):
        # optional BGE blend for ambiguous band only (Sec 26)
        if bge_scores is not None and t in bge_scores and 0.3 <= float(p) <= 0.7:
            p = 0.7 * float(p) + 0.3 * float(bge_scores[t])
        blended.append(float(p))
    # per-source pre-filter, then singleton gate for the final keep set
    pre = {t for t, p in zip(tids, blended)
           if p >= (s2_thr if t.startswith("S2-") else s3_thr)}
    if not pre:
        return set()
    gated = gate(tids, np.asarray(blended), s1rec,
                 {t: trecs[t] for t in tids if t in (trecs or {})}
                 if isinstance(trecs, dict) else trecs,
                 thr=min(s2_thr, s3_thr), sing_thr=sing_thr)
    return pre & gated
