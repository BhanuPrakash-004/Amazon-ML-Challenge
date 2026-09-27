"""Recall-preserving cheap candidate pruning + adaptive final pruning.

Stages (spec section 6/15/16):
  raw retrieval pool 100-300 (union output, never written directly)
  -> cheap prune to INTERNAL_KEEP_K=25 (before expensive features)
  -> adaptive final ~8-12 preferred, 15 hard ceiling (written to
     candidate_pairs.tsv).

``prune()`` is the legacy V2/V3 baseline (keep_k=150) and is preserved
byte-for-byte in behavior for ``--no-adaptive`` SAFE BASELINE runs.

Adaptive rule: always keep exact + multi-block (>=2) evidence before any
cap; keep high-semantic weak-lexical only if among top recovery; drop weak
one-channel candidates when enough strong ones exist. Never force-fill to
15 when only 3 strong exist.
"""
import numpy as np


def cheap_scores(cids, evidence, meta, e5_scores=None):
    s = []
    for c in cids:
        ev = evidence.get(c, {})
        rank, score, n = meta.get(c, (999, 0.0, 1))
        v = 0.0
        v += 5.0 * ev.get("exact_name", 0) + 4.0 * ev.get("exact_address", 0)
        v += 2.0 * min(n, 4)
        v += float(score)
        if e5_scores and c in e5_scores:
            v += 3.0 * float(e5_scores[c])
        v += -0.01 * min(rank, 500)
        s.append(v)
    return np.asarray(s, dtype=np.float32)


def prune(cids, evidence, meta, keep_k=150, e5_scores=None):
    if len(cids) <= keep_k:
        return cids
    # must-keep: exact evidence or >=3 supporting blocks
    must = [c for c in cids if evidence.get(c, {}).get("exact_name") or evidence.get(c, {}).get("exact_address")
            or meta.get(c, (0, 0, 0))[2] >= 3]
    rest = [c for c in cids if c not in set(must)]
    sc = cheap_scores(rest, evidence, meta, e5_scores)
    order = np.argsort(-sc)
    keep_rest = [rest[i] for i in order[:max(0, keep_k - len(must))]]
    kept = must + keep_rest
    # restore global order
    pos = {c: i for i, c in enumerate(cids)}
    return sorted(kept, key=lambda c: pos[c])[:keep_k]


def _is_strong(c, evidence, meta, e5_scores=None):
    ev = evidence.get(c, {})
    n = meta.get(c, (999, 0.0, 1))[2]
    if ev.get("exact_name") or ev.get("exact_address"):
        return True
    if n >= 2:
        return True
    if e5_scores and float(e5_scores.get(c, 0.0)) >= 0.5:
        return True
    return False


def adaptive_prune_for_s1(cids, evidence, meta, e5_scores=None, probs=None,
                           internal_k=25, final_max=15):
    """Two-stage adaptive prune for one S1.

    Returns final list (<= final_max) preserving recall-critical evidence.
    Never pads weak candidates: if only 3 strong exist, returns 3.
    """
    if not cids:
        return []
    # Stage 1: cheap prune to internal pool (keeps must-keep + top cheap).
    pool = prune(cids, evidence, meta, keep_k=internal_k, e5_scores=e5_scores)
    if len(pool) <= final_max:
        # Still apply strong-first ordering but keep all (do not pad).
        return pool
    # Stage 2: split strong vs weak.
    strong = [c for c in pool if _is_strong(c, evidence, meta, e5_scores)]
    weak = [c for c in pool if c not in set(strong)]
    if probs:
        # rank weak by model prob then cheap score so semantic recovery
        # with high prob survives over lexically-similar distractors.
        wscore = {}
        for c in weak:
            wscore[c] = float(probs.get(c, 0.0))
        weak = sorted(weak, key=lambda c: -wscore[c])
    else:
        sc = cheap_scores(weak, evidence, meta, e5_scores)
        weak = [weak[i] for i in np.argsort(-sc)]
    # Keep all strong up to final_max; fill remainder with best weak only
    # if strong set is small (preserve recall, avoid blind fill).
    if len(strong) >= final_max:
        # Too many strong: keep top final_max by cheap score.
        sc = cheap_scores(strong, evidence, meta, e5_scores)
        strong = [strong[i] for i in np.argsort(-sc)[:final_max]]
        return strong
    room = final_max - len(strong)
    # Do not fill more weak than needed: keep at most room, but drop
    # tail weak one-channel candidates when strong already cover >=3.
    keep_weak = weak[:room] if len(strong) < 3 else weak[:room]
    kept = strong + keep_weak
    pos = {c: i for i, c in enumerate(pool)}
    return sorted(kept, key=lambda c: pos[c])[:final_max]


def candidate_stats(cand_dict, truth=None):
    """Stats for candidate_stats.json (spec section 17)."""
    import numpy as np
    sizes = np.asarray([len(v) for v in cand_dict.values()], dtype=np.float64)
    if len(sizes) == 0:
        sizes = np.asarray([0.0])
    out = {
        "n_s1": int(len(cand_dict)),
        "average_candidates": float(sizes.mean()),
        "median_candidates": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "p95": float(np.percentile(sizes, 95)),
        "p99": float(np.percentile(sizes, 99)),
        "max": int(sizes.max()),
        "min": int(sizes.min()),
        "s1_coverage": float(sum(1 for v in cand_dict.values() if len(v) > 0) / max(len(cand_dict), 1)),
    }
    if truth is not None:
        hits = tot = 0
        per = []
        for s, t in truth.items():
            t = set(t or [])
            if not t:
                per.append(1.0)
                continue
            c = set(cand_dict.get(s, []))
            r = len(t & c) / max(len(t), 1)
            per.append(r)
            hits += len(t & c)
            tot += len(t)
        out["blocking_recall_micro"] = float(hits / max(tot, 1))
        out["blocking_recall_macro"] = float(sum(per) / max(len(per), 1))
    return out


def check_candidate_budget(stats, max_avg=15.0, strict=True):
    """Fail loudly if average exceeds ceiling (spec section 17)."""
    avg = float(stats.get("average_candidates", 0.0))
    if avg > max_avg:
        msg = ("CANDIDATE BUDGET EXCEEDED: avg=%.2f > %.2f. "
               "Final candidate_pairs.tsv must average 8-15. "
               "Tighten adaptive caps or pass --no-adaptive only for baseline." % (avg, max_avg))
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg, flush=True)
    return avg
