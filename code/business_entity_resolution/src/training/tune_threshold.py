"""F0.5 threshold tuning (Sec 22). Coarse grid 0.30-0.95 then refine;
separate S2/S3 thresholds."""
from .. import config as C
from ..evaluation.f05 import macro_f05


def tune(val_items, probs, truth, grid=None):
    """val_items: list of (s1, [tids]); probs aligned; truth: s1->set."""
    grid = grid or C.THRESHOLD_GRID
    best_t, best_f = 0.5, -1.0
    for t in grid:
        pred = {}
        for (s1, tids), ps in zip(val_items, probs):
            pred[s1] = {t_ for t_, p in zip(tids, ps) if p >= t}
        f = macro_f05(truth, pred)
        if f > best_f:
            best_f, best_t = f, t
    # refine
    for dt in (-0.04, -0.02, 0.02, 0.04):
        t = round(min(0.97, max(0.2, best_t + dt)), 2)
        pred = {}
        for (s1, tids), ps in zip(val_items, probs):
            pred[s1] = {t_ for t_, p in zip(tids, ps) if p >= t}
        f = macro_f05(truth, pred)
        if f > best_f:
            best_f, best_t = f, t
    # per-source thresholds
    best_s2, best_s3 = best_t, best_t
    bf = best_f
    for t2 in [best_t - 0.05, best_t, best_t + 0.05]:
        for t3 in [best_t - 0.05, best_t, best_t + 0.05]:
            pred = {}
            for (s1, tids), ps in zip(val_items, probs):
                keep = set()
                for t_, p in zip(tids, ps):
                    th = t2 if t_.startswith("S2-") else t3
                    if p >= th:
                        keep.add(t_)
                pred[s1] = keep
            f = macro_f05(truth, pred)
            if f > bf:
                bf, best_s2, best_s3 = f, round(t2, 2), round(t3, 2)
    return {"threshold": best_t, "s2_threshold": best_s2,
            "s3_threshold": best_s3, "f05": bf}
