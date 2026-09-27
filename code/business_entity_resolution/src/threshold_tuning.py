"""F0.5 threshold tuning (macro per-S1). Global + margin + singleton protection."""

from .evaluate import macro_f05


def tune_decision(val_s1, val_cands, val_scores, truth, grid=None, margins=(0.1, 0.15, 0.2)):
    from . import config as C
    from .decision import decide_for_s1
    grid = grid or C.THRESHOLD_GRID
    best = (0.5, margins[1], -1.0)
    hist = []
    for thr in grid:
        for m in margins:
            pred = {}
            for s in val_s1:
                c = val_cands.get(s, [])
                sc = val_scores.get(s, [])
                pred[s] = set(decide_for_s1(list(c), list(sc), None, thr, m))
            f = macro_f05(truth, pred)
            hist.append((thr, m, f))
            if f > best[2]:
                best = (thr, m, f)
    hist.sort(key=lambda x: -x[2])
    return {"threshold": best[0], "margin": best[1], "f05": best[2], "history": hist}
