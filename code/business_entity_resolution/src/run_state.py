"""run_state.json + 9h time-budget monitoring (spec sections 28-31).

Lightweight helper reusing checkpoint.py manifests. Every stage calls
update(); if elapsed + ETA exceeds budget, caller switches
FULL -> FAST (disable BGE) -> SAFE BASELINE (lexical + best threshold).
"""
import json
import os
import time

BUDGET_S = 9 * 3600
WARN_S = int(8 * 3600)
MODES = ("FULL", "FAST", "SAFE_BASELINE")


def _path(output_root):
    return os.path.join(output_root, "run_state.json")


def init(output_root, mode="FULL"):
    os.makedirs(output_root, exist_ok=True)
    st = {"mode": mode, "stage": "init", "status": "running",
          "t0": time.time(), "rows_done": 0, "rows_total": 0,
          "elapsed_seconds": 0, "estimated_remaining_seconds": 0}
    with open(_path(output_root), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    return st


def update(output_root, stage, rows_done, rows_total, t0=None, status="running"):
    now = time.time()
    el = now - (t0 or now)
    rate = (rows_done / el) if el > 0 and rows_done else 0.0
    rem = ((rows_total - rows_done) / rate) if rate > 0 else 0.0
    st = {"stage": stage, "status": status, "rows_done": int(rows_done),
          "rows_total": int(rows_total), "elapsed_seconds": float(el),
          "estimated_remaining_seconds": float(rem)}
    try:
        with open(_path(output_root), encoding="utf-8") as f:
            old = json.load(f)
        st["mode"] = old.get("mode", "FULL")
        st["t0"] = old.get("t0", t0 or now)
        total_el = now - st["t0"]
        st["total_elapsed_seconds"] = float(total_el)
        if total_el + rem > BUDGET_S:
            st["budget_warning"] = "TIME LIMIT WARNING: projected over 9h; consider FAST/SAFE_BASELINE"
    except Exception:
        pass
    os.makedirs(output_root, exist_ok=True)
    with open(_path(output_root), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    if st.get("budget_warning"):
        print("TIME LIMIT WARNING at stage %s: %s" % (stage, st["budget_warning"]), flush=True)
    return st


def pick_mode(elapsed_s, e5_done=True, bge_done=False):
    """Fallback ladder: FULL -> FAST -> SAFE_BASELINE by elapsed time."""
    if elapsed_s > 7 * 3600:
        return "SAFE_BASELINE"
    if elapsed_s > 5.5 * 3600 and not bge_done:
        return "FAST"
    return "FULL"
