"""Small shared helpers (no heavy deps)."""
import hashlib
import os
import random
import time


def fmt_hms(sec):
    sec = max(0, int(sec))
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def seed_all(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def count_data_rows(path):
    """Fast data-row count (excl header) via binary newline count."""
    try:
        n = 0
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                n += blk.count(b"\n")
        return max(0, n - 1)
    except Exception:
        return 0


def sha1_of_file(path, max_bytes=1 << 20):
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def log_progress(done, total, t0, label):
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    if total:
        pct = 100.0 * done / total
        eta = (total - done) / rate if rate > 0 else -1
        print("[%s] %d/%d (%.1f%%) | %.0f rows/s | elapsed %s | ETA %s"
              % (label, done, total, pct, rate, fmt_hms(el), fmt_hms(eta) if eta >= 0 else "--"),
              flush=True)
    else:
        print("[%s] %d rows | %.0f rows/s | elapsed %s" % (label, done, rate, fmt_hms(el)), flush=True)
