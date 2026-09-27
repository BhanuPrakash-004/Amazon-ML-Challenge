"""Probability calibration (Sec 21). Isotonic or Platt on held-out S1 val.
Use calibrated score only if it improves val F0.5."""
import numpy as np


def fit_calibrator(yva, pva, method="isotonic"):
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(pva, yva)
            return ("isotonic", ir)
        lr = LogisticRegression()
        lr.fit(np.asarray(pva).reshape(-1, 1), yva)
        return ("platt", lr)
    except Exception:
        return ("none", None)


def apply(cal, p):
    kind, m = cal
    p = np.asarray(p, dtype=float)
    if kind == "isotonic":
        return np.asarray(m.predict(p), dtype=float)
    if kind == "platt":
        return np.asarray(m.predict_proba(p.reshape(-1, 1))[:, 1], dtype=float)
    return p
