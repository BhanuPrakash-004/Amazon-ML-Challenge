"""Tokenization + rarity tables (Sec 7). Separate name/addr frequencies."""
import math
import re
from collections import Counter

GENERIC = {"company", "limited", "road", "street", "services", "india",
           "united", "states", "private", "enterprises", "traders"}


def tokens(s):
    return str(s).split() if s else []


def build_freq(name_norm_lists, addr_norm_lists):
    fn, fa = Counter(), Counter()
    for t in name_norm_lists:
        fn.update(set(t))
    for t in addr_norm_lists:
        fa.update(set(t))
    return fn, fa


def idf(n, df):
    return math.log(n / (1.0 + df))


def rare_tokens(toks, freq, cap=300, max_toks=12):
    scored = sorted(set(toks), key=lambda t: freq.get(t, 0))
    out = [t for t in scored if freq.get(t, 0) <= cap][:max_toks]
    return out
