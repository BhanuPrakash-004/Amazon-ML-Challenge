"""Scalable rare-token blocking (v2) for 10M-scale data.

Design (laptop-friendly, 16GB RAM, 12 cores):
- Per-country partitioning (open-set: any label works; France handled automatically).
- Two passes, streaming, chunked:
    Pass 1: DF counting (Counter) over S23 tokens (normalized name+address words).
    Pass 2: postings only for RARE tokens (DF <= df_cap). Common tokens
            ("limited","road",...) are never indexed -> memory stays <1GB/country.
- Query: for each S1 take up to `n_tok` rarest tokens (must be rare, else fallback
  to rarest available), concat posting arrays (numpy), sort+accumulate IDF weights
  (numba), keep top-K. Plus exact-name and PIN boosters.
- No sklearn fit, no dense matrix, vocab-free -> handles unseen France tokens.

Memory: rare postings ~15M ints/country (~60MB as int32) + dict overhead.
Speed: ~0.05-0.2ms/query (numpy+numba), ~10min for 3.9M queries on 12 cores.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import numba

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

from .normalize import normalize_name, normalize_address, normalize_country, core_name


def norm_tokens(name, addr):
    return (normalize_name(name) + " " + normalize_address(addr)).split()


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized frame for pair building (name_norm/addr_norm/country_norm)."""
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["addr_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].map(normalize_country)
    return df


def build_df_counter(s23_df_iter, token_col="tok"):
    """Pass 1: count DF over an iterable of token lists (or DataFrame chunks)."""
    df = Counter()
    for toks in s23_df_iter:
        df.update(set(toks))
    return df


def build_rare_index(s23_ids, s23_toks, df, df_cap):
    """Pass 2: dict rare_token -> np int32 array of positions; idf dict."""
    post = defaultdict(list)
    for pos, toks in enumerate(s23_toks):
        for t in set(toks):
            if df.get(t, 0) <= df_cap:
                post[t].append(pos)
    post_np = {}
    for t, lst in post.items():
        post_np[t] = np.asarray(lst, dtype=np.int32)
    # idf over full N (for scoring)
    # N passed implicitly via df? compute outside; here use placeholder, caller rescales
    return post_np


if HAS_NUMBA:
    @numba.njit
    def _accum_sorted(sorted_ids, sorted_ws, out_ids, out_scores):
        n = len(sorted_ids)
        w = 0
        i = 0
        while i < n:
            j = i + 1
            s = sorted_ws[i]
            while j < n and sorted_ids[j] == sorted_ids[i]:
                s += sorted_ws[j]
                j += 1
            out_ids[w] = sorted_ids[i]
            out_scores[w] = s
            w += 1
            i = j
        return w
else:
    _accum_sorted = None


def topk_for_query(qtoks, df, idf, post_np, K, n_tok=8):
    """Return list[(pos, score)] top-K. Pure numpy/numba, no Python per-doc loop."""
    uniq = set(qtoks)
    if not uniq:
        return []
    ranked = sorted(uniq, key=lambda t: idf.get(t, -1), reverse=True)
    # prefer rare tokens actually indexed; allow fallback to rarest even if common
    cands = [t for t in ranked if t in post_np][:n_tok]
    if not cands:
        # no rare token indexed: fallback to rarest 3 (may be common, large lists -> cap)
        cands = ranked[:3]
    arrs, warrs = [], []
    for t in cands:
        a = post_np.get(t)
        if a is None or len(a) == 0:
            continue
        # cap huge posting lists (common-token fallback) to first 20000 to bound time
        if len(a) > 20000:
            a = a[:20000]
        arrs.append(a)
        warrs.append(np.full(len(a), idf.get(t, 0.0), dtype=np.float32))
    if not arrs:
        return []
    all_ids = np.concatenate(arrs)
    all_ws = np.concatenate(warrs)
    order = np.argsort(all_ids, kind="mergesort")
    sids = all_ids[order]
    sws = all_ws[order]
    if HAS_NUMBA:
        out_ids = np.empty(len(sids), dtype=np.int32)
        out_sc = np.empty(len(sids), dtype=np.float32)
        w = _accum_sorted(sids, sws, out_ids, out_sc)
        out_ids, out_sc = out_ids[:w], out_sc[:w]
    else:
        # numpy fallback via unique
        uniq_ids, idx = np.unique(sids, return_index=True)
        # accumulate via bincount on positions? use pandas groupby fallback (slower)
        import pandas as pd

        dfm = pd.DataFrame({"id": sids, "w": sws})
        g = dfm.groupby("id", sort=False)["w"].sum()
        out_ids = g.index.to_numpy(dtype=np.int32)
        out_sc = g.to_numpy(dtype=np.float32)
    if len(out_ids) <= K:
        top_idx = np.argsort(-out_sc)
    else:
        part = np.argpartition(-out_sc, K - 1)[:K]
        top_idx = part[np.argsort(-out_sc[part])]
    return [(int(out_ids[i]), float(out_sc[i])) for i in top_idx]


class RareTokenBlocker:
    """Per-country blocker. Fit on S23 DataFrame (entity_id, business_name,
    business_address, country). Query with S1 DataFrame."""

    def __init__(self, df_cap=2000, top_k=100, n_tok=12):
        self.df_cap = df_cap
        self.top_k = top_k
        self.n_tok = n_tok
        self.per_country = {}  # c -> dict(post_np, idf, ids, names, pins...)

    def fit(self, s23: pd.DataFrame):
        s23 = s23.copy()
        s23["business_name"] = s23["business_name"].fillna("").astype(str)
        s23["business_address"] = s23["business_address"].fillna("").astype(str)
        s23["country"] = s23["country"].fillna("").astype(str)
        s23["nn"] = s23["business_name"].map(normalize_name)
        s23["aa"] = s23["business_address"].map(normalize_address)
        s23["cc"] = s23["country"].map(normalize_country)
        s23["tok"] = (s23["nn"] + " " + s23["aa"]).str.split()
        for c, grp in s23.groupby("cc"):
            ids = grp["entity_id"].tolist()
            toks = grp["tok"].tolist()
            nns = grp["nn"].tolist()
            N = len(grp)
            df = Counter()
            for t in toks:
                df.update(set(t))
            idf = {t: math.log(N / (1 + d)) for t, d in df.items()}
            post_np = build_rare_index(ids, toks, df, self.df_cap)
            # exact-name + core-name + pin lookups for boosters
            name_to_pos = defaultdict(list)
            core_to_pos = defaultdict(list)
            for i, nm in enumerate(nns):
                if nm:
                    name_to_pos[nm].append(i)
                    cn = core_name(nm)
                    if cn:
                        core_to_pos[cn].append(i)
            pin_to_pos = defaultdict(list)
            pat = re.compile(r"\d{5,6}")
            for i, a in enumerate(grp["business_address"].astype(str)):
                for p in set(pat.findall(a)):
                    pin_to_pos[p].append(i)
            self.per_country[c] = {
                "ids": ids, "toks": toks, "nns": nns,
                "df": df, "idf": idf, "post": post_np,
                "name_to_pos": dict(name_to_pos), "core_to_pos": dict(core_to_pos),
                "pin_to_pos": dict(pin_to_pos),
            }
        return self

    def query(self, s1: pd.DataFrame):
        """Multi-pass retrieval + cheap re-rank (fixes full-10M recall drop).

        Passes per S1 (same per-country index):
          A: full toks (name+address), top-2K
          B: name-only toks, top-K
          C: address-only toks, top-K
        Union (max IDF score), re-rank top-2K by 0.7*IDF_norm + 0.3*name-Jaccard,
        keep top-K. Boosters: exact name + core-name + PIN (up to 10 extra).
        """
        s1 = s1.copy()
        s1["business_name"] = s1["business_name"].fillna("").astype(str)
        s1["business_address"] = s1["business_address"].fillna("").astype(str)
        s1["country"] = s1["country"].fillna("").astype(str)
        s1["nn"] = s1["business_name"].map(normalize_name)
        s1["aa"] = s1["business_address"].map(normalize_address)
        s1["cc"] = s1["country"].map(normalize_country)
        s1["tok"] = (s1["nn"] + " " + s1["aa"]).str.split()
        s1["ntok"] = s1["nn"].str.split()
        s1["atok"] = s1["aa"].str.split()
        pat = re.compile(r"\d{5,6}")
        out_rows = []
        total_q = len(s1)
        import time as _time
        t_q = _time.time()
        for _qi, (_, row) in enumerate(s1.iterrows(), 1):
            c = row["cc"]
            blk = self.per_country.get(c)
            if blk is None:
                out_rows.append((row["entity_id"], []))
                continue
            K = self.top_k
            # multi-pass union (max score per doc)
            scored = {}
            for qt, kk in ((row["tok"], 2 * K), (row["ntok"], K), (row["atok"], K)):
                for p, s in topk_for_query(list(qt), blk["df"], blk["idf"], blk["post"], kk, self.n_tok):
                    if s > scored.get(p, 0.0):
                        scored[p] = s
            # cheap re-rank: name-Jaccard blend (uses stored nns, no extra I/O)
            if scored:
                qset = set(row["nn"].split())
                mx = max(scored.values()) or 1.0
                reranked = []
                for p, s in scored.items():
                    bset = set(blk["nns"][p].split())
                    j = (len(qset & bset) / len(qset | bset)) if (qset or bset) else 1.0
                    reranked.append((p, 0.7 * (s / mx) + 0.3 * j))
                reranked.sort(key=lambda x: -x[1])
                cand_pos = [p for p, _ in reranked[:K]]
                scored = {p: s for p, s in reranked}
            else:
                cand_pos = []
            # boosters: exact name + core-name + pin (up to 10 extra)
            boost = []
            for j in blk["name_to_pos"].get(row["nn"], []):
                if j not in scored and j not in boost:
                    boost.append(j)
            cn = core_name(row["nn"])
            for j in blk.get("core_to_pos", {}).get(cn, []):
                if j not in scored and j not in boost:
                    boost.append(j)
                if len(boost) >= 10:
                    break
            for p in set(pat.findall(str(row["business_address"]))):
                for j in blk["pin_to_pos"].get(p, []):
                    if j not in scored and j not in boost:
                        boost.append(j)
                    if len(boost) >= 10:
                        break
            for j in boost[:10]:
                if j not in cand_pos:
                    cand_pos.append(j)
            cand_ids = [blk["ids"][j] for j in cand_pos]
            seen, ded = set(), []
            for x in cand_ids:
                if x not in seen:
                    seen.add(x)
                    ded.append(x)
            out_rows.append((row["entity_id"], ded))
            if _qi % 10000 == 0 or _qi == total_q:
                _el = _time.time() - t_q
                _rate = _qi / _el if _el > 0 else 0
                _eta = (total_q - _qi) / _rate if _rate > 0 else -1
                print("[query] %d/%d (%.1f%%) | %.0f q/s | elapsed %d:%02d | ETA %s | candidates avg %.1f"
                      % (_qi, total_q, 100.0 * _qi / total_q, _rate, int(_el // 60), int(_el % 60),
                         ("%d:%02d" % (int(_eta // 60), int(_eta % 60)) if _eta >= 0 else "--"),
                         sum(len(x[1]) for x in out_rows) / max(len(out_rows), 1)), flush=True)
        return pd.DataFrame(out_rows, columns=["source1_entity_id", "candidate_entity_ids"])
