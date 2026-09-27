"""Semantic / retrieval / rarity / structural features (Sec 18)."""
import numpy as np

from ..preprocessing.normalize import name_toks, addr_toks


def sem_feats(meta, dense_cos=None):
    return {
        "dense_cosine": float(meta.get("ann_score", dense_cos if dense_cos is not None else 0.0)),
        "ann_rank": float(meta.get("ann_rank", 99)),
        "ann_hit": 1.0 if meta.get("hit_ann") else 0.0,
        "shared_block_count": float(meta.get("block_count", 0)),
    }


def rare_feats(a, b, idfn=None, idfa=None):
    ra = set(name_toks(a)) & set(name_toks(b))
    rb = set(addr_toks(a)) & set(addr_toks(b))
    if idfn:
        w = sum(idfn.get(t, 0.0) for t in ra)
    else:
        w = float(len(ra))
    if idfa:
        u = sum(idfa.get(t, 0.0) for t in rb)
    else:
        u = float(len(rb))
    return {"rare_name_overlap": float(w), "rare_address_overlap": float(u)}


def struct_feats(tid, fam_count=1):
    return {
        "source_indicator": 0.0 if tid.startswith("S2-") else 1.0,
        "target_family_indicator": float(min(fam_count, 10)),
        "target_duplicate_count": float(max(fam_count - 1, 0)),
    }


FEATURE_ORDER = [
    "name_exact", "name_core_exact", "name_translit_exact", "name_compact_exact",
    "fuzz_ratio", "wratio", "token_sort_ratio", "token_set_ratio",
    "name_token_jaccard", "name_length_ratio",
    "address_exact", "address_ratio", "address_token_sort_ratio",
    "address_token_set_ratio", "address_token_jaccard", "address_length_ratio",
    "house_exact", "house_normalized_exact", "shared_numeric_count",
    "numeric_jaccard", "postal_exact", "postal_conflict", "house_conflict",
    "numeric_conflict", "dense_cosine", "ann_rank", "ann_hit", "shared_block_count",
    "rare_name_overlap", "rare_address_overlap",
    "source_indicator", "target_family_indicator", "target_duplicate_count",
    "source_address_missing", "target_address_missing", "both_address_present",
]


def row_vector(a, b, tid, meta, fam_count=1, idfn=None, idfa=None):
    from .name_features import name_feats
    from .address_features import addr_feats
    from .numeric_features import num_feats
    d = {}
    d.update(name_feats(a, b))
    d.update(addr_feats(a, b))
    d.update(num_feats(a, b))
    d.update(sem_feats(meta))
    d.update(rare_feats(a, b, idfn, idfa))
    d.update(struct_feats(tid, fam_count))
    return [float(d.get(k, 0.0)) for k in FEATURE_ORDER]
