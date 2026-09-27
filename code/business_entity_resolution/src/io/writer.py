"""Output writers (Sec 30). Exactly one row per test S1, matches ⊆ candidates."""
import csv


def _sort_ids(ids):
    return sorted(set(ids), key=lambda x: (0 if x.startswith("S2-") else 1, x))


def write_matching(path, s1_ids_in_order, pred_dict):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1 in s1_ids_in_order:
            ids = [x for x in _sort_ids(pred_dict.get(s1, set()))
                   if x.startswith("S2-") or x.startswith("S3-")]
            w.writerow([s1, ",".join(ids)])


def write_candidates(path, s1_ids_in_order, cand_dict):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1 in s1_ids_in_order:
            v = cand_dict.get(s1, [])
            ids = _sort_ids(v) if not isinstance(v, str) else \
                _sort_ids([x.strip() for x in v.split(",") if x.strip()])
            ids = [x for x in ids if x.startswith("S2-") or x.startswith("S3-")]
            w.writerow([s1, ",".join(ids)])
