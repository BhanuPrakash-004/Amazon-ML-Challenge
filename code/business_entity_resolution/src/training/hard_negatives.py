"""Hard-negative mining (Sec 17). Same-name/core/house/postal/rare/ANN/address
but wrong target. pos + 2-4x hard negatives (no trivial random flood)."""
import random


def mine(s1rec, merged, trec_by_id, positives, neg_per_pos=3, seed=42):
    rnd = random.Random(seed + hash(s1rec.get("name_norm", "")) % 1000)
    pos = set(positives)
    cands = [t for t in merged if t in trec_by_id and t not in pos]
    if not cands:
        return []
    def key(t):
        b = trec_by_id[t]
        s = 0
        if b.get("name_norm") == s1rec.get("name_norm"):
            s += 5
        if b.get("name_core") == s1rec.get("name_core"):
            s += 4
        if b.get("house_number") and b["house_number"] == s1rec.get("house_number"):
            s += 3
        if b.get("postal_candidate") and b["postal_candidate"] == s1rec.get("postal_candidate"):
            s += 2
        return s
    cands.sort(key=key, reverse=True)
    want = min(len(cands), max(neg_per_pos * max(len(pos), 1), 2))
    hard = cands[:want]
    rest = [t for t in cands[want:] if rnd.random() < 0.1][: max(0, want - len(hard))]
    return hard + rest
