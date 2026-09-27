"""Checkpoint / resume manifests for multi-session Kaggle execution.

Layout:
  cache/checkpoints/<stage>.json  -> {"shards": {shard_id: {...}}, ...}
Every completed shard records: shard id, start/end row, checksum, status,
model/config hash. On restart completed shards are skipped.
"""
import json
import os
import time


def _path(cache_root, stage):
    return os.path.join(cache_root, "checkpoints", "%s.json" % stage)


def load_manifest(cache_root, stage):
    p = _path(cache_root, stage)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"stage": stage, "shards": {}}


def save_manifest(cache_root, stage, manifest):
    p = _path(cache_root, stage)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    os.replace(tmp, p)


def mark_done(cache_root, stage, shard_id, info=None):
    m = load_manifest(cache_root, stage)
    m["shards"][str(shard_id)] = dict(info or {}, status="done", ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_manifest(cache_root, stage, m)


def is_done(cache_root, stage, shard_id):
    m = load_manifest(cache_root, stage)
    s = m["shards"].get(str(shard_id))
    return bool(s and s.get("status") == "done")


def pending_shards(cache_root, stage, all_shards):
    return [s for s in all_shards if not is_done(cache_root, stage, s)]
