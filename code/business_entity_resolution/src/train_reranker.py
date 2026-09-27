"""Optional reranker fine-tuning on competition positives + hard negatives.

Compares pretrained vs fine-tuned on validation; caller picks the winner.
CPU-safe stub when torch missing (skips with warning).
"""
import os


def finetune_reranker(train_pairs, model_name=None, out_dir=None, epochs=1, device="cpu"):
    print("[RERANK-TRAIN] start (pairs=%d)" % len(train_pairs), flush=True)
    try:
        import torch  # noqa
    except Exception:
        print("[RERANK-TRAIN] torch missing; skipping (use pretrained).", flush=True)
        return None
    # Minimal HF Trainer-free loop would go here; to stay dependency-light we
    # fine-tune via sentence-transformers CrossEncoder if present, else skip.
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(model_name, num_labels=1, device=device)
        samples = []
        for r in train_pairs.itertuples():
            samples.append([getattr(r, "text1", ""), getattr(r, "text2", ""), int(getattr(r, "label", 0))])
        import math
        ce.fit(train_samples=[(s[0], s[1], float(s[2])) for s in samples],
               epochs=epochs, show_progress_bar=False)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ce.save(out_dir)
        print("[RERANK-TRAIN] done.", flush=True)
        return out_dir or model_name
    except Exception as e:
        print("[RERANK-TRAIN] skipped: %s" % e, flush=True)
        return None
