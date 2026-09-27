"""Optional E5 fine-tuning (InfoNCE, 1-2 epochs, mixed precision, grad accum).

Uses only competition training labels. Caller must REBUILD indexes after
fine-tuning and compare pretrained vs fine-tuned on validation.
"""
import os


def finetune_e5(train_triplets, model_name=None, out_dir=None, epochs=1, device="cpu"):
    """train_triplets: list of (anchor, positive, [negatives]). Returns out path or None."""
    print("[E5-TRAIN] start (triplets=%d)" % len(train_triplets), flush=True)
    try:
        import torch  # noqa
        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader
    except Exception as e:
        print("[E5-TRAIN] skipped (missing deps): %s" % e, flush=True)
        return None
    from . import config as C
    model_name = model_name or C.E5_MODEL_NAME
    mdl = SentenceTransformer(model_name, device=device)
    ex = []
    for a, p, negs in train_triplets:
        for n in (negs or [])[:8]:
            ex.append(InputExample(texts=["query: " + a, "passage: " + p, "passage: " + n]))
    if not ex:
        print("[E5-TRAIN] no examples; skipping.", flush=True)
        return None
    loader = DataLoader(ex, batch_size=8, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(mdl)
    mdl.fit(train_objectives=[(loader, loss)], epochs=epochs, show_progress_bar=False)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        mdl.save(out_dir)
        print("[E5-TRAIN] saved %s" % out_dir, flush=True)
        return out_dir
    return model_name
