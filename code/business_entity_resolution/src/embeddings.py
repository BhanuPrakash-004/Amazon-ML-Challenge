"""Local E5 embedding inference (multilingual-e5-small, MIT).

- E5 format: query-prefixed S1, passage-prefixed S2/S3.
- Mixed precision, sharded output, auto OOM recovery (batch/=2, min 8).
- DataParallel when >1 GPU helps; single-GPU default. CPU fallback.
- Lazy imports: module imports fine without torch/transformers.
"""
import os

import numpy as np


def _l2norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return (a / n).astype(np.float32)


def embed_texts(texts, model_name=None, device="cpu", batch=96, max_len=256,
                is_query=False, fp16=True):
    """Embed a list of strings. Returns float32 L2-normalized array."""
    from . import config as C
    model_name = model_name or C.E5_MODEL_NAME
    prefix = "query: " if is_query else "passage: "
    texts = [(prefix + (t or "")) for t in texts]
    # Lazy heavy imports
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=False)
    mdl.eval()
    try:
        if device.startswith("cuda") and torch.cuda.device_count() > 1 and len(texts) > 5000:
            mdl = torch.nn.DataParallel(mdl)
    except Exception:
        pass
    mdl.to(device)
    use_amp = device.startswith("cuda")
    out = []
    bs = max(8, int(batch))
    i = 0
    while i < len(texts):
        cur = min(bs, len(texts) - i)
        try:
            enc = tok(texts[i:i + cur], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    h = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.float().cpu().numpy())
            i += cur
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 8:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                bs = max(8, bs // 2)
                continue
            raise
    if not out:
        return np.zeros((0, 384), dtype=np.float32)
    return _l2norm(np.concatenate(out, axis=0))
