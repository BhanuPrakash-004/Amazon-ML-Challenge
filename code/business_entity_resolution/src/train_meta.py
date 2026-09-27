"""V3 train_meta entry: meta-model over validation-safe base outputs."""
import json
import os

import numpy as np

from .meta_model import train_meta, save_meta, META_COLS


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default=None)
    a = ap.parse_args()
    from . import config as C
    mr = a.model_root or C.MODEL_DIR
    print("train_meta: expects base predictions; see train_pipeline for end-to-end (stub ok).", flush=True)
    print("META_COLS=%s" % META_COLS, flush=True)


if __name__ == "__main__":
    main()
