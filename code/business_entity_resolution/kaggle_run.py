"""Kaggle 3h end-to-end runner (2xT4, 30GB RAM, offline business data).

Public dataset (input-only):
  /kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset
Fallbacks: $DATA_ROOT, student_resource/dataset, dataset.

Stages with budget (Sec 3):
  0-30m  load/normalize/det-index | 20-60m ANN (GPU0=S2, GPU1=S3) |
  50-90m train candidates+LightGBM | 80-160m test candidates+scoring |
  150-175m post/output | 175-180m validate.
Safety margin 5min; CPU-FAISS fallback; never queries internet for business data.

Usage (Kaggle):
  !python -m src.main --mode full --K 22   # from code/business_entity_resolution
  # or:
  !python kaggle/run_kaggle.py --K 22
"""
import argparse
import os
import sys
import time

CANDIDATES = [
    "/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset",
    "/kaggle/input/amazon-ml-challenge-2026/dataset",
    os.environ.get("DATA_ROOT", ""),
    "student_resource/dataset",
    "../../student_resource/dataset",
    "dataset",
]


def resolve_data():
    for p in CANDIDATES:
        if p and os.path.isdir(os.path.join(p, "train")) and \
           os.path.isdir(os.path.join(p, "test")):
            return os.path.abspath(p)
    # repo-relative search
    cur = os.path.abspath(os.getcwd())
    for _ in range(5):
        for sub in ("student_resource/dataset", "dataset"):
            q = os.path.join(cur, sub)
            if os.path.isdir(os.path.join(q, "train")):
                return q
        cur = os.path.dirname(cur)
    raise FileNotFoundError("dataset/train+test not found. Attach public dataset as Input.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=22)
    ap.add_argument("--batch-size", type=int, default=150000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu-ids", default="0,1")
    ap.add_argument("--no-ann", action="store_true")
    ap.add_argument("--experiment-only", action="store_true")
    ap.add_argument("--train-sample", type=int, default=250000)
    a = ap.parse_args(argv)
    t0 = time.time()
    data = resolve_data()
    print("[kaggle] data=%s" % data, flush=True)
    try:
        import torch
        print("[kaggle] cuda=%s n=%d" % (torch.cuda.is_available(),
                                        torch.cuda.device_count()), flush=True)
    except Exception as e:
        print("[kaggle] torch: %s" % str(e)[:150], flush=True)
    # file: <repo>/code/business_entity_resolution/kaggle_run.py
    here = os.path.abspath(__file__)
    pkg = os.path.dirname(here)  # <repo>/code/business_entity_resolution
    repo = os.path.dirname(os.path.dirname(pkg))  # <repo>
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    os.chdir(pkg)
    args_output = os.path.join(repo, "output")
    from src.main import main as pipe_main
    args = ["--train-path", os.path.join(data, "train"),
            "--test-path", os.path.join(data, "test"),
            "--output-path", args_output,
            "--model-path", os.path.join(pkg, "models"),
            "--K", str(a.K), "--batch-size", str(a.batch_size),
            "--workers", str(a.workers), "--gpu-ids", a.gpu_ids,
            "--train-sample", str(a.train_sample)]
    if a.no_ann:
        args.append("--no-ann")
    if a.experiment_only:
        args += ["--experiment"]
        return pipe_main(args)
    # default: full train+predict within budget
    rc = pipe_main(["--mode", "full"] + args)
    print("[kaggle] total %.0fs" % (time.time() - t0), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
