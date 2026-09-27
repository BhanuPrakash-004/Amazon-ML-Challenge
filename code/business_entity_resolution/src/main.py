"""CLI entrypoint (Sec 33.19): --train-path --test-path --output-path --K
--batch-size --workers --gpu-ids. Prints runtime for every major stage.
"""
import argparse
import os
import sys
import time

from . import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(description="Amazon Business Entity Resolution (K=22 default)")
    ap.add_argument("--train-path", default=None)
    ap.add_argument("--test-path", default=None)
    ap.add_argument("--output-path", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--K", type=int, default=C.K)
    ap.add_argument("--batch-size", type=int, default=C.S1_CHUNK)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu-ids", default="0,1")
    ap.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    ap.add_argument("--no-ann", action="store_true")
    ap.add_argument("--train-sample", type=int, default=C.TRAIN_S1_SAMPLE)
    ap.add_argument("--experiment", action="store_true",
                    help="run Blocking A/B/C/D experiment then exit")
    a = ap.parse_args(argv)
    gpu_ids = [int(x) for x in str(a.gpu_ids).split(",") if x.strip() != ""]
    t0 = time.time()
    if a.experiment:
        from .experiment_blocking import run as exp_run
        exp_run(train_dir=a.train_path or C.TRAIN_DIR, K=a.K, no_ann=a.no_ann)
        return 0
    if a.mode in ("train", "full"):
        from .pipeline.train import run as train_run
        train_run(train_dir=a.train_path or C.TRAIN_DIR,
                  model_dir=a.model_path or C.MODEL_DIR, K=a.K,
                  s1_sample=a.train_sample, no_ann=a.no_ann)
    if a.mode in ("predict", "full"):
        from .pipeline.predict import run as pred_run
        pred_run(test_dir=a.test_path or C.TEST_DIR,
                 model_dir=a.model_path or C.MODEL_DIR,
                 output_dir=a.output_path or C.OUTPUT_DIR, K=a.K,
                 batch=a.batch_size, workers=a.workers, gpu_ids=gpu_ids,
                 no_ann=a.no_ann)
        # Sec 31: mandatory validator
        _validate(a.output_path or C.OUTPUT_DIR, a.test_path or C.TEST_DIR)
    print("[main] total %.1fs" % (time.time() - t0), flush=True)
    return 0


def _validate(output_dir, test_dir):
    import subprocess
    for cand in (os.path.join(C.ROOT, "student_resource", "utils", "validate_submission.py"),
                 os.path.join(os.path.dirname(__file__), "..", "..", "..",
                              "student_resource", "utils", "validate_submission.py"),
                 "student_resource/utils/validate_submission.py",
                 "utils/validate_submission.py"):
        if os.path.isfile(cand):
            v = cand
            break
    else:
        print("[main] validator not found, skipping", flush=True)
        return
    m = os.path.join(output_dir, "matching_results.tsv")
    c = os.path.join(output_dir, "candidate_pairs.tsv")
    r = subprocess.run([sys.executable, v, "--matching", m, "--candidate", c,
                        "--test-dir", test_dir])
    if r.returncode != 0:
        print("[main] VALIDATOR FAILED", flush=True)
    else:
        print("[main] validator PASS", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
