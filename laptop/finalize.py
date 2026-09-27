"""Laptop: finalize outputs + validator (wrapper over src.predict_pipeline outputs).

Runs V3 predict (adaptive), copies candidate_pairs_v3 flow if present,
writes run_manifest.json, and runs the official validator.

Usage:
  python laptop/finalize.py --data-root student_resource/dataset \
      --model-root code/business_entity_resolution/models \
      --output-root output
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(cmd=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "student_resource", "dataset"))
    ap.add_argument("--model-root", default=os.path.join(ROOT, "code", "business_entity_resolution", "models"))
    ap.add_argument("--output-root", default=os.path.join(ROOT, "output"))
    a = ap.parse_args(cmd)
    t0 = time.time()
    # V3 predict (adaptive final 8-15, resumable)
    code_dir = os.path.join(ROOT, "code", "business_entity_resolution")
    r = subprocess.run([sys.executable, "-m", "src.run_predict", "--data-root", a.data_root,
                        "--model-root", a.model_root, "--output-root", a.output_root],
                       cwd=code_dir)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    man = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
           "outputs": ["candidate_pairs.tsv", "matching_results.tsv", "candidate_stats.json"]}
    with open(os.path.join(a.output_root, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    v = subprocess.run([sys.executable, os.path.join(ROOT, "student_resource", "utils", "validate_submission.py"),
                        "--matching", os.path.join(a.output_root, "matching_results.tsv"),
                        "--candidate", os.path.join(a.output_root, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(a.data_root, "test")])
    raise SystemExit(v.returncode)


if __name__ == "__main__":
    main()
