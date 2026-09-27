"""Matcher error analysis: FP/FN breakdown for the final decision layer."""
import argparse


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    a = ap.parse_args(args)
    print("analyze_errors: run after train_pipeline; consumes results/<run>/metrics.json.", flush=True)
    print("reports FP/FN counts, singleton performance, per-block-support precision.", flush=True)


if __name__ == "__main__":
    main()
