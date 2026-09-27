"""Laptop: train Stage-1 matcher (wrapper preserving V2 baseline + V3 adaptive).

V2 baseline (conservative, K=100 rare-token):
  python run_train_v2.py --sample-s1 100000 --val-s1 20000
V3 adaptive (multi-block union + hard negatives + adaptive 8-15):
  cd code/business_entity_resolution && python -m src.run_train --no-e5 --no-reranker

This wrapper just forwards to src.train_pipeline with adaptive defaults.

Usage:
  python laptop/train_stage1.py --no-e5 --no-reranker
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
os.chdir(ROOT)

from src.train_pipeline import main

if __name__ == "__main__":
    main()
