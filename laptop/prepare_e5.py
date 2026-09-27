"""Laptop: prepare E5 transfer package (thin wrapper over scripts/create_transfer_e5.py).

Usage (repo root):
  python laptop/prepare_e5.py --data-root student_resource/dataset --out transfer_e5 --split test
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.create_transfer_e5 import main

if __name__ == "__main__":
    main()
