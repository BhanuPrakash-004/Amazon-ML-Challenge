"""Laptop: export ambiguous BGE workload (wrapper over scripts/create_transfer_bge.py).

Usage:
  python laptop/export_bge.py --scored output/stage1_scored.parquet --out transfer_bge --top-n 10
"""
import sys
sys.path.insert(0, ".")
from scripts.create_transfer_bge import main

if __name__ == "__main__":
    main()
