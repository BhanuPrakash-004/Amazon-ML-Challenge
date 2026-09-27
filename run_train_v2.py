"""Train entry-point (full scale v2). Usage: python run_train_v2.py [--sample-s1 100000 ...]"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
os.chdir(ROOT)

from src.train_v2 import main

if __name__ == "__main__":
    main()
