"""Predict entry-point (full scale v2). Usage: python run_predict_v2.py"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code", "business_entity_resolution"))
os.chdir(ROOT)

from src.predict_v2 import main

if __name__ == "__main__":
    main()
