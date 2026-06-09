import os
import sys

# Make `src` importable when tests are collected from the repo root
# (e.g. `pytest data_pipeline/tests`).
sys.path.insert(0, os.path.dirname(__file__))
