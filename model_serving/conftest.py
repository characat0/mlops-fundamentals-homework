import os
import sys

# Make `app` importable when tests are collected from the repo root
# (e.g. `pytest model_serving/tests`).
sys.path.insert(0, os.path.dirname(__file__))
