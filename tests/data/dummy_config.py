"""
Load config dict used across the dummy generation scripts.
"""
import json

with open("dummy_config.json") as f:
    config = json.load(f)
