"""
Load config dict used across the dummy generation scripts.
"""
import yaml

with open("dummy_config.yaml") as f:
    config = yaml.safe_load(f)
