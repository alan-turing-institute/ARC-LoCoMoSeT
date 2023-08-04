import json
from pathlib import Path

with open(Path(__file__, "..", "imagenet1k_scores.json").resolve()) as f:
    imagenet1k_scores = json.load(f)
