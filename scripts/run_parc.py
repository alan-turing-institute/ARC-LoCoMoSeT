# Run a PARC experiment
# Author: Edmund Dable-Heath
"""
    Run a comparative experiment for a given pair (dataset, model) for a number of
    different proce set sizes, random seeds and feature reduction levels. Results saved
    to file path of form results/results_YYYYMMDD-HHMMSS.json by default.
"""

import argparse

import yaml


def run_parc(config: dict):
    """Run comparative PARC experiment for given pair (dataset, model). Results saved
    to file path of form results/results_YYYYMMDD-HHMMSS.json by default.

    Args:
        config (dict): _description_
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PARC metrics")
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)
