"""
Entry point for running experiments per model for each metric stated in config.
"""
import argparse

import datasets

from locomoset.metrics.classes import MetricConfig
from locomoset.metrics.experiment import run_config

# TODO - IMPLEMENT THIS PROPERLY
datasets.disable_caching()


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    config = MetricConfig.read_yaml(args.configfile)

    run_config(config)


if __name__ == "__main__":
    main()
