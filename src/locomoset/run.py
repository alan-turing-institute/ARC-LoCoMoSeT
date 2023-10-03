"""
    Entry point for running experiments per model for each metric stated in config.
"""

import argparse

import yaml


def run(**config):
    """Run comparative metric experiment for a given pair (model, dataset) for stated
    metrics"""


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    with open(args.configfile, "r") as f:
        config = yaml.safe_load(f)

    run(config)


if __name__ == "__main__":
    main()
