"""
    Generate config files for indidividual runs of the MetricExperimentClass
"""

import argparse

from locomoset.run.config_classes import TopLevelMetricConfig


def gen_configs(config: TopLevelMetricConfig):
    """Generate and save config files for MetricExperimentClass run from a top level
    config
    """
    config.generate_sub_configs()
    config.save_sub_configs()


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    config = TopLevelMetricConfig.read_yaml(args.configfile)

    gen_configs(config)


if __name__ == "__main__":
    main()
