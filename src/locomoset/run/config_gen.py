"""
    Generate config files for indidividual runs of the MetricExperimentClass
"""

import argparse

from locomoset.metrics.classes import TopLevelMetricConfig

# from locomoset.models.train_config_class import TopLevelFineTuningConfig


def gen_configs(config: TopLevelMetricConfig):
    """Generate and save config files for MetricExperimentClass run from a top level
    config
    """
    config.generate_sub_configs()
    config.save_sub_configs()
    if config.use_bask:
        config.create_bask_job_script(len(config.sub_configs))


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
