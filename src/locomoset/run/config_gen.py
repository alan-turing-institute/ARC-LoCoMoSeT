"""
    Generate config files for indidividual runs of the MetricExperimentClass
"""

import argparse

from locomoset.metrics.classes import TopLevelMetricConfig
from locomoset.models.classes import TopLevelFineTuningConfig
from locomoset.run.config_classes import TopLevelConfig


def gen_configs(config: TopLevelConfig, alt_config: TopLevelConfig | None = None):
    """Generate and save config files for config classes run from a top level
    config. Will generate two sets of configs with the same date time marker if a
    second config file is given.
    """
    config.generate_sub_configs()
    config.save_sub_configs()
    if config.use_bask:
        config.create_bask_job_script(len(config.sub_configs))

    if alt_config is not None:
        alt_config.config_gen_dtime = config.config_gen_dtime
        alt_config.generate_sub_configs()
        alt_config.save_sub_configs()
        if alt_config.use_bask:
            alt_config.create_bask_job_script(len(alt_config.sub_configs))


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics scans with various parameter values"
    )
    parser.add_argument("configfile", help="Path to config file")
    parser.add_argument(
        "-t",
        "--type",
        help="Which type of config to generate, metric or train, leave blank for both",
    )
    args = parser.parse_args()

    alt_config = None
    if args.type == "metrics":
        config = TopLevelMetricConfig.read_yaml(args.configfile)
        config.config_type = "metrics"
    elif args.type == "train":
        config = TopLevelFineTuningConfig.read_yaml(args.configfile)
        config.config_type = "train"
    else:
        config = TopLevelMetricConfig.read_yaml(args.configfile)
        config.config_type = "metrics"
        alt_config = TopLevelFineTuningConfig.read_yaml(args.configfile)
        alt_config.config_type = "train"

    gen_configs(config, alt_config)


if __name__ == "__main__":
    main()
