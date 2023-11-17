import argparse

from locomoset.models.classes import FineTuningConfig
from locomoset.models.train import run_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model given a training config."
    )
    parser.add_argument("configfile", help="Path to config file")
    args = parser.parse_args()
    config = FineTuningConfig.read_yaml(args.configfile)
    run_config(config)


if __name__ == "__main__":
    main()
