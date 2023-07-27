"""
    Script to generate config files based on pre-config yaml files containing the
    models, datasets and parameters.
"""
import itertools as it

import yaml


def make_config(config_dict: dict):
    """Make a yaml file for the corresponding config set up with filename:

    <metric>_<dataset>_<model>.yaml

    Args:
        dict: dictionary containing the config details.
    """
    filename1 = f"{config_dict['metric']}_{config_dict['dataset_name']}"
    filename2 = f"_{config_dict['model_name']}.yaml"
    with open(filename1 + filename2, "w") as file:
        yaml.safe_dump(config_dict, file)


def main():
    with open("config/pre_config_lists.yaml", "r") as file:
        pre_config_lists = yaml.safe_load(file)

    keys, values = zip(*pre_config_lists.items())
    _ = [make_config(dict(zip(keys, v))) for v in it.product(*values)]


if __name__ == "__main__":
    main()
