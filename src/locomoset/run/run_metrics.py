"""
Entry point for running experiments per model for each metric stated in config.
"""
import argparse

from locomoset.metrics.experiment import ModelMetricsExperiment
from locomoset.run.config_classes import MetricConfig


def run_config(config: MetricConfig):
    """Run comparative metric experiment for a given pair (model, dataset) for stated
    metrics. Results saved to file path of form results/results_YYYYMMDD-HHMMSS.json by
    default.

    Args:
        config: Loaded configuration dictionary including the following keys:
            - models: a list of HuggingFace model names to experiment with.
            - dataset_name: Name of HuggingFace dataset to use.
            - dataset_split: Dataset split to use.
            - n_samples: List of how many samples (images) to compute the metric with.
            - random_state: List of random seeds to compute the metric with (used for
                subsetting the data and dimensionality reduction).
            - metrics: Which metrics to experiment on.
            - metric_kwargs: dictionary of entries {metric_name: **metric_kwargs}
                        containing parameters for each metric.
            - (Optional) save_dir: Directory to save results, "results" if not set.
    """

    if config.use_wandb:
        config.init_wandb()

    model_experiment = ModelMetricsExperiment(config.to_dict())
    model_experiment.run_experiment()

    if config.local_save:
        model_experiment.save_results()

    if config.use_wandb:
        print(config.wandb_args)
        model_experiment.log_wandb_results()


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
