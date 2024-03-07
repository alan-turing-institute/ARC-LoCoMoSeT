"""
    Utility functions for pulling and parsing data.
"""

import json
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import wandb
import yaml
from matplotlib.axes import Axes


def load_results(file_paths: list[str], n_samples=None) -> (list[dict], str):
    """Load results files from metric scans.

    Args:
        file_paths: Paths to results JSON files, or a single directory containing them.
        n_samples: If set, only load results for this many samples.

    Returns:
        (Loaded results JSON files, path to directory to save results)
    """
    # If the user passes a directory, glob for all JSON files in that directory.
    if len(file_paths) == 1 and os.path.isdir(file_paths[0]):
        save_dir = file_paths[0]
        file_paths = glob(os.path.join(file_paths[0], "*.json"))
    else:
        save_dir = "."

    results = []
    for rf in file_paths:
        with open(rf, "r") as f:
            this_file = json.load(f)
            if n_samples is not None and this_file["n_samples"] != n_samples:
                continue
            results.append(this_file)
    return results, save_dir


def load_results_wandb(group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from weights and biases into dataframes for metric and train
    experiments for plotting and analysis.

    Args:
        group_name: baskerville/wandb run group name.

    Returns:
        tuple of dataframes of results, (metric_df, training_df)
    """
    # Instantiate wandb api
    api = wandb.Api()

    # Load training data
    train_runs = api.runs(
        path="turing-arc/locomoset",
        filters={"group": group_name, "jobType": "train"},
    )

    summary_train, config_train, name_train = [], [], []
    for run in train_runs:
        summary_train.append(run.summary._json_dict)
        config_train.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_train.append(run.name)

    train_df = pd.DataFrame(
        {"summary": summary_train, "config": config_train, "name": name_train}
    )

    # Load metric data
    metric_runs = api.runs(
        path="turing-arc/locomoset",
        filters={
            "group": group_name,
            "jobType": "metrics",
        },
    )

    summary_metrics, config_metrics, name_metrics = [], [], []
    for run in metric_runs:
        summary_metrics.append(run.summary._json_dict)
        config_metrics.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_metrics.append(run.name)

    metrics_df = pd.DataFrame(
        {"summary": summary_metrics, "config": config_metrics, "name": name_metrics}
    )

    return metrics_df, train_df


def parse_results_dataframes(
    metrics_df: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse a results dataframe into a dictionaries for plotting and analysis.

    Args:
        metrics_df: dataframe containing metric experiment results. Defaults to
                            None.
        train_df: dataframe containing training experiment results. Defaults to
                            None.

    Returns:
        tuple of dictionaries containing metric and training results for analysis and
        plotting, (metric_results, training_results), with structures:

        metric_results = {
            task_specific_metric: {n_samples: {model: score}},
            task_agnostic_metric: {model: score}
        }

        train_results = {
            model: validation_accuracy
        }
    """
    train_results = {}
    if train_df is not None:
        for _, row in train_df.iterrows():
            if row.summary.get("eval/accuracy") is not None:
                train_results[
                    row.config["locomoset"].get("model_name")
                ] = row.summary.get("eval/accuracy")

    metric_results = {}

    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            for met in row.summary["metric_scores"].keys():
                if metric_results.get(met) is not None:
                    if met == "n_pars":
                        metric_results[met][row.summary["model_name"]] = row.summary[
                            "metric_scores"
                        ][met]["score"]
                    else:
                        if (
                            metric_results[met].get(row.summary["n_samples"])
                            is not None
                        ):
                            metric_results[met][row.summary["n_samples"]][
                                row.summary["model_name"]
                            ] = row.summary["metric_scores"][met]["score"]
                        else:
                            metric_results[met][row.summary["n_samples"]] = {}
                            metric_results[met][row.summary["n_samples"]][
                                row.summary["model_name"]
                            ] = row.summary["metric_scores"][met]["score"]
                else:
                    metric_results[met] = {}
                    if met == "n_pars":
                        metric_results[met][row.summary["model_name"]] = row.summary[
                            "metric_scores"
                        ][met]["score"]
                    else:
                        if (
                            metric_results[met].get(row.summary["n_samples"])
                            is not None
                        ):
                            metric_results[met][row.summary["n_samples"]][
                                row.summary["model_name"]
                            ] = row.summary["metric_scores"][met]["score"]
                        else:
                            metric_results[met][row.summary["n_samples"]] = {}
                            metric_results[met][row.summary["n_samples"]][
                                row.summary["model_name"]
                            ] = row.summary["metric_scores"][met]["score"]

    return metric_results, train_results


def load_imagenet_acc(path: str) -> dict[str, float]:
    """Load the imagenet accuracy for a collection of models from a yaml.

    Args:
        path: path to imagenet accuracy yaml

    Returns:
        dictionary of imagenet accuracy, {model: validation_accuracy}
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_results(results: list[dict], actuals: dict | None = None) -> dict[str, dict]:
    """Parse results into a dict containing metric scores, the no. of images
    (samples) used to compute them, and optionally the model's actual fine-tuned
    performance for each metric present in the input results list.

    Args:
        results: Loaded results JSON files.
        actuals: Loaded mapping from model names to fine-tuned performance scores.

    Returns:
        Dict of extracted results for plotting.
    """
    parsed_results = {}
    keys_to_add = ["scores", "n_samples", "actuals"]

    for r in results:
        metric = r["metric"]
        model = r["model_name"]
        if metric not in parsed_results:
            parsed_results[metric] = {}
            for k in keys_to_add:
                parsed_results[metric][k] = {}
        if model not in parsed_results[metric]["scores"]:
            for k in keys_to_add:
                parsed_results[metric][k][model] = []

        parsed_results[metric]["scores"][model].append(r["result"]["score"])
        parsed_results[metric]["n_samples"][model].append(r["n_samples"])
        if actuals is not None:
            parsed_results[metric]["actuals"][model].append(actuals[model])

    return parsed_results


def plot_results(
    metric_scores: dict[str, list[float]],
    other_scores: dict[str, list[int]],
    metric_axis: str,
    metric_label: str,
    other_label: str,
    title: str,
    save_path: str,
    log_scale: bool = False,
    # show_legend: bool = False,
    **scatter_args,
) -> Axes:
    """Make a plot of metric scores vs. another quantity.

    Args:
        metric_scores: Dict of model_name: list of metric scores.
        other_scores: Dict of model_name: other score to plot metric scores against.
        metric_axis: Whether to plot metric scores on the "x" or "y" axis.
        metric_label: Axis label for metric scores axies.
        other_label: Axis label for other scores axis.
        title: Plot title.
        save_path: Path where plot will be saved.
        **scatter_args: Additional keyword arguments to pass to plt.scatter.

    Returns:
        Plot axes.
    """
    fig, ax = plt.subplots(1, 1)
    for model in metric_scores:
        if metric_axis == "x":
            ax.scatter(
                metric_scores[model], other_scores[model], label=model, **scatter_args
            )
        elif metric_axis == "y":
            ax.scatter(
                other_scores[model], metric_scores[model], label=model, **scatter_args
            )
        else:
            raise ValueError(f"{metric_axis} must be 'x' or 'y'.")

    # if show_legend:
    #     ax.legend()

    if metric_axis == "x":
        ax.set_xlabel(metric_label)
        ax.set_ylabel(other_label)
        if log_scale:
            ax.set_yscale("log")
    else:
        ax.set_xlabel(other_label)
        ax.set_ylabel(metric_label)
        if log_scale:
            ax.set_xscale("log")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Saved figure to {save_path}")

    return ax
