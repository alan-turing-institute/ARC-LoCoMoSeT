"""
    From a wandb group name plot collection of graphs of statistics.
"""

import argparse
import os

import numpy as np

from locomoset.plots import utils

# from pathlib import Path


def plot_scores_vs_val(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str
) -> None:
    """Plot metric scores vs. validation accuracy with and without outliers

    Args:
        metric_scores: metric scores dictionary
        val_acc: validation accuracy dictionary
        metric_name: metric name
        title: plot title
    """
    utils.plot_results(
        metric_scores=metric_scores,
        other_scores=val_acc,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="validation accuracy",
        title=title,
    )

    # remove outliers
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    title_without_outliers = title + ", no outliers"
    utils.plot_results(
        metric_scores=metric_scores,
        other_scores=val_acc_no_outliers,
        metric_axis="y",
        metric_label=f"{metric_name} score",
        other_label="validation accuracy",
        title=title_without_outliers,
    )


def plot_scores_vs_val_ranked(
    metric_scores: dict, val_acc: dict, metric_name: str, title: str
) -> None:
    """Plot ranked metric scores vs ranked validation accuracy with and without outliers

    Args:
        metric_scores: metric scores dictionaruy
        val_acc: validation accuracy dictionary
        metric_name: metric name
        title: plot title
    """
    sorted_val_acc = {
        k: v for k, v in sorted(val_acc.items(), key=lambda item: item[1], reverse=True)
    }
    val_acc_rank = {k: idx + 1 for idx, k in enumerate(sorted_val_acc.keys())}
    sorted_metric_scores = {
        k: v
        for k, v in sorted(
            metric_scores.items(), key=lambda item: item[1], reverse=True
        )
    }
    metric_rank = {k: idx + 1 for idx, k in enumerate(sorted_metric_scores.keys())}
    ranked_title = title + ", ranked"
    utils.plot_results(
        metric_scores=metric_rank,
        other_scores=val_acc_rank,
        metric_axis="y",
        metric_label=f"ranked {metric_name} score",
        other_label="ranked validation accuracy",
        title=ranked_title,
    )
    val_acc_no_outliers = {
        k: v
        for k, v in val_acc.items()
        if v > np.mean(list(val_acc.values())) - 3 * np.var(list(val_acc.values()))
    }
    sorted_val_acc_no_outliers = {
        k: v
        for k, v in sorted(
            val_acc_no_outliers.items(), key=lambda item: item[1], reverse=True
        )
    }
    val_acc_rank_no_outliers = {
        k: idx + 1 for idx, k in enumerate(sorted_val_acc_no_outliers.keys())
    }
    metric_scores_no_outliers = {
        model: metric_scores[model] for model in val_acc_no_outliers.keys()
    }
    sorted_metric_scores_no_outliers = {
        k: v
        for k, v in sorted(
            metric_scores_no_outliers.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    metric_rank_no_outliers = {
        k: idx + 1 for idx, k in enumerate(sorted_metric_scores_no_outliers)
    }
    ranked_no_outliers_title = title + ", without outliers, ranked"
    utils.plot_results(
        metric_scores=metric_rank_no_outliers,
        other_scores=val_acc_rank_no_outliers,
        metric_axis="y",
        metric_label=f"ranked {metric_name} score",
        other_label="ranked validation accuracy",
        title=ranked_no_outliers_title,
    )


def generate_plots(
    group_name: str, save_path: str, imagenet_scores_path: str | None = None
) -> None:
    """Generate the following plots for a set of results from weights and biases:

        - Metric scores vs. validation accuracy
        - (optional) Imagenet validation accuracy vs. validation accuracy
        - Ranked Metric scores vs. ranked validation accuracy
        - Ranked Imagenet validation accuracy vs. ranked validation accuracy
        - Metric scores vs. n samples
        - Correlation(metric scores, validation accuracy) vs. n samples
        - All above with outlier models (by validation accuracy score) remove

    Where applicable the metric scores will only use the max number of samples for
    plots.

    Args:
        group_name: weights and biases group name
        save_path: path to save the plots
        imagenet_scores_path (optional): path to imagenet scores, if given the plots
                                         will be made
    """
    metrics_df, train_df = utils.load_results_wandb(group_name)
    metric_res, train_res = utils.parse_results_dataframes(metrics_df, train_df)

    # Metric scores vs validation accuracy
    for metric in metric_res.keys():
        title = f"{metric} score vs validation accuracy"
        if metric == "n_pars":
            metric_scores = metric_res[metric]
        else:
            max_n_samples = max(metric_res[metric].keys())
            metric_scores = metric_res[metric][max_n_samples]
            title += f" n={max_n_samples}"

        plot_scores_vs_val(metric_scores, train_res, metric, title)
        plot_scores_vs_val_ranked(metric_scores, train_res, metric, title)

    # Image net graphs
    if imagenet_scores_path is not None:
        imagenet_scores = utils.load_imagenet_acc(imagenet_scores_path)
        title = "ImageNet validation accuracy vs validation accuracy"
        plot_scores_vs_val(
            imagenet_scores, train_res, metric_name="ImageNet_val_acc", title=title
        )
        plot_scores_vs_val_ranked(
            imagenet_scores, train_res, metric_name="ImageNet_val_acc", title=title
        )


def main() -> None:
    desc1 = "Create a collection of plots of metric scores vs. other quantities from a"
    desc2 = " weights and biases group name"
    parser = argparse.ArgumentParser(description=desc1 + desc2)
    parser.add_argument("wanbd_group_name", help="Weights and biases group name")
    parser.add_argument("imagenet_scores_path", help="Path to ImageNet accuracy scores")
    args = parser.parse_args()
    save_path = f"results/{args.wandb_group_name}"
    os.mkdir(save_path)
    generate_plots(args.wandb_group_name, save_path, args.imagenet_scores_path)
