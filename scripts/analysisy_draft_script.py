"""
    A collection of helper functions for analysis.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr


def pull_metric_and_train_data(group_name: str) -> Tuple(dict, dict):
    """Pull metric and train data from weights and biases, turning it into the requisite
    dict structure:

    train_results = {
        n_samples: {
            model_name: validation_score
        }
    }

    metric_results = {
        metric: {
            n_samples: {
                mode_name: metric_score
            }
        }
            OR IF METRIC = n_pars,
        n_pars: {
            model_name: metric score
        }
    }

    """

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    train_runs = api.runs(
        path="turing-arc/locomoset",
        filters={"group": group_name, "jobType": "train"},
    )

    summary_train, config_train, name_train = [], [], []
    for run in train_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_train.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_train.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_train.append(run.name)

    train_df = pd.DataFrame(
        {"summary": summary_train, "config": config_train, "name": name_train}
    )

    train_res = {}

    for _, row in train_df.iterrows():
        if row.summary.get("eval/accuracy") is not None:
            if train_res.get(row.config["locomoset"]["n_samples"]) is not None:
                train_res[row.config["locomoset"]["n_samples"]][
                    row.config["locomoset"].get("model_name")
                ] = row.summary.get("eval/accuracy")
            else:
                train_res[row.config["locomoset"]["n_samples"]] = {}
                train_res[row.config["locomoset"]["n_samples"]][
                    row.config["locomoset"].get("model_name")
                ] = row.summary.get("eval/accuracy")

    metric_runs = api.runs(
        path="turing-arc/locomoset",
        filters={
            "group": group_name,
            "jobType": "metrics",
        },
    )

    summary_metrics, config_metrics, name_metrics = [], [], []
    for run in metric_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_metrics.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_metrics.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_metrics.append(run.name)

    metrics_df = pd.DataFrame(
        {"summary": summary_metrics, "config": config_metrics, "name": name_metrics}
    )

    metric_results = {}

    for _, row in metrics_df.iterrows():
        for met in row.summary["metric_scores"].keys():
            if metric_results.get(met) is not None:
                if met == "n_pars":
                    metric_results[met][row.summary["model_name"]] = row.summary[
                        "metric_scores"
                    ][met]["score"]
                else:
                    if metric_results[met].get(row.summary["n_samples"]) is not None:
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
                    if metric_results[met].get(row.summary["n_samples"]) is not None:
                        metric_results[met][row.summary["n_samples"]][
                            row.summary["model_name"]
                        ] = row.summary["metric_scores"][met]["score"]
                    else:
                        metric_results[met][row.summary["n_samples"]] = {}
                        metric_results[met][row.summary["n_samples"]][
                            row.summary["model_name"]
                        ] = row.summary["metric_scores"][met]["score"]

    return train_res, metric_results


def plotter(
    val_acc: dict,
    metric_scores: dict,
    metric: str,
    n_samples: int | None = None,
    remove_outliers: bool = False,
    outlier_threshold: float = 0.5,
    ranked: bool = False,
    xy: bool = False,
    dataset_name: str | None = None,
):
    """Plot validation accuracy vs. metric scores, with or without outlier, with ranking
    or not"""
    # if metric == "n_pars":
    #     n_samples = None
    # else:
    #     if n_samples is None:
    #         n_samples = 30000

    if ranked:
        sorted_val_acc = {
            k: v
            for k, v in sorted(val_acc.items(), key=lambda item: item[1], reverse=True)
        }
        if remove_outliers:
            for model in val_acc.keys():
                if val_acc[model] < outlier_threshold:
                    sorted_val_acc.pop(model)
            new_sorted_val_acc = {
                k: v
                for k, v in sorted(
                    sorted_val_acc.items(), key=lambda item: item[1], reverse=True
                )
            }
            sorted_val_acc = new_sorted_val_acc
        v_rank = {k: idx + 1 for idx, k in enumerate(sorted_val_acc.keys())}
        # print(v_rank)

        if n_samples is not None:
            sorted_met = {
                k: v
                for k, v in sorted(
                    metric_scores[metric][n_samples].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
        else:
            sorted_met = {
                k: v
                for k, v in sorted(
                    metric_scores[metric].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
        if remove_outliers:
            for model in val_acc.keys():
                if val_acc[model] < outlier_threshold:
                    sorted_met.pop(model)
            new_sorted_met = {
                k: v
                for k, v in sorted(
                    sorted_met.items(), key=lambda item: item[1], reverse=True
                )
            }
            sorted_met = new_sorted_met
        m_rank = {k: idx + 1 for idx, k in enumerate(sorted_met.keys())}
        # print(m_rank)

        m_scores = []
        v_acc = []
        for model in v_rank.keys():
            m_scores.append(m_rank[model])
            v_acc.append(v_rank[model])
        # print(m_scores)
        # print(v_acc)

    else:
        m_scores = []
        v_acc = []
        for model in val_acc.keys():
            if remove_outliers:
                if val_acc[model] < outlier_threshold:
                    continue

            v_acc.append(val_acc[model])

            if n_samples is not None:
                m_scores.append(metric_scores[metric][n_samples][model])
            else:
                m_scores.append(metric_scores[metric][model])

    fig, ax = plt.subplots(1)
    ax.scatter(m_scores, v_acc)
    corr = np.round(spearmanr(m_scores, v_acc)[0], 2)

    if ranked:
        ax.set_ylabel("validation accuracy ranking")
        ax.set_xlabel(f"{metric} score ranking")
    else:
        ax.set_ylabel("validation accuracy")
        ax.set_xlabel(f"{metric} score")

    if xy:
        x = np.linspace(
            min(min(m_scores), min(v_acc)), max(max(m_scores), max(v_acc)), 10000
        )
        y = x
        ax.plot(x, y)

    title = f"Val Acc vs. {metric} score, correlation: {corr}"
    if dataset_name is not None:
        title = dataset_name + "; " + title
    if n_samples is not None:
        title += f", {n_samples} samples"
    if remove_outliers:
        title += ", no outliers"
    if ranked:
        title += ", ranked"

    fig.suptitle(title)

    plt.show()


def plot_all(vals, mets, met, ranked=False, xy=False, dataset_name=None):
    for n in vals.keys():
        if met == "n_pars":
            n = None
        plotter(
            val_acc=vals[n],
            metric_scores=mets,
            metric=met,
            n_samples=n,
            ranked=ranked,
            xy=xy,
            dataset_name=dataset_name,
        )
