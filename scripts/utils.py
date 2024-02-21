"""
    A collection of helper functions for analysis.
"""

import os

import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy.stats import bootstrap, spearmanr

# 1) Fns for pulling data --------------------------------------------------------------


def define_dataset_combinations(dataset_name):
    m = constants.MAX_SIZES[dataset_name]
    sample_list = [s for s in constants.SAMPLES if s < m]
    sample_list.append(m)
    return sample_list


def _pull_runs(
    api: wandb.Api,
    job_type: str,
    dataset_name: str,
    n_samples: int,
    n_metric_samples: int | None,
) -> wandb.Api.runs:
    filters = {
        "jobType": job_type,
        "created_at": {"$gt": "2024-01-01"},
        "State": "finished",
        "config.locomoset.dataset_name": dataset_name,
        "config.locomoset.n_samples": n_samples,
    }
    if n_metric_samples is not None:
        filters["config.locomoset.metrics_samples"] = n_metric_samples
    return api.runs(
        "turing-arc/locomoset",
        filters=filters,
    )


def pull_runs(
    api: wandb.Api,
    job_type: str,
    dataset_name: str,
    n_samples: int,
    n_metric_samples: None,
) -> list[wandb.Api.run]:
    runs = _pull_runs(api, job_type, dataset_name, n_samples, n_metric_samples)
    runs = [run for run in runs if len(run.summary.keys()) > 0]
    if job_type == "metrics":
        runs = [run for run in runs if ("metrics_samples" in run.summary.keys())]
    return runs


def _unpack_run(
    run: wandb.Api.run,
    type: str,
) -> dict:
    run_dict = {
        # "run_name": run.name,
        # "group": run.group,
        "dataset_name": run.config["locomoset"]["dataset_name"],
        "model_name": run.config["locomoset"]["model_name"],
        "n_samples": run.config["locomoset"]["n_samples"],
    }
    if type == "train":
        run_dict = {
            **run_dict,
            "test_accuracy": run.summary["test/accuracy"],
            "train_runtime": run.summary["train/train_runtime"],
        }
    if type == "metrics":
        run_dict = {
            **run_dict,
            "n_metric_samples": run.summary["metrics_samples"],
            **run.summary["metric_scores"],
            "inference_times": run.summary["inference_times"]["features"],
        }
    if type == "preproc":
        run_dict = {**run_dict, "preproc_times": run.summary["times"]["total"]}
    return run_dict


def unpack_runs(
    runs: list[wandb.Api.run],
    type: str,
) -> pd.DataFrame:
    df = pd.DataFrame([_unpack_run(run, type) for run in runs])
    col_list = ["dataset_name", "model_name", "n_samples"]
    if type == "metrics":
        col_list.append("n_metric_samples")
    df = df.loc[df[col_list].duplicated() == False]  # noqa: E712
    return df


def get_data(api, job_type, datasets):
    runs = []
    for d in datasets:
        samples = define_dataset_combinations(d)
        for s in samples:
            if job_type == "train":
                runs = runs + pull_runs(api, job_type, d, s, None)
            if job_type == "preproc":
                runs = runs + pull_runs(api, job_type, d, s, None)
            if job_type == "metrics":
                for ms in samples:
                    runs = runs + pull_runs(api, job_type, d, s, ms)
    data = unpack_runs(runs, job_type)
    return data


# 2) Fns for making correlations -------------------------------------------------------


def bootstrap_corr(x, y, corr_fn):
    def get_corr_coef(x, y):
        return corr_fn(x, y)[0]

    mean, p = corr_fn(x, y)
    ci = bootstrap(
        (x, y), get_corr_coef, vectorized=False, paired=True
    ).confidence_interval
    return mean, p, ci


# 3) Fns for outputting tables ---------------------------------------------------------


def select_main_results(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function used for selecting main best case scenario results: largest dataset
    size, use full dataset for metric

    Args:
        table: _description_
    """
    return table.loc[
        table["n_metric_samples"].isin(set(constants.MAX_SIZES.values())), :
    ]


def select_sample_results(table, dataset_name):
    """
    Function used for selecting second set of results

    Args:
        table: _description_
    """
    return table.loc[
        (table["n_samples"] == table["n_metric_samples"])
        & (table["dataset"] == dataset_name),
        :,
    ]


# 4) Fns for LaTeX Tables --------------------------------------------------------------


def mean_pick_fn(x):
    if x.iloc[0] == "Kendall's Tau":
        return (
            np.argmax(
                [
                    x.iloc[1]["mean"],
                    x.iloc[2]["mean"],
                    x.iloc[3]["mean"],
                    x.iloc[4]["mean"],
                ]
            )
            + 1
        )
    else:
        return (
            np.argmin(
                [
                    x.iloc[1]["mean"],
                    x.iloc[2]["mean"],
                    x.iloc[3]["mean"],
                    x.iloc[4]["mean"],
                ]
            )
            + 1
        )


def latexify(
    table: pd.DataFrame,
    labels: list[str],
    t_name: str,
    t_label: str,
    type: str,
):
    # Dimensions of table
    rows, cols = table.shape
    columns = table.columns

    # Obtain best metric list for highlighting in bold
    if type == "corr":
        best_metric = (
            table.apply(
                lambda x: np.argmax(
                    [x.iloc[1][0], x.iloc[2][0], x.iloc[3][0], x.iloc[4][0]]
                ),
                axis=1,
            ).values
            + 1
        )
    if type == "regret":
        best_value = table.apply(
            lambda x: np.min([x.iloc[1], x.iloc[2], x.iloc[3], x.iloc[4]]), axis=1
        ).values
        best_metric = [
            np.where(table.iloc[x] == val) for x, val in enumerate(best_value)
        ]
    if type == "mean":
        best_metric = table.apply(lambda x: mean_pick_fn(x), axis=1).values

    # Make first part of table
    t_head = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{t_name}}}\n"
        f"\\label{{tab:{t_label}}}\n"
        "\\setlength\\tabcolsep{1.5pt}\n"
        f"\\begin{{tabular}}{{c|{''.join(['c' for _ in range(1,cols)])}}}\n"
    )

    # Add column headings
    for i in range(cols):
        if i != (cols - 1):
            t_head = t_head + f"\\textbf{{{labels[i]}}}" + " & "
        if i == (cols - 1):
            t_head = t_head + f"\\textbf{{{labels[i]}}}" + " \\\\\n\\hline\n"

    # Table content
    t_content = ""
    for i in range(rows):
        for j in range(cols):
            # Append value
            if j == 0:
                if not isinstance(table.loc[i, columns[j]], str):
                    val = str(table.loc[i, columns[j]])
                else:
                    val = table.loc[i, columns[j]]
                t_content = t_content + val
            if j > 0:
                val = table.loc[i, columns[j]]
                if type == "corr":
                    if best_metric[i] == j:
                        t_content = (
                            t_content
                            + "\\makecell{"
                            + f"\\textbf{{{round(val[0], 2):.2f}}} \\\\[0pt] "
                            + f"({round(val[1][0], 2):.2f}, {round(val[1][1], 2):.2f})"
                            + "}"
                        )
                    else:
                        t_content = (
                            t_content
                            + "\\makecell{"
                            + f"{round(val[0], 2):.2f} \\\\[0pt] "
                            + f"({round(val[1][0], 2):.2f}, {round(val[1][1], 2):.2f})"
                            + "}"
                        )
                if type == "regret":
                    if np.isin(j, best_metric[i]):
                        t_content = (
                            t_content
                            + "\\makecell{"
                            + f"\\textbf{{{round(val, 2):.2f}}}"
                            + "}"
                        )
                    else:
                        t_content = (
                            t_content + "\\makecell{" + f"{round(val, 2):.2f}" + "}"
                        )
                if type == "mean":
                    if best_metric[i] == j:
                        t_content = (
                            t_content
                            + "\\makecell{"
                            + f"\\textbf{{{round(val['mean'], 2):.2f}}} \\\\[0pt] "
                            + f"({round(val['lower'], 2):.2f}, "
                            + f"{round(val['lower'], 2):.2f})"
                            + "}"
                        )
                    else:
                        t_content = (
                            t_content
                            + "\\makecell{"
                            + f"{round(val['mean'], 2):.2f} \\\\[0pt] "
                            + f"({round(val['lower'], 2):.2f}, "
                            + f"{round(val['lower'], 2):.2f})"
                            + "}"
                        )

            # Append either & or \\ for next line
            if j != (cols - 1):
                t_content = t_content + " & "
            if j == (cols - 1):
                t_content = t_content + " \\\\\n"
                if i < (rows - 1):
                    t_content = t_content + "\\hline\n"

    # Table tail
    if type == "corr":
        t_tail = (
            "\\end{tabular}\n"
            "\\caption*{\\\\\\textit{Values presented with bootstrapped 95\\% "
            "confidence intervals.\\\\\nLargest value for each row presented in bold.}}"
            "\n\\end{table}"
        )
    if type == "regret":
        t_tail = (
            "\\end{tabular}\n"
            "\\caption*{\\\\\\textit{Largest value for each row presented in bold.}}"
            "\n\\end{table}"
        )
    if type == "mean":
        t_tail = (
            "\\end{tabular}\n"
            "\\caption*{\\\\\\textit{Values presented with 95\\% confidence "
            "intervals.\\\\\nLargest value for each row presented in bold.}}"
            "\n\\end{table}"
        )

    # Make table
    tex_table = t_head + t_content + t_tail

    # Save
    with open(os.path.join("tables", t_label + ".tex"), "w") as f:
        f.write(tex_table)


# 5) Fns for outputting plots ----------------------------------------------------------


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
