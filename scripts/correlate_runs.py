from copy import deepcopy

import pandas as pd
import wandb
from scipy.stats import bootstrap, kendalltau, spearmanr
from tqdm import tqdm

api = wandb.Api()


def print_run_correls(dataset_name, n_samples):
    project_name = "turing-arc/locomoset"
    all_filters = {
        "config.locomoset.dataset_name": dataset_name,
        "config.locomoset.n_samples": n_samples,
        "created_at": {"$gt": "2024-01-01T"},
        "State": "finished",
    }
    train_filters = deepcopy(all_filters)
    train_filters["jobType"] = "train"

    metrics_filters = deepcopy(all_filters)
    metrics_filters["jobType"] = "metrics"
    # should be multiple metrics_samples per n_samples value in full analysis
    metrics_filters["config.locomoset.metrics_samples"] = n_samples

    if "/" in dataset_name:
        short_name = dataset_name.split("/")[-1]
    else:
        short_name = dataset_name

    print("======", short_name, "======")

    train_runs = api.runs(project_name, filters=train_filters)
    print(f"{len(train_runs)=}")
    metrics_runs = api.runs(project_name, filters=metrics_filters)
    print(f"{len(metrics_runs)=}")

    train_results = pd.DataFrame(
        [
            {
                "model_name": run.config["locomoset"]["model_name"],
                "test/accuracy": run.summary["test/accuracy"],
            }
            for run in tqdm(train_runs)
        ]
    )

    metrics_results = pd.DataFrame(
        [
            {
                "model_name": run.config["locomoset"]["model_name"],
                "n_pars": run.summary["metric_scores"]["n_pars"]["score"],
                "renggli": run.summary["metric_scores"]["renggli"]["score"],
                "LogME": run.summary["metric_scores"]["LogME"]["score"],
            }
            for run in tqdm(metrics_runs)
        ]
    )

    # take mean if there are multiple runs for the same model
    # THIS IS A QUCK WORKAROUND AND SHOULD BE DEALT WITH PROPERLY IN FUTURE
    train_results = train_results.groupby("model_name").mean()
    metrics_results = metrics_results.groupby("model_name").mean()

    results = train_results.join(metrics_results)
    results.to_csv(f"results_{short_name}.csv")

    def bootstrap_corr(x, y, corr_fn):
        def get_corr_coef(x, y):
            return corr_fn(x, y)[0]

        mean, p = corr_fn(x, y)
        ci = bootstrap(
            (x, y), get_corr_coef, vectorized=False, paired=True
        ).confidence_interval
        return mean, p, ci

    print("Spearman")
    for metric in ["n_pars", "renggli", "LogME"]:
        mean, p, ci = bootstrap_corr(
            results["test/accuracy"], results[metric], spearmanr
        )
        print(f"{metric}: {mean:.2f} ({ci[0]:.2f} - {ci[1]:.2f}) {p=:.2f}")

    print("\nKendall Tau")
    for metric in ["n_pars", "renggli", "LogME"]:
        mean, p, ci = bootstrap_corr(
            results["test/accuracy"], results[metric], kendalltau
        )
        print(f"{metric}: {mean:.2f}  ({ci[0]:.2f} - {ci[1]:.2f}) {p=:.2f}")


if __name__ == "__main__":
    datasets = [
        "/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/binary_datasets/bin_faucet",
        "/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/binary_datasets/bin_watch",
        "/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/binary_datasets/bin_tree",
        "pcuenq/oxford-pets",
        "huggan/wikiart",
        "aharley/rvl_cdip",
    ]
    n_samples = [2303, 2825, 26004, 4803, 50000, 50000]
    for ds, ns in zip(datasets, n_samples):
        print_run_correls(ds, ns)
        print("\n")
