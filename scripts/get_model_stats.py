import pandas as pd
import tqdm
import wandb

api = wandb.Api()
results = []
for run in tqdm.tqdm(api.runs("turing-arc/locomoset")):
    try:
        job_type = run.config["locomoset"]["wandb_args"]["job_type"]
        runtime = run.summary["_runtime"]
        n_gpus = run.metadata["gpu_count"]
        dataset_name = run.config["locomoset"]["dataset_name"]
        model_name = run.config["locomoset"]["model_name"]
        group = run.group
        if job_type == "metrics":
            n_samples = run.summary["n_samples"]
            epochs = None
            max_epochs = None
        else:
            dataset_samples = {  # size of train
                "pcuenq/oxford-pets": 0.75 * 7349,
                "huggan/wikiart": 0.75 * 81444,
                "aharley/rvl_cdip": 320000,
            }
            n_samples = run.summary["train/epoch"] * dataset_samples[dataset_name]
            epochs = run.summary["train/epoch"]
            max_epochs = run.config["locomoset"]["training_args"]["num_train_epochs"]

        results.append(
            {
                "path": run.path,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "job_type": job_type,
                "group": group,
                "n_gpus": n_gpus,
                "runtime": runtime,
                "n_samples": n_samples,
                "epochs": epochs,
                "max_epochs": max_epochs,
            }
        )
    except Exception:
        print(run.group, run.name, "failed")

df = pd.DataFrame(results)
df.to_csv("model_stats.csv", index=False)
