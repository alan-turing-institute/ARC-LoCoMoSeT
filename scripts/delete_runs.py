import wandb

api = wandb.Api()

runs = api.runs(
    path="turing-arc/locomoset",
    filters={"group": "pcuenq/oxford-pets_20231110-083306-254553", "jobType": "train"},
)
print(f"Found {len(runs)} runs")
print("Deleting checkpoints")
for run in runs:
    print("=" * 20)
    print(run.name)
    print("=" * 20)
    for artifact in run.logged_artifacts():
        if artifact.type == "model" and "latest" not in artifact.aliases:
            print("DELETE", artifact.type, artifact.name)
            artifact.delete(delete_aliases=True)
        else:
            print("KEEP", artifact.type, artifact.name)
