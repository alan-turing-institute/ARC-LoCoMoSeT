# LoCoMoSeT: Low-Cost Model Selection for Transformers

This project aims to contrast and compare multiple low-cost metrics for selecting pre-trained models for fine-tuning on a downstream task. In particular, we intend to suggest which of the current metrics gives the most accurate prediction - prior to fine-tuning - of how well a transformer-architecture based pre-trained model for image classification will perform once fine tuned for a new data set.

## Status

This project is a work in progress under active development. The work packages are:

- WP1: Implement one metric and sanity check it on ImageNet ✅
- WP2: Implement further metrics and tune metric hyperparameters ⏳
- WP3: Fine-tune models and test the metrics on other datasets
- WP4: Write a report

## Links

**TODO**

## Installation

1. Clone this repository

2. Install with `pip` (or see the developer setup below for using a poetry environment instead):

   ```bash
   pip install .
   ```

3. If using LogME, NCE or LEEP clone the following repository <https://github.com/thuml/LogME> into `src`:

   ```bash
   git clone https://github.com/thuml/LogME.git src/locomoset/LogME
   ```

## Usage

### Download ImageNet

ImageNet-1k is gated so you need to login with a HuggingFace token to download it (they're under <https://huggingface.co/settings/tokens> in your account settings). Log in to the HuggingFace CLI:

```bash
huggingface-cli login
```

Once you've done this, head on over to <https://huggingface.co/datasets/imagenet-1k>, read the terms and conditions and if happy to proceed agree to them. Then run:

```bash
python -c "import datasets; datasets.load_dataset('imagenet-1k')"
```

But note this will take a long time (hours).

### Config Files

To run either metrics or training in LoCoMoSeT (see below), metrics and/or training config files are required. Examples are given in [example_metrics_config.yaml](/configs/example_metrics_config.yaml) and [example_train_config.yaml](/configs/example_train_config.yaml) for metrics and training configs respectively.

Both kinds of config should contain:

- `caches`: Contains two entries (`models` and `datasets`) showing where to cache HuggingFace models and datasets respectively
- `dataset_name`: Name of the dataset on HuggingFace
- `model_name`: Name of the model to be used on HuggingFace
- `random_state`: Seed for random number generation
- `run_name`: Name for the wandb run
- `save_dir`: Directory in which to save results
- `use_wandb`: Set to `true` to log results to wandb

If `use_wandb` is `true`, then under `wandb_args` the following shoud additionally be specified:

- `entity`: Wandb entity name
- `project`: Wandb project name
- `job_type`: Job type to group wandb runs with. Should be `metrics` or `train`
- `log_model`: How to handle model logging in wandb

Metrics configs should additionally contain:

- `dataset_split`: A single dataset split or list of splits (`train`, `val`, or `test`) over which the metric should be computed.
- `local_save`: Set to `true` to locally save a copy of the results
- `metrics`: A list of metrics implemented in src/locomost/metrics to be used
- `n_samples`: Number of images from the dataset to compute the metrics with

Train configs should additionally contain the following nested under `dataset_args`:

- `train_split`: Name of the data split to train on
- `val_split`: Name of the data split to evaluate on. If the same as `train_split`, the `train_split` will itself be randomly split for training and evaluation

Along with several further arguments nested under `training_args`:

- `eval_steps`: Steps between each evaluation
- `evaluation_strategy`: HuggingFace evaluation strategy
- `logging_strategy`: HuggingFace logging strategy
- `num_train_epochs`: Number of epochs to train model for
- `output_dir`: Directory to store outputs in
- `overwrite_output_dir`: Whether to overwrite the output directory
- `save_strategy`: HuggingFace saving strategy
- `use_mps_device`: Whether to use MPS

Since in practice you will likely wish to run many jobs together, LoCoMoSeT provides support for top-level configs from which you can generate many lower-level configs. Top-level configs can contain parameters for metrics scans, model training, or both. Broadly, this should contain the arguments laid out above, with some additional arguments and changes.

The additional arguments are:

- `config_dir`: Location to store subconfigs
- `slurm_template_name`: Name of the slurm template to be used. If set to `null`, it will be picked from [src/locomoset/config](/src/locomoset/config)
- `use_bask`: Set to `True` if you wish to run the jobs on baskerville (HPC used in our research - for uses outside the Turing, this means a slurm script will be generated alongside the configs)

If `use_bask` is `True`, then you should include the following additional arguments nested under `bask`. They should be further nested under `train` and/or `metrics` as required:

- `job_name`: Baskerville job name
- `walltime`: Maximum runtime for the Baskerville job. Format is dd-hh:mm:ss
- `node_number`: Number of nodes to use
- `gpu_number`: Number of GPUs to use
- `cpu_per_gpu`: Number of CPUs per GPU

The changes are:

- `models`: Replaces `model`, contains a list of HuggingFace model names
- `dataset_names`: Replaces `dataset_name`, contains a list of HuggingFace dataset
- `random_states`: Replaces `random_state`, contains a list of seeds to generate scripts over.

To generate configs from the top level config, run

```bash
locomoset_gen_configs <top_level_config_file_path>
```

This will generate training and/or metrics configs across all combinations of model, dataset, and random state. `locomoset_gen_configs` will automatically detect whether your top-level config contains training and/or metrics-related arguments and will generate both kinds of config accordingly.

### Run a metric scan

With the environment activated (`poetry shell`):

```bash
locomoset_run_metrics <config_file_path>
```

For an example config file see [configs/config_wp1.yaml](configs/config_example.yaml).

This script will compute metrics scores for a given model, dataset, and random state.

### Train a model

With the environment activated (`poetry shell`):

```bash
locomoset_run_train <config_file_path>
```

This script will train a model for a given model name, dataset, and random state.

### Save plots

#### Metric Scores vs. No. of Images

This plot shows how the metric values (y-axis) change with the number of images (samples) used to compute them (x-axis). Ideally the metric should converge to some fixed value which does not change much after the number of images is increased. The number of images it takes to get a reliable performance prediction determines how long it takes to compute the metric, so metrics that converge after seeing fewer images are preferable.

To make a plot of metric scores vs. actual fine-tuned performance performance:

```bash
locomoset_plot_vs_samples <PATH_TO_RESULTS_DIR>
```

Where  `<PATH_TO_RESULTS_DIR>` is the path to a directory containing JSON files produced by a metric scan (see above).

You can also run `locomoset_plot_vs_samples --help` to see the arguments.

#### Metric Scores vs. Fine-Tuned Performance

This plot shows the predicted performance score for each model from one of the low-cost metrics on the x-axis, and the actual fine-tuned performance of the models on that dataset on the y-axis. A high quality metric should have high correlation between its score (which is meant to reflect the transferability of the model to the new dataset) and the actual fine-tuned model performance.

To make this plot:

```bash
locomoset_plot_vs_actual <PATH_TO_RESULTS_DIR> --scores_file <path_to_scores_file> --n_samples <n_samples>
```

Where:

- `<PATH_TO_RESULTS_DIR>` is the path to a directory containing JSON files produced by a metric scan (see above).
- `<path_to_scores_file>` is a mapping between model names and fine-tuned performance on ImageNet-1k, such as the file [configs/scores_imagenet1k.yaml](configs/scores_imagenet1k.yaml) in this repo.
- `<n_samples>` sets the no. of samples (images) the metric was computed with to plot. Usually a metrics scan includes results with different numbers of images, but for this plot the different metrics should be compared using a fixed no. of images only.

You can also run `locomoset_plot_vs_actual --help` to see the arguments.

## Development

### Developer Setup

1. Install dependencies with Poetry

   ```bash
   poetry install
   ```

2. If using LogME, NCE or LEEP clone the following repository <https://github.com/thuml/LogME> into `src`:

   ```bash
   git clone https://github.com/thuml/LogME.git src/locomoset/LogME
   ```

3. Install pre-commit hooks:

   ```bash
   poetry run pre-commit install --install-hooks
   ```

### Common Commands/Tasks

- To run commands in the poetry virtual environment (in a terminal), either:
  - Prefix the command you want to run with `poetry run`
    - e.g. `poetry run python myscript.py`
  - Enter the virtual environment with `poetry shell` and then run commands as normal
    - then exit the virtual environment with `exit`

- To run tests:

  ```bash
  poetry run pytest
  ```

- To run linters:
  - If you have setup pre-commit `flake8`, `black`, and `isort` will run automatically before making commits
  - Or you can run them manually:

    ```bash
    poetry run black .
    poetry run isort .
    poetry run flake8
    ```
