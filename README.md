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

## Usage

### Download ImageNet

ImageNet-1k is gated so you need to login with a HuggingFace token to download it (they're under <https://huggingface.co/settings/tokens> in your account settings). Once you have a token:

```bash
huggingface-cli login
python -c "import datasets; datasets.load_dataset('imagenet-1k')"
```

But note this will take a long time (hours).

### Run a metric scan

With the environment activated (`poetry shell`):

```bash
locomoset_run_metrics <config_file_path>
```

For an example config file see [configs/config_wp1.yaml](configs/config_example.yaml).

This script will compute metrics scores for all permutations of the model names, no. images, random seeds, and metric names specified. Results will be saved to the directory specified in the config file.

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

2. Install pre-commit hooks:

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
