# LoCoMoSeT: Low-Cost Model Selection for Transformers

**Status:** Work in progress (one out of four work packages completed)

This project aims to contrast and compare multiple low-cost metrics for selecting pre-trained models for fine-tuning on a downstream task. In particular, we intend to suggest which of the current metrics gives the most accurate prediction - prior to fine-tuning - of how well a transformer-architecture based pre-trained model for image classification will perform once fine tuned for a new data set.

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

This script will compute metrics scores for all permutations of the model names, no. images, random seeds, and metric names specified.

### Save plots

Currently implemented as separate scripts only, not in the main locomoset package.

To make a plot of metric scores vs. no of images and actual performance vs. metric scores:

```bash
cd scripts
python plot_vs_actual.py <PATH_TO_RESULTS_DIR> --scores_file <path_to_scores_file> --n_samples <n_samples>
python plot_vs_samples.py <PATH_TO_RESULTS_DIR>
```

Where `<PATH_TO_RESULTS_DIR>` is the path to a directory containing JSON files produced by a metric scan.

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
