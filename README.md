# LoCoMoSeT: Low-Cost Model Selection for Transformers

This project aims to contrast and compare multiple low-cost metrics for selecting pre-trained models for fine-tuning on a downstream task. In particular, we intend to suggest which of the current metrics gives the most accurate prediction - prior to fine-tuning - of how well a transformer-architecture based pre-trained model for image classification will perform once fine tuned for a new data set.

## Links

**TODO**

## Installation

1. Clone this repository

2. Install with `pip`:

   ```bash
   pip install .
   ```

## Usage

**TODO**

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

- To add dependencies to the poetry environment:

   ```bash
   poetry add <PACKAGE_NAME>
   ```

  See [the poetry documentation](https://python-poetry.org/docs/basic-usage/#specifying-dependencies) for more details on specifying dependencies.

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

- Your source code files should go in the `src/todo_packagename` directory (with `todo_packagename` replaced with the name of your package). These will be available as a python package, i.e. you can do `from todo_mypackagename.myfile import myfunction` etc.

- Add tests (in files with names like `test_*.py` and with functions with names starting `test_*`) the `tests/` directory.
