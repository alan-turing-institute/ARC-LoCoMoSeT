#!/bin/bash
#SBATCH --account vjgo8416-locomoset
#SBATCH --qos turing
#SBATCH --job-name locomoset_metric_experiment
#SBATCH --time 0-0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --array=1-10
#SBATCH --output ./slurm_train_logs/locomoset_metric_experiment-%j.out

# Load required modules here (pip etc.)
module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# This should be edited by the setup script to point in the right direction.
CONFIGSPATH=/PATH/TO/CONFIGS

# Define the path to your Conda environment (modify as appropriate)
/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/locomosetenv
conda activate ${CONDA_ENV_PATH}

# Run script
python3 src/locomoset/run/run_metrics.py ${CONFIGSPATH}/config_${SLURM_ARRAY_TASK_ID}.yaml