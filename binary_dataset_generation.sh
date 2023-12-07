#!/bin/bash
#SBATCH --account vjgo8416-locomoset
#SBATCH --qos turing
#SBATCH --job-name binary_dataset_generation
#SBATCH --time 0-12:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --output ./slurm_logs/locomoset_binary_dataset_gen-%j.out

# Load required modules here (pip etc.)
module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# This should be edited by the setup script to point in the
# right direction for the configs and run call
CONFIGSPATH=configs/binary_dataset.yaml
RUN_CALL=locomoset_gen_bin_datasets

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH=/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/locomosetenv
conda activate ${CONDA_ENV_PATH}

# Run script
$RUN_CALL ${CONFIGSPATH}
