#!/bin/bash
#SBATCH --account vjgo8416-locomoset
#SBATCH --qos turing
#SBATCH --job-name locomoset_metric_experiment
#SBATCH --time 0-0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10

# This should be edited by the setup script to point in the right direction.
CONFIGSPATH=/PATH/TO/CONFIGS

# Load required modules here (pip etc.)
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install .

# Run script
python3 src/locomoset/run/run_metrics.py ${CONFIGSPATH}/config_${SLURM_ARRAY_TASK_ID}.yaml
