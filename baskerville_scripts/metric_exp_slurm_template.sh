#!/bin/bash
#SBATCH --job-name=myarrayjob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10

# This should be edited by the setup script to point in the right direction.
CONFIGSPATH=/PATH/TO/CONFIGS

# Load required modules here (pip etc.)
module pip

# Run script
python3 src/locomoset/run/run_metrics.py ${CONFIGSPATH}/config_${SLURM_ARRAY_TASK_ID}.yaml
