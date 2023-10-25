#!/bin/bash
#SBATCH --account vjgo8416-locomoset
#SBATCH --qos turing
#SBATCH --gpus=1
#SBATCH --time 00:05:00
#SBATCH --array=1-4
#SBATCH --cpus-per-gpu=36
# Load required modules​
module purge
module load baskerville
module restore system
module load bask-apps/test
module load Miniconda3/4.10.3
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"
echo ${SLURM_ARRAY_TASK_ID}
