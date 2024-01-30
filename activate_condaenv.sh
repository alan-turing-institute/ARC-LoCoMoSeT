module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# This should be edited by the setup script to point in the
# right direction for the configs and run call
CONFIGSPATH=configs/20231121-190822-362963/config_train_${SLURM_ARRAY_TASK_ID}.yaml
RUN_CALL=locomoset_run_train

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH=/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/locomosetenv
conda activate ${CONDA_ENV_PATH}

