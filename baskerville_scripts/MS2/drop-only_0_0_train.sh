#!/bin/sh
#SBATCH --account vjgo8416-mod-sim-2
#SBATCH --qos turing
#SBATCH --time 0-0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name ms2-drop-only_0_0
#SBATCH --output ./slurm_train_logs/drop-only_0_0-train-%j.out

module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH=/bask/projects/v/vjgo8416-mod-sim-2/ms2env

conda activate ${CONDA_ENV_PATH}
python scripts/train_models.py --experiment_groups_path /Users/pswatton/Documents/Code/arc-model-similarities-phase-2/configs/experiment_groups --experiment_group drop-only --dmpair_config_path /Users/pswatton/Documents/Code/arc-model-similarities-phase-2/configs/dmpair_kwargs.yaml --trainer_config_path /Users/pswatton/Documents/Code/arc-model-similarities-phase-2/configs/trainer.yaml --seed_index 0 --dataset_index 0
