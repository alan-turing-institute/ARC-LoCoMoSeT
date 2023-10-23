cd /bask/projects/v/vjgo8416-mod-sim-2
module restore system
module load bask-apps/test
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

CONDA_ENV_PATH="/bask/projects/v/vjgo8416-mod-sim-2/ms2env"


conda activate ${CONDA_ENV_PATH}
