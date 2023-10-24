#!/bin/bash

#####
# Set-up for a baskerville run of the metric experiments
#####
#
# For a given top level config file this script will create the requisite config files for each
# specific set of parameters, as well as an adapted .sbatch array-job script that points in the
# right direction of the config files in addition to setting up the requisite size of the array.
#

# Get path to top level config file from command line argument
TOPLEVELCONFIG=$1

# Generate specific configs and get path to directory containing them as well as number of configs
echo "Generating configs"
PYOUTPUTS=($(python3 src/locomoset/run/config_gen.py ${TOPLEVELCONFIG}))
CONFIGSPATH=${PYOUTPUTS[0]}
NUMCONFIGS=${PYOUTPUTS[1]}


# Copy the slurm template script and edit it with above details
echo "Setting up baskerville job script"
cd baskerville_scripts
DT=$(date '+%Y%m%d-%H%M%S')
cp metric_exp_slurm_template.sh metric_exp_job_${DT}.sh
sed -i -e "s/#SBATCH --array=1-10/#SBATCH --array=1-${NUMCONFIGS}/g" metric_exp_job_${DT}.sh
sed -i -e "s%CONFIGSPATH=/PATH/TO/CONFIGS%CONFIGSPATH=${CONFIGSPATH}%g" metric_exp_job_${DT}.sh
