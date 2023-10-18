#!/bin/bash

# testing running the pipline of generating config files and running the metric experiments

# Get path to top level config from command line argument
TOPLEVELCONFIG=$1

# Generate specific configs and get path to directory containing them
echo "Generating configs"
CONFIGSPATH=$(python3 src/locomoset/run/config_gen.py ${TOPLEVELCONFIG})

# Iterate over configs in config folder and run experiments
echo "Running metric experiments"
for FILE in ${CONFIGSPATH}/*;
do
    python3 src/locomoset/run/run_metrics.py ${FILE}
done
