#!/bin/bash

# testing running the pipline of generating config files and running the metric experiments
CONFIGPATH=$1
echo "Generating configs"
python3 src/locomoset/run/config_gen.py ${CONFIGPATH}
CONFIGPATHDATE=`date +"%Y%m%d"`
echo "Running metric experiments"
for FILE in configs/${CONFIGPATHDATE}*/*;
do
    python3 src/locomoset/run/run_metrics.py ${FILE}
done
