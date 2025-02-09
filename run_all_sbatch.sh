#!/bin/bash

# Specify the directory containing the shell scripts
directory="/home/iai/oc9627/StembryoNet/sbatch_files"

# Iterate over all shell scripts in the directory
for file in $directory/*.sh; do
    # Submit each shell script using sbatch
    sbatch "$file"
done