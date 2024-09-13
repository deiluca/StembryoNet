import os
from os.path import join as opj
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR

# SLURM script template for job submission
slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=30000
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/StembryoNet/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/StembryoNet/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\n\n'''

# Directory where SLURM scripts will be saved
slurm_scripts_dir = "sbatch_files"

# Task parameters
nclasses = 1

# Architecture and training parameters
model_depth = 18
epochs = 200
batch_size = 16
lr = 1e-3
opt = 'adam'
wd = 1e-4
use_autofind_lr = False

cj_bn = 0.0
t_minus = 0

# Define time points as (hour, time point)
time_points = [
    (0, 0)
    # Uncomment below for additional time points
    # (5, 9), (10, 17), (15, 26), (20, 34),
    # (25, 43), (30, 51), (35, 60), (40, 68),
    # (45, 77), (50, 85), (55, 94), (60, 102),
    # (65, 111), (70, 119), (75, 128), (80, 136),
    # (85, 145), (90, 152)
]

# Iterate over the (time point, hour) pairs
for hour, _ in time_points:
    tname = f'{hour}h'
    for dname in ['fluorinfocus']:
        # Construct paths for the dataset and results
        d = opj(OUTDIR_ROOT, f'resnet_{tname}_{dname}')
        resd = opj(d, 'results')
        
        # Loop over cross-validation folds and splits
        for cv in range(NR_CVS):
            for split in range(NR_SPLITS):
                # Construct output directory name and path
                outdir_name = f'resnet{model_depth}_2d_epochs{epochs}_lr-{lr}_bs{batch_size}_{dname}_{tname}'
                outdir = opj(*[resd, outdir_name, f'cv{cv}_split{split}'])
                os.makedirs(outdir, exist_ok=True)  # Create output directory if it doesn't exist

                # Define directories for training, validation, and test sets
                train_dir = opj(d, f'cv{cv}_split{split}_training')
                val_dir = opj(d, f'cv{cv}_split{split}_validation')
                test_dir = opj(d, f'cv{cv}_split{split}_test')     
                
                # Construct the SLURM command to execute the training script
                cmd = f'''python train_main.py {model_depth}\\
 {nclasses}\\
 {epochs}\\
 {train_dir}\\
 {val_dir}\\
 -ts {test_dir}\\
 --batch_size {batch_size}\\
 -g 1\\
 -tr\\
 -s {outdir}\\
 --ce_weights 0 3.368932038834951\\
 -lr {lr}\\
 -tb_outdir {outdir}\\
 -g 1\\
 -tr\\
 --weight_decay {wd}\\
 --optimizer {opt}\\
 --cj_bn {cj_bn}'''
                
                # Append additional options if specified
                if use_autofind_lr:
                    cmd += '\\\n --use_autofindlr'
                
                # Print the constructed command for debugging purposes
                print(slurm_backbone + cmd)
                
                # Write the SLURM script to a file
                with open(opj(SBATCH_DIR, outdir_name + f'_cv{cv}_split{split}.sh'), "w") as f:
                    f.write(slurm_backbone + cmd)