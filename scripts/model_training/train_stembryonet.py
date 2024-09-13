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

# Iterate over the dataset type, dataset name, and channels
for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    for t_minus in [0]:  # Iterate over time minus values (in this case, only 0)
        # Construct paths for the dataset and results
        d = opj(OUTDIR_ROOT, f'stembryonet_{dname}')
        resd = opj(d, 'results')
        
        # Loop over cross-validation folds and splits
        for cv in range(NR_CVS):
            for split in range(NR_SPLITS):
                # Construct output directory name and path
                outdir_name = f'stembryonet_{model_depth}_2d_epochs{epochs}_lr-{lr}_bs{batch_size}_{dname}'
                outdir = opj(*[resd, outdir_name, f'cv{cv}_split{split}'])
                os.makedirs(outdir, exist_ok=True)
                
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
                with open(opj(SBATCH_DIR, outdir_name + f'cv{cv}_split{split}.sh'), "w") as f:
                    f.write(slurm_backbone + cmd)