import os
from os.path import join as opj
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR

# SLURM script template for job submission
slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=60000
#SBATCH --time=3:00:00
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

# Training parameters
epochs = 20
nclasses = 1
t_start, t_end  = 111, 153
img_div255 = False
img_norm = True
cj_bn = 0.0
lr = 'autofind'
lr_dummy = 1e-5
tsync_diff = 0

# Optimization parameters
weight_decay = 1e-4
opt = 'adam'

# Loop over dataset types, names, and channels
for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    # Construct paths for the dataset and results
    d = opj(OUTDIR_ROOT, f'mvit_{dname}')
    resd = opj(d, 'results')

    # Loop over cross-validation folds and splits
    for cv in range(NR_CVS):
        for split in range(NR_SPLITS):                              
            # Construct output directory name and path
            outdir_name = 'mvit'
            outdir = opj(*[resd, outdir_name, f'cv{cv}_split{split}'])
            os.makedirs(outdir, exist_ok=True)  # Create output directory if it doesn't exist

            # Define directories for training, validation, and test sets
            train_dir = opj(d, f'cv{cv}_split{split}_training')
            val_dir = opj(d, f'cv{cv}_split{split}_validation')
            test_dir = opj(d, f'cv{cv}_split{split}_test')     

            # Construct the SLURM command to execute the training script
            cmd = f'python train_main_mvit.py mvit\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0 3.368932038834951\\\n -lr {lr_dummy}\\\n -tb_outdir {outdir}\\\n --weight_decay {weight_decay}\\\n --optimizer {opt}\\\n --cj_bn {cj_bn}'
            
            # Append additional options based on parameters
            if not img_norm:
                cmd += '\\\n --no_img_norm'
            if not img_div255:
                cmd += '\\\n --no_img_div255'
            if lr == 'autofind':
                cmd += '\\\n --use_autofindlr'
            print(slurm_backbone + cmd)
            # Write the SLURM script to a file
            with open(opj(SBATCH_DIR, outdir_name+f'_cv{cv}_split{split}.sh'), "w") as f:
                f.write(slurm_backbone + cmd)