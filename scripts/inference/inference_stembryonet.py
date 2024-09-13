import os
from os.path import join as opj
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')  # Add the custom script path for importing local modules
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR  # Import constants for output directories and cross-validation

# SLURM job submission script template for running jobs on the cluster
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


# Define training hyperparameters
epochs = 200              # Number of training epochs
nclasses = 1              # Number of classes (classification task)
model_depth = 18          # Depth of the model (ResNet18)
batch_size = 1024         # Batch size for inference
lr = 1e-3                 # Learning rate
opt = 'adam'              # Optimizer (Adam)
wd = 1e-4                 # Weight decay for regularization
use_autofind_lr = False   # Whether to automatically find the learning rate
tmin, tmax = 111, 153     # Time range for inference data

inputdir = opj(OUTDIR_ROOT, 'stembryonet_fluorinfocus/results/stembryonet_18_2d_epochs200_lr-0.001_bs16_fluorinfocus') 

# Loop over test modes ('val' for validation and 'test' for testing)
for test_mode in ['val', 'test']:
    image_dir = opj(OUTDIR_ROOT, 'stembryonet_inference_fluorinfocus') 

    for cv in range(NR_CVS):  # Loop over cross-validation splits
        for split in range(NR_SPLITS):  # Loop over dataset splits
            # Define directory for model checkpoints
            ckpth_dir = opj(*[inputdir, f'cv{cv}_split{split}'])
            sbatch_name = f'tmin{tmin}_tmax{tmax}_{test_mode}'  # Name for SLURM batch script

            # Define output directory for inference results based on test mode
            if test_mode == 'val':
                outdir = opj(ckpth_dir, 'inference_stembryonet_everyt_val')
            else:
                outdir = opj(ckpth_dir, 'inference_stembryonet_everyt_test')
            
            os.makedirs(outdir, exist_ok=True)  # Create the output directory if it doesn't exist

            # Select the best model checkpoint (.ckpt file) for evaluation
            ckpth_best = opj(ckpth_dir, [x for x in os.listdir(ckpth_dir) if x.endswith('.ckpt')][0])
            print(ckpth_best)  # Print the path of the best checkpoint for debugging

            # Define directories for training, validation, and test data
            train_dir = opj(inputdir, f'split{split}_training')
            val_dir = opj(inputdir, f'split{split}_validation')
            if test_mode == 'val':
                test_dir = opj(image_dir, f'cv{cv}_split{split}_validation')
            else:
                test_dir = opj(image_dir, f'cv{cv}_split{split}_test')

            # Construct the command for training and evaluation
            cmd = f'''python train_main.py {model_depth}\\
 {nclasses}\\
 {epochs}\\
 {train_dir}\\
 {val_dir}\\
 --batch_size {batch_size}\\
 -ts {test_dir}\\
 --test_only\\
 -g 1 -tr -s {outdir}\\
 --ce_weights 0.5 0.5\\
 -lr {lr} -tb_outdir {outdir}\\
 --ckpth_best {ckpth_best}'''  # Additional parameters like batch size, learning rate, checkpoint, etc.

            # Add the flag for automatic learning rate finding if enabled
            if use_autofind_lr:
                cmd += '\\\n --use_autofindlr'

            # Print the combined SLURM script and training command
            print(slurm_backbone + cmd)
            
            # Write the SLURM batch script to a file for submission
            with open(opj(SBATCH_DIR, sbatch_name+f'cv{cv}_split{split}.sh'), "w") as f:
                f.write(slurm_backbone + cmd)