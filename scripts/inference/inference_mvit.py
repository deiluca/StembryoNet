import os
from os.path import join as opj
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')  # Add custom script path to sys.path for importing local modules
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR  # Import constants related to output directories and cross-validation splits

# SLURM job submission script template for executing jobs on the cluster
slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=60000
#SBATCH --time=0:10:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/StembryoNet/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/StembryoNet/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\n\n'''   # Activate Python virtual environment

slurm_scripts_dir = "sbatch_files"  # Directory to store SLURM scripts

# Define training hyperparameters
epochs = 20               # Number of training epochs
nclasses = 1              # Number of classes (for classification)
t_start, t_end = 111, 153 # Time range for the training process
img_div255 = False        # Whether to divide image pixel values by 255
img_norm = True           # Whether to normalize images
cj_bn = 0.0               # Color jitter batch normalization parameter
lr = 'autofind'           # Learning rate (can be 'autofind' or a specific value)
tsync_diff = 0            # Parameter for time synchronization difference
weight_decay = 1e-4  # Weight decay for optimization
opt = 'adam'         # Optimizer to use (e.g., Adam)
if lr == 'autofind':
    lr_dummy = 1e-5  # Dummy learning rate to use if 'autofind' is enabled
else:
    lr_dummy = lr    # Use the specified learning rate if 'autofind' is not enabled

# Loop over dataset types, dataset names, and channels for cross-validation and split
for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    d = opj(OUTDIR_ROOT, f'mvit_{dname}')  # Define output directory based on dataset name
    resd = opj(d, 'results')               # Define results directory for the current dataset

    for cv in range(NR_CVS):               # Loop over cross-validation splits
        for split in range(NR_SPLITS):     # Loop over dataset splits
            
            outdir_name = 'mvit'  # Define output directory name
            outdir = opj(*[resd, outdir_name, f'cv{cv}_split{split}'])  # Output path for results

            os.makedirs(outdir, exist_ok=True)  # Create the output directory if it doesn't exist

            # Check if the output file already exists; skip if it does
            if os.path.isfile(opj(outdir, 'outputs.csv')):
                continue
            
            # Locate the best checkpoint file (with .ckpt extension) in the output directory
            ckpth_best = opj(outdir, [x for x in os.listdir(outdir) if x.endswith('.ckpt')][0])

            # Define directories for training, validation, and test sets
            train_dir = opj(d, f'cv{cv}_split{split}_training')
            val_dir = opj(d, f'cv{cv}_split{split}_validation')
            test_dir = opj(d, f'cv{cv}_split{split}_test')

            # Define the command for training the model, specifying the model, directories, and parameters
            cmd = f'python train_main_mvit.py mvit\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0 3.368932038834951\\\n -lr {lr_dummy}\\\n -tb_outdir {outdir}\\\n --weight_decay {weight_decay}\\\n --optimizer {opt}\\\n --cj_bn {cj_bn}\\\n --test_only\\\n --ckpth_best {ckpth_best}'

            # Append additional flags to the command if certain conditions are met
            if not img_norm:
                cmd += '\\\n --no_img_norm'    # Add flag to skip image normalization
            if not img_div255:
                cmd += '\\\n --no_img_div255'  # Add flag to skip dividing image values by 255
            if lr == 'autofind':
                cmd += '\\\n --use_autofindlr' # Add flag to automatically find learning rate

            # Print and write the SLURM script combined with the training command
            print(slurm_backbone + cmd)
            with open(opj(SBATCH_DIR, outdir_name+f'_cv{cv}_split{split}.sh'), "w") as f:
                f.write(slurm_backbone + cmd)