import os
from os.path import join as opj
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR, DF_PATH
from utils import generate_sbatch_file

# SLURM script template for job submission
slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1                          # Request one node
#SBATCH --gres=gpu:1g.5gb:1                # Request 1 GPU with 5GB memory
#SBATCH --mem=10000                        # Request 10GB of memory
#SBATCH --time=0:20:00                     # Set time limit for 20 minutes
#SBATCH --mail-type=FAIL                   # Send an email on job failure
#SBATCH --mail-user=luca.deininger@kit.edu # Email to send notifications
#SBATCH --job-name=Embryos                 # Job name
#SBATCH --output=/home/iai/oc9627/StembryoNet/logs/%x.%j.out  # Output file path
#SBATCH --error=/home/iai/oc9627/StembryoNet/logs/%x.%j.err   # Error file path
#SBATCH --partition=normal                 # Job partition
#SBATCH --constraint="LSDF"                # Constraint on LSDF resources
source ~/aidaretreat23/bin/activate\n\n'''  # Activate the virtual environment

# Ensure the SBATCH directory exists
os.makedirs(SBATCH_DIR, exist_ok=True)

# Set the minimum and maximum time points
tmin, tmax = 111, 153

# Loop over different image data types, dataset names, and channels
for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    # Define output directory based on dataset name
    outdir = opj(OUTDIR_ROOT, f'mvit_{dname}')
    
    # Iterate over cross-validation folds and splits
    for cv in range(NR_CVS):
        for split in range(NR_SPLITS):
            # Define the SLURM script filename
            sbatchfilename = f'create_dataset_mvit_{dname}_cv{cv}_split{split}.sh'
            
            # Command to execute Python code within the SLURM script
            cmd = f'''
python << END
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')
import pandas as pd
from utils import save_imgs_selected_general_5cvs

# Load dataset splits from CSV
df_splits = pd.read_csv('{DF_PATH}')
outdir = '{outdir}'
split_col = f'cv{cv}_split{split}'

# Filter the dataset into training, validation, and test sets
train = df_splits[df_splits[split_col] == 'train']
val = df_splits[df_splits[split_col] == 'val']
test = df_splits[df_splits[split_col] == 'test']

# Call function to save images/videos from the dataset with the specified parameters
save_imgs_selected_general_5cvs({cv}, train, val, test, {split}, outdir, t_minus=0, t_col='plain', img_col='{dtype}', label_col='Score_str', id_col='embryo_id', channels={channels}, tmin={tmin}, tmax={tmax}, save_type='video')
END'''

            # Generate the SLURM script using the provided template and command
            generate_sbatch_file(SBATCH_DIR, sbatchfilename, slurm_backbone, cmd)