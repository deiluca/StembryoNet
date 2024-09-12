import os
from os.path import join as opj
from constants import OUTDIR_ROOT, NR_CVS, NR_SPLITS, SBATCH_DIR, DF_PATH
from utils import generate_sbatch_file

slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --mem=10000
#SBATCH --time=0:20:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/StembryoNet/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/StembryoNet/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\n\n'''

os.makedirs(SBATCH_DIR, exist_ok=True)

for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    outdir = opj(OUTDIR_ROOT, f'stembryonet_{dname}')
    for cv in range(NR_CVS):
        for split in range(NR_SPLITS):
            sbatchfilename = f'create_dataset_stembryonet_{dname}_cv{cv}_split{split}.sh'
            cmd = f'''
python << END
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')
import pandas as pd
from utils import save_imgs_selected_general_5cvs

df_splits = pd.read_csv('{DF_PATH}')
outdir = '{outdir}'
split_col = f'cv{cv}_split{split}'

train = df_splits[df_splits[split_col] == 'train']
val = df_splits[df_splits[split_col] == 'val']
test = df_splits[df_splits[split_col] == 'test']

save_imgs_selected_general_5cvs({cv}, train, val, test, {split}, outdir, t_minus=0, t_col='Uniformly_sampled_timepoint', img_col='{dtype}', label_col='Score_str', id_col='embryo_id', channels={channels})
END'''

            generate_sbatch_file(SBATCH_DIR, sbatchfilename, slurm_backbone, cmd)