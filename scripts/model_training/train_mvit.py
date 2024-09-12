import os
from os.path import join as opj


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

slurm_scripts_dir = "sbatch_files"

epochs = 20
nclasses = 1
t_start, t_end  = 111, 153
img_div255 = False
img_norm = True
cj_bn = 0.0
lr = 'autofind'
lr_dummy = 1e-5
tsync_diff = 0

weight_decay = 1e-4
opt = 'adam'
             
d = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/mvit'
resd = opj(d, 'results')
for cv in range(5):
    for i in range(5):                                
        outdir_name = f'epochs{epochs}_mvit'
        outdir = opj(*[resd, outdir_name, f'cv{cv}_split{i}'])
        os.makedirs(outdir, exist_ok=True)
        train_dir = opj(d, f'cv{cv}_split{i}_training')
        val_dir = opj(d, f'cv{cv}_split{i}_validation')
        test_dir = opj(d, f'cv{cv}_split{i}_test')     
        cmd = f'python train_main_mvit.py mvit\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0 3.368932038834951\\\n -lr {lr_dummy}\\\n -tb_outdir {outdir}\\\n --weight_decay {weight_decay}\\\n --optimizer {opt}\\\n --cj_bn {cj_bn}'
        if not img_norm:
            cmd += '\\\n --no_img_norm'
        if not img_div255:
            cmd += '\\\n --no_img_div255'
        if lr == 'autofind':
            cmd += '\\\n --use_autofindlr'
        print(slurm_backbone+cmd)
        with open(opj(slurm_scripts_dir, outdir_name+f'_cv{cv}_split{i}.sh'), "w") as f:
            f.write(slurm_backbone+cmd)