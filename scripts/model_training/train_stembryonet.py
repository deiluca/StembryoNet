import os
from os.path import join as opj


slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=30000
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
#SBATCH --dependency=afterok:1446697
source ~/aidaretreat23/bin/activate\n\n'''

slurm_scripts_dir = "sbatch_files"

# task
nclasses = 1

# architecture and training
model_depth = 18
epochs = 200
batch_size = 16
lr = 1e-3
opt = 'adam'
wd = 1e-4
use_autofind_lr = False
cj_bn = 0.0
t_minus = 0
for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    for t_minus in [0]:
        d = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/stembryonet_{dname}'
        resd = opj(d, 'results')
        for cv in range(5):
            for i in range(5):
            # print(f"training split{i}")
                outdir_name = f'resnet{model_depth}_2d_epochs{epochs}_lr-{lr}_bs{batch_size}_{dname}'
                outdir = opj(*[resd, outdir_name, f'cv{cv}_split{i}'])
                # os.makedirs(outdir, exist_ok=True)
                train_dir = opj(d, f'cv{cv}_split{i}_training')
                val_dir = opj(d, f'cv{cv}_split{i}_validation')
                test_dir = opj(d, f'cv{cv}_split{i}_test')     
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
                if use_autofind_lr:
                    cmd += '\\\n --use_autofindlr'
                print(slurm_backbone+cmd)
                with open(opj(slurm_scripts_dir, outdir_name+f'cv{cv}_split{i}.sh'), "w") as f:
                    f.write(slurm_backbone+cmd)
