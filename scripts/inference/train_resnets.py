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
#SBATCH --output=/home/iai/oc9627/StembryoNet/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/StembryoNet/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
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
# hour, tp
time_points = [
    (0, 0),
    (5, 9),
    (10, 17),
    (15, 26),
    (20, 34),
    (25, 43),
    (30, 51),
    (35, 60),
    (40, 68),
    (45, 77),
    (50, 85),
    (55, 94),
    (60, 102),
    (65, 111),
    (70, 119),
    (75, 128),
    (80, 136),
    (85, 145),
    (90, 152)]
jobctr, dirctr = 0, 0

# Iterate over the (tp, hour) pairs
for hour, _ in time_points:
    tname = f'unsynced{hour}h'
    for dname in ['fluorinfocus']:
        d = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/resnet{tname}_{dname}'
        resd = opj(d, 'results')
        for cv in range(5):
            for i in range(5):
            # print(f"training split{i}")
                slurm_scripts_dir = f'sbatch_files_early_class_ds125/{dirctr}'
                os.makedirs(slurm_scripts_dir, exist_ok=True)
                outdir_name = f'resnet{model_depth}_2d_epochs{epochs}_lr-{lr}_bs{batch_size}_{dname}_{tname}'
                outdir = opj(*[resd, outdir_name, f'cv{cv}_split{i}'])
                if os.path.isfile(opj(outdir, 'outputs.csv')):
                    continue
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
                jobctr += 1
                if jobctr%100==0:
                    dirctr+=1
