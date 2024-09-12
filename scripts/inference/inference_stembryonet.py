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

import os
from os.path import join as opj
slurm_scripts_dir = "sbatch_files"

epochs = 200
nclasses = 1

model_depth = 18
epochs = 200
batch_size = 1024
lr = 1e-3
opt = 'adam'
wd = 1e-4
use_autofind_lr = False
tmin, tmax = 111, 153
inputdir = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/stembryonet_fluorinfocus_tminus0/results/resnet18_2d_epochs200_lr-0.001_bs16_fluorinfocus'

for test_mode in ['val', 'test']:
    image_dir = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/stembryonet_fluorinfocus_tmin111_tmax153'
    # for cv in range(5):
    #     for split in range(5):
    for cv in range(5):
        for split in range(5):
        # print(f"training split{i}")
            ckpth_dir = opj(*[inputdir, f'cv{cv}_split{split}'])
            sbatch_name = f'tmin{tmin}_tmax{tmax}_{test_mode}'
            if test_mode == 'val':
                outdir = opj(ckpth_dir, 'inference_synced_model_everyt_val')
            else:
                outdir = opj(ckpth_dir, 'inference_synced_model_everyt_test')
            os.makedirs(outdir, exist_ok=True)
            print(os.listdir(ckpth_dir))
            ckpth_best = opj(ckpth_dir, [x for x in os.listdir(ckpth_dir) if x.endswith('.ckpt')][0])
            print(ckpth_best)
            train_dir = opj(inputdir, f'split{split}_training')
            val_dir = opj(inputdir, f'split{split}_validation')
            if test_mode == 'val':
                test_dir = opj(image_dir, f'cv{cv}_split{split}_validation')
            else:
                test_dir = opj(image_dir, f'cv{cv}_split{split}_test')                
            cmd = f'''python resnet_classifier_haicore_2class.py {model_depth}\\
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
 --ckpth_best {ckpth_best}\\
 --imgs_4channel'''
            if use_autofind_lr:
                cmd += '\\\n --use_autofindlr'
            print(slurm_backbone+cmd)
            with open(opj(slurm_scripts_dir, sbatch_name+f'cv{cv}_split{split}.sh'), "w") as f:
                f.write(slurm_backbone+cmd)
