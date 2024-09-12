import os
from os.path import join as opj


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
source ~/aidaretreat23/bin/activate\n\n'''

slurm_scripts_dir = "sbatch_files"

epochs = 20
nclasses = 1
t_start, t_end  = 111, 153
img_div255 = False
img_norm = True
cj_bn = 0.0
lr = 'autofind'
tsync_diff = 0

d = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/mvit'
resd = opj(d, 'results')

weight_decay = 1e-4
opt = 'adam'
if lr == 'autofind':
    lr_dummy = 1e-5
else:
    lr_dummy = lr                
for unif_temp_sub, vit_img_size, batch_size in [(16, 224, 1)]:
    for clip_dur in [8]:
        for cv in range(5):
            for i in range(5):
                # print(f"training split{i}")
                outdir_name = f'epochs{epochs}_mvitPRETRAINED_t{t_start}_to_t{t_end}'
                outdir = opj(*[resd, outdir_name, f'cv{cv}_split{i}'])
                if os.path.isfile(opj(outdir, 'outputs.csv')):
                    continue
                os.makedirs(outdir, exist_ok=True)
                ckpth_best = opj(outdir, [x for x in os.listdir(outdir) if x.endswith('.ckpt')][0])

                train_dir = opj(d, f'cv{cv}_split{i}_training')
                val_dir = opj(d, f'cv{cv}_split{i}_validation')
                test_dir = opj(d, f'cv{cv}_split{i}_test')
                cmd = f'python train_main_mvit.py mvit\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n --batch_size {batch_size}\\\n --unif_temp_sub {unif_temp_sub}\\\n --clip_duration {clip_dur}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0 3.368932038834951\\\n -lr {lr_dummy}\\\n -tb_outdir {outdir}\\\n --vit_img_size {vit_img_size}\\\n --weight_decay {weight_decay}\\\n --optimizer {opt}\\\n --cj_bn {cj_bn}\\\n --test_only\\\n --ckpth_best {ckpth_best}'
                if not img_norm:
                    cmd += '\\\n --no_img_norm'
                if not img_div255:
                    cmd += '\\\n --no_img_div255'
                if lr == 'autofind':
                    cmd += '\\\n --use_autofindlr'
                print(slurm_backbone+cmd)
                with open(opj(slurm_scripts_dir, outdir_name+f'_cv{cv}_split{i}.sh'), "w") as f:
                    f.write(slurm_backbone+cmd)