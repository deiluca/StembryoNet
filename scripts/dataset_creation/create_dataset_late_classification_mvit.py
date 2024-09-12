import os
from os.path import join as opj


slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --mem=10000
#SBATCH --time=1:40:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\n\n'''

slurm_scripts_dir = "sbatch_files"

df_path = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/dataset_info.csv'
t_start, t_end = 111, 153

outdir = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/mvit'
for cv in range(5):
    for split in range(5):
        sbatchfilename = f'create_dataset_videos_cv{cv}_split{split}.sh'
        cmd = f'''
python << END
import numpy as np
import pandas as pd
from os.path import join as opj
import os
import skimage.io as skio
import tifffile as tiff
import cv2
import skimage.io as skio

width = 288
height = 288
channel = 1
 
fps = 22

def save_imgs_selected_general5cvs_videos_(df, outdir, t_col='Uniformly_sampled_timepoint', all_channels=True, t_start=0, t_end=20, img_col='img', label_col = 'Score_new_str', id_col='embryo_id_y', channels = [0]):
    for i, row in df.iterrows():
        outdir2 = opj(outdir, row[label_col])
        os.makedirs(outdir2, exist_ok=True)
        embryo_id = row[id_col]
        # if os.path.isfile(opj(outdir2, embryo_id+'.mp4')):
            # print(embryo_id, 'already extracted, continuing')
            # continue
        img = skio.imread(row[img_col].replace('/mnt/lsdf_iai-aida/Daten_Deininger/', '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/'))
        if len(img.shape)==4:
            img = img[:, :, :, channels]
        elif len(img.shape)==3:
            pass

        video = cv2.VideoWriter(opj(outdir2, embryo_id+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height), True)
        print(embryo_id, t_start, t_end)
        for t in range(t_start, t_end):
            x = img[t, ...]
            video.write(x)
 
        video.release()
        
def save_imgs_selected_general_5cvs_videos(cv_nr, train, val, test, i, outdir, all_channels=False, t_col='Uniformly_sampled_timepoint_new', t_start=0, t_end=20, img_col='img', label_col = 'Score_new_str', id_col='embryo_id_y', channels = [0]):
    outdir_imgs = outdir

    outdir_imgs_train = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_training')
    outdir_imgs_val = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_validation')
    outdir_imgs_test = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_test')

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_imgs_train, exist_ok=True)
    os.makedirs(outdir_imgs_val, exist_ok=True)
    os.makedirs(outdir_imgs_test, exist_ok=True)

    save_imgs_selected_general5cvs_videos_(train, outdir_imgs_train, t_col=t_col, t_start=t_start, t_end = t_end, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)
    save_imgs_selected_general5cvs_videos_(val, outdir_imgs_val, t_col=t_col, t_start=t_start, t_end = t_end, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)
    save_imgs_selected_general5cvs_videos_(test, outdir_imgs_test, t_col=t_col, t_start=t_start, t_end = t_end, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)

df_splits = pd.read_csv('{df_path}')

print(df_splits['Score_str'].value_counts())
outdir = '{outdir}'
os.makedirs(outdir, exist_ok=True)

# print(f'split {split}')
split_col = f'cv{cv}_split{split}'
train = df_splits[df_splits[split_col]=='train']
val = df_splits[df_splits[split_col]=='val']
test = df_splits[df_splits[split_col]=='test']
for cv in range(5):
    for split in range(5):
        save_imgs_selected_general_5cvs_videos({cv}, 
                                    train, 
                                    val, 
                                    test, 
                                    {split}, 
                                    outdir, 
                                    t_col = 'plain',
                                    img_col='bf_f_infocus', 
                                    label_col = 'Score_str', 
                                    id_col='embryo_id',
                                    channels = [1, 2, 3],
                                    t_start = {t_start},
                                    t_end = {t_end})
END'''
        print(slurm_backbone+cmd)
        with open(opj(slurm_scripts_dir, sbatchfilename), "w") as f:
            f.write(slurm_backbone+cmd)
