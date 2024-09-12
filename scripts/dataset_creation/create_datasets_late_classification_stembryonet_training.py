import os
from os.path import join as opj


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

slurm_scripts_dir = "sbatch_files"
t_minus = 0
df_path = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/dataset_info.csv'

for dtype, dname, channels in [('bf_f_infocus', 'fluorinfocus', '[1, 2, 3]')]:
    for t_minus in [0]:
        outdir = f'/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/stembryonet_{dname}'
        for cv in range(5):
            for split in range(5):
                sbatchfilename = f'create_dataset_synced_{dname}_cv{cv}_split{split}.sh'
                cmd = f'''
python << END
import numpy as np
import pandas as pd
from os.path import join as opj
import os
import skimage.io as skio
import tifffile as tiff
import skimage.io as skio

def save_imgs_selected_general5cvs_(df, outdir, t_col='Uniformly_sampled_timepoint', t_minus=0, all_channels=True, t_fix = None, img_col='img', label_col = 'Score_new_str', id_col='embryo_id_y', channels = [0]):
    for i, row in df.iterrows():
        outdir2 = opj(outdir, row[label_col])
        os.makedirs(outdir2, exist_ok=True)
        embryo_id = row[id_col]
        if t_col == 'Uniformly_sampled_timepoint':
            t = int(row[t_col]) - 1 - t_minus # Paolo used 1-indexing
            t = t-1 if t == 153 else t
        elif t_col == 'plain':
            assert t_fix is not None
            t= t_fix
        else:
            sys.exit()
        img = skio.imread(row[img_col].replace('/mnt/lsdf_iai-aida/Daten_Deininger/', '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/'))
        print(embryo_id, t, row[label_col], img.shape)
        if len(img.shape)==4:
            img = np.squeeze(img[t, :, :, channels])
        elif len(img.shape)==3:
            img = np.squeeze(img[t, :, :])
        else:
            sys.exit('img.shape != [3, 4]')
        if channels == [0]:
            tiff.imwrite(opj(outdir2, embryo_id+'.tif'), img, imagej=True)
        else:
            skio.imsave(opj(outdir2, embryo_id+'_'+str(t)+'.tif'), img, plugin='tifffile', imagej=True)

        
def save_imgs_selected_general_5cvs(cv_nr, train, val, test, i, outdir, t_minus=0, all_channels=False, t_col='Uniformly_sampled_timepoint_new', t=None, img_col='img', label_col = 'Score_new_str', id_col='embryo_id_y', channels = [0]):
    outdir_imgs = outdir

    outdir_imgs_train = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_training')
    outdir_imgs_val = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_validation')
    outdir_imgs_test = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_test')

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_imgs_train, exist_ok=True)
    os.makedirs(outdir_imgs_val, exist_ok=True)
    os.makedirs(outdir_imgs_test, exist_ok=True)

    save_imgs_selected_general5cvs_(train, outdir_imgs_train, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)
    save_imgs_selected_general5cvs_(val, outdir_imgs_val, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)
    save_imgs_selected_general5cvs_(test, outdir_imgs_test, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels)

df_splits = pd.read_csv('{df_path}')

print(df_splits['Score_str'].value_counts())
outdir = '{outdir}'
os.makedirs(outdir, exist_ok=True)

# print(f'split {split}')
split_col = f'cv{cv}_split{split}'
train = df_splits[df_splits[split_col]=='train']
val = df_splits[df_splits[split_col]=='val']
test = df_splits[df_splits[split_col]=='test']
save_imgs_selected_general_5cvs({cv}, 
                                train, 
                                val, 
                                test, 
                                {split}, 
                                outdir, 
                                t_minus={t_minus},
                                t_col = 'Uniformly_sampled_timepoint',
                                img_col='{dtype}', 
                                label_col = 'Score_str', 
                                id_col='embryo_id',
                                channels = {channels})
END'''
                print(slurm_backbone+cmd)
                with open(opj(slurm_scripts_dir, sbatchfilename), "w") as f:
                    f.write(slurm_backbone+cmd)
