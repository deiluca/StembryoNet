import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statannot
from matplotlib import rc

import numpy as np

from os.path import join as opj
import os
import pandas as pd

from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

import math
from utils import get_model_performance, get_stembryonet_threshold

dname = 'fluorinfocus'
dir_resnet = '/mnt/lsdf_iai-aida/Daten_Deininger/Daten_Deininger/projects/embryo_project/datasets/resnet_90h_fluorinfocus/results/resnet18_2d_epochs200_lr-0.001_bs16_fluorinfocus_90h' 
dir_mvit = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/datasets/mvit_fluorinfocus/results/mvit'
dir_stembryonet = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/datasets/stembryonet_fluorinfocus/results/stembryonet_18_2d_epochs200_lr-0.001_bs16_fluorinfocus' 


stembryonet_threshold_file = f'stembryonet_best_validation_threshold_{dname}.csv'
get_stembryonet_threshold(base_d = dir_stembryonet,
                    inf_dir = 'inference_synced_model_everyt_val',
                    outfile = stembryonet_threshold_file)

rc('font',**{'family':'sans-serif','sans-serif':'Arial'})

df_resrandom_binary = pd.read_csv('random_model_performance.csv')

df_resnet = get_model_performance(base_d = dir_resnet, 
                                  model_name= r'$ResNet_{90 h}$')

df_mvit = get_model_performance(base_d = dir_mvit, 
                                model_name=r'$MViT_{65-90 h}$', 
                                target_col='target_2')

df_stembryonet = get_model_performance(base_d = dir_stembryonet, 
                                 model_name='StembryoNet', 
                                 stembryonet=True, 
                                  inf_dir = 'inference_synced_model_everyt_test',
                                 threshold_file=stembryonet_threshold_file)

df_res = pd.concat([df_resrandom_binary, df_resnet, df_mvit, df_stembryonet])

plt.figure(figsize=(3.3, 3.5))
metric, metric_name = 'Accuracy', 'Accuracy'
    
sns.boxplot(data=df_res, y=metric, x='model', color='lightgrey', fliersize=2.0, palette=['#E6E6E6', '#C8C8C8', '#969696', 'grey'])

plt.ylim((0.38, 0.95))

plt.tick_params(axis='x', rotation=20)

box_pairs = [
            (r'$ResNet_{90 h}$', r'$MViT_{65-90 h}$'),
            (r'$ResNet_{90 h}$', 'StembryoNet'),
            ('StembryoNet', r'$MViT_{65-90 h}$'),
            ('Random', 'StembryoNet')
             ]

test_results = statannot.add_stat_annotation(plt.gca(), box_pairs=box_pairs, 
                      data=df_res,   
                      x='model', 
                      y=metric, 
                        test='t-test_ind', text_format='star',
                        loc='outside', verbose=1, comparisons_correction=None)
plt.ylabel(metric_name)
y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9]

plt.gca().set_yticks(y_ticks)
plt.gca().yaxis.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.2)

plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig('fig2b.png', dpi=1000)