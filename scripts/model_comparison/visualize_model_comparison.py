import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statannot
from matplotlib import rc
from os.path import join as opj
from utils import get_model_performance, get_stembryonet_threshold
import sys
sys.path.append('/home/iai/oc9627/StembryoNet/scripts/dataset_creation')
from constants import OUTDIR_ROOT, DATA_NAME, DIR_STEMBRYONET_INF_VAL, DIR_STEMBRYONET_INF_TEST

# Define variables for directories and dataset name
dname = 'fluorinfocus'

# Directories for ResNet, MViT, and StembryoNet model results
dir_resnet = opj(OUTDIR_ROOT, f'resnet_90h_{DATA_NAME}/results/resnet18_{DATA_NAME}_90h')
dir_mvit = opj(OUTDIR_ROOT, f'mvit_{DATA_NAME}/results/mvit_{DATA_NAME}')
dir_stembryonet = opj(OUTDIR_ROOT, f'stembryonet_{DATA_NAME}/results/stembryonet_{DATA_NAME}')

# Filename for StembryoNet threshold results
stembryonet_threshold_file = f'stembryonet_best_validation_threshold_{DATA_NAME}.csv'

# Get the StembryoNet threshold using a utility function
get_stembryonet_threshold(
    base_d=dir_stembryonet,
    inf_dir=DIR_STEMBRYONET_INF_VAL,
    outfile=stembryonet_threshold_file
)

# Set the font configuration for plotting
rc('font', **{'family': 'sans-serif', 'sans-serif': 'Arial'})

# Get performance for a random model
df_resrandom_binary = pd.read_csv('random_model_performance.csv')

# Get performance for ResNet
df_resnet = get_model_performance(
    base_d=dir_resnet,
    model_name=r'$ResNet_{90 h}$'
)

# Get performance for MViT
df_mvit = get_model_performance(
    base_d=dir_mvit,
    model_name=r'$MViT_{65-90 h}$',
    target_col='target_2'
)

# Get performance for StembryoNet
df_stembryonet = get_model_performance(
    base_d=dir_stembryonet,
    model_name='StembryoNet',
    stembryonet=True,
    inf_dir=DIR_STEMBRYONET_INF_TEST,
    threshold_file=stembryonet_threshold_file
)

# Concatenate performance data from all models into one dataframe
df_res = pd.concat([df_resrandom_binary, df_resnet, df_mvit, df_stembryonet])

# Set up the figure size for the plot
plt.figure(figsize=(3.3, 3.5))

# Define metric and its label for plotting
metric, metric_name = 'Accuracy', 'Accuracy'

# Create a boxplot for model performance comparison
sns.boxplot(
    data=df_res, y=metric, x='model',
    color='lightgrey', fliersize=2.0,
    palette=['#E6E6E6', '#C8C8C8', '#969696', 'grey']
)

# Set the limits for the y-axis (accuracy)
plt.ylim((0.38, 0.95))

# Rotate the x-axis labels slightly for better readability
plt.tick_params(axis='x', rotation=20)

# Define pairs of models for statistical comparison
# box_pairs = [
#     (r'$ResNet_{90 h}$', r'$MViT_{65-90 h}$'),
#     (r'$ResNet_{90 h}$', 'StembryoNet'),
#     ('StembryoNet', r'$MViT_{65-90 h}$'),
#     ('Random', 'StembryoNet')
# ]

# # Add statistical annotations (t-test) to the boxplot
# test_results = statannot.add_stat_annotation(
#     plt.gca(), box_pairs=box_pairs,
#     data=df_res, x='model', y=metric,
#     test='t-test_ind', text_format='star',
#     loc='outside', verbose=1, comparisons_correction=None
# )

# Set the label for the y-axis
plt.ylabel(metric_name)

# Define custom y-ticks for the plot
y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
plt.gca().set_yticks(y_ticks)

# Add a light grid to the y-axis
plt.gca().yaxis.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.2)

# Remove the x-axis label
plt.xlabel('')

# Remove the top and right borders of the plot
sns.despine()

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('fig2b.png', dpi=1000)