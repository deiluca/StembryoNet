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


def sigmoid(x):
    """
    Calculate the sigmoid of a given value.

    Args:
        x (float): The input value.

    Returns:
        float: The sigmoid of the input value.
    """
    return 1 / (1 + math.exp(-x))

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of predictions against the targets.

    Args:
        predictions (np.ndarray): The predicted values.
        targets (np.ndarray): The true target values.

    Returns:
        float: The mean accuracy of the predictions.
    """
    return (predictions == targets).mean()

def find_best_threshold(predicted_raw, targets):
    """
    Determine the threshold that maximizes the F1 score for binary predictions.

    Args:
        predicted_raw (np.ndarray): Raw predicted values.
        targets (np.ndarray): True target values.

    Returns:
        tuple: The best threshold and the corresponding F1 score.
    """
    best_threshold = None
    best_f1_score = 0
    
    # Sort and find unique thresholds from the raw predictions
    thresholds = np.unique(predicted_raw)
    
    # Evaluate each threshold
    for threshold in thresholds:
        # Convert raw predictions to binary predictions based on the current threshold
        binary_predictions = (predicted_raw >= threshold).astype(int)
        
        # Calculate the F1 score for the current binary predictions
        current_f1_score = f1_score(targets, binary_predictions)
        
        # Update the best threshold if the current F1 score is higher
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_threshold = threshold
    
    return best_threshold, best_f1_score

def get_stembryonet_threshold(base_d, outfile='bla.csv', inf_dir='fasfd'):
    """
    Calculate the best thresholds for StembryoNet across cross-validation splits.

    Args:
        base_d (str): The base directory containing data files.
        outfile (str): The output file to save results (default: 'bla.csv').
        inf_dir (str): The inference directory (default: 'fasfd').

    Returns:
        pd.DataFrame: A DataFrame containing cross-validation results and thresholds.
    """
    cvs, splits, f1s, thresholds_f1 = [], [], [], []
    
    # Loop through cross-validation and splits
    for cv in range(5):
        for split in range(5):
            # Construct the file path to the outputs
            d = opj(base_d, f'cv{cv}_split{split}/{inf_dir}/outputs.csv')
            x = pd.read_csv(d)
            
            # Apply sigmoid function to predicted values
            x['predicted_sigmoid'] = x['predicted'].apply(sigmoid)

            # Extract time and embryo ID from filename
            x['t'] = x['filename'].apply(lambda x: int(os.path.basename(x).replace('.tif', '').split('_')[-1]))
            x['embryo_id'] = x['filename'].apply(lambda x: os.path.basename(x)[:7])
            
            # Group data by embryo ID and aggregate relevant columns
            x['pred_normal'] = x['predicted'].apply(sigmoid)
            x = x.groupby(['embryo_id']).aggregate({'pred_normal': list, 't': list, 'target': list, 'target_2': list, 'filename': list}).reset_index()
            x['predicted'] = x['pred_normal'].apply(np.max)
            x['filename'] = x['filename'].apply(lambda x: x[0])
            x['target'] = x['target'].apply(lambda x: x[0])
            x['target_2'] = x['target_2'].apply(lambda x: x[0])
    
            # Apply sigmoid to the predicted values again
            x['predicted_sigmoid'] = x['predicted'].apply(sigmoid)
    
            # Find the best threshold and F1 score
            predicted_raw = x['predicted_sigmoid']
            targets = x['target']
            best_threshold_f1, best_f1_score = find_best_threshold(predicted_raw, targets)
    
            # Store results
            cvs.append(cv)
            splits.append(split)
            f1s.append(best_f1_score)
            thresholds_f1.append(best_threshold_f1)
    
    # Create a DataFrame from the results and save it to a CSV file
    df = pd.DataFrame.from_dict({'cv': cvs, 'split': splits, 'val_f1': f1s, 'threshold_f1': thresholds_f1})
    df.to_csv(outfile, index=False)
    return df

def get_model_performance(base_d, model_name, target_col='target', stembryonet=False, inf_dir=None, threshold_file=None):
    """
    Calculate accuracy and F1 score for a given model across cross-validation splits.

    Args:
        base_d (str): The base directory containing data files.
        model_name (str): The name of the model being evaluated.
        target_col (str): The column name for the target values (default: 'target').
        stembryonet (bool): Flag indicating if the model is StembryoNet (default: False).
        inf_dir (str): The inference directory (only relevant if stembryonet is True).

    Returns:
        pd.DataFrame: A DataFrame containing model performance metrics for each split.
    """
    model_names, splits, accs, f1s = [], [], [], []
    
    # If using StembryoNet, read the thresholds from the previously calculated file
    if stembryonet:
        df_thresholds = pd.read_csv(threshold_file)

    # Loop through cross-validation and splits
    for cv in range(5):
        for split in range(5):
            # Construct the file path based on whether it is StembryoNet or not
            if stembryonet:
                d = opj(base_d, f'cv{cv}_split{split}/{inf_dir}/outputs.csv')
            else:
                d = opj(base_d, f'cv{cv}_split{split}/outputs.csv')
                
            x = pd.read_csv(d)

            if stembryonet:
                # Extract time and embryo ID for StembryoNet
                x['t'] = x['filename'].apply(lambda x: int(os.path.basename(x)[8:11]))
                x['embryo_id'] = x['filename'].apply(lambda x: os.path.basename(x)[:7])
                x['pred_normal'] = x['predicted'].apply(sigmoid)
                
                # Group data by embryo ID and aggregate
                x = x.groupby(['embryo_id']).aggregate({'pred_normal': list, 't': list, 'target': list, 'target_2': list, 'filename': list}).reset_index()
                x['predicted'] = x['pred_normal'].apply(np.max) # maximum predicted probability for class normal
                x['filename'] = x['filename'].apply(lambda x: x[0])
                x['target'] = x['target'].apply(lambda x: x[0])
                x['target_2'] = x['target_2'].apply(lambda x: x[0])

            # Apply sigmoid to the predicted values
            x['predicted_sigmoid'] = x['predicted'].apply(sigmoid)

            # Determine the binary predictions based on the chosen threshold
            if stembryonet:
                threshold = df_thresholds[(df_thresholds['cv'] == cv) & (df_thresholds['split'] == split)]['threshold_f1'].item()
                x['predicted_binary'] = x['predicted_sigmoid'].apply(lambda x: 1 if x >= threshold else 0)
            else:
                x['predicted_binary'] = x['predicted_sigmoid'].apply(lambda x: 1 if x >= 0.5 else 0)

            # Calculate the number of correct predictions and accuracy
            correct_predictions = (x[target_col] == x['predicted_binary']).sum()
            total_predictions = len(x)
            accuracy = correct_predictions / total_predictions
            
            # Append accuracy and F1 score to the results
            accs.append(accuracy)
            f1 = f1_score(x[target_col], x['predicted_binary'])
            f1s.append(f1)

            splits.append(split)
            model_names.append(model_name)

    # Create a DataFrame with the performance results
    df_res = pd.DataFrame.from_dict({'model': model_names, 'split': splits, 'Accuracy': accs, 'f1_score': f1s})
    return df_res
