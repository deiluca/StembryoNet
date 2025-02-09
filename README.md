# AI-based approach to dissect the variability of mouse stem cell-derived embryo models

This repository allows the reproduction of the results in the paper "AI-based approach to dissect the variability of mouse stem cell-derived embryo models".


### Prerequisites
Make sure you have Python 3.9.16 installed on your system. This codebase has been developed on a linux machine and a HPC cluster running with the slurm workload manager.

### Optional: Installing virtualenv
To create an isolated Python environment, it's recommended to use virtualenv. Install it using:
```bash
pip install virtualenv
```
# Setup Instructions

Follow these steps to set up the project and install the necessary dependencies:
# 1. Clone the repository

```bash
git clone https://github.com/deiluca/StembryoNet.git
cd StembryoNet
```

# 2. Create a virtual environment
In the project directory, run the following command to create a virtual environment:
```bash
virtualenv venv
```
This will create a new folder called venv in your project directory.

# 3. Activate the virtual environment
```bash
source venv/bin/activate
```


# 4. Install dependencies
With the virtual environment activated, install the required dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```
# 5. Dataset creation
First, download the data from [Zenodo](https://zenodo.org/records/14605093).

This repository enables the creation of datasets for 5-times repeated 5-fold cross-validation, supporting various models and data types:


### Supported Models and Training Tasks
| Model       | Task Description                                             |
|------------|---------------------------------------------------------------|
| StembryoNet | Advanced-stage ETiX embryo classification                |
| MViT        | Advanced-stage ETiX embryo classification                |
| ResNet      | ETiX embryo classification at distinct time points from 0 to 90 hours in 5-hour intervals |

### Available Data Types

| Data Type               | Description                                           | `dataset_info.csv` Column | Channels  |
|-------------------------|-------------------------------------------------------|---------------------------|-----------|
| Fluor-in-focus (default) | Fluorescence in-focus images                      | `bf_f_infocus`            | 1, 2, 3   |
| Fluorescence z-sum   | Fluorescence z-sum projection images                 | `if_zsum`                 | 0, 1, 2   |
| BF-in-focus          | Brightfield in-focus images                          | `bf_f_infocus`            | 0         |

To create datasets, use the scripts located in [scripts/dataset_creation](scripts/dataset_creation). These scripts generate SLURM batch files that facilitate dataset generation.
```python
# Generate datasets for StembryoNet training
python scripts/dataset_creation/stembryonet_training.py

# Generate datasets for StembryoNet inference
python scripts/dataset_creation/stembryonet_inference.py

# Generate datasets for ResNet training and inference
python scripts/dataset_creation/resnet_training_inference.py

# Generate datasets for MViT (Multiscale Vision Transformer) training and inference
python scripts/dataset_creation/mvit_training_inference.py
```
Then, submit all scripts using SBATCH by running:

```bash
bash run_all_sbatch.sh
```

Configure parameters such as data location, SLURM batch file directory, and the number of cross-validation splits in [scripts/dataset_creation/constants.py](scripts/dataset_creation/constants.py).


# 6. Model training and inference
Use scripts in 
First, remove any SBATCH files.
```bash
rm sbatch_files/*
```

For training and inference, use the scripts located in [scripts/model_training](scripts/model_training) and [scripts/inference](scripts/inference). These scripts generate SLURM batch files that facilitate model training and inference.

### Training
```python
# Train StembryoNet model
python scripts/model_training/train_stembryonet.py

# Train ResNet models
python scripts/model_training/train_resnets.py

# Train MViT (Multiscale Vision Transformer) model
python scripts/model_training/train_mvit.py
```
Then, submit all scripts using SBATCH by running:

```bash
bash run_all_sbatch.sh
```

### Inference
```python
# Inference StembryoNet model
python scripts/inference/inference_stembryonet.py

# Inference MVit
python scripts/inference/inference_mvit.py
```
ResNet inference is integrated into the training process, eliminating the need for a separate inference step.

# 7. Model comparison
```python
# Generate boxplot of model accuracies for ResNet, MViT, and StembryoNet
python scripts/model_comparison/visualize_model_comparison.py
```




