# AI-based approach to dissect the variability of mouse stem cell-derived embryo models

This repository allows the reproduction of the results in the paper "AI-based approach to dissect the variability of mouse stem cell-derived embryo models".

### Prerequisites
Make sure you have Python 3.9.16 installed on your system. You can download Python [here](https://www.python.org/downloads/).

### Optional: Installing virtualenv
To create an isolated Python environment, it's recommended to use virtualenv. Install it using:
```
pip install virtualenv
```
# Setup Instructions

Follow these steps to set up the project and install the necessary dependencies:
# 1. Clone the repository

```
git clone https://github.com/deiluca/StembryoNet.git
cd StembryoNet
```

# 2. Create a virtual environment
In the project directory, run the following command to create a virtual environment:
```
virtualenv venv
```
This will create a new folder called venv in your project directory.

# 3. Activate the virtual environment
```
source venv/bin/activate
```

# 4. Install dependencies
With the virtual environment activated, install the required dependencies from the requirements.txt file:

```
pip install -r requirements.txt
```
# 4. Dataset creation
This repository allows the creation of datasets (5-times repeated 5-fold cross validation) for
- Training and inference
    - StembryoNet: advanced stage ETiX embryo classification
    - MViT: advanced stage ETiX embryo classification
    - ResNet: ETiX embryo classification at distinct time points from 0 to 90 hours in steps of 5 hours

- Available data types
    - fluor-in-focus (default): fluorescence in-focus images, use dataset_info.csv column 'bf_f_infocus' and channels 1, 2, 3
    - fluorescence z-sum: fluorescence z-sum projection images, use dataset_info.csv column 'if_zsum' and channels 0, 1, 2
    - bf-in-focus: brightfield in-focus images, use dataset_info.csv column 'bf_f_infocus' and channel 0

For creation of datasets use scripts in scripts/dataset_creation

# 4. Model training
Use scripts in scripts/model_training and scripts/inference


blabla





