# Drug–Target Interaction Prediction Project

## Overview
This project focuses on developing a machine learning pipeline for predicting **drug–target interactions (DTIs)** using simulated molecular and protein feature data. DTI prediction is an important task in computational drug discovery, helping researchers rapidly identify promising drug candidates and reduce experimental screening costs.

The project includes modules for data simulation, model training, evaluation, hyperparameter tuning, visualization, and optional chemical fingerprint generation using RDKit.

---

## Objectives
- Simulate a dataset representing molecular and target-related numerical features.
- Train a machine learning model to predict drug–target interaction likelihood.
- Evaluate model performance with standard metrics.
- Apply hyperparameter tuning for optimization.
- Visualize performance metrics.
- Demonstrate chemical fingerprint generation using RDKit (optional).

---

## Methodology

### 1. Data Simulation
A synthetic dataset is generated with:
- 1000 samples
- 20 numerical features
- Binary labels (1 = interaction, 0 = no interaction)

Certain features simulate biologically meaningful signals such as docking-like scores and hydrophobicity, with added noise for realism.

### 2. Machine Learning Model
The project uses a **Random Forest Classifier**, selected for its robustness on tabular biological data. Data preprocessing steps include:
- Train–test split
- Standard scaling
- Stratified sampling

### 3. Evaluation Metrics
The following metrics are used to assess model performance:
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Metrics are saved in JSON format.

### 4. Hyperparameter Tuning
`RandomizedSearchCV` is applied to optimize:
- Number of trees
- Maximum depth
- Minimum samples per split
- Feature selection strategy

### 5. Visualization
A script generates bar plots for key evaluation metrics and saves them as `.png` files.

### 6. RDKit Fingerprint Example (Optional)
If RDKit is installed, the `rdkit_example.py` script demonstrates:
- Converting SMILES to molecular objects
- Generating Morgan fingerprints (ECFP-like)
- Producing binary feature vectors for machine learning
