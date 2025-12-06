# 🤖 AI Project: Gene Expression & Digit Recognition

This project includes two machine learning applications: 
(1) clustering biological samples using gene expression data (GEO analysis), 
and (2) digit recognition using a neural network (MLP Classifier – MNIST).

## 📊 1. Gene Expression Analysis (GEO)
In this section:
- Data downloaded using GEOparse
- Expression matrix created
- Normalization performed with StandardScaler
- Clustering with K-Means (n=3)
- Dimensionality reduction using PCA (2D)
- Visualization saved

Outputs are saved in the `plots/` directory.

## ✏️ 2. Digit Recognition (MLP Classifier)
In this section:
- sklearn "digits" dataset loaded
- Train/Test split created (80% - 20%)
- Model trained using MLPClassifier
- Accuracy calculated

## 🚀 Run
```
pip install -r requirements.txt
python run_analysis.py
```
