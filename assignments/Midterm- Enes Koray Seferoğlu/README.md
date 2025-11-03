# 🧬 Breast Cancer Classification (Colab)

A complete machine learning workflow for **Breast Cancer (Wisconsin Diagnostic)** dataset.  
This project demonstrates the full pipeline from **data exploration (EDA)** to **model training and prediction** —  
replicable both on **Google Colab** and **locally (VS Code / terminal)**.

---

## 📚 About the Dataset

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569  
- **Features:** 30 numerical attributes derived from cell nucleus morphology  
- **Target:**
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

Each sample represents measurements taken from a fine needle aspirate (FNA) of a breast mass.  
The goal is to predict whether the mass is **malignant** or **benign** based on its features.

---

## 🧠 Project Overview

| Stage | Script | Description |
|--------|--------|-------------|
| 🔍 **EDA** | `src/eda.py` | Generates visualizations and descriptive statistics (`reports/figures/`) |
| 🧩 **Training** | `src/train.py` | Trains multiple models and saves the best one as `models/best_model.joblib` |
| 🎯 **Prediction** | `src/predict.py` | Loads the trained model and performs sample predictions via CLI |

---

## ▶️ Run on Google Colab
```python
from google.colab import files
uploaded = files.upload()  # Bilgisayarından yükleme penceresi açılacak

import zipfile, os
zip_name = [k for k in uploaded.keys() if k.endswith('.zip')][0]
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('/content')
!ls -lah /content/project_breast_cancer

%cd /content/project_breast_cancer
!pip install -q -r requirements.txt

!python src/eda.py

from IPython.display import Image, display
from pathlib import Path

for p in sorted(Path('reports/figures').glob('*.png')):
    display(Image(filename=str(p)))


!python src/train.py

import json
print(json.dumps(json.load(open('reports/metrics.json','r',encoding='utf-8')), indent=2, ensure_ascii=False))

!python src/predict.py --input "mean radius=14.0,mean texture=20.0,mean perimeter=90.0,mean area=600.0,mean smoothness=0.1"
