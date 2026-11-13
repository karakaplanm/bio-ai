# 🧬 VHL Histopathology AI: ccRCC Mutation & Phenotype Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A Deep Learning framework to detect **VHL (Von Hippel-Lindau)** mutation-associated phenotypes in **Clear Cell Renal Cell Carcinoma (ccRCC)** from H&E stained histopathology patches with **Grad-CAM visual explainability**.

---

## 🔬 Biological Background

* **VHL (Von Hippel-Lindau):** A classic tumor suppressor gene located on chromosome 3p25.
* **Cellular Mechanism:** In healthy tissues, the VHL protein acts as the recognition unit of an E3 ubiquitin ligase targeting HIF-1α / HIF-2α for degradation in the presence of oxygen.
* **Cancer Link (ccRCC):** In over 70% of clear cell renal cell carcinomas, VHL is inactivated, driving:
  1. **Clear Cell Morphology:** Dysregulated lipid and glycogen storage that dissolve during histological fixation, yielding optically clear cytoplasm.
  2. **Microvascular Arborization:** VEGF-induced neo-vascular networks ("chicken-wire" vasculature).

---

## 📁 Project Structure

```text
vhl-histopathology-ai/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── main.py                            # CLI pipeline runner
├── notebooks/
│   └── VHL_Histopathology_Colab.ipynb # Google Colab 1-Click notebook
├── src/
│   ├── config.py                      # Hyperparameters and paths
│   ├── dataset.py                     # Custom PyTorch Dataset & H&E transforms
│   ├── model.py                       # ResNet18 Transfer Learning architecture
│   ├── train.py                       # Training loop with validation
│   ├── evaluate.py                    # Metrics, Confusion Matrix, and ROC Curve
│   ├── gradcam.py                     # Grad-CAM visual explainability module
│   └── data_download.py               # Sample patch generator & TCGA instructions
├── data/                              # Data directories
└── reports/                           # Output plots and Grad-CAM visualizations
```

---

## 🚀 Quick Start

### 1. Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/VHL_Histopathology_Colab.ipynb`.
3. Set runtime to **GPU (T4)** and click **Run all**.

### 2. Local Execution
```bash
# 1. Navigate to directory
cd vhl-histopathology-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (sample creation, training, evaluation, Grad-CAM)
python main.py --mode all
```

---

## 📊 Outputs & Reports
* `reports/confusion_matrix.png`: Model classification accuracy on test patches.
* `reports/roc_curve.png`: Diagnostic sensitivity vs specificity (ROC-AUC).
* `reports/gradcam_vhl_explanation.png`: Heatmap highlighting clear cell cytoplasm regions.

---

## 📤 GitHub Upload (GitHub Desktop)
1. Open **GitHub Desktop** on your desktop.
2. Click **File** ➔ **Add Local Repository**.
3. Select the `vhl-histopathology-ai` folder.
4. Click **Publish repository** to push it to your GitHub profile.

