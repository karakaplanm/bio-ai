# 🧫 Blood Cell Classification (Midterm Project)

A complete deep learning workflow for **classifying blood cells**  
(lymphocyte – neutrophil – monocyte) from **peripheral blood smear images**  
using a simple Convolutional Neural Network (CNN).

---

# 🔬 Biological Background

White blood cells (leukocytes) are essential components of the immune system.  
In a peripheral blood smear, different leukocyte types can be recognized by:

- **Nuclear morphology** (round vs multilobed)  
- **Cytoplasmic color** (basophilic vs eosinophilic)  
- **Granules** (present or absent)

This project demonstrates a simple CNN architecture for automated leukocyte recognition.

---

# 🧠 Model

The model architecture is defined in `src/cnn.py`:

- **2× Conv2D + ReLU + MaxPool layers**  
- **1× Fully connected hidden layer (128 units)**  
- **Output layer with 3 units →**  
  `[lymphocyte, neutrophil, monocyte]`

**Input size:** `224×224` RGB.

---

# 🩸 Data

Place a blood smear image inside:

```
data/blood_smear_example.jpg
```

You may replace this with your own dataset.

---

# ▶️ How to Run

## 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Train the CNN (synthetic dataset for the midterm)
```bash
python code-of-midterm.py
```

This creates:
```
models/simple_cnn.pth
```

## 3️⃣ Visualize the blood smear and RGB channels
```bash
python -m src.goster
python -m src.rgb
```

## 4️⃣ Generate activation heatmap
```bash
python -m src.heat_map
```

This saves:
```
reports/heatmap.png
```

## 5️⃣ Predict on a blood smear image
```bash
python predict.py --image data/blood_smear_example.jpg
```

---

# 💡 Midterm Focus

This repository demonstrates:

- Biomedical deep learning project structure  
- CNN model implementation (PyTorch)  
- Simple image processing (RGB channels)  
- Model interpretability (activation heatmaps)

This repository follows a typical ML pipeline:  
**data → preprocessing → model → training → visualization → prediction**

---

# 📂 Repository Structure

```text
blood-cell-classification/
│
├── data/
│   └── blood_smear_example.jpg
│
├── models/
│   └── simple_cnn.pth
│
├── notebooks/
│   └── training_notebook.ipynb
│
├── reports/
│   ├── heatmap.png
│   └── rgb_channels.png
│
├── src/
│   ├── cnn.py
│   ├── goster.py
│   ├── rgb.py
│   ├── heat_map.py
│   └── __init__.py
│
├── .gitignore
├── code-of-midterm.py
├── predict.py
├── requirements.txt
└── README.md
```
