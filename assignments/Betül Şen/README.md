# Machine Learning Analysis: Predicting Drug Response Using Simulated Gene Expression Data

This project was created as part of my assignment for the **Bio-AI course**.  
The goal of this study is to simulate a realistic biological dataset and build a machine learning model that predicts how cancer cell lines respond to different drugs based on gene expression levels.

Although the dataset used in this project is artificially generated, the workflow reflects real bioinformatics and computational biology analysis steps.

---

## 📌 1. Project Overview

In this project, I:

- Simulated **gene expression** data for 100 cancer cell lines  
- Created **drug response (IC50)** values for 5 different cancer drugs  
- Selected one specific drug (“Cisplatin”) for predictive modeling  
- Preprocessed and scaled the data  
- Trained a **Random Forest Regressor** model  
- Performed **hyperparameter tuning** using GridSearchCV  
- Evaluated the model using R² and MSE  
- Generated predictions for example samples  

The entire workflow mimics a real-world pharmacogenomics pipeline.

---

## 📌 2. Simulated Dataset Details

### **Gene Expression Matrix**
- Shape: **100 cell lines × 500 genes**
- Values generated randomly between 0 and 10
- Represents normalized gene expression (similar to RNA-Seq TPM values)

### **Drug Response Dataset**
- 5 drugs: *Cisplatin, Paclitaxel, Doxorubicin, Gefitinib, Sorafenib*
- For each drug:
  - IC50 values simulated with biological noise  
  - Weak correlations introduced with gene averages  
  - 100 samples per drug

### **Selected Drug**
The model was trained only on:


---

## 📌 3. Methods and Tools Used

| Step | Description |
|------|-------------|
| **Data Simulation** | NumPy random generation with biological noise |
| **Data Processing** | Pandas manipulation, feature selection |
| **Scaling** | StandardScaler |
| **Train/Test Split** | 80% training – 20% testing |
| **Model** | Random Forest Regressor |
| **Hyperparameter Tuning** | GridSearchCV (3-fold CV) |
| **Metrics** | R² score, MSE |

---

## 📌 4. Model Performance

After tuning, the model achieved:

- **Best Parameters:**  
  - `n_estimators = 200`  
  - `max_depth = 20`  

- **Performance:**  
  - **Test MSE:** around *0.90–1.20* depending on random seed  
  - **Test R²:** typically *0.40–0.55*  

These values are reasonable considering:
- The dataset is **fully synthetic**
- The biological signal is intentionally weak

---

## 📌 5. Example Predictions

The model was used to predict IC50 values for example test samples.  
Results look like this:

Sample 1 → 4.87
Sample 2 → 5.13
Sample 3 → 4.65
Sample 4 → 5.02
Sample 5 → 4.91

## 📌 6. File Structure

BetulSen/
├── analysis.py # Main machine learning workflow
└── readme.md # Project explanation (this file)


---

## 📌 7. Conclusion

This project demonstrates how machine learning can be applied to biological datasets such as gene expression and drug response profiles.  
Although the dataset is simulated, the methods used here are directly applicable to real-world datasets like:

- GDSC (Genomics of Drug Sensitivity in Cancer)
- CCLE (Cancer Cell Line Encyclopedia)
- TCGA-derived expression matrices

Through this assignment, I gained hands-on experience with:

- Data simulation  
- Preprocessing for ML  
- Model selection and evaluation  
- Hyperparameter tuning  
- Interpreting ML results in a biological context  

---

## 📌 8. Requirements

To run the script:

```bash
pip install pandas numpy scikit-learn
Then:

python analysis.py

