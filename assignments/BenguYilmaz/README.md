# 💊 Predicting Drug Response in Cancer Cells

## 📝 Project Description

This project uses **gene expression data** to predict how cancer cells respond to a drug.
We try to guess the **IC50 value**, which shows how **sensitive** or **resistant** a cell is to a drug.

We use **simulated (fake) data** to show how machine learning works. With real data, the model can make real predictions.

---

## 📊 Dataset

* **Gene Expression:** 100 cancer cell lines, 500 genes per cell.
* **Drug Response:** IC50 values for 5 drugs.
* **Selected Drug:** Cisplatin
* **Samples Used:** 100

The data is generated randomly for demonstration purposes.

---

## 🛠️ Method

1. Load gene expression and drug response data.
2. Select the drug you want to predict.
3. Match the cell lines between gene data and drug data.
4. Scale the data to make features comparable.
5. Train a **Random Forest Regressor** to predict IC50.
6. Test the model on new samples and evaluate performance.

---

## 📈 Results

* **Best Parameters:** `max_depth=20`, `n_estimators=100`
* **Mean Squared Error (MSE):** 2.44
* **R² Score:** -0.11 → Low because the data is fake
* **Example Predictions:** IC50 ≈ 4.7

> 🔹 Interpretation: The model is working, but because the data is simulated, predictions do not have real biological meaning.

---

## 🚀 How to Use

1. Run the Python script: `drug_response_prediction.py`
2. The script will generate fake data and train a model.
3. You can change the drug name in the code to test other drugs.
4. Check the predicted IC50 values for new cell samples.

---

## ⚠️ Notes

* This is a **demonstration project** for learning purposes.
* With **real data** (from GDSC, CCLE, or CTRP), the model can learn **real patterns**.
* This project shows how **machine learning can predict drug responses from gene expression**.