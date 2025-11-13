# ==========================================
# Predicting Drug Response in Cancer Cells Using Machine Learning
# Simulated (Fake) Dataset Example
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------
# 1. YAPAY (SİMÜLE EDİLMİŞ) VERİ OLUŞTUR
# ------------------------------------------

np.random.seed(42)

# 100 hücre hattı, 500 gen, 5 farklı ilaç
cell_lines = [f"CL_{i}" for i in range(100)]
genes = [f"Gene_{j}" for j in range(500)]
drugs = ["Cisplatin", "Paclitaxel", "Doxorubicin", "Gefitinib", "Sorafenib"]

# Gen ekspresyonu verisi
expr = pd.DataFrame(
    np.random.rand(len(cell_lines), len(genes)) * 10,
    index=cell_lines,
    columns=genes
)

# İlaç yanıt verisi (her ilaç için IC50)
rows = []
for drug in drugs:
    for cl in cell_lines:
        ic50 = (
            np.random.normal(loc=5, scale=1.5)
            - 0.02 * np.mean(expr.loc[cl])  # zayıf bir biyolojik ilişki
        )
        rows.append([cl, drug, ic50])

drug = pd.DataFrame(rows, columns=["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"])

print("Fake Expression Shape:", expr.shape)
print("Fake Drug Response Shape:", drug.shape)

# ------------------------------------------
# 2. BELİRLİ BİR İLAÇ SEÇ
# ------------------------------------------
drug_name = "Cisplatin"
drug_sub = drug[drug["DRUG_NAME"] == drug_name]

# Ortak hücre hatlarını bul
common_cells = list(set(expr.index) & set(drug_sub["CELL_LINE_NAME"]))

# Gen ekspresyonu (X) ve IC50 (y)
X = expr.loc[common_cells]
y = drug_sub.set_index("CELL_LINE_NAME").loc[common_cells]["LN_IC50"]

print(f"\nSelected Drug: {drug_name}")
print(f"Samples Used: {len(y)}")

# ------------------------------------------
# 3. VERİYİ EĞİTİM / TEST OLARAK AYIR
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------
# 4. MODEL EĞİTİMİ
# ------------------------------------------
rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None]
}

grid = GridSearchCV(rf, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# ------------------------------------------
# 5. DEĞERLENDİRME
# ------------------------------------------
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Best Parameters: {grid.best_params_}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R²: {r2:.4f}")

# ------------------------------------------
# 6. ÖRNEK TAHMİN
# ------------------------------------------
example = pd.DataFrame(X_test.iloc[:5])
example_pred = best_model.predict(scaler.transform(example))
print("\nExample Predictions (IC50):")
for i, pred in enumerate(example_pred):
    print(f"Sample {i+1}: {pred:.3f}")
