# ====================================================
# Drug Response Prediction Using Random Forest
# Synthetic Dataset Example for Assignment
# ====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. SYNTHETIC DATA GENERATION
# -----------------------------

np.random.seed(42)

cell_lines = [f"CellLine_{i}" for i in range(100)]
genes = [f"Gene_{j}" for j in range(500)]
drugs = ["DrugA", "DrugB", "DrugC"]

expr_df = pd.DataFrame(
    np.random.rand(100, 500) * 10,
    index=cell_lines,
    columns=genes
)

rows = []
for drug in drugs:
    for cl in cell_lines:
        gene_mean = np.mean(expr_df.loc[cl])
        ic50 = (
            np.random.normal(5, 1.2)
            - 0.01 * gene_mean
        )
        rows.append([cl, drug, ic50])

drug_df = pd.DataFrame(rows, columns=["CELL_LINE", "DRUG", "IC50"])

# -----------------------------
# 2. DRUG SELECTION
# -----------------------------

selected_drug = "DrugA"
sub = drug_df[drug_df["DRUG"] == selected_drug]

common = list(set(expr_df.index) & set(sub["CELL_LINE"]))

X = expr_df.loc[common]
y = sub.set_index("CELL_LINE").loc[common]["IC50"]

# -----------------------------
# 3. SPLIT & SCALE
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -----------------------------
# 4. MODEL & GRID SEARCH
# -----------------------------

rf = RandomForestRegressor(random_state=42)

params = {
    "n_estimators": [150, 250],
    "max_depth": [15, 30]
}

grid = GridSearchCV(rf, params, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train_s, y_train)

model = grid.best_estimator_

# -----------------------------
# 5. EVALUATION
# -----------------------------

pred = model.predict(X_test_s)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n===== MODEL RESULTS =====")
print("Best Params:", grid.best_params_)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# -----------------------------
# 6. SAMPLE PREDICTIONS
# -----------------------------

sample = X_test.iloc[:3]
sample_pred = model.predict(scaler.transform(sample))

print("\nSample Predictions:")
for i, p in enumerate(sample_pred, 1):
    print(f"Prediction {i}: {p:.4f}")
