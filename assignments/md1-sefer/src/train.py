import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

BASE = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE/"data"/"breast_cancer.csv")

X = df.drop(columns=['target']).values
y = df['target'].values

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Candidate models (iris örneği ile benzer çeşitlilik)
candidates = {
    "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "svc": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))]),
    "mlp": Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42))])
}

results = {}
best_name, best_model, best_f1 = None, None, -1.0

for name, model in candidates.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)
    pre = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)

    results[name] = {"f1": float(f1), "accuracy": float(acc), "precision": float(pre), "recall": float(rec)}
    if f1 > best_f1:
        best_name, best_model, best_f1 = name, model, f1

# save metrics
reports_dir = BASE / "reports"
reports_dir.mkdir(exist_ok=True, parents=True)
with open(reports_dir/"metrics.json", "w", encoding="utf-8") as f:
    json.dump({"validation": results, "best_model": best_name}, f, indent=2)

# save best model
models_dir = BASE / "models"
models_dir.mkdir(exist_ok=True, parents=True)
joblib.dump(best_model, models_dir/"best_model.joblib")

print("Eğitim tamamlandı.")
print("Best model:", best_name, "F1:", round(best_f1, 4))
print("Metrikler -> reports/metrics.json")
print("Model -> models/best_model.joblib")
