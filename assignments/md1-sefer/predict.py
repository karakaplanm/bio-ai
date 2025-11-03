import argparse, sys, json
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
model_path = BASE/"models"/"best_model.joblib"
clf = joblib.load(model_path)

# Özellik adları CSV'den okunur ki sıraya uyulsun
df = pd.read_csv(BASE/"data"/"breast_cancer.csv")
feature_names = [c for c in df.columns if c != "target"]

parser = argparse.ArgumentParser(description="Breast Cancer Classifier (0=malignant, 1=benign)")
parser.add_argument("--input", type=str, required=False, help='Örn: "mean radius=14.0,mean texture=20.0" şeklinde')

args = parser.parse_args()

if not args.input:
    # Eğer input yoksa ortalama bir satırı alıp örnek tahmin yapalım
    x = df[feature_names].iloc[[0]]
else:
    # "feat=val,feat=val" metnini sözlüğe çevir
    pairs = [p.strip() for p in args.input.split(",") if p.strip()]
    kv = {}
    for p in pairs:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = float(v.strip())
    # Tüm feature'lar için değer topla, eksik olanı ortalama ile doldur
    row = {}
    means = df[feature_names].mean()
    for f in feature_names:
        row[f] = kv.get(f, means[f])
    x = pd.DataFrame([row])[feature_names]

pred = int(clf.predict(x)[0])
proba = getattr(clf, "predict_proba", None)
out = {"prediction": pred}
if proba is not None:
    p = clf.predict_proba(x)[0].tolist()
    out["proba"] = p

print(json.dumps(out, ensure_ascii=False, indent=2))
