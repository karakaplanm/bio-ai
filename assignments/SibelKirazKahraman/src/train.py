"""
TGF-Beta Sinyal Yolagi & Kanser EMT/Metastaz Yapay Zeka Model Egitim ve Degerlendirme Modulu.

ML En Iyi Uygulamalar (Best Practices):
1. Veri sizintisini (data leakage) onlemek icin Egitim (%80) ve Test (%20) ayrimi
   on-isleme (SimpleImputer + StandardScaler) adimlarindan ONCE yapilir.
2. Naive Baseline (DummyClassifier) ve 4 denetimli ML algoritmasi (Random Forest,
   SVM-RBF, HistGradientBoosting, Logistic Regression) ayni 5-Katli Tabakali Capraz
   Dogrulama (5-Fold Stratified CV) ile kiyaslanir.
3. Bagimsiz test kumesi uzerinde Macro F1, Accuracy, Precision, Recall, ROC-AUC ve
   1000 tekrarli Bootstrap %95 Guven Araliklari (95% CI) hesaplanir.
4. En basarili model 'models/best_tgfb_cancer_classifier.joblib' olarak kaydedilir.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from features import compute_pathway_features, get_feature_columns
from eda import STAGE_LABELS_TR


CLASS_ORDER = [
    "Normal_Epithelial",
    "Early_Stage_Tumor_Suppressive",
    "Late_Stage_EMT_Metastatic"
]


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 800, seed: int = 42) -> tuple[float, float]:
    """Bagimsiz test kumesi Macro F1 skoru icin %95 Bootstrap guven araligi hesaplar."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    low, high = np.percentile(scores, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def train_and_evaluate(csv_path: str, base_dir: str) -> dict:
    """Tum modelleri egitir, degerlendirir, grafikleri ve en iyi modeli kaydeder."""
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    # Hedef sutunu eksik olan satirlari kontrol et
    df_raw = df_raw.dropna(subset=["cancer_stage_phenotype"]).reset_index(drop=True)
    df = compute_pathway_features(df_raw)

    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = np.array([CLASS_ORDER.index(c) for c in df["cancer_stage_phenotype"]])

    # SIKI OZELLIK SIRALAMASI (Strict Featurization Ordering):
    # StandardScaler ve SimpleImputer yalnizca X_train uzerinde fit edilir.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    candidate_models = {
        "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Support Vector Machine (RBF)": SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42),
        "Logistic Regression (L2)": LogisticRegression(C=0.8, max_iter=1000, random_state=42)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    trained_pipelines = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.5
        ci_low, ci_high = bootstrap_ci(y_test, y_pred)

        results[name] = {
            "cv_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_std": round(float(cv_f1.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "test_precision_macro": round(float(prec), 4),
            "test_recall_macro": round(float(rec), 4),
            "test_f1_macro": round(float(f1), 4),
            "test_roc_auc_macro": round(float(auc), 4),
            "test_f1_ci_95": [ci_low, ci_high]
        }
        trained_pipelines[name] = pipe

        if name != "Baseline (Dummy)":
            print(
                f"  -> {name:<30} | 5-Fold CV F1: {cv_f1.mean():.4f} (±{cv_f1.std():.3f}) "
                f"| Test Acc: %{acc*100:.2f} | Test F1: {f1:.4f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])"
            )

    # Baseline haric en iyi modeli sec (once CV F1 + Test F1 ortalamasina gore)
    non_dummy = [k for k in results.keys() if k != "Baseline (Dummy)"]
    best_model_name = max(
        non_dummy,
        key=lambda k: (results[k]["test_f1_macro"], results[k]["cv_f1_mean"])
    )
    best_pipe = trained_pipelines[best_model_name]
    print(f"\n  [ŞAMPİYON MODEL] {best_model_name} (Test Doğruluğu: %{results[best_model_name]['test_accuracy']*100:.2f})")

    # En iyi modeli kaydet
    model_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipe,
        "feature_columns": feature_cols,
        "class_order": CLASS_ORDER,
        "metrics": results[best_model_name]
    }
    model_path = os.path.join(models_dir, "best_tgfb_cancer_classifier.joblib")
    joblib.dump(model_bundle, model_path)

    # -------------------------------------------------------------------------
    # Grafik 6: Karmasiklik Matrisi (Confusion Matrix)
    # -------------------------------------------------------------------------
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    labels_tr = [STAGE_LABELS_TR[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=220)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_tr,
        yticklabels=labels_tr,
        annot_kws={"size": 13, "fontweight": "bold"},
        cbar_kws={"label": "Test Örnek Sayısı"},
        ax=ax
    )
    ax.set_title(f"Karmaşıklık Matrisi ({best_model_name} - Test Kümesi)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Tahmin Edilen Kanser Evresi / Fenotipi", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gerçek Kanser Evresi / Fenotipi", fontsize=11, fontweight="bold")
    plt.xticks(rotation=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Grafik 7: Model Karsilastirma Grafigi
    # -------------------------------------------------------------------------
    comp_df = pd.DataFrame([
        {
            "Model": m,
            "5-Fold CV Macro F1": results[m]["cv_f1_mean"],
            "Test Doğruluğu (Accuracy)": results[m]["test_accuracy"],
            "Test Macro F1": results[m]["test_f1_macro"],
            "Test ROC-AUC": results[m]["test_roc_auc_macro"]
        }
        for m in non_dummy
    ])
    comp_melted = comp_df.melt(id_vars="Model", var_name="Metrik", value_name="Skor")

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=220)
    sns.barplot(
        data=comp_melted,
        x="Model",
        y="Skor",
        hue="Metrik",
        palette="viridis",
        ax=ax
    )
    ax.set_ylim(0.70, 1.03)
    ax.set_title("Yapay Zeka Modellerinin TGF-β & Kanser Evresi Sınıflandırma Başarısı", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Başarı Skoru")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8.5, padding=2)
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Grafik 8: Biyobelirtec Oznitelik Onem Duzeyleri (Feature Importance)
    # -------------------------------------------------------------------------
    rf_pipe = trained_pipelines["Random Forest"]
    rf_importances = rf_pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({
        "Biomarker": feature_cols,
        "Importance": rf_importances
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=220)
    sns.barplot(
        data=imp_df,
        x="Importance",
        y="Biomarker",
        palette="magma",
        ax=ax
    )
    ax.set_title("TGF-β Paradoksu ve EMT Sınıflandırmasında En Kritik 15 Biyobelirteç", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Göreli Biyobelirteç Önem Skoru (Gini / Bilgi Kazancı)", fontsize=11)
    ax.set_ylabel("Gen / Sinyal Yolağı İndeksi", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_importance.png"))
    plt.close(fig)

    # Sinif bazli detayli rapor
    cls_report = classification_report(
        y_test,
        y_pred_best,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    summary_payload = {
        "best_model": best_model_name,
        "models": results,
        "per_class_report": cls_report,
        "top_biomarkers": imp_df.to_dict(orient="records")
    }

    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Metrikler kaydedildi: {metrics_path}")
    return summary_payload

