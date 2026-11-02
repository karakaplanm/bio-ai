"""
Model Training, Cross-Validation, and Evaluation Pipeline
Trains and evaluates multiple AI/ML models on insect DNA barcoding features:
- Random Forest Classifier
- HistGradientBoosting Classifier
- Support Vector Machine (RBF Kernel)
- Logistic Regression (L2 Regularized)

Outputs:
- 5-Fold Stratified Cross-Validation benchmarks
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1)
- reports/metrics.json
- reports/figures/confusion_matrix.png
- reports/figures/model_comparison.png
- reports/figures/feature_importance.png
- models/best_insect_classifier.joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from features import SequenceFeatureExtractor


def train_and_evaluate(csv_path, output_dir):
    """
    Main training and benchmark routine.
    """
    print("=" * 65)
    print("  MACHINE LEARNING TRAINING & BENCHMARK - INSECT CLASSIFICATION ")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    X_raw = df['sequence'].values
    y_raw = df['order'].values

    # Encode taxonomic labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # Extract biological alignment-free sequence features
    print(f"\n[+] Extracting k-mer (k=3) and physicochemical features from {len(X_raw)} sequences...")
    extractor = SequenceFeatureExtractor(k=3, include_physicochemical=True)
    X = extractor.fit_transform(X_raw)
    feature_names = extractor.get_feature_names_out()
    print(f"[+] Total feature dimensions: {X.shape[1]}")

    # Train / Test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Candidate models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=5.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_pipelines = {}

    print("\n--- 5-Fold Stratified Cross-Validation Results ---")
    for name, clf in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_macro')
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results[name] = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "y_pred": y_pred
        }

        print(f"{name:24s} | CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.3f}) | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

    # Select best model based on Test F1 Score
    best_name = max(results, key=lambda k: results[k]['test_f1'])
    best_pipe = fitted_pipelines[best_name]
    best_result = results[best_name]
    print(f"\n[*] Best Selected Model: {best_name} with Test F1 = {best_result['test_f1']:.4f}")

    # Reports directory
    fig_dir = os.path.join(output_dir, "reports", "figures")
    rep_dir = os.path.join(output_dir, "reports")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, best_result['y_pred'])
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="white")
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix: {best_name} (Insect Order Classification)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Insect Order", fontsize=11, fontweight='semibold')
    plt.ylabel("True Insect Order", fontsize=11, fontweight='semibold')
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrix: {cm_path}")

    # 2. Model Benchmark Comparison Plot
    df_res = pd.DataFrame([
        {"Model": k, "Metric": "Accuracy", "Score": v['test_accuracy']}
        for k, v in results.items()
    ] + [
        {"Model": k, "Metric": "Macro F1", "Score": v['test_f1']}
        for k, v in results.items()
    ] + [
        {"Model": k, "Metric": "CV F1 (Mean)", "Score": v['cv_f1_mean']}
        for k, v in results.items()
    ])

    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    sns.barplot(
        data=df_res,
        x='Model',
        y='Score',
        hue='Metric',
        palette='crest',
        edgecolor='black',
        linewidth=0.6
    )
    plt.title("Performance Comparison Across AI Algorithms", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score", fontsize=11, fontweight='semibold')
    plt.xlabel("Algorithm", fontsize=11, fontweight='semibold')
    plt.legend(frameon=True)
    plt.tight_layout()
    comp_path = os.path.join(fig_dir, "model_comparison.png")
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved model comparison plot: {comp_path}")

    # 3. Feature Importance Analysis (Random Forest)
    rf_clf = fitted_pipelines["Random Forest"].named_steps['classifier']
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    sns.barplot(
        x=top_scores,
        y=top_features,
        palette='viridis',
        edgecolor='black',
        linewidth=0.6
    )
    plt.title("Top 15 Most Discriminative Sequence Features (Random Forest)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Gini Feature Importance", fontsize=11, fontweight='semibold')
    plt.ylabel("Feature (k-mer / Physicochemical)", fontsize=11, fontweight='semibold')
    plt.tight_layout()
    fi_path = os.path.join(fig_dir, "feature_importance.png")
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved feature importance plot: {fi_path}")

    # Detailed per-class classification report
    report_dict = classification_report(
        y_test,
        best_result['y_pred'],
        target_names=class_names,
        output_dict=True
    )

    # Save metrics JSON
    metrics_summary = {
        "best_model": best_name,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "classes": class_names,
        "models_benchmark": {
            k: {
                "cv_f1_mean": v["cv_f1_mean"],
                "cv_f1_std": v["cv_f1_std"],
                "test_accuracy": v["test_accuracy"],
                "test_precision": v["test_precision"],
                "test_recall": v["test_recall"],
                "test_f1": v["test_f1"]
            }
            for k, v in results.items()
        },
        "best_model_per_class_report": report_dict
    }

    metrics_path = os.path.join(rep_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[+] Saved metrics report: {metrics_path}")

    # Save best model bundle
    bundle = {
        "model_name": best_name,
        "pipeline": best_pipe,
        "extractor": extractor,
        "label_encoder": label_encoder,
        "classes": class_names,
        "feature_names": feature_names
    }
    model_path = os.path.join(models_dir, "best_insect_classifier.joblib")
    joblib.dump(bundle, model_path)
    print(f"[+] Saved serialized model bundle: {model_path}")

    return metrics_summary


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_file = os.path.join(base_dir, "data", "insect_phylogeny_data.csv")
    train_and_evaluate(csv_file, base_dir)
