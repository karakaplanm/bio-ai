"""
TGF-Beta Sinyal Yolagi & Kanser EMT/Metastaz Klinik Karar Destek Tahmin Modulu (CLI).

Yeni bir tumor biyopsisi veya RNA-Seq gen ekspresyon profili verildiginde:
1. Sinyal yolagi skorlarini (Kanonik SMAD, EMT, Cadherin Switch, TGF-Beta Paradoks Indeksi) hesaplar,
2. Egitilmis en basarili yapay zeka modeli ile kanser evresini / fenotipini tahmin eder,
3. Biyolojik yorum ve klinik risk degerlendirmesi uretir.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import compute_pathway_features
from eda import STAGE_LABELS_TR


def load_model(model_path: str = None) -> dict:
    """Kaydedilmis en iyi modeli yukler."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "models", "best_tgfb_cancer_classifier.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}. Lutfen once 'python code-of-midterm.py' calistirin.")
    return joblib.load(model_path)


def predict_sample(sample_df: pd.DataFrame, bundle: dict) -> dict:
    """Tek bir hasta RNA-Seq profili uzerinde evre ve TGF-Beta paradoks tahmini yapar."""
    enriched_df = compute_pathway_features(sample_df)
    feature_cols = bundle["feature_columns"]
    X_sample = enriched_df[feature_cols].values

    pipe = bundle["pipeline"]
    class_order = bundle["class_order"]

    pred_idx = int(pipe.predict(X_sample)[0])
    probs = pipe.predict_proba(X_sample)[0]
    pred_class = class_order[pred_idx]

    row = enriched_df.iloc[0]
    if pred_class == "Normal_Epithelial":
        interpretation = "Bazal TGF-β sinyali, yüksek E-kaderin (CDH1) ve korunmuş epitel doku mimarisi."
    elif pred_class == "Early_Stage_Tumor_Suppressive":
        interpretation = (
            "Aktif kanonik TGF-β -> SMAD2/3/4 yolağı; yüksek p21 (CDKN1A) ve p15 (CDKN2B) ile "
            "tümör baskılayıcı hücre döngüsü freni aktif."
        )
    else:
        interpretation = (
            "TGF-β PARADOKSU AKTİF! SMAD4 kaybı/baskılanması ve MAPK/AKT üzerinden EMT (SNAI1, ZEB1, VIM) "
            "ve matriks yıkımı (MMP2/MMP9) tetiklenmiş; yüksek metastaz ve immün kaçış riski."
        )

    return {
        "sample_id": str(row.get("sample_id", "CUSTOM_SAMPLE")),
        "predicted_class": pred_class,
        "predicted_label_tr": STAGE_LABELS_TR.get(pred_class, pred_class),
        "confidence_percent": round(float(probs[pred_idx] * 100.0), 2),
        "probabilities": {
            class_order[i]: round(float(probs[i] * 100.0), 2) for i in range(len(class_order))
        },
        "canonical_smad_score": float(row["Canonical_SMAD_Score"]),
        "emt_transcription_score": float(row["EMT_Transcription_Score"]),
        "cadherin_switch_ratio": float(row["Cadherin_Switch_Ratio"]),
        "tgfb_paradox_index": float(row["TGFB_Paradox_Index"]),
        "biological_interpretation": interpretation
    }


def main():
    parser = argparse.ArgumentParser(description="TGF-Beta Kanser & EMT Evresi Tahmin Araci")
    parser.add_argument("--csv", type=str, default=os.path.join(BASE_DIR, "data", "tgfb_cancer_expression_data.csv"))
    parser.add_argument("--index", type=int, default=10, help="Test edilecek ornek satir indeksi")
    args = parser.parse_args()

    bundle = load_model()
    df = pd.read_csv(args.csv)
    sample_row = df.iloc[[args.index]].copy()
    res = predict_sample(sample_row, bundle)

    print("\n" + "=" * 72)
    print("  TGF-β SİNYAL YOLAĞI & KANSER EMT/METASTAZ KLİNİK TAHMİN RAPORU")
    print("=" * 72)
    print(f"Örnek Kimliği             : {res['sample_id']}")
    print(f"Gerçek Evre (Veri Seti)   : {sample_row['cancer_stage_phenotype'].iloc[0]}")
    print(f"Tahmin Edilen Fenotip     : {res['predicted_label_tr']} ({res['predicted_class']})")
    print(f"Model Güven Skoru         : %{res['confidence_percent']}")
    print(f"Kanonik SMAD Skoru        : {res['canonical_smad_score']:.3f}")
    print(f"EMT Transkripsiyon Skoru  : {res['emt_transcription_score']:.3f}")
    print(f"Cadherin Switch Oranı     : {res['cadherin_switch_ratio']:.3f}")
    print(f"TGF-β Paradoks İndeksi    : {res['tgfb_paradox_index']:.3f}")
    print(f"Biyolojik Yorum           : {res['biological_interpretation']}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

