"""
================================================================================
  Biyolojide Yapay Zeka - Vize Projesi (Bio-AI Midterm Project)
  Proje: TGF-β Sinyal Yolağı Paradoksu, EMT ve Kanser Metastaz Sınıflandırması
  Repository: bio-ai / assignments / TGFB_Cancer_Pathway_AI
================================================================================

Bu ana çalıştırıcı betik (master script), tüm projeyi baştan sona otomatik yürütür:
1. TGF-β sinyal yolağı ve EMT gen ekspresyon veri setinin üretilmesi / doğrulanması (360 örnek, 24 gen)
2. Sinyal yolağı biyolojik öznitelik mühendisliği (Kanonik SMAD, Cadherin Switch, TGF-β Paradoks İndeksi)
3. Keşifsel Veri Analizi (EDA), K-Means Silhouette kümeleme, PCA ve korelasyon grafiklerinin üretimi
4. Çoklu Yapay Zeka Modellerinin (SVM, Random Forest, HistGradientBoosting, Logistic Regression)
   5-Katlı Tabakalı Çapraz Doğrulama (5-Fold Stratified CV) ve %95 Bootstrap Güven Aralığı ile değerlendirilmesi
5. En başarılı modelin kaydedilmesi ve örnek tümör biyopsisi üzerinde klinik tahmin gösterimi
"""

import os
import sys
import time
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from generate_dataset import generate_dataset
from eda import run_eda
from train import train_and_evaluate
from predict import load_model, predict_sample


def main():
    start_time = time.time()
    print("\n" + "=" * 78)
    print("  BIOLOGY & ARTIFICIAL INTELLIGENCE - MIDTERM PROJECT")
    print("  TGF-β Signaling Paradox, EMT & Cancer Metastasis Classification")
    print("=" * 78)

    data_dir = os.path.join(BASE_DIR, "data")
    reports_dir = os.path.join(BASE_DIR, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    csv_file = os.path.join(data_dir, "tgfb_cancer_expression_data.csv")

    os.makedirs(figures_dir, exist_ok=True)

    # ---------------------------------------------------------
    # ADIM 1: Veri Seti Kontrolü ve Üretimi
    # ---------------------------------------------------------
    print("\n[ADIM 1/4] TGF-β ve EMT gen ekspresyon veri seti hazırlanıyor...")
    if not os.path.exists(csv_file):
        generate_dataset(n_per_class=120, seed=42, output_dir=data_dir)
    else:
        print(f"  [OK] Veri seti mevcut: {csv_file}")

    # ---------------------------------------------------------
    # ADIM 2: Keşifsel Veri Analizi (EDA) & Yolak Grafikleri
    # ---------------------------------------------------------
    print("\n[ADIM 2/4] Keşifsel Veri Analizi (EDA), PCA ve Korelasyon Analizi yürütülüyor...")
    eda_stats = run_eda(csv_file, figures_dir)

    # ---------------------------------------------------------
    # ADIM 3: Yapay Zeka Modellerinin Eğitimi & Kıyaslanması
    # ---------------------------------------------------------
    print("\n[ADIM 3/4] Yapay Zeka Modelleri 5-Katlı Çapraz Doğrulama ile Eğitiliyor...")
    train_summary = train_and_evaluate(csv_file, BASE_DIR)

    # ---------------------------------------------------------
    # ADIM 4: Örnek Klinik Tümör Profili Üzerinde Tahmin Gösterimi
    # ---------------------------------------------------------
    print("\n[ADIM 4/4] Örnek Tümör Biyopsisi Üzerinde Klinik Tahmin Gerçekleştiriliyor...")
    bundle = load_model()
    df = pd.read_csv(csv_file)

    # İleri evre metastatik bir örnek seçelim
    late_indices = df.index[df["cancer_stage_phenotype"] == "Late_Stage_EMT_Metastatic"].tolist()
    sample_row = df.iloc[[late_indices[0]]].copy()
    pred_res = predict_sample(sample_row, bundle)

    print("\n" + "-" * 68)
    print(f"Test Örneği       : {pred_res['sample_id']}")
    print(f"Gerçek Fenotip    : {sample_row['cancer_stage_phenotype'].iloc[0]}")
    print(f"Model Tahmini     : {pred_res['predicted_label_tr']}")
    print(f"Güven Skoru       : %{pred_res['confidence_percent']}")
    print(f"Kanonik SMAD Skor : {pred_res['canonical_smad_score']:.3f}")
    print(f"EMT Skoru         : {pred_res['emt_transcription_score']:.3f}")
    print(f"Cadherin Switch   : {pred_res['cadherin_switch_ratio']:.3f}")
    print(f"TGF-β Paradoks    : {pred_res['tgfb_paradox_index']:.3f}")
    print(f"Klinik Yorum      : {pred_res['biological_interpretation']}")
    print("-" * 68)

    elapsed = time.time() - start_time
    print(f"\n[*] TÜM VİZE PROJESİ PIPELINE'I {elapsed:.2f} SANİYEDE BAŞARIYLA TAMAMLANDI!")
    print(f"[*] Üretilen Grafikler : {figures_dir}")
    print(f"[*] Kaydedilen Model   : {os.path.join(BASE_DIR, 'models', 'best_tgfb_cancer_classifier.joblib')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

