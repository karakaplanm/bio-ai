"""
================================================================================
  Biyolojide Yapay Zeka - Vize Projesi (Bio-AI Midterm Project)
  Proje: Böcek Filogenisi ve DNA Barkodlama ile Taksonomik Sınıflandırma
  Author: Ahmet
  Repository: bio-ai / assignments / Ahmet_Insect_Phylogeny
================================================================================

Bu ana çalıştırıcı betik (master script), tüm projeyi baştan sona otomatik yürütür:
1. Veri seti doğrulaması ve yüklenmesi (COI mitokondriyal gen dizileri)
2. Keşifsel Veri Analizi (EDA) ve grafiklerin üretimi (GC%, k-mer spektrumu)
3. Hizalamasız Moleküler Filogenetik Ağaç (UPGMA Dendrogramı) ve PCA Analizi
4. Çoklu Yapay Zeka / Makine Öğrenmesi Modellerinin 5-Katlı Çapraz Doğrulaması
5. En başarılı modelin kaydedilmesi ve test kümesinde değerlendirilmesi
6. Örnek DNA dizisi üzerinde otomatik taksonomik tahmin gösterimi
"""

import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure src directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from generate_dataset import generate_dataset
from features import SequenceFeatureExtractor, extract_features_df
from eda import run_eda
from phylogeny import plot_phylogenetic_dendrogram, plot_pca_phylogeny
from train import train_and_evaluate
from predict import load_model, predict_sequence, parse_fasta


def main():
    start_time = time.time()
    print("\n" + "=" * 75)
    print("  BIOLOGY & ARTIFICIAL INTELLIGENCE - MIDTERM PROJECT")
    print("  Insect Phylogenetics & Alignment-Free Taxonomic Classification")
    print("=" * 75)

    data_dir = os.path.join(BASE_DIR, "data")
    reports_dir = os.path.join(BASE_DIR, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    csv_file = os.path.join(data_dir, "insect_phylogeny_data.csv")
    fasta_file = os.path.join(data_dir, "insect_coi_sequences.fasta")

    os.makedirs(figures_dir, exist_ok=True)

    # ---------------------------------------------------------
    # ADIM 1: Veri Seti Kontrolü
    # ---------------------------------------------------------
    print("\n[ADIM 1/5] Veri seti kontrol ediliyor...")
    if not os.path.exists(csv_file) or not os.path.exists(fasta_file):
        print("  -> Veri seti bulunamadı, yeniden üretiliyor...")
        generate_dataset(n_samples_per_order=50, output_dir=data_dir)
    else:
        print(f"  [OK] Veri seti hazır: {csv_file}")
        print(f"  [OK] FASTA dosyası hazır: {fasta_file}")

    # ---------------------------------------------------------
    # ADIM 2: Keşifsel Veri Analizi (EDA)
    # ---------------------------------------------------------
    print("\n[ADIM 2/5] Keşifsel Veri Analizi (EDA) yürütülüyor...")
    run_eda(csv_file, figures_dir)

    # ---------------------------------------------------------
    # ADIM 3: Filogenetik Ağaç ve PCA Kümeleme
    # ---------------------------------------------------------
    print("\n[ADIM 3/5] Moleküler Filogenetik Ağaç (UPGMA) ve PCA hesaplanıyor...")
    X_df, extractor = extract_features_df(csv_file)
    feature_cols = [c for c in X_df.columns if c not in ['order', 'species', 'sample_id']]
    X = X_df[feature_cols].values
    y = X_df['order'].values

    dendro_path = os.path.join(figures_dir, "phylogenetic_dendrogram.png")
    pca_path = os.path.join(figures_dir, "pca_phylogeny_clusters.png")

    plot_phylogenetic_dendrogram(X, y, output_path=dendro_path)
    plot_pca_phylogeny(X, y, output_path=pca_path)
    print("  [OK] Filogenetik grafikler başarıyla oluşturuldu.")

    # ---------------------------------------------------------
    # ADIM 4: Model Eğitimi & Çapraz Doğrulama (Benchmark)
    # ---------------------------------------------------------
    print("\n[ADIM 4/5] Yapay Zeka Modelleri Eğitiliyor & Kıyaslanıyor...")
    metrics = train_and_evaluate(csv_file, BASE_DIR)

    # ---------------------------------------------------------
    # ADIM 5: Test Tahmini Gösterimi
    # ---------------------------------------------------------
    print("\n[ADIM 5/5] Örnek Dizi Üzerinde Test Tahmini Gerçekleştiriliyor...")
    bundle = load_model()
    headers, seqs = parse_fasta(fasta_file)
    
    test_idx = 155  # Hymenoptera example
    sample_hdr = headers[test_idx]
    sample_seq = seqs[test_idx]

    pred_res = predict_sequence(sample_seq, bundle)

    print("\n" + "-" * 50)
    print(f"Test Dizisi : {sample_hdr}")
    print(f"Uzunluk     : {pred_res['sequence_length']} bp")
    print(f"GC İçeriği  : %{pred_res['gc_percent']}")
    print(f"Tahmin      : {pred_res['predicted_order']}")
    print(f"Güven Skoru : %{pred_res['confidence']}")
    print("-" * 50)

    elapsed = time.time() - start_time
    print(f"\n[*] TÜM VİZE PROJESİ PIPELINE'I {elapsed:.2f} SANİYEDE TAMAMLANDI!")
    print(f"[*] Raporlar & Grafikler: {figures_dir}")
    print(f"[*] Eğitilen Model: {os.path.join(BASE_DIR, 'models', 'best_insect_classifier.joblib')}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
