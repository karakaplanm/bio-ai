"""
================================================================================
  BİYOLOJİDE YAPAY ZEKA (BIO-AI) - VİZE PROJESİ
  Öğrenci: Duygu Helin Toprak (03220321035)
  Proje: Preimplantasyon Genetik Taramada (PGT-A) Kromozomal Kopya Sayısı (CNV)
         Profillerinin Yapay Zeka ile Sınıflandırılması
  Konu: Tüp Bebek (IVF), Düşük Kapsamlı NGS (lpWGS) ve Embriyo Anöploidi Tespiti
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 24 İnsan Kromozomu (22 Otozom + 2 Eşey Kromozomu)
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# Sınıf İsimleri
CLASS_NAMES = ["Euploid (Öploid - Sağlıklı)", "Aneuploid (Anöploid - Anormal)", "Mosaic (Mozaik - Parsiyel)"]


# ------------------------------------------------------------------------------
# 1. BİYOLOJİK VERİ SİMÜLASYONU (LOW-PASS NGS & PGT-A DATA GENERATION)
# ------------------------------------------------------------------------------

def generate_pgta_dataset(n_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    """
    Tüp bebekte (IVF) 5. gün blastokist trophektoderm biyopsisinden elde edilen
    düşük kapsamlı yeni nesil dizileme (Low-pass NGS ~0.05x) verilerini simüle eder.

    Biyolojik Dağılım:
    - 0: Euploid (Öploid / Normal Diploid 2n - %50):
         Kopya sayısı oranı ~1.0 (normal diploid), X/Y eşeye bağlı (46,XX veya 46,XY).
    - 1: Aneuploid (Anöploid / Trizomi veya Monozomi - %35):
         Anne yaşıyla ilişkili olarak Trizomi 21 (Down), Trizomi 18 (Edwards),
         Trizomi 13 (Patau), Trizomi 16 veya Monozomi X (Turner). Kopya oranı ~1.5 veya ~0.5.
    - 2: Mosaic (Mozaik Embriyo - %15):
         Biyopsi hücrelerinin bir kısmı normal, bir kısmı anöploid (%30-%50 oran).
         Kopya oranı ~1.25 veya ~0.75.
    """
    np.random.seed(random_seed)
    data = []

    # Biyolojik sınıf oranları
    n_euploid = int(n_samples * 0.50)
    n_aneuploid = int(n_samples * 0.35)
    n_mosaic = n_samples - n_euploid - n_aneuploid

    sample_id = 1

    # 1. ÖPLOİD (SAĞLIKLI) EMBRİYOLAR
    for _ in range(n_euploid):
        # Anne yaşı (26-38 yaş aralığı)
        maternal_age = int(np.random.normal(loc=31, scale=3.5))
        maternal_age = max(24, min(42, maternal_age))
        is_female = np.random.rand() > 0.5
        biopsy_day = np.random.choice([5, 6], p=[0.75, 0.25])

        row = {
            "sample_id": f"EMB_{sample_id:04d}",
            "maternal_age": maternal_age,
            "biopsy_day": biopsy_day,
            "sex": "XX" if is_female else "XY"
        }

        # 22 Otozom için normal diploid okuma oranı (Gürültü ~ normal(1.0, 0.05))
        for chrom in AUTOSOMES:
            row[f"ratio_{chrom}"] = np.random.normal(loc=1.00, scale=0.045)

        # Eşey kromozomları
        if is_female:
            row["ratio_chrX"] = np.random.normal(loc=1.00, scale=0.045)
            row["ratio_chrY"] = np.random.normal(loc=0.02, scale=0.015)
        else:
            row["ratio_chrX"] = np.random.normal(loc=0.50, scale=0.035)
            row["ratio_chrY"] = np.random.normal(loc=0.50, scale=0.035)

        row["target_class"] = 0  # Euploid
        data.append(row)
        sample_id += 1

    # 2. ANÖPLOİD (KROMOZOM ANOMALİSİ) EMBRİYOLAR
    # Trizomiye en yatkın kromozomlar: 21, 16, 18, 15, 22, 13
    common_aneuploid_chroms = ["chr21", "chr16", "chr18", "chr15", "chr22", "chr13", "chrX"]

    for _ in range(n_aneuploid):
        # İleri anne yaşı anöploidi riskini belirgin şekilde artırır
        maternal_age = int(np.random.normal(loc=37, scale=3.8))
        maternal_age = max(28, min(45, maternal_age))
        is_female = np.random.rand() > 0.5
        biopsy_day = np.random.choice([5, 6], p=[0.55, 0.45])

        row = {
            "sample_id": f"EMB_{sample_id:04d}",
            "maternal_age": maternal_age,
            "biopsy_day": biopsy_day,
            "sex": "XX" if is_female else "XY"
        }

        # Temel 22 otozom (diploid)
        for chrom in AUTOSOMES:
            row[f"ratio_{chrom}"] = np.random.normal(loc=1.00, scale=0.050)

        if is_female:
            row["ratio_chrX"] = np.random.normal(loc=1.00, scale=0.050)
            row["ratio_chrY"] = np.random.normal(loc=0.02, scale=0.015)
        else:
            row["ratio_chrX"] = np.random.normal(loc=0.50, scale=0.040)
            row["ratio_chrY"] = np.random.normal(loc=0.50, scale=0.040)

        # Anomali Enjeksiyonu (Trizomi = +1 kopya / Monozomi = -1 kopya)
        affected_chrom = np.random.choice(common_aneuploid_chroms)
        is_trisomy = np.random.rand() > 0.35  # %65 trizomi, %35 monozomi

        if affected_chrom == "chrX":
            if not is_trisomy and is_female:
                # 45,X Turner Sendromu
                row["ratio_chrX"] = np.random.normal(loc=0.50, scale=0.040)
            else:
                row["ratio_chrX"] = np.random.normal(loc=1.50, scale=0.050)
        else:
            if is_trisomy:
                # Trizomi (3 Kopya: Normal 1.0 -> 1.50)
                row[f"ratio_{affected_chrom}"] = np.random.normal(loc=1.50, scale=0.055)
            else:
                # Monozomi (1 Kopya: Normal 1.0 -> 0.50)
                row[f"ratio_{affected_chrom}"] = np.random.normal(loc=0.50, scale=0.045)

        row["target_class"] = 1  # Aneuploid
        data.append(row)
        sample_id += 1

    # 3. MOZAİK EMBRİYOLAR (PARSİYEL KOPYA DEĞİŞİMİ)
    for _ in range(n_mosaic):
        maternal_age = int(np.random.normal(loc=34, scale=3.5))
        maternal_age = max(26, min(43, maternal_age))
        is_female = np.random.rand() > 0.5
        biopsy_day = np.random.choice([5, 6], p=[0.60, 0.40])

        row = {
            "sample_id": f"EMB_{sample_id:04d}",
            "maternal_age": maternal_age,
            "biopsy_day": biopsy_day,
            "sex": "XX" if is_female else "XY"
        }

        for chrom in AUTOSOMES:
            row[f"ratio_{chrom}"] = np.random.normal(loc=1.00, scale=0.048)

        if is_female:
            row["ratio_chrX"] = np.random.normal(loc=1.00, scale=0.048)
            row["ratio_chrY"] = np.random.normal(loc=0.02, scale=0.015)
        else:
            row["ratio_chrX"] = np.random.normal(loc=0.50, scale=0.038)
            row["ratio_chrY"] = np.random.normal(loc=0.50, scale=0.038)

        # Mozaik Anomali (%30-%50 arası anormal hücre oranı)
        affected_chrom = np.random.choice(common_aneuploid_chroms[:5])
        mosaic_gain = np.random.rand() > 0.5

        if mosaic_gain:
            # Mozaik Trizomi (1.25 kopya seviyesi)
            row[f"ratio_{affected_chrom}"] = np.random.normal(loc=1.26, scale=0.050)
        else:
            # Mozaik Monozomi (0.75 kopya seviyesi)
            row[f"ratio_{affected_chrom}"] = np.random.normal(loc=0.74, scale=0.045)

        row["target_class"] = 2  # Mosaic
        data.append(row)
        sample_id += 1

    df = pd.DataFrame(data).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# 2. BİYOİNFORMATİK ÖZNİTELİK MÜHENDİSLİĞİ (GENOMIC FEATURE EXTRACTION)
# ------------------------------------------------------------------------------

def engineer_genomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham kromozom okuma oranlarından klinikte kullanılan biyoinformatik 
    ve kalite kontrol (QC) metriklerini hesaplar:
    1. Kromozom Başına Z-Skorları: Z = (Ratio - 1.0) / Baseline_Std
    2. MAPD (Median Absolute Pairwise Difference): NGS dizileme gürültü indeksi
    3. Max |Z-Skoru|: En yüksek sapma gösteren kromozomun sapma katsayısı
    4. Anöploidi Yükü (Aneuploidy Load): Sapmaların mutlak toplamı
    5. Anne Yaşı ve Biyopsi Günü Entegrasyonu
    """
    df_feat = df.copy()

    # 1. Z-Skorları (22 Otozom için)
    baseline_std = 0.05
    for chrom in AUTOSOMES:
        col = f"ratio_{chrom}"
        df_feat[f"zscore_{chrom}"] = (df_feat[col] - 1.0) / baseline_std

    # 2. MAPD Skoru (NGS Dizileme Kalitesi / Gürültü Seviyesi)
    autosome_cols = [f"ratio_{chrom}" for chrom in AUTOSOMES]
    ratios_matrix = df_feat[autosome_cols].values
    pairwise_diffs = np.abs(np.diff(ratios_matrix, axis=1))
    df_feat["mapd_noise"] = np.median(pairwise_diffs, axis=1)

    # 3. Maksimum Mutlak Z-Skoru (En anormal kromozomu yakalar)
    zscore_cols = [f"zscore_{chrom}" for chrom in AUTOSOMES]
    df_feat["max_abs_zscore"] = np.max(np.abs(df_feat[zscore_cols].values), axis=1)

    # 4. Genel Anöploidi Yükü (Toplam Sapma Derecesi)
    df_feat["aneuploidy_load"] = np.sum(np.abs(ratios_matrix - 1.0) > 0.15, axis=1)

    # 5. Varyans Katsayısı (Kromozomlar arası dalgalanma)
    df_feat["autosome_variance"] = np.var(ratios_matrix, axis=1)

    return df_feat


# ------------------------------------------------------------------------------
# 3. ÇOKLU MODEL EĞİTİMİ, ÇAPRAZ DOĞRULAMA VE TEST
# ------------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, output_dir: str = "."):
    """
    Yapay zeka modellerini eğitir, 5-Katlı Çapraz Doğrulama ile kıyaslar
    ve test kümesinde değerlendirir.
    """
    print("\n" + "=" * 75)
    print("  [ADIM 1/4] BİYOİNFORMATİK ÖZNİTELİK MÜHENDİSLİĞİ")
    print("=" * 75)

    df_featured = engineer_genomic_features(df)

    # Modele girecek öznitelik sütunları
    ignore_cols = ["sample_id", "sex", "target_class"]
    feature_cols = [c for c in df_featured.columns if c not in ignore_cols]

    X = df_featured[feature_cols].values
    y = df_featured["target_class"].values

    print(f"[*] Toplam Embriyo Sayısı       : {len(df)}")
    print(f"[*] Öploid (Sağlıklı 2n)       : {sum(y == 0)}")
    print(f"[*] Anöploid (Anormal)          : {sum(y == 1)}")
    print(f"[*] Mozaik (Parsiyel)           : {sum(y == 2)}")
    print(f"[*] Kullanılan Genomik Özellik : {len(feature_cols)}")

    # %80 Eğitim, %20 Bağımsız Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standartlaştırma
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Kıyaslanacak 4 Farklı Model
    models = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel='rbf', C=2.0, probability=True, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42)
    }

    print("\n" + "=" * 75)
    print("  [ADIM 2/4] 5-KATLI ÇAPRAZ DOĞRULAMA (5-FOLD STRATIFIED CV)")
    print("=" * 75)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
        print(f" • {name:<30}: 5-Fold Macro F1 = {scores.mean():.4f} (± {scores.std():.4f})")

    print("\n" + "=" * 75)
    print("  [ADIM 3/4] BAĞIMSIZ TEST KÜMESİNDE PERFORMANS METRİKLERİ (%20 Test)")
    print("=" * 75)

    results = []
    trained_models = {}
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1": f1,
            "Macro ROC-AUC": auc
        })

    metrics_df = pd.DataFrame(results).sort_values(by="Macro ROC-AUC", ascending=False)
    print(metrics_df.to_string(index=False))

    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[★] En Başarılı Model: {best_model_name}")

    best_pred = best_model.predict(X_test_scaled)
    print("\n--- Sınıflandırma Raporu (En İyi Model) ---")
    print(classification_report(y_test, best_pred, target_names=["Euploid", "Aneuploid", "Mosaic"]))

    # --------------------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME VE GRAFİKLER
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  [ADIM 4/4] BİYOİNFORMATİK & MODEL GRAFİKLERİ ÇİZİLİYOR")
    print("=" * 75)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(wspace=0.25, hspace=0.32)

    # 1. Grafik: Çoklu Sınıf Karışıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=["Euploid", "Aneuploid", "Mosaic"],
                yticklabels=["Euploid", "Aneuploid", "Mosaic"])
    axes[0, 0].set_title(f"Karışıklık Matrisi ({best_model_name.split()[0]})", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Yapay Zeka Tahmini", fontsize=11)
    axes[0, 0].set_ylabel("Gerçek Klinik Teşhis", fontsize=11)

    # 2. Grafik: En İyi Model İçin Sınıf Bazında ROC-AUC Eğrileri (One-vs-Rest)
    y_prob_best = best_model.predict_proba(X_test_scaled)
    for class_idx, class_label in enumerate(["Euploid", "Aneuploid", "Mosaic"]):
        y_true_binary = (y_test == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_best[:, class_idx])
        auc_val = roc_auc_score(y_true_binary, y_prob_best[:, class_idx])
        axes[0, 1].plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {auc_val:.3f})")

    axes[0, 1].plot([0, 1], [0, 1], color='navy', linestyle='--', label='Rastgele Sınıflandırma')
    axes[0, 1].set_title("Sınıf Bazında ROC-AUC Eğrileri", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("False Positive Rate (1 - Özgüllük)", fontsize=11)
    axes[0, 1].set_ylabel("True Positive Rate (Duyarlılık / Recall)", fontsize=11)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Grafik: Random Forest Genomik Özellik Önemi (Feature Importance)
    rf_model = trained_models["Random Forest Classifier"]
    importances = rf_model.feature_importances_
    top_indices = np.argsort(importances)[-12:]

    axes[1, 0].barh(range(len(top_indices)), importances[top_indices], color='#2b8cbe')
    axes[1, 0].set_yticks(range(len(top_indices)))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in top_indices], fontsize=10)
    axes[1, 0].set_xlabel("Gini Önem Düzeyi (Feature Importance)", fontsize=11)
    axes[1, 0].set_title("En Belirleyici 12 Genomik & Klinik Özellik", fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # 4. Grafik: In Silico Karyotip CNV Profili (Biyolojik İllüstrasyon)
    # Örnek 4 embriyonun 22 otozomdaki kopya sayı oranı dalgalanması
    sample_euploid = df[df["target_class"] == 0].iloc[0]
    sample_aneuploid = df[df["target_class"] == 1].iloc[0]
    sample_mosaic = df[df["target_class"] == 2].iloc[0]

    x_vals = np.arange(len(AUTOSOMES))
    euploid_ratios = [sample_euploid[f"ratio_{c}"] for c in AUTOSOMES]
    aneuploid_ratios = [sample_aneuploid[f"ratio_{c}"] for c in AUTOSOMES]
    mosaic_ratios = [sample_mosaic[f"ratio_{c}"] for c in AUTOSOMES]

    axes[1, 1].plot(x_vals, euploid_ratios, 'o-', color='#2ca02c', label="Öploid (Normal Diploid)", lw=1.5)
    axes[1, 1].plot(x_vals, aneuploid_ratios, 's-', color='#d62728', label="Anöploid (Trizomi/Monozomi)", lw=1.5)
    axes[1, 1].plot(x_vals, mosaic_ratios, '^-', color='#ff7f0e', label="Mozaik (Parsiyel Sapma)", lw=1.5)

    axes[1, 1].axhline(1.0, color='black', linestyle=':', label="Diploid Referans (2n=1.0)")
    axes[1, 1].axhline(1.5, color='red', linestyle='--', alpha=0.5, label="Trizomi Eşiği (3n=1.5)")
    axes[1, 1].axhline(0.5, color='blue', linestyle='--', alpha=0.5, label="Monozomi Eşiği (1n=0.5)")

    axes[1, 1].set_xticks(x_vals)
    axes[1, 1].set_xticklabels([c.replace("chr", "") for c in AUTOSOMES], fontsize=8)
    axes[1, 1].set_xlabel("Kromozom Numarası (1-22)", fontsize=11)
    axes[1, 1].set_ylabel("Normalize Okuma Oranı (Copy Ratio)", fontsize=11)
    axes[1, 1].set_title("Örnek Embriyoların 24 Kromozom CNV Profili", fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "pgt_a_classification_report.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Biyoinformatik & Model Grafiği Kaydedildi: {chart_path}")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "scaler": scaler,
        "feature_cols": feature_cols
    }


# ------------------------------------------------------------------------------
# 4. KLİNİK VAKA ANALİZİ VE TAHMİN DEMOSU (CLINICAL INFERENCE)
# ------------------------------------------------------------------------------

def predict_embryo_sample(raw_dict: dict, model, scaler, feature_cols: list):
    """
    Klinikten gelen tek bir embriyonun 24 kromozom NGS okuma oranlarını 
    ve klinik parametrelerini değerlendirerek teşhis ve öneri üretir.
    """
    df_single = pd.DataFrame([raw_dict])
    df_feat = engineer_genomic_features(df_single)
    
    feat_vector = np.array([df_feat[c].values[0] for c in feature_cols]).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vector)

    probabilities = model.predict_proba(feat_scaled)[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]

    # Klinik öneri mantığı
    recommendations = {
        0: "✅ BİRİNCİL TRANSFER ADAYI: Embriyo öploid (kromozomal olarak normal) bulunmuştur.",
        1: "❌ TRANSFER EDİLEMEZ: Embriyoda tam anöploidi tespit edilmiştir. İmplantasyon başarısızlığı veya düşük riski çok yüksektir.",
        2: "⚠️ KOŞULLU / GENETİK DANIŞMANLIK GEREKLİ: Embriyo mozaik bulunmuştur. Diğer öploid embriyolar önceliklidir; gerekirse hekim onayıyla değerlendirilebilir."
    }

    return {
        "predicted_class": predicted_class,
        "class_label": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {
            "Euploid": probabilities[0],
            "Aneuploid": probabilities[1],
            "Mosaic": probabilities[2]
        },
        "recommendation": recommendations[predicted_class],
        "max_zscore": df_feat["max_abs_zscore"].values[0],
        "mapd_noise": df_feat["mapd_noise"].values[0]
    }


def demonstrate_clinical_cases(model, scaler, feature_cols: list):
    """
    Gerçek dünya IVF kliniğinde karşılaşılan 4 tipik vakanın analizi.
    """
    print("\n" + "=" * 75)
    print("  [KLİNİK VAKA GÖSTERİMİ] YENİ EMBRİYO BİYOPSİ TESTLERİ")
    print("=" * 75)

    cases = [
        {
            "vaka_adi": "Vaka 1: Sağlıklı 5. Gün Dişi Embriyo (46,XX)",
            "anne_yasi": 29,
            "biopsy_day": 5,
            "sex": "XX",
            "anomali": None
        },
        {
            "vaka_adi": "Vaka 2: Down Sendromlu Erkek Embriyo (47,XY,+21 - Trizomi 21)",
            "anne_yasi": 39,
            "biopsy_day": 6,
            "sex": "XY",
            "anomali": ("chr21", 1.52)  # Trizomi 21
        },
        {
            "vaka_adi": "Vaka 3: Turner Sendromlu Embriyo (45,X - Monozomi X)",
            "anne_yasi": 34,
            "biopsy_day": 5,
            "sex": "XX",
            "anomali": ("chrX", 0.48)   # Monozomi X
        },
        {
            "vaka_adi": "Vaka 4: Mozaik Trizomi 16 Embriyosu (46,XX / 47,XX,+16)",
            "anne_yasi": 36,
            "biopsy_day": 6,
            "sex": "XX",
            "anomali": ("chr16", 1.27)  # %40 Mozaik kazanç
        }
    ]

    for case in cases:
        sample = {
            "sample_id": "TEST_SAMPLE",
            "maternal_age": case["anne_yasi"],
            "biopsy_day": case["biopsy_day"],
            "sex": case["sex"]
        }

        # Varsayılan normal diploid oranlar
        for chrom in AUTOSOMES:
            sample[f"ratio_{chrom}"] = np.random.normal(1.0, 0.04)

        if case["sex"] == "XX":
            sample["ratio_chrX"] = np.random.normal(1.0, 0.04)
            sample["ratio_chrY"] = 0.02
        else:
            sample["ratio_chrX"] = 0.50
            sample["ratio_chrY"] = 0.50

        # Varsa anomaliyi ekle
        if case["anomali"]:
            chrom, val = case["anomali"]
            sample[f"ratio_{chrom}"] = val

        res = predict_embryo_sample(sample, model, scaler, feature_cols)

        print(f"\n● {case['vaka_adi']}")
        print(f"  Anne Yaşı: {case['anne_yasi']} | Biyopsi Günü: Gün {case['biopsy_day']}")
        print(f"  Yapay Zeka Teşhisi : >>> {res['class_label']} <<< (Güven: %{res['confidence']*100:.2f})")
        print(f"  Olasılık Dağılımı  : Öploid: %{res['probabilities']['Euploid']*100:.1f} | Anöploid: %{res['probabilities']['Aneuploid']*100:.1f} | Mozaik: %{res['probabilities']['Mosaic']*100:.1f}")
        print(f"  Maksimum |Z-Skoru| : {res['max_zscore']:.2f} (Eşik: >3.0 anlamlı anomali)")
        print(f"  Klinik Rapor       : {res['recommendation']}")


# ------------------------------------------------------------------------------
# 5. ANA FONKSİYON (MAIN)
# ------------------------------------------------------------------------------

def main():
    print("""
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║        BİYOLOJİDE YAPAY ZEKA (BIO-AI) DERSİ - VİZE PROJESİ              ║
    ║   Preimplantasyon Genetik Taramada (PGT-A) Kromozomal Kopya Sayısı      ║
    ║        (CNV) Profillerinin Yapay Zeka ile Sınıflandırılması             ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """)

    # 1. Biyolojik NGS Veri Setini Üret
    print("[*] Tüp bebek (IVF) PGT-A NGS veri seti simüle ediliyor...")
    df = generate_pgta_dataset(n_samples=650, random_seed=42)

    # 2. Modelleri Eğit, Karşılaştır ve Grafikleri Oluştur
    bundle = train_and_evaluate(df, output_dir=".")

    # 3. Klinik Vaka Analizini Çalıştır
    demonstrate_clinical_cases(
        model=bundle["best_model"],
        scaler=bundle["scaler"],
        feature_cols=bundle["feature_cols"]
    )

    print("\n" + "=" * 75)
    print(" [✓] Vize projesi başarıyla yürütüldü!")
    print(" [✓] Yüksek çözünürlüklü grafik 'pgt_a_classification_report.png' olarak kaydedildi.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

