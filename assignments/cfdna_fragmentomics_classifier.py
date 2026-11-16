"""
================================================================================
  BİYOLOJİDE YAPAY ZEKA (BIO-AI) - VİZE PROJESİ
  Öğrenci: Sude Merve Bulut (03190321083)
  Proje: Likit Biyopside Hücre Dışı DNA (cfDNA) Fragmantasyon Profilleri ve
         Kısa/Uzun Okuma Oranları (DELFI) ile Erken Evre Kanser Tespiti
  Konu: Kanser Biyolojisi, Epigenetik Nükleozom Pozisyonlanması ve Likit Biyopsi
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Genomik Kollar (39 Otozomal Kromozom Kolu - p ve q kolları)
CHROMOSOME_ARMS = [
    "1p", "1q", "2p", "2q", "3p", "3q", "4p", "4q", "5p", "5q",
    "6p", "6q", "7p", "7q", "8p", "8q", "9p", "9q", "10p", "10q",
    "11p", "11q", "12p", "12q", "13q", "14q", "15q", "16p", "16q",
    "17p", "17q", "18p", "18q", "19p", "19q", "20p", "20q", "21q", "22q"
]

CLASS_NAMES = [
    "Sağlıklı Kontrol (Healthy)",
    "Erken Evre Akciğer Kanseri (NSCLC - Evre I/II)",
    "Erken Evre Kolorektal Kanser (CRC - Evre I/II)"
]


# ------------------------------------------------------------------------------
# 1. LİKİT BİYOPSİ cfDNA FRAGMANOMİK VERİ SİMÜLASYONU (DELFI METODOLOJİSİ)
# ------------------------------------------------------------------------------

def generate_cfdna_dataset(n_samples: int = 720, random_seed: int = 104) -> pd.DataFrame:
    """
    Plazma örneklerinden izole edilen hücre dışı DNA'nın (cfDNA) düşük kapsamlı
    tüm genom dizilemesi (lpWGS) ile elde edilen fragman boyut dağılımlarını simüle eder.

    Biyolojik Temel (Cristiano et al., Nature 2019 - DELFI):
    - Sağlıklı bireylerde cfDNA fragmanları mononükleozomal ~167 bp etrafında sıkı kümelenir.
      Kısa (100-150 bp) / Uzun (151-220 bp) fragman oranı (SL Ratio) genom boyunca stabildir (~0.18).
    - Kanser hastalarında (ctDNA varlığı) kromatin yeniden modellenmesi ve epigenetik
      açık kromatin bölgeleri nedeniyle fragman uzunlukları düzensizleşir; kısa fragmanlar
      (100-150 bp) artar ve kanser türüne özgü kromozom kollarında belirgin sapmalar görülür.
    """
    np.random.seed(random_seed)
    data = []

    n_healthy = int(n_samples * 0.45)
    n_nsclc = int(n_samples * 0.30)
    n_crc = n_samples - n_healthy - n_nsclc

    sample_id = 1

    # 1. SAĞLIKLI KONTROL BİREYLER
    for _ in range(n_healthy):
        age = int(np.clip(np.random.normal(54, 9.5), 30, 80))
        cfdna_conc = np.clip(np.random.normal(8.5, 2.2), 3.0, 16.0)  # ng/mL plazma
        modal_bp = int(np.random.normal(167, 1.5))  # Klasik nükleozomal tepe noktası

        row = {
            "sample_id": f"LIQ_{sample_id:04d}",
            "patient_age": age,
            "cfdna_ng_ml": round(cfdna_conc, 2),
            "modal_fragment_bp": modal_bp
        }

        # 39 kromozom kolu için Short/Long (100-150bp / 151-220bp) fragman oranları
        base_profile = np.random.normal(0.18, 0.015, size=len(CHROMOSOME_ARMS))
        for idx, arm in enumerate(CHROMOSOME_ARMS):
            row[f"sl_ratio_{arm}"] = round(max(0.08, base_profile[idx]), 4)

        # 5' Uç Motifi (End-Motif CCCA frekansı - sağlıklı bireylerde yüksek)
        row["motif_CCCA_freq"] = round(np.random.normal(0.024, 0.002), 5)
        row["target_class"] = 0
        data.append(row)
        sample_id += 1

    # 2. ERKEN EVRE AKCİĞER KANSERİ (NSCLC)
    nsclc_hotspots = ["1q", "3q", "5p", "7p", "8q", "17p", "19q"]
    for _ in range(n_nsclc):
        age = int(np.clip(np.random.normal(63, 8.0), 40, 85))
        cfdna_conc = np.clip(np.random.normal(11.8, 3.8), 6.0, 35.0)
        modal_bp = int(np.random.normal(164.5, 2.6))  # Kısalmaya doğru kayma

        row = {
            "sample_id": f"LIQ_{sample_id:04d}",
            "patient_age": age,
            "cfdna_ng_ml": round(cfdna_conc, 2),
            "modal_fragment_bp": modal_bp
        }

        # Genel genomik gürültü ve fragman kısalması artışı
        base_profile = np.random.normal(0.205, 0.038, size=len(CHROMOSOME_ARMS))
        for idx, arm in enumerate(CHROMOSOME_ARMS):
            val = base_profile[idx]
            if arm in nsclc_hotspots:
                val += np.random.normal(0.068, 0.030)
            row[f"sl_ratio_{arm}"] = round(max(0.10, val), 4)

        # DNASE1L3 aktivite azalmasına bağlı CCCA uç motif düşüşü
        row["motif_CCCA_freq"] = round(np.random.normal(0.020, 0.0028), 5)
        row["target_class"] = 1
        data.append(row)
        sample_id += 1

    # 3. ERKEN EVRE KOLOREKTAL KANSER (CRC)
    crc_hotspots = ["7p", "8q", "13q", "17p", "18q", "20p", "20q"]
    for _ in range(n_crc):
        age = int(np.clip(np.random.normal(61, 8.5), 38, 84))
        cfdna_conc = np.clip(np.random.normal(12.4, 4.1), 6.5, 38.0)
        modal_bp = int(np.random.normal(164.0, 2.8))

        row = {
            "sample_id": f"LIQ_{sample_id:04d}",
            "patient_age": age,
            "cfdna_ng_ml": round(cfdna_conc, 2),
            "modal_fragment_bp": modal_bp
        }

        base_profile = np.random.normal(0.210, 0.040, size=len(CHROMOSOME_ARMS))
        for idx, arm in enumerate(CHROMOSOME_ARMS):
            val = base_profile[idx]
            if arm in crc_hotspots:
                val += np.random.normal(0.072, 0.032)
            row[f"sl_ratio_{arm}"] = round(max(0.10, val), 4)

        row["motif_CCCA_freq"] = round(np.random.normal(0.0195, 0.0028), 5)
        row["target_class"] = 2
        data.append(row)
        sample_id += 1

    df = pd.DataFrame(data).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------------
# 2. FRAGMANOMİK BİYOİNFORMATİK ÖZNİTELİK MÜHENDİSLİĞİ
# ------------------------------------------------------------------------------

def extract_fragmentomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham kromozom kolu Short/Long oranlarından biyoinformatik özet metrikler türetir:
    1. Global Ortalama Kısa/Uzun Oranı (Mean SL Ratio)
    2. Kromozom Kolları Arası Düzensizlik (Genomic Instability Std)
    3. Shannon Fragmantasyon Entropisi (Kromatin heterojenite ölçüsü)
    4. Sağlıklı Referans Profiline Korelasyon (Healthy Reference Correlation)
    5. Hotspot Sapma Skorları (NSCLC vs CRC spesifik kromozom indeksleri)
    """
    df_feat = df.copy()
    arm_cols = [f"sl_ratio_{arm}" for arm in CHROMOSOME_ARMS]
    sl_matrix = df_feat[arm_cols].values

    # 1. Global Ortalama ve Standart Sapma
    df_feat["global_mean_sl"] = np.mean(sl_matrix, axis=1)
    df_feat["global_std_sl"] = np.std(sl_matrix, axis=1)

    # 2. Shannon Entropisi (Fragman dağılım düzensizliği)
    row_sums = np.sum(sl_matrix, axis=1, keepdims=True)
    prob_matrix = sl_matrix / row_sums
    df_feat["fragment_shannon_entropy"] = entropy(prob_matrix, axis=1)

    # 3. Sağlıklı Referans Vektörüne Kosinüs Benzerliği
    healthy_ref = np.full(len(CHROMOSOME_ARMS), 0.18)
    norms = np.linalg.norm(sl_matrix, axis=1) * np.linalg.norm(healthy_ref)
    df_feat["healthy_cosine_sim"] = np.dot(sl_matrix, healthy_ref) / norms

    # 4. Kanser Türü Spesifik Hotspot İndeksleri
    nsclc_cols = [f"sl_ratio_{arm}" for arm in ["1q", "3q", "5p", "19q"]]
    crc_cols = [f"sl_ratio_{arm}" for arm in ["13q", "18q", "20p", "20q"]]

    df_feat["nsclc_signature_index"] = df_feat[nsclc_cols].mean(axis=1) - df_feat["global_mean_sl"]
    df_feat["crc_signature_index"] = df_feat[crc_cols].mean(axis=1) - df_feat["global_mean_sl"]

    # 5. Kısa Fragman Yükü (Eşik > 0.25 olan kromozom kolu sayısı)
    df_feat["aberrant_arms_count"] = np.sum(sl_matrix > 0.25, axis=1)

    return df_feat


# ------------------------------------------------------------------------------
# 3. MODEL EĞİTİMİ, ÇAPRAZ DOĞRULAMA VE DEĞERLENDİRME
# ------------------------------------------------------------------------------

def train_and_evaluate_models(df: pd.DataFrame, output_dir: str = "."):
    print("\n" + "=" * 78)
    print("  [ADIM 1/4] LİKİT BİYOPSİ FRAGMANOMİK ÖZNİTELİK ÇIKARIMI")
    print("=" * 78)

    df_featured = extract_fragmentomic_features(df)
    ignore_cols = ["sample_id", "target_class"]
    feature_cols = [c for c in df_featured.columns if c not in ignore_cols]

    X = df_featured[feature_cols].values
    y = df_featured["target_class"].values

    print(f"[*] Toplam Plazma Örneği Sayısı       : {len(df)}")
    print(f"[*] Sağlıklı Kontrol (Healthy)        : {sum(y == 0)}")
    print(f"[*] Erken Evre Akciğer Kanseri (NSCLC): {sum(y == 1)}")
    print(f"[*] Erken Evre Kolorektal Kanser (CRC): {sum(y == 2)}")
    print(f"[*] Çıkarılan Fragmanomik Özellik     : {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=104, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=200, max_depth=12, random_state=104),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=104),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=104),
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, C=1.5, random_state=104)
    }

    print("\n" + "=" * 78)
    print("  [ADIM 2/4] 5-KATLI TABAKALI ÇAPRAZ DOĞRULAMA (STRATIFIED 5-FOLD CV)")
    print("=" * 78)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=104)
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
        print(f" • {name:<30}: 5-Fold Macro F1 = {scores.mean():.4f} (± {scores.std():.4f})")

    print("\n" + "=" * 78)
    print("  [ADIM 3/4] BAĞIMSIZ TEST KÜMESİ PERFORMANS KARŞILAŞTIRMASI (%20 Test)")
    print("=" * 78)

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro Precision": precision_score(y_test, y_pred, average='macro'),
            "Macro Recall": recall_score(y_test, y_pred, average='macro'),
            "Macro F1": f1_score(y_test, y_pred, average='macro'),
            "Macro ROC-AUC": roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        })

    metrics_df = pd.DataFrame(results).sort_values(by="Macro ROC-AUC", ascending=False)
    print(metrics_df.to_string(index=False))

    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[★] En Yüksek Performanslı Model: {best_model_name}")

    best_pred = best_model.predict(X_test_scaled)
    print("\n--- Detaylı Sınıflandırma Raporu ---")
    print(classification_report(y_test, best_pred, target_names=["Healthy", "NSCLC (Akciğer)", "CRC (Kolorektal)"]))

    # --------------------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME PANELİ
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  [ADIM 4/4] FRAGMANOMİK & MODEL PERFORMANS GRAFİKLERİ OLUŞTURULUYOR")
    print("=" * 78)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # 1. Grafik: Karışıklık Matrisi
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 0],
                xticklabels=["Sağlıklı", "Akciğer (NSCLC)", "Kolorektal (CRC)"],
                yticklabels=["Sağlıklı", "Akciğer (NSCLC)", "Kolorektal (CRC)"])
    axes[0, 0].set_title(f"Karışıklık Matrisi ({best_model_name})", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Yapay Zeka Tahmini", fontsize=11)
    axes[0, 0].set_ylabel("Gerçek Klinik Tanı", fontsize=11)

    # 2. Grafik: Çoklu Sınıf ROC-AUC Eğrileri
    y_prob_best = best_model.predict_proba(X_test_scaled)
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    labels = ["Sağlıklı Kontrol", "Erken Evre Akciğer (NSCLC)", "Erken Evre Kolorektal (CRC)"]
    for idx, (label, col) in enumerate(zip(labels, colors)):
        y_bin = (y_test == idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob_best[:, idx])
        auc_score = roc_auc_score(y_bin, y_prob_best[:, idx])
        axes[0, 1].plot(fpr, tpr, lw=2.2, color=col, label=f"{label} (AUC = {auc_score:.3f})")

    axes[0, 1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Rastgele Tahmin')
    axes[0, 1].set_title("Sınıf Bazında ROC-AUC Eğrileri (One-vs-Rest)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Yanlış Pozitif Oranı (1 - Özgüllük)", fontsize=11)
    axes[0, 1].set_ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=11)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    # 3. Grafik: En Önemli 12 Fragmanomik Biyobelirteç
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = trained_models["Random Forest Classifier"].feature_importances_

    top_idx = np.argsort(importances)[-12:]
    axes[1, 0].barh(range(len(top_idx)), importances[top_idx], color='#008080')
    axes[1, 0].set_yticks(range(len(top_idx)))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in top_idx], fontsize=10)
    axes[1, 0].set_xlabel("Öznitelik Önem Skoru", fontsize=11)
    axes[1, 0].set_title("En Belirleyici 12 cfDNA Fragmanomik Biyobelirteç", fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)

    # 4. Grafik: 39 Kromozom Kolu Boyunca DELFI Fragman Profili (Kısa/Uzun Oranı)
    s_healthy = df[df["target_class"] == 0].iloc[0]
    s_nsclc = df[df["target_class"] == 1].iloc[0]
    s_crc = df[df["target_class"] == 2].iloc[0]

    x_arms = np.arange(len(CHROMOSOME_ARMS))
    axes[1, 1].plot(x_arms, [s_healthy[f"sl_ratio_{a}"] for a in CHROMOSOME_ARMS], 'o-', color='#2ca02c', label="Sağlıklı Kontrol (Stabil)", lw=1.5, markersize=4)
    axes[1, 1].plot(x_arms, [s_nsclc[f"sl_ratio_{a}"] for a in CHROMOSOME_ARMS], 's-', color='#1f77b4', label="Akciğer Kanseri (NSCLC)", lw=1.5, markersize=4)
    axes[1, 1].plot(x_arms, [s_crc[f"sl_ratio_{a}"] for a in CHROMOSOME_ARMS], '^-', color='#d62728', label="Kolorektal Kanser (CRC)", lw=1.5, markersize=4)

    axes[1, 1].axhline(0.18, color='black', linestyle=':', label="Sağlıklı Referans Tabanı (0.18)")
    axes[1, 1].set_xticks(x_arms[::2])
    axes[1, 1].set_xticklabels([CHROMOSOME_ARMS[i] for i in range(0, len(CHROMOSOME_ARMS), 2)], rotation=45, fontsize=8)
    axes[1, 1].set_xlabel("Kromozom Kolları (1p - 22q)", fontsize=11)
    axes[1, 1].set_ylabel("Kısa / Uzun Fragman Oranı (SL Ratio)", fontsize=11)
    axes[1, 1].set_title("Genom Boyu cfDNA Fragmantasyon Profili (DELFI)", fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "cfdna_fragmentomics_report.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Analiz Raporu Grafiği Kaydedildi: {chart_path}")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "scaler": scaler,
        "feature_cols": feature_cols
    }


# ------------------------------------------------------------------------------
# 4. KLİNİK LİKİT BİYOPSİ VAKA ANALİZİ (CLINICAL INFERENCE)
# ------------------------------------------------------------------------------

def analyze_liquid_biopsy(sample_dict: dict, model, scaler, feature_cols: list):
    df_single = pd.DataFrame([sample_dict])
    df_feat = extract_fragmentomic_features(df_single)

    vec = np.array([df_feat[c].values[0] for c in feature_cols]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)

    probs = model.predict_proba(vec_scaled)[0]
    pred_idx = int(np.argmax(probs))

    clinical_actions = {
        0: "✅ DÜŞÜK RİSK: cfDNA fragman dağılımı normal nükleozomal sınırlar içindedir. Rutin yıllık tarama önerilir.",
        1: "⚠️ YÜKSEK AKCİĞER KANSERİ (NSCLC) SİNYALİ: 1q/3q/5p fragman kısalması saptandı. Düşük doz toraks BT (LDCT) önerilir.",
        2: "⚠️ YÜKSEK KOLOREKTAL KANSER (CRC) SİNYALİ: 13q/18q/20q fragman düzensizliği saptandı. Acil kolonoskopi yönlendirmesi önerilir."
    }

    return {
        "diagnosis": CLASS_NAMES[pred_idx],
        "confidence": probs[pred_idx],
        "probs": probs,
        "global_sl": df_feat["global_mean_sl"].values[0],
        "aberrant_arms": int(df_feat["aberrant_arms_count"].values[0]),
        "action": clinical_actions[pred_idx]
    }


def run_clinical_demo(model, scaler, feature_cols: list):
    print("\n" + "=" * 78)
    print("  [KLİNİK VAKA SİMÜLASYONU] PLAZMA LİKİT BİYOPSİ TARAMA TESTLERİ")
    print("=" * 78)

    np.random.seed(77)
    cases = [
        {
            "name": "Vaka 1: 48 Yaşında Asemptomatik Sağlıklı Birey",
            "age": 48, "conc": 7.8, "modal": 167, "motif": 0.025,
            "base_sl": 0.175, "hotspots": []
        },
        {
            "name": "Vaka 2: 64 Yaşında Sigara Öykülü Hasta (Evre I Akciğer Kanseri Şüphesi)",
            "age": 64, "conc": 14.9, "modal": 162, "motif": 0.016,
            "base_sl": 0.24, "hotspots": ["1q", "3q", "5p", "7p", "19q"]
        },
        {
            "name": "Vaka 3: 59 Yaşında Gaitada Gizli Kan Pozitif Hasta (Evre I Kolorektal Şüphesi)",
            "age": 59, "conc": 17.2, "modal": 161, "motif": 0.015,
            "base_sl": 0.25, "hotspots": ["13q", "18q", "20p", "20q", "8q"]
        }
    ]

    for c in cases:
        sample = {
            "sample_id": "CLINICAL_TEST",
            "patient_age": c["age"],
            "cfdna_ng_ml": c["conc"],
            "modal_fragment_bp": c["modal"],
            "motif_CCCA_freq": c["motif"]
        }
        for arm in CHROMOSOME_ARMS:
            val = np.random.normal(c["base_sl"], 0.015)
            if arm in c["hotspots"]:
                val += 0.11
            sample[f"sl_ratio_{arm}"] = val

        res = analyze_liquid_biopsy(sample, model, scaler, feature_cols)
        print(f"\n● {c['name']}")
        print(f"  Plazma cfDNA: {c['conc']} ng/mL | Modal Boyut: {c['modal']} bp | Anormal Kol Sayısı: {res['aberrant_arms']}/39")
        print(f"  Yapay Zeka Kararı : >>> {res['diagnosis']} <<< (Güven: %{res['confidence']*100:.2f})")
        print(f"  Olasılıklar       : Sağlıklı: %{res['probs'][0]*100:.1f} | Akciğer: %{res['probs'][1]*100:.1f} | Kolorektal: %{res['probs'][2]*100:.1f}")
        print(f"  Klinik Yönlendirme: {res['action']}")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║          BİYOLOJİDE YAPAY ZEKA (BIO-AI) DERSİ - VİZE PROJESİ              ║
    ║   Likit Biyopside Hücre Dışı DNA (cfDNA) Fragmantasyon Profilleri ile     ║
    ║              Erken Evre Kanser Tespiti ve Doku Kökeni Analizi             ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    df = generate_cfdna_dataset(n_samples=720, random_seed=104)
    bundle = train_and_evaluate_models(df, output_dir=".")
    run_clinical_demo(bundle["best_model"], bundle["scaler"], bundle["feature_cols"])

    print("\n" + "=" * 78)
    print(" [✓] Tüm analizler tamamlandı. Grafik 'cfdna_fragmentomics_report.png' olarak kaydedildi.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
