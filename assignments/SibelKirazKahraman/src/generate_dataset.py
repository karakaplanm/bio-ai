"""
TGF-Beta Sinyal Yolagi ve Kanser (EMT / Metastaz) Sentetik RNA-Seq Veri Seti Ureticisi.

Biyolojik Arka Plan (TGF-Beta Paradoksu):
1. Normal_Epithelial (Saglikli Epitel):
   - Bazal TGF-B sinyali, yuksek epitel belirtecler (CDH1/E-kaderin, OCLN),
     dusuk mezenkimal genler (VIM, SNAI1, ZEB1, MMP9).
2. Early_Stage_Tumor_Suppressive (Erken Evre Kanser - Tumor Baskilayici Faz):
   - Aktif kanonik TGFB1 -> TGFBR1/2 -> SMAD2/3/4 sinyali.
   - Hucre dongusu tutuklanmasi (CDKN1A/p21, CDKN2B/p15 yuksek) ve apoptoz (BCL2L11/BIM).
   - Epitel dokusu korunur (CDH1 yuksek), metastaz baskilanir.
3. Late_Stage_EMT_Metastatic (Ileri Evre Kanser - Tumor Tetikleyici / EMT Fazi):
   - Yuksek TGFB1/TGFB2 ekspresyonuna ragmen SMAD4 kaybi/azalmasi ve SMAD7 artisi.
   - Kanonik olmayan MAPK1 (ERK) ve AKT1 yolaklarinin hiperaktivasyonu.
   - EMT transkripsiyon faktorlerinin (SNAI1, ZEB1, TWIST1) aktivasyonu,
     E-kaderin (CDH1) kaybi, N-kaderin (CDH2), Vimentin (VIM), Fibronektin (FN1),
     matriks metalloproteinazlar (MMP2, MMP9) ve PD-L1 (CD274) artisi.
"""

import os
import numpy as np
import pandas as pd


GENE_LIST = [
    # TGF-Beta Ligandlari ve Reseptorleri
    "TGFB1", "TGFB2", "TGFBR1", "TGFBR2",
    # Kanonik SMAD Sinyal Proteinleri
    "SMAD2", "SMAD3", "SMAD4", "SMAD7",
    # Tumor Baskilayici Hedef Genler (Hucre Dongusu Freni & Apoptoz)
    "CDKN1A", "CDKN2B", "BCL2L11",
    # Kanonik Olmayan (Non-Canonical) Sagkalim / Kinaz Yolaklari
    "MAPK1", "AKT1",
    # Epitel Belirtecler (Hucreler Arasi Baglanti)
    "CDH1", "OCLN",
    # EMT Transkripsiyon Faktorleri ve Mezenkimal / Metastaz Genleri
    "SNAI1", "ZEB1", "TWIST1", "CDH2", "VIM", "FN1", "MMP2", "MMP9", "CD274"
]


def generate_dataset(n_per_class: int = 120, seed: int = 42, output_dir: str = "data") -> pd.DataFrame:
    """
    TGF-Beta sinyal yolagi gen ekspresyon profillerini (log2(TPM + 1)) ve klinik/molekuler
    degiskenleri iceren biyolojik olarak gercekci veri setini uretir.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    classes = [
        "Normal_Epithelial",
        "Early_Stage_Tumor_Suppressive",
        "Late_Stage_EMT_Metastatic"
    ]

    # Her sinif icin ortalama gen ekspresyon profilleri (log2(TPM+1) olceginde, 0-12 arasi)
    # Biyolojik literaturdeki (TCGA Pan-Cancer TGF-B & EMT imzasi) kaliplari yansitir.
    profiles = {
        "Normal_Epithelial": {
            "TGFB1": 4.2, "TGFB2": 3.5, "TGFBR1": 5.0, "TGFBR2": 5.4,
            "SMAD2": 5.2, "SMAD3": 5.1, "SMAD4": 5.8, "SMAD7": 4.0,
            "CDKN1A": 4.8, "CDKN2B": 4.5, "BCL2L11": 4.3,
            "MAPK1": 4.6, "AKT1": 4.5,
            "CDH1": 8.4, "OCLN": 7.8,
            "SNAI1": 2.4, "ZEB1": 2.6, "TWIST1": 2.2,
            "CDH2": 2.8, "VIM": 3.5, "FN1": 3.8,
            "MMP2": 2.9, "MMP9": 2.5, "CD274": 2.7
        },
        "Early_Stage_Tumor_Suppressive": {
            "TGFB1": 6.8, "TGFB2": 5.6, "TGFBR1": 6.5, "TGFBR2": 6.7,
            "SMAD2": 6.9, "SMAD3": 7.1, "SMAD4": 6.6, "SMAD7": 5.1,
            "CDKN1A": 7.9, "CDKN2B": 7.4, "BCL2L11": 6.8,
            "MAPK1": 5.5, "AKT1": 5.4,
            "CDH1": 7.1, "OCLN": 6.5,
            "SNAI1": 3.8, "ZEB1": 3.6, "TWIST1": 3.3,
            "CDH2": 3.9, "VIM": 4.6, "FN1": 4.9,
            "MMP2": 4.1, "MMP9": 3.8, "CD274": 3.9
        },
        "Late_Stage_EMT_Metastatic": {
            "TGFB1": 8.9, "TGFB2": 8.1, "TGFBR1": 7.4, "TGFBR2": 5.8,
            "SMAD2": 5.9, "SMAD3": 6.1, "SMAD4": 3.4, "SMAD7": 7.5,
            "CDKN1A": 4.1, "CDKN2B": 3.7, "BCL2L11": 3.2,
            "MAPK1": 8.2, "AKT1": 8.0,
            "CDH1": 3.6, "OCLN": 3.3,
            "SNAI1": 8.1, "ZEB1": 7.9, "TWIST1": 7.5,
            "CDH2": 8.0, "VIM": 8.8, "FN1": 8.5,
            "MMP2": 7.8, "MMP9": 8.2, "CD274": 7.4
        }
    }

    records = []
    sample_idx = 1

    for cls_name in classes:
        means = profiles[cls_name]
        for _ in range(n_per_class):
            # Hasta bazli biyolojik heterojenite (latent pathway aktivasyon dalgalanmasi)
            tgfb_drive = rng.normal(0.0, 0.75)
            emt_drive = rng.normal(0.0, 0.85)
            smad_integrity = rng.normal(0.0, 0.70)

            # Siniflar arasi gercekci gecis/ortusme (sınır vakalar ~%8)
            is_borderline = rng.random() < 0.08

            row = {
                "sample_id": f"TCGA_TGFB_{sample_idx:03d}",
                "cancer_stage_phenotype": cls_name,
            }

            # SMAD4 somatik mutasyon / delesyon durumu
            if cls_name == "Normal_Epithelial":
                smad4_mut = int(rng.random() < 0.02)
            elif cls_name == "Early_Stage_Tumor_Suppressive":
                smad4_mut = int(rng.random() < 0.12)
            else:
                smad4_mut = int(rng.random() < 0.68)
            row["smad4_mutation_status"] = smad4_mut

            for gene in GENE_LIST:
                base_val = means[gene]
                noise = rng.normal(0.0, 1.15)

                # Yolak ici biyolojik kovaryans yapisi
                if gene in ["TGFB1", "TGFB2", "TGFBR1"]:
                    val = base_val + 0.6 * tgfb_drive + noise
                elif gene in ["SMAD2", "SMAD3", "SMAD4", "CDKN1A", "CDKN2B", "BCL2L11"]:
                    val = base_val + 0.65 * smad_integrity + noise
                    if gene == "SMAD4" and smad4_mut == 1:
                        val -= rng.uniform(1.2, 2.4)
                elif gene in ["CDH1", "OCLN"]:
                    val = base_val - 0.70 * emt_drive + noise
                elif gene in ["SNAI1", "ZEB1", "TWIST1", "CDH2", "VIM", "FN1", "MMP2", "MMP9", "CD274", "MAPK1", "AKT1"]:
                    val = base_val + 0.70 * emt_drive + 0.30 * tgfb_drive + noise
                else:
                    val = base_val + noise

                # Sinir (borderline) tumor orneklerinde ara fenotip etkisi
                if is_borderline:
                    val = 0.65 * val + 0.35 * rng.uniform(4.0, 6.5)

                row[gene] = round(float(np.clip(val, 0.2, 13.5)), 4)

            sample_idx += 1
            records.append(row)

    df = pd.DataFrame(records)
    # Ornekleri karistir
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    csv_path = os.path.join(output_dir, "tgfb_cancer_expression_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [OK] TGF-Beta & Kanser veri seti kaydedildi: {csv_path} ({len(df)} ornek, {len(GENE_LIST)} gen)")
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_dataset(output_dir=os.path.join(base_dir, "data"))

