"""
TGF-Beta Sinyal Yolagi Biyolojik Oznitelik Muhendisligi (Pathway Feature Engineering).

Ham gen ekspresyon verilerinden (24 gen) biyolojik anlam tasiyan sinyal yolagi aktivite
skorlarini hesaplar:
1. Canonical_SMAD_Score: Kanonik SMAD2/3/4 ve p21/p15 tumor baskilayici aktivite skoru
2. NonCanonical_Kinase_Score: MAPK1 (ERK) ve AKT1 sagkalim/onkogenik sinyal skoru
3. EMT_Transcription_Score: SNAI1, ZEB1, TWIST1 anahtar EMT transkripsiyon skoru
4. Cadherin_Switch_Ratio: CDH2 (N-kaderin) / CDH1 (E-kaderin) gecis orani
5. Mesenchymal_Invasion_Score: VIM, FN1, MMP2, MMP9 dokusal invazyon ve metastaz skoru
6. TGFB_Paradox_Index: TGF-Beta ligandlarinin kanonik baskilama yerine EMT/metastaz yonune kayma orani
"""

import numpy as np
import pandas as pd
from generate_dataset import GENE_LIST


ENGINEERED_FEATURES = [
    "Canonical_SMAD_Score",
    "NonCanonical_Kinase_Score",
    "EMT_Transcription_Score",
    "Cadherin_Switch_Ratio",
    "Mesenchymal_Invasion_Score",
    "TGFB_Paradox_Index"
]


def compute_pathway_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri cercevesine biyolojik sinyal yolagi indekslerini ekler ve dondurur.
    """
    out = df.copy()

    # 1. Kanonik SMAD ve Hucre Dongusu Fren Skoru
    out["Canonical_SMAD_Score"] = (
        out[["SMAD2", "SMAD3", "SMAD4", "CDKN1A", "CDKN2B", "BCL2L11"]].mean(axis=1)
        - 0.35 * out["SMAD7"]
    ).round(4)

    # 2. Kanonik Olmayan MAPK/ERK ve PI3K/AKT Kinaz Skoru
    out["NonCanonical_Kinase_Score"] = out[["MAPK1", "AKT1"]].mean(axis=1).round(4)

    # 3. EMT Transkripsiyon Faktorleri Aktivasyon Skoru
    out["EMT_Transcription_Score"] = out[["SNAI1", "ZEB1", "TWIST1"]].mean(axis=1).round(4)

    # 4. Cadherin Switch Orani (N-Cadherin / E-Cadherin)
    out["Cadherin_Switch_Ratio"] = (
        (out["CDH2"] + 0.5) / (out["CDH1"] + 0.5)
    ).round(4)

    # 5. Mezenkimal Invasyon ve Ekstraselluler Matriks Yikim Skoru
    out["Mesenchymal_Invasion_Score"] = (
        out[["VIM", "FN1", "MMP2", "MMP9", "CD274"]].mean(axis=1)
        - out[["CDH1", "OCLN"]].mean(axis=1)
    ).round(4)

    # 6. TGF-Beta Paradoks Indeksi (Ligand Yuksekligi x (EMT - Kanonik SMAD))
    ligand_level = out[["TGFB1", "TGFB2"]].mean(axis=1)
    out["TGFB_Paradox_Index"] = (
        ligand_level * (out["EMT_Transcription_Score"] - out["Canonical_SMAD_Score"]) / 10.0
    ).round(4)

    return out


def get_feature_columns() -> list[str]:
    """Model egitiminde kullanilacak tum gen ve yolak ozniteliklerinin listesini dondurur."""
    return GENE_LIST + ["smad4_mutation_status"] + ENGINEERED_FEATURES

