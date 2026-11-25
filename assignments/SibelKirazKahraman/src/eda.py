"""
TGF-Beta Sinyal Yolagi & Kanser EMT/Metastaz Kesifsel Veri Analizi (EDA) ve Gorsellestirme Modulu.

Uretilen Bilimsel Grafikler (reports/figures/):
1. pathway_expression_heatmap.png : 24 TGF-B ve EMT geninin evrelere gore Z-skor ekspresyon isi haritasi
2. tgfb_paradox_violin.png        : CDKN1A (p21), SMAD4, CDH1 (E-kaderin) ve VIM (Vimentin) dagilimlari
3. cadherin_switch_paradox.png    : TGF-Beta Paradoksu (Kanonik SMAD Skoru vs Mezenkimal Invasyon Skoru)
4. pca_pathway_clusters.png       : PCA 2B uzayinda tumor evrelerinin kumelenmesi & K-Means Silhouette analizi
5. correlation_network_heatmap.png: Genler arasi Pearson korelasyon matrisi (Epitel vs EMT antagonizmasi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from generate_dataset import GENE_LIST
from features import compute_pathway_features, get_feature_columns


STAGE_LABELS_TR = {
    "Normal_Epithelial": "Normal Epitel",
    "Early_Stage_Tumor_Suppressive": "Erken Evre (Tümör Baskılayıcı)",
    "Late_Stage_EMT_Metastatic": "İleri Evre (EMT / Metastatik)"
}

STAGE_COLORS = {
    "Normal_Epithelial": "#2b83ba",
    "Early_Stage_Tumor_Suppressive": "#fdae61",
    "Late_Stage_EMT_Metastatic": "#d7191c"
}


def run_eda(csv_path: str, figures_dir: str) -> dict:
    """Tum kesifsel veri analizi kontrollerini ve grafiklerini uretir."""
    os.makedirs(figures_dir, exist_ok=True)
    df_raw = pd.read_csv(csv_path)

    # Veri Semasi ve Eksik Deger Kontrolu
    missing_count = int(df_raw.isnull().sum().sum())
    print(f"  -> Veri seti boyutu: {df_raw.shape[0]} örnek, {df_raw.shape[1]} sütun (Eksik değer: {missing_count})")

    df = compute_pathway_features(df_raw)
    stage_order = [
        "Normal_Epithelial",
        "Early_Stage_Tumor_Suppressive",
        "Late_Stage_EMT_Metastatic"
    ]

    sns.set_theme(style="whitegrid", font_scale=1.05)

    # -------------------------------------------------------------------------
    # 1. Gen Ekspresyon Profili Isi Haritasi (Z-Score Heatmap)
    # -------------------------------------------------------------------------
    mean_by_stage = df.groupby("cancer_stage_phenotype")[GENE_LIST].mean().loc[stage_order]
    z_matrix = (mean_by_stage - mean_by_stage.mean(axis=0)) / (mean_by_stage.std(axis=0) + 1e-8)
    z_matrix.index = [STAGE_LABELS_TR[s] for s in z_matrix.index]

    fig, ax = plt.subplots(figsize=(15, 5.2), dpi=220)
    sns.heatmap(
        z_matrix,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        cbar_kws={"label": "Standartlaştırılmış Ekspresyon (Z-Skoru)"},
        ax=ax
    )
    ax.set_title("TGF-β Sinyal Yolağı ve EMT Genlerinin Kanser Evrelerine Göre Ekspresyon Özeti", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("TGF-β / SMAD / EMT Hedef Genleri", fontsize=11, fontweight="bold")
    ax.set_ylabel("Biyolojik Fenotip / Evre", fontsize=11, fontweight="bold")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pathway_expression_heatmap.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Anahtar Biyobelirtecler Keman Grafigi (Violin Plot)
    # -------------------------------------------------------------------------
    key_genes = [
        ("CDKN1A", "CDKN1A (p21 - Hücre Döngüsü Freni)"),
        ("SMAD4", "SMAD4 (Kanonik Tümör Baskılayıcı)"),
        ("CDH1", "CDH1 (E-Kaderin - Epitel Bütünlük)"),
        ("VIM", "VIM (Vimentin - EMT & Metastaz)")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=220)
    for ax, (gene, title) in zip(axes.flat, key_genes):
        sns.violinplot(
            data=df,
            x="cancer_stage_phenotype",
            y=gene,
            order=stage_order,
            palette=STAGE_COLORS,
            inner="box",
            cut=0,
            ax=ax
        )
        ax.set_xticklabels([STAGE_LABELS_TR[s] for s in stage_order], rotation=12)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Ekspresyon: log₂(TPM + 1)")
    fig.suptitle("TGF-β Paradoksu: Erken Evre Baskılamadan İleri Evre EMT'ye Geçiş", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "tgfb_paradox_violin.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. TGF-Beta Paradoksu & Cadherin Switch Sacilim Grafigi
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=220)

    for stage in stage_order:
        sub = df[df["cancer_stage_phenotype"] == stage]
        axes[0].scatter(
            sub["Canonical_SMAD_Score"],
            sub["Mesenchymal_Invasion_Score"],
            label=STAGE_LABELS_TR[stage],
            color=STAGE_COLORS[stage],
            alpha=0.78,
            edgecolors="k",
            linewidths=0.4,
            s=48
        )
    axes[0].set_title("TGF-β Paradoksu: Kanonik SMAD vs. Mezenkimal İnvazyon", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Kanonik SMAD Tümör Baskılayıcı Skoru")
    axes[0].set_ylabel("Mezenkimal İnvazyon & Metastaz Skoru")
    axes[0].legend(frameon=True)

    for stage in stage_order:
        sub = df[df["cancer_stage_phenotype"] == stage]
        axes[1].scatter(
            sub["CDH1"],
            sub["CDH2"],
            label=STAGE_LABELS_TR[stage],
            color=STAGE_COLORS[stage],
            alpha=0.78,
            edgecolors="k",
            linewidths=0.4,
            s=48
        )
    axes[1].set_title("Cadherin Switch: E-Kaderin (CDH1) Kaybı ve N-Kaderin (CDH2) Artışı", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("CDH1 (E-Kaderin) Ekspresyonu [log₂(TPM+1)]")
    axes[1].set_ylabel("CDH2 (N-Kaderin) Ekspresyonu [log₂(TPM+1)]")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cadherin_switch_paradox.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. PCA Boyut Indirgeme & Denetimsiz K-Means Silhouette Analizi
    # -------------------------------------------------------------------------
    feat_cols = get_feature_columns()
    X_scaled = StandardScaler().fit_transform(df[feat_cols].values)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = round(float(silhouette_score(X_scaled, labels)), 4)
    best_k = max(sil_scores, key=sil_scores.get)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=220)
    for stage in stage_order:
        mask = (df["cancer_stage_phenotype"] == stage).values
        axes[0].scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=STAGE_LABELS_TR[stage],
            color=STAGE_COLORS[stage],
            alpha=0.80,
            edgecolors="k",
            linewidths=0.4,
            s=50
        )
    axes[0].set_title(
        f"PCA 2B Yolak Uzayı (Açıklanan Varyans: PC1 %{pca.explained_variance_ratio_[0]*100:.1f}, PC2 %{pca.explained_variance_ratio_[1]*100:.1f})",
        fontsize=11,
        fontweight="bold"
    )
    axes[0].set_xlabel(f"Temel Bileşen 1 (PC1 - %{pca.explained_variance_ratio_[0]*100:.1f})")
    axes[0].set_ylabel(f"Temel Bileşen 2 (PC2 - %{pca.explained_variance_ratio_[1]*100:.1f})")
    axes[0].legend(frameon=True)

    ks = list(sil_scores.keys())
    vals = [sil_scores[k] for k in ks]
    axes[1].plot(ks, vals, marker="o", linewidth=2.2, color="#2b83ba")
    axes[1].axvline(best_k, linestyle="--", color="#d7191c", label=f"Optimal Küme Sayısı k={best_k} (Skor={sil_scores[best_k]:.3f})")
    axes[1].set_title("Denetimsiz K-Means Kümeleme Silhouette Skoru Analizi", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Küme Sayısı (k)")
    axes[1].set_ylabel("Ortalama Silhouette Skoru")
    axes[1].set_xticks(ks)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pca_pathway_clusters.png"))
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. Genler Arasi Korelasyon Agi Matrisi
    # -------------------------------------------------------------------------
    corr_genes = [
        "TGFB1", "TGFBR1", "SMAD2", "SMAD3", "SMAD4", "SMAD7",
        "CDKN1A", "CDKN2B", "CDH1", "OCLN",
        "MAPK1", "AKT1", "SNAI1", "ZEB1", "TWIST1", "CDH2", "VIM", "MMP9", "CD274"
    ]
    corr_mat = df[corr_genes].corr()
    fig, ax = plt.subplots(figsize=(11.5, 9.5), dpi=220)
    sns.heatmap(
        corr_mat,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        linewidths=0.4,
        cbar_kws={"label": "Pearson Korelasyon Katsayısı (r)"},
        ax=ax
    )
    ax.set_title("TGF-β / SMAD / EMT Gen Ağı Pearson Korelasyon Matrisi", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_network_heatmap.png"))
    plt.close(fig)

    print(f"  [OK] 5 adet bilimsel EDA grafiği '{figures_dir}' klasörüne kaydedildi.")
    return {
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "missing_values": missing_count,
        "pca_var_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "silhouette_scores": sil_scores,
        "optimal_k": int(best_k)
    }

