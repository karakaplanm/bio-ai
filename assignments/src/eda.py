"""
Exploratory Data Analysis (EDA) Module for Insect Phylogeny & DNA Barcoding
Analyzes:
- GC-content and nucleotide composition differences across insect orders
- k-mer frequency profiles and order-level discriminative motifs
- Correlation structure among biological sequence properties
- Generates publication-ready figures in reports/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from features import SequenceFeatureExtractor, extract_features_df


def plot_gc_distribution(df, output_path=None):
    """
    Plots violin + swarm plot of GC content across insect orders.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    order_order = df.groupby('order')['gc_percent'].median().sort_values().index
    palette = sns.color_palette("muted", n_colors=len(order_order))

    ax = sns.violinplot(
        data=df,
        x='order',
        y='gc_percent',
        order=order_order,
        palette=palette,
        inner="quartile",
        cut=0,
        linewidth=1.2
    )

    sns.stripplot(
        data=df,
        x='order',
        y='gc_percent',
        order=order_order,
        color='black',
        alpha=0.45,
        jitter=0.2,
        size=5
    )

    plt.title("Mitochondrial COI GC% Content Distribution Across Insect Orders", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Insect Taxonomic Order", fontsize=11, fontweight='semibold')
    plt.ylabel("GC Content (%)", fontsize=11, fontweight='semibold')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Saved GC distribution plot: {output_path}")
    plt.close()


def plot_kmer_profiles(X_df, output_path=None):
    """
    Compares average frequency of key diagnostic trinucleotide k-mers across insect orders.
    """
    key_kmers = ['AAA', 'AAT', 'ATT', 'TTT', 'TTA', 'AAC', 'CCG', 'GCA', 'CGC', 'CGT']
    
    # Calculate order-level mean for these key kmers
    summary = X_df.groupby('order')[key_kmers].mean().reset_index()
    melted = pd.melt(summary, id_vars=['order'], var_name='kmer', value_name='frequency')

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("Set2", n_colors=len(X_df['order'].unique()))
    sns.barplot(
        data=melted,
        x='kmer',
        y='frequency',
        hue='order',
        palette=palette,
        edgecolor='black',
        linewidth=0.6
    )

    plt.title("Trinucleotide (3-mer) Frequency Spectrum Across Insect Taxonomic Orders", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Trinucleotide Motif (k-mer)", fontsize=11, fontweight='semibold')
    plt.ylabel("Mean Normalized Frequency", fontsize=11, fontweight='semibold')
    plt.legend(title="Insect Order", frameon=True)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Saved k-mer profile plot: {output_path}")
    plt.close()


def plot_correlation_heatmap(X_df, output_path=None):
    """
    Plots correlation heatmap of physicochemical features and selected prominent k-mers.
    """
    features_to_plot = [
        "gc_percent", "at_skew", "gc_skew", "shannon_entropy", "cpg_ratio", "tpa_ratio",
        "AAA", "AAT", "ATT", "TTT", "TTA", "AAC", "CCG", "GCA"
    ]
    
    existing_cols = [c for c in features_to_plot if c in X_df.columns]
    corr_matrix = X_df[existing_cols].corr()

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8}
    )

    plt.title("Correlation Matrix of Key Biological & k-mer Sequence Properties", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Saved correlation heatmap: {output_path}")
    plt.close()


def run_eda(csv_path, figures_dir):
    """
    Runs full EDA pipeline and outputs statistical report.
    """
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS (EDA) - INSECT PHYLOGENY  ")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"\n[+] Total Sequences: {len(df)}")
    print(f"[+] Insect Orders ({df['order'].nunique()}): {list(df['order'].unique())}")
    print("\n--- Summary Statistics by Order (GC%) ---")
    summary = df.groupby('order')['gc_percent'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    print(summary)

    print("\n[+] Extracting sequence k-mer and physicochemical features...")
    X_df, extractor = extract_features_df(csv_path)

    os.makedirs(figures_dir, exist_ok=True)
    plot_gc_distribution(df, os.path.join(figures_dir, "gc_content_violin.png"))
    plot_kmer_profiles(X_df, os.path.join(figures_dir, "kmer_distribution.png"))
    plot_correlation_heatmap(X_df, os.path.join(figures_dir, "correlation_heatmap.png"))

    print("\n[+] EDA complete. All figures generated successfully in reports/figures/")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "..", "data", "insect_phylogeny_data.csv")
    fig_dir = os.path.join(current_dir, "..", "reports", "figures")
    run_eda(csv_file, fig_dir)
