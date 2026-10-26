"""
Phylogenetic Analysis & Clustering Module (Alignment-Free Phylogenetics)
Implements:
- Pairwise genetic distance matrix computation (k-mer Euclidean / Cosine)
- UPGMA (Average Linkage) Hierarchical Clustering Dendrogram
- Principal Component Analysis (PCA) projection of evolutionary clades
- Order-level phylogenetic tree generation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA


def compute_order_centroids(X, y):
    """
    Computes mean feature vector for each insect order.
    """
    orders = sorted(list(set(y)))
    centroids = []
    for order in orders:
        mask = (y == order)
        centroids.append(X[mask].mean(axis=0))
    return orders, np.array(centroids)


def plot_phylogenetic_dendrogram(X, y, output_path=None, title="Molecular Phylogeny of Insect Orders (COI k-mer UPGMA Tree)"):
    """
    Constructs a phylogenetic tree dendrogram based on order centroids using UPGMA (average linkage).
    """
    orders, centroids = compute_order_centroids(X, y)
    
    # Calculate pairwise Euclidean genetic distance between order centroids
    dist_matrix = pdist(centroids, metric='euclidean')
    # UPGMA corresponds to average linkage
    tree_linkage = linkage(dist_matrix, method='average')

    plt.figure(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    ddata = dendrogram(
        tree_linkage,
        labels=orders,
        leaf_rotation=0,
        leaf_font_size=12,
        orientation='right',
        color_threshold=0.5 * max(tree_linkage[:, 2])
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Evolutionary Genetic Distance (k-mer divergence)", fontsize=11, labelpad=10)
    plt.ylabel("Insect Taxonomic Order", fontsize=11, labelpad=10)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Saved phylogenetic dendrogram: {output_path}")
    plt.close()


def plot_pca_phylogeny(X, y, output_path=None):
    """
    Visualizes evolutionary clades in 2D PCA space with 95% confidence ellipses / cluster hulls.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Order': y
    })

    plt.figure(figsize=(10, 7), dpi=300)
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y)))

    sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue='Order',
        palette=palette,
        s=80,
        alpha=0.85,
        edgecolor='k',
        linewidth=0.6
    )

    # Plot order centroids
    for order in np.unique(y):
        sub = df_pca[df_pca['Order'] == order]
        mean_x = sub['PC1'].mean()
        mean_y = sub['PC2'].mean()
        plt.scatter(mean_x, mean_y, s=250, c='black', marker='X', edgecolors='white', linewidths=1.5, zorder=10)
        plt.annotate(
            order,
            (mean_x, mean_y),
            textcoords="offset points",
            xytext=(8, 5),
            fontweight='bold',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    plt.title("Evolutionary Sequence Divergence: PCA of Insect Orders (COI Gene)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel(f"Principal Component 1 ({var1:.1f}% explained variance)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var2:.1f}% explained variance)", fontsize=11)
    plt.legend(title="Insect Order", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[+] Saved PCA phylogeny plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    from features import extract_features_df
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "..", "data", "insect_phylogeny_data.csv")
    X_df, extractor = extract_features_df(csv_file)
    feature_cols = [c for c in X_df.columns if c not in ['order', 'species', 'sample_id']]
    X = X_df[feature_cols].values
    y = X_df['order'].values

    rep_fig = os.path.join(current_dir, "..", "reports", "figures")
    plot_phylogenetic_dendrogram(X, y, output_path=os.path.join(rep_fig, "phylogenetic_dendrogram.png"))
    plot_pca_phylogeny(X, y, output_path=os.path.join(rep_fig, "pca_phylogeny_clusters.png"))
