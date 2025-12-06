import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import GEOparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("📥 Downloading GEO dataset: GSE68849 ...")
gse = GEOparse.get_GEO("GSE68849", destdir="data")

samples = []
for gsm_name, gsm in gse.gsms.items():
    expr = gsm.table["VALUE"].astype(float).tolist()
    samples.append(expr)

df = pd.DataFrame(samples)
print("✔ Expression matrix created.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(df["PC1"], df["PC2"], c=df["cluster"])
plt.title("PCA – Gene Expression Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("plots/pca_clusters.png", dpi=300)
plt.close()

print("📊 GEO analysis completed and plot saved to plots/.")

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = MLPClassifier(
    hidden_layer_sizes=(15, 13),
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

print("🤖 Training MLP Classifier...")
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"✔ Digits Model Accuracy: {accuracy:.2%}")

print("🎉 All analyses completed.")
