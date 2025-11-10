# --- 1. GEREKLİ KÜTÜPHANELERİ YÜKLEME ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# --- 2. VERİYİ YÜKLEME VE İNCELEME ---

print("Adım 1: Malatya müşteri segmentasyon veri seti yükleniyor...")
# Veri setini UTF-8-sig kodlamasıyla okuyoruz
df = pd.read_csv("malatya_musteri_segmentasyonu.csv", encoding="utf-8-sig")

print("\nVeri setinin ilk 5 satırı:")
print(df.head())

print("\nVeri seti hakkında özet bilgi:")
df.info()

print("\nEksik değer kontrolü:")
print(df.isnull().sum())

# --- 3. ÖZELLİKLERİ TANIMLAMA ---
# Kümelemede tüm veriyi (X) kullanırız, ayrı bir hedef (y) yoktur.
X = df.copy()

# --- 4. KATEGORİK VERİYİ DÖNÜŞTÜRME VE ÖLÇEKLENDİRME ---
# Sayısal ve kategorik sütunları ayır
numeric_features = ["yas", "aylik_gelir_TL", "harcama_skoru", "kredi_kart_sayisi"]
categorical_features = ["cinsiyet", "medeni_durum", "semt", "arac_sahibi_mi"]

# Her sütun türü için işlem adımları
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Dönüştürme işlemlerini birleştir
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough" # Kalan sütunları (varsa) işlemez
)

# --- 5. OPTİMAL KÜME SAYISINI BULMA (Dirsek Yöntemi) ---
print("\nAdım 2: Optimal küme sayısı (k) için Dirsek Yöntemi...")
# Önce veriyi dönüştürelim
X_processed = preprocessor.fit_transform(X)

wcss = [] # Küme İçi Hata Kareleri Toplamı
K = range(1, 11)
for k in K:
    kmeans_elbow = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42, algorithm='lloyd')
    kmeans_elbow.fit(X_processed)
    wcss.append(kmeans_elbow.inertia_)

# Dirsek Grafiği
plt.figure(figsize=(10, 6))
plt.plot(K, wcss, 'bo-', markerfacecolor='red', markersize=8)
plt.title('Optimal K için Dirsek Yöntemi (Elbow Method)')
plt.xlabel('Küme Sayısı (k)')
plt.ylabel('WCSS (Hata Kareleri Toplamı)')
plt.grid(True)
plt.show()

# --- 6. MODEL KURMA VE PIPELINE OLUŞTURMA ---
# Dirsek grafiğine göre optimal k'yi 5 olarak varsayıyoruz
optimal_k = 5

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clusterer", KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42, algorithm='lloyd'))
])

# --- 7. MODELİ EĞİTME ---
print(f"\nAdım 3: Model k={optimal_k} için eğitiliyor...")
# Kümeleme tüm veriye (X) uygulanır
model.fit(X)
print("Model başarıyla eğitildi!")

# --- 8. MODEL SONUÇLARINI DEĞERLENDİRME (KÜME ANALİZİ) ---
print("\nAdım 4: Kümeler analiz ediliyor...")
# Oluşturulan kümeleri (etiketleri) orijinal DataFrame'e ekle
df['cluster'] = model.named_steps['clusterer'].labels_

print("\n--- KÜME ÖZELLİK ORTALAMALARI (SAYISAL) ---")
numeric_profile = df.groupby('cluster')[numeric_features].mean()
print(numeric_profile.to_markdown(floatfmt=",.0f"))

print("\n--- KÜME ÖZELLİKLERİ (KATEGORİK - En Sık Görülen) ---")
# agg(pd.Series.mode) kullanarak her kümedeki en yaygın kategorik değeri buluyoruz
categorical_profile = df.groupby('cluster')[categorical_features].agg(lambda x: pd.Series.mode(x)[0])
print(categorical_profile.to_markdown())

# --- 9. GÖRSELLEŞTİRME ---
print("\nAdım 5: Kümeler görselleştiriliyor...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="aylik_gelir_TL",
    y="harcama_skoru",
    hue="cluster",
    palette="deep", # Farklı renk paleti
    s=100, # Nokta boyutu
    alpha=0.7 # Şeffaflık
)

# Küme merkezlerinin ortalamalarını (centroid) grafiğe ekle
cluster_centers = numeric_profile[['aylik_gelir_TL', 'harcama_skoru']]
plt.scatter(
    cluster_centers['aylik_gelir_TL'],
    cluster_centers['harcama_skoru'],
    s=300,
    c='red',
    marker='X',
    label='Küme Merkezi (Ortalama)'
)

plt.title('Müşteri Segmentasyonu (Gelir vs. Harcama Skoru)')
plt.xlabel('Aylık Gelir (TL)')
plt.ylabel('Harcama Skoru (1-100)')
plt.legend(title='Küme')
plt.grid(True)
plt.show()

print("\nProje başarıyla tamamlandı! 🎉")