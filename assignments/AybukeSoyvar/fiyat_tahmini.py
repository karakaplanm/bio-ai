# --- 1. GEREKLİ KÜTÜPHANELERİ YÜKLEME ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 2. VERİYİ YÜKLEME VE İNCELEME ---

print("Adım 1: Malatya konut veri seti yükleniyor...")
df = pd.read_csv("malatya_konut_verisi.csv", encoding="utf-8-sig")

print("\nVeri setinin ilk 5 satırı:")
print(df.head())

print("\nVeri seti hakkında özet bilgi:")
print(df.info())

print("\nEksik değer kontrolü:")
print(df.isnull().sum())

# --- 3. ÖZELLİKLERİ VE HEDEFİ AYIRMA ---
X = df.drop("fiyat (TL)", axis=1)
y = df["fiyat (TL)"]

# --- 4. KATEGORİK VERİYİ DÖNÜŞTÜRME VE ÖLÇEKLENDİRME ---
# Sayısal ve kategorik sütunları ayır
numeric_features = ["metrekare", "oda_sayısı", "bina_yasi", "merkez_uzaklık_km"]
categorical_features = ["semt"]

# Her sütun türü için işlem adımları
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Dönüştürme işlemlerini birleştir
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# --- 5. EĞİTİM VE TEST SETLERİNE AYIRMA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nEğitim verisi: {X_train.shape[0]} satır")
print(f"Test verisi: {X_test.shape[0]} satır")

# --- 6. MODEL KURMA VE PIPELINE OLUŞTURMA ---
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# --- 7. MODELİ EĞİTME ---
print("\nAdım 2: Model eğitiliyor...")
model.fit(X_train, y_train)
print("Model başarıyla eğitildi!")

# --- 8. MODELİ DEĞERLENDİRME ---
print("\nAdım 3: Model değerlendiriliyor...")
predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n--- MODEL PERFORMANS SONUÇLARI ---")
print(f"R² Skoru: {r2:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae:,.0f} TL")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:,.0f} TL")

print("\nYorum:")
print(f"Model fiyat değişkenliğinin %{r2*100:.2f}'sini açıklıyor.")
print(f"Tahminler ortalama ±{mae:,.0f} TL hata ile yapılıyor.")

# --- 9. GÖRSELLEŞTİRME ---
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.6, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", lw=2)
plt.title("Gerçek Fiyatlar vs Tahmin Edilen Fiyatlar (Malatya)")
plt.xlabel("Gerçek Fiyatlar (TL)")
plt.ylabel("Tahmin Edilen Fiyatlar (TL)")
plt.grid(True)
plt.show()

print("\nProje başarıyla tamamlandı! 🎉")
