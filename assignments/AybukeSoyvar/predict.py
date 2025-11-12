import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- PARAMETRELER (Buraları düzenleyin) ---

# 1. Kaydedilen modelin yolu
MODEL_PATH = 'bacteria_colony_model.h5'

# 2. Tahmin edilecek yeni resmin yolu
IMG_PATH = 'C:/Users/yugur/OneDrive/Belgeler/AybukeSoyvar/test/testEuglena.jpg' # ❗ Kendi resminizin yolunu yazın

# 3. Modelin beklediği resim boyutu (Eğitimdekiyle aynı olmalı)
IMG_SIZE = (224, 224)

# 4. Sınıf isimleri (EĞİTİM SIRASINDAKİ İLE AYNI SIRADA OLMALI)
# Bu bilgiyi 'train_data.class_indices' çıktısından alabilirsiniz.
# Örnek: {'Amoeba': 0, 'Euglena': 1, 'Hydra': 2, ...}
# BURAYI KENDİ SINIFLARINIZA GÖRE DÜZENLEYİN!
# ❗ (Sizin paylaştığınız resme göre sıralıyorum, kontrol edin)
CLASS_NAMES = [
    'Amoeba', 'Euglena', 'Hydra', 'Paramecium', 
    'Rod Bacteria', 'Spherical Bacteria', 'Spiral Bacteria', 'Yeast'
]
# ---

# 1. Adım: Modeli Yükle
print(f"Model yükleniyor: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenemedi. {e}")
    exit()

# 2. Adım: Yeni Görüntüyü Yükle ve Ön İşle
print(f"Görüntü yükleniyor: {IMG_PATH}")
try:
    img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
except FileNotFoundError:
    print(f"HATA: Resim bulunamadı: {IMG_PATH}")
    exit()

# Görüntüyü bir 'array' (dizi) haline getir
img_array = image.img_to_array(img)

# Görüntüyü modelin beklediği formata (batch) genişlet
# (224, 224, 3) -> (1, 224, 224, 3)
img_batch = np.expand_dims(img_array, axis=0)

# Görüntüyü 'rescale' et (Eğitimdeki gibi 1./255)
# ÖNEMLİ: MobileNetV2'nin kendi preprocess_input'unu kullanmadıysanız
# (ki sizin kodunuzda kullanmamış, 1./255 yapmıştınız) bu gereklidir.
img_preprocessed = img_batch / 255.0

# 3. Adım: Tahmin Yap
print("Tahmin yapılıyor...")
prediction = model.predict(img_preprocessed)

# 4. Adım: Sonuçları Yorumla
# 'prediction' şöyle bir şey olacaktır: [[0.01, 0.05, 0.92, 0.01, ...]]
# En yüksek olasılığa sahip sınıfın indeksini bul
predicted_class_index = np.argmax(prediction[0])
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = np.max(prediction[0]) * 100

print("\n--- TAHMİN SONUCU ---")
print(f"Görüntü: {os.path.basename(IMG_PATH)}")
print(f"Tahmin Edilen Sınıf: {predicted_class_name}")
print(f"Eminlik Yüzdesi: %{confidence:.2f}")

# (İsteğe bağlı) Tüm olasılıkları göster
print("\nTüm Sınıf Olasılıkları:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: %{prediction[0][i] * 100:.2f}")