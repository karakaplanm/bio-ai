import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras

# Eğitilmiş modeli yükle (önce rakam_tanima.py çalıştırılmalı)
model = keras.models.load_model("rakam_modeli.keras")

# Görseli oku
img = Image.open("el_yazim.png").convert("L")  # Gri tonlama
img = ImageOps.invert(img)  # Siyah zemin, beyaz rakam yaptıysan gerekmez
a = np.array(img)

# MNIST ön işlemesi: rakamı kırp, 20x20 kutuya sığdır,
# 28x28 tuvale yerleştir ve kütle merkezini ortala
ys, xs = np.where(a > 50)
a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
h, w = a.shape
olcek = 20 / max(h, w)
yeni_w, yeni_h = max(1, round(w * olcek)), max(1, round(h * olcek))
a = np.array(Image.fromarray(a).resize((yeni_w, yeni_h), Image.LANCZOS))

tuval = np.zeros((28, 28), dtype=np.float32)
y0, x0 = (28 - yeni_h) // 2, (28 - yeni_w) // 2
tuval[y0:y0 + yeni_h, x0:x0 + yeni_w] = a

# Kütle merkezini (14, 14) noktasına kaydır
yy, xx = np.indices(tuval.shape)
toplam = tuval.sum()
dy = int(round(14 - (yy * tuval).sum() / toplam))
dx = int(round(14 - (xx * tuval).sum() / toplam))
tuval = np.roll(tuval, (dy, dx), axis=(0, 1))

plt.imshow(tuval, cmap="gray")
plt.show()

# Normalize et ve modele uygun şekle getir
img_array = (tuval / 255.0).reshape(1, 784)  # 28x28 -> 784

# Tahmin et
pred = model.predict(img_array)
print("Tahmin edilen rakam:", np.argmax(pred))
print(f"Güven: {pred.max():.2%}")
