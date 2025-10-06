import numpy as np
from PIL import Image, ImageOps

# Görseli oku
img = Image.open("el_yazim.png").convert("L")  # Gri tonlama
img = ImageOps.invert(img)  # Siyah zemin, beyaz rakam yaptıysan gerekmez
img = img.resize((28, 28))  # MNIST boyutu
plt.imshow(img, cmap="gray")
plt.show()

# Normalize et ve modele uygun şekle getir
img_array = np.array(img).astype("float32") / 255.0
img_array = img_array.reshape(1, 784)  # 28x28 -> 784

# Tahmin et
pred = model.predict(img_array)
print("Tahmin edilen rakam:", np.argmax(pred))

