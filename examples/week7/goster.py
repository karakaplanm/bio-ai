"""
Basit bir histoloji veya mikroskopi görüntüsünü yükleyip gösteren Python betiği.
OpenCV kullanarak görüntüyü BGR formatında okur, RGB'ye dönüştür ve matplotlib ile görüntüler.
Kurulum:
pip install opencv-python --break-system-package
pip install matplotlib
"""

import cv2
import matplotlib.pyplot as plt

# Örnek: histology veya microscopy görüntü dosyasının yolu
# Örn: "data/histo_images/sample_01.png"
image_path = "leaf.png"

# OpenCV BGR formatında okur
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# BGR -> RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Histology / Microscopy Image")
plt.axis("off")
plt.show()
#plt.savefig('displayed_image.png', dpi=150, bbox_inches='tight')
#print("Image displayed and saved as 'displayed_image.png'")
#plt.close()
