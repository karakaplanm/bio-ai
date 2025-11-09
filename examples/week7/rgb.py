import cv2
import matplotlib.pyplot as plt

image_path = "leaf.png"
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

r = img_rgb[:, :, 0]
g = img_rgb[:, :, 1]
b = img_rgb[:, :, 2]

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original RGB")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(r, cmap="gray")
plt.title("Red channel")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(g, cmap="gray")
plt.title("Green channel")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(b, cmap="gray")
plt.title("Blue channel")
plt.axis("off")

plt.tight_layout()
plt.show()
