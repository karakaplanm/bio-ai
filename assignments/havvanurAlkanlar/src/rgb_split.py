import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/petri_example.jpg")
r, g, b = cv2.split(img)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(r, cmap="gray")
plt.title("Red Channel")
plt.subplot(1,3,2)
plt.imshow(g, cmap="gray")
plt.title("Green Channel")
plt.subplot(1,3,3)
plt.imshow(b, cmap="gray")
plt.title("Blue Channel")
plt.show()
