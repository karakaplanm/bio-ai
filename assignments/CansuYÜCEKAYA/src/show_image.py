import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/petri_example.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title("Petri Image")
plt.axis("off")
plt.show()
