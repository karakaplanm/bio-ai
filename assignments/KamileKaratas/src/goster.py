"""
goster.py

Simple script to load and display a peripheral blood smear image.
"""

import os
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = os.path.join("data", "blood_smear_example.jpg")


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Convert BGR (OpenCV) -> RGB (matplotlib)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title("Peripheral Blood Smear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
