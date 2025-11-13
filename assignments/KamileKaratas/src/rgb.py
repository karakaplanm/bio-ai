"""
rgb.py

Visualize individual RGB channels of the blood smear image.
This highlights nucleus vs cytoplasm vs background differences.
"""

import os
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = os.path.join("data", "blood_smear_example.jpg")
OUTPUT_PATH = os.path.join("reports", "rgb_channels.png")


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

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
    plt.title("Red channel (nuclei contrast)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(g, cmap="gray")
    plt.title("Green channel (cytoplasm)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(b, cmap="gray")
    plt.title("Blue channel (granules / dark areas)")
    plt.axis("off")

    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"RGB channel figure saved to: {OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
