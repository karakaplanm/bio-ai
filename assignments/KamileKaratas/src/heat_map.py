"""
heat_map.py

Generate a simple activation heatmap from the last convolutional layer
to visualize what the CNN focuses on in the blood smear image.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.cnn import SimpleBloodCellCNN


IMAGE_PATH = os.path.join("data", "blood_smear_example.jpg")
OUTPUT_PATH = os.path.join("reports", "heatmap.png")


def preprocess(img_bgr):
    """Resize + normalize image for the CNN."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor


def main():
    # Load image
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Preprocess
    x = preprocess(img_bgr)

    # Load model
    model = SimpleBloodCellCNN(num_classes=3)
    model_path = os.path.join("models", "simple_cnn.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("Loaded trained model.")
    else:
        print("WARNING: No trained model found. Using random weights.")

    model.eval()

    # Hook to capture last conv layer output
    conv_output = None

    def hook_fn(module, input, output):
        nonlocal conv_output
        conv_output = output.detach()

    hook = model.conv2.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    hook.remove()

    # Average activations
    heat = conv_output.squeeze(0)
    heat = heat.mean(dim=0).numpy()

    # Normalize
    heat = np.maximum(heat, 0)
    heat /= heat.max()

    # Resize to match image
    heat_resized = cv2.resize(heat, (img_bgr.shape[1], img_bgr.shape[0]))

    # Generate heatmap overlay
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heat_resized), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(heatmap_color, 0.4, img_bgr, 0.6, 0)

    # Save result
    os.makedirs("reports", exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Heatmap saved to: {OUTPUT_PATH}")

    # Show
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Activation Heatmap (CNN Focus Regions)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
