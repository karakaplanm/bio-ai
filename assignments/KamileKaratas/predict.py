"""
predict.py

Use the trained CNN model to classify a blood smear image.
"""

import argparse
import os
import cv2
import torch
import numpy as np

from src.cnn import SimpleBloodCellCNN


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to blood smear image"
    )
    args = parser.parse_args()

    image_path = args.image

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

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

    # Prediction
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()
        pred_class = np.argmax(probabilities)

    class_names = ["lymphocyte", "neutrophil", "monocyte"]
    predicted_label = class_names[pred_class]

    print("\nPrediction Result:")
    print(f"Predicted class: {predicted_label}")
    print("Probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:<10}: {probabilities[i]:.4f}")


if __name__ == "__main__":
    main()
