"""
code-of-midterm.py

Main training script for a simple CNN that classifies blood cells
(lymphocyte / neutrophil / monocyte) from peripheral blood smear images.

For the midterm, we use a synthetic (random) dataset to keep the focus
on the deep learning pipeline rather than dataset collection.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.cnn import SimpleBloodCellCNN


def main():
    # Device configuration (CPU is fine for the midterm)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_classes = 3  # lymphocyte, neutrophil, monocyte
    num_samples = 60
    batch_size = 8
    num_epochs = 5
    learning_rate = 1e-4

    # ------------------------------------------------------------------
    # 1. Create a dummy dataset (3-channel, 224x224 images)
    # ------------------------------------------------------------------
    print("Creating synthetic blood smear dataset...")
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # 2. Model, loss, optimizer
    # ------------------------------------------------------------------
    model = SimpleBloodCellCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images_batch, labels_batch in dataloader:
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"- Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        )

    print("Training completed!")

    # ------------------------------------------------------------------
    # 4. Save model
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "simple_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
