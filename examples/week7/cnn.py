import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Use CPU to avoid memory issues
device = torch.device("cpu")
print(f"Using device: {device}")




class SimpleHistologyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 224 -> 112 -> 56
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> (16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))   # -> (32, 56, 56)
        x = torch.flatten(x, 1)                # -> (N, 32*56*56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create dummy dataset for demonstration
print("Creating dummy histology dataset...")
batch_size = 4
num_samples = 20

# Generate random images (3 channels, 224x224) and binary labels
dummy_images = torch.randn(num_samples, 3, 224, 224)
dummy_labels = torch.randint(0, 2, (num_samples,))

# Create dataset and dataloader
dataset = TensorDataset(dummy_images, dummy_labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Dataset created with {num_samples} samples, batch size: {batch_size}")

# Create model
model = SimpleHistologyCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Starting training...")
num_epochs = 5  # Derste örnek için kısa tut
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

print("Training completed!")
