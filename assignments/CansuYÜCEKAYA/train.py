import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from src.model import BacteriaCNN
import numpy as np

class SimplePetriDataset(Dataset):
    def __init__(self, size=300):
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])

    def __len__(self):
        return self.size

    def generate_image(self, label):
        img = 255 * np.ones((128, 128, 3), dtype=np.uint8)
        color = [(0,0,0), (255,255,255), (0,255,0)][label]
        radius = [20,25,30][label]
        center = (64,64)
        cv2.circle(img, center, radius, color, -1)
        return img

    def __getitem__(self, idx):
        label = idx % 3
        img = self.generate_image(label)
        img = self.transform(img)
        return img, label

def train():
    dataset = SimplePetriDataset(size=300)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BacteriaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining Started...\n")
    for epoch in range(10):
        total_loss = 0
        for img, label in loader:
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fungal_cnn.pth")
    print("\nTraining completed. Model saved → models/fungal_cnn.pth\n")

if __name__ == "__main__":
    train()
