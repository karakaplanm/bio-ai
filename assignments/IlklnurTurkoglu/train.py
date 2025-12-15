
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
from src.model import BloodCellCNN

class SyntheticBloodCellDataset(Dataset):
    def __init__(self, size=300):
        self.size = size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.size

    def generate_image(self, label):
        img = 255 * np.ones((128,128,3), dtype=np.uint8)
        center = (64,64)
        if label == 0:
            cv2.circle(img, center, 25, (255,0,0), -1)
        elif label == 1:
            cv2.circle(img, center, 30, (180,0,180), -1)
        else:
            cv2.circle(img, center, 8, (0,0,255), -1)
        return img

    def __getitem__(self, idx):
        label = idx % 3
        img = self.transform(self.generate_image(label))
        return img, label

def train():
    dataset = SyntheticBloodCellDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = BloodCellCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for img, label in loader:
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss {total_loss:.4f}")

    torch.save(model.state_dict(), "models/blood_cell_cnn.pth")

if __name__ == "__main__":
    train()
