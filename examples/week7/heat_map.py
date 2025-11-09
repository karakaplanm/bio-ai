# UYARI: Bu sadeleştirilmiş bir iskelettir. Gerçek Grad-CAM için ek adımlar gerekir.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Device configuration - use CPU to avoid memory issues
device = torch.device('cpu')
print(f"Using device: {device}")

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # For 64x64 input: after conv1+pool -> 32x32, after conv2+pool -> 16x16
        # So final feature map size is 32 * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(self.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = x.view(-1, 32 * 16 * 16)  # Flatten: 32 channels * 16 * 16
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model and move to device
model = SimpleCNN(num_classes=10).to(device)
print("Model created successfully")

# Load and preprocess the leaf image
def load_leaf_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found!")
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 64x64 to match our model input
    image = image.resize((64, 64))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    tensor_image = transform(image)
    return tensor_image, "leaf"  # Return tensor and a label

# Load the leaf image
print("Loading leaf.png...")
img, label = load_leaf_image("leaf.png")
print(f"Image loaded successfully. Shape: {img.shape}")

# Tek bir görüntü alalım
model.eval()
input_tensor = img.unsqueeze(0).to(device)

# Son konv katmanın çıktısını yakalamak için hook
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv2.register_forward_hook(get_activation('conv2'))

# Forward
output = model(input_tensor)
pred_prob = F.softmax(output, dim=1)
pred_class = torch.argmax(pred_prob, dim=1).item()
print("Predicted class:", pred_class)

# Aktivasyonları al
act = activation['conv2'][0]  # (C, H, W)
# Basitçe tüm kanalların ortalamasını alıp ısı haritası gibi görelim
heatmap = act.mean(dim=0).cpu().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= heatmap.max() + 1e-8

plt.figure(figsize=(10, 5))

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Original Leaf Image\n(Resized to 64x64)")
plt.axis("off")

# Plot heatmap
plt.subplot(1, 2, 2)
plt.imshow(heatmap, cmap='jet')
plt.title(f"Activation Heatmap\n(Model Prediction: Class {pred_class})")
plt.axis("off")

plt.tight_layout()
plt.savefig('activation_heatmap.png', dpi=150, bbox_inches='tight')
print("Heatmap saved as 'activation_heatmap.png'")
plt.close()
