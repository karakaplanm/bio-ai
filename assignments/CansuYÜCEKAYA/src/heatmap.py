import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from src.model import BacteriaCNN
import os

def generate_heatmap(path="data/petri_example.jpg"):
    model = BacteriaCNN()
    model.load_state_dict(torch.load("models/fungal_cnn.pth"))
    model.eval()

    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128))
    ])

    x = transform(rgb).unsqueeze(0)
    x.requires_grad = True

    output = model(x)
    prediction = torch.argmax(output)

    model.zero_grad()
    output[0,prediction].backward()

    heatmap = x.grad[0].mean(0).detach().numpy()
    heatmap = np.maximum(heatmap,0)

    plt.imshow(heatmap, cmap="hot")
    plt.axis("off")
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/heatmap.png")
    print("Heatmap saved → reports/heatmap.png")

if __name__ == "__main__":
    generate_heatmap()
