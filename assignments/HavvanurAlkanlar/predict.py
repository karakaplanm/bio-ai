import torch
import cv2
import argparse
from torchvision import transforms
from src.model import BacteriaCNN

LABELS = ["staphylococcus", "ecoli", "pseudomonas"]

def predict(image_path):
    model = BacteriaCNN()
    model.load_state_dict(torch.load("models/bacteria_cnn.pth"))
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output).item()
    print(f"Prediction → {LABELS[pred]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predict(args.image)
