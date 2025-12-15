
import torch
import cv2
import argparse
from torchvision import transforms
from src.model import BloodCellCNN

LABELS = ["red_blood_cell","white_blood_cell","platelet"]

def predict(image_path):
    model = BloodCellCNN()
    model.load_state_dict(torch.load("models/blood_cell_cnn.pth"))
    model.eval()
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        pred = torch.argmax(model(tensor)).item()
    print("Prediction ->", LABELS[pred])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predict(args.image)
