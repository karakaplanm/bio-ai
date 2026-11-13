from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from src import config

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        logits = self.model(input_tensor)
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * self.activations[0], dim=0)
        cam = torch.clamp(cam, min=0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

def explain_and_visualize(model, image_path: Path, device=config.DEVICE, output_path=config.REPORTS_DIR / "gradcam_vhl_explanation.png"):
    raw_image = Image.open(image_path).convert("RGB")
    raw_np = np.array(raw_image.resize((224, 224)), dtype=np.float32) / 255.0

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_input = tf(raw_image).unsqueeze(0).to(device)

    target_layer = getattr(model, "target_conv_layer", model.backbone.layer4[-1])
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(tensor_input, target_class=1)

    resized_cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * raw_np + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_np)
    axes[0].set_title("H&E Tissue Patch")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Biological Focus Area")
    axes[2].axis("off")

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved Grad-CAM visualization to: {output_path}")
    plt.close()

