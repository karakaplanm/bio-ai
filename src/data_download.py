import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src import config

def generate_synthetic_he_patch(is_vhl_mutant: bool, size: tuple = (224, 224)) -> Image.Image:
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    if is_vhl_mutant:
        base_color = np.array([238, 225, 230], dtype=np.float32)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.float32)
        img = Image.fromarray(np.clip(base_color + noise, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for _ in range(35):
            x, y = np.random.randint(10, size[0] - 20), np.random.randint(10, size[1] - 20)
            r = np.random.randint(8, 20)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(250, 248, 250), outline=(210, 180, 195))
        for _ in range(40):
            nx, ny = np.random.randint(5, size[0] - 10), np.random.randint(5, size[1] - 10)
            nr = np.random.randint(3, 7)
            draw.ellipse([nx - nr, ny - nr, nx + nr, ny + nr], fill=(70, 25, 95))
        for _ in range(6):
            pts = [(np.random.randint(0, size[0]), np.random.randint(0, size[1])) for _ in range(4)]
            draw.line(pts, fill=(180, 50, 70), width=1)
    else:
        base_color = np.array([215, 150, 185], dtype=np.float32)
        noise = np.random.normal(0, 12, img_array.shape).astype(np.float32)
        img = Image.fromarray(np.clip(base_color + noise, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for _ in range(55):
            nx, ny = np.random.randint(5, size[0] - 10), np.random.randint(5, size[1] - 10)
            nr = np.random.randint(2, 5)
            draw.ellipse([nx - nr, ny - nr, nx + nr, ny + nr], fill=(60, 20, 85))
    return img

def create_sample_dataset(num_per_class: int = 50, output_dir: Path = config.SAMPLE_DATA_DIR):
    for class_name, is_mutant in [("VHL_WildType", False), ("VHL_Mutant", True)]:
        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_per_class):
            patch = generate_synthetic_he_patch(is_vhl_mutant=is_mutant)
            patch.save(target_dir / f"patch_{i:04d}.png")
    print(f"Created {num_per_class * 2} sample patches in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-patches", type=int, default=50)
    args = parser.parse_args()
    create_sample_dataset(num_per_class=args.num_patches)

