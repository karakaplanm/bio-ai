from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src import config

def get_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class HistopathologyDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def prepare_datasets_from_dir(data_dir: Path, test_size: float = 0.2, val_size: float = 0.1, random_state: int = config.SEED):
    image_paths, labels = [], []
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_folder = data_dir / class_name
        if not class_folder.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for file_path in class_folder.glob(ext):
                image_paths.append(file_path)
                labels.append(class_idx)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No patches found in {data_dir}.")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(test_size + val_size), random_state=random_state, stratify=labels
    )
    rel_test = test_size / (test_size + val_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=rel_test, random_state=random_state, stratify=temp_labels
    )

    train_tf, val_tf = get_transforms()
    return (
        HistopathologyDataset(train_paths, train_labels, transform=train_tf),
        HistopathologyDataset(val_paths, val_labels, transform=val_tf),
        HistopathologyDataset(test_paths, test_labels, transform=val_tf)
    )

def get_dataloaders(train_ds, val_ds, test_ds=None, batch_size: int = config.BATCH_SIZE):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
    return train_loader, val_loader, test_loader

