import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from src import config
from src.model import build_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5
    return val_loss, val_acc, val_auc

def run_training(train_loader, val_loader, num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE, device=config.DEVICE, save_path=config.CHECKPOINT_DIR / "best_vhl_model.pt"):
    print(f"Initial training on device: {device}")
    model = build_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_auc = validate(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] Train Loss: {t_loss:.4f} Acc: {t_acc:.3f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.3f} AUC: {v_auc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": v_loss}, save_path)
            print(f"  Saved best model checkpoint to {save_path.name}")

    return history

