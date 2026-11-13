import argparse
from pathlib import Path
import torch
from src import config
from src.data_download import create_sample_dataset
from src.dataset import prepare_datasets_from_dir, get_dataloaders
from src.train import run_training
from src.evaluate import evaluate_model
from src.model import build_model
from src.gradcam import explain_and_visualize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["sample", "train", "eval", "explain", "all"])
    parser.add_argument("--data-dir", default=str(config.SAMPLE_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    print("=" * 60)
    print(" 🧬 VHL Histopathology AI Pipeline (PyTorch)")
    print("=" * 60)

    if args.mode in ["sample", "all"]:
        if not (data_path / "VHL_Mutant").exists() or len(list((data_path / "VHL_Mutant").glob("*.png"))) == 0:
            create_sample_dataset(num_per_class=40, output_dir=data_path)

    if args.mode in ["train", "eval", "all"]:
        train_ds, val_ds, test_ds = prepare_datasets_from_dir(data_path)
        train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)

    checkpoint_file = config.CHECKPOINT_DIR / "best_vhl_model.pt"

    if args.mode in ["train", "all"]:
        run_training(train_loader, val_loader, num_epochs=args.epochs, device=config.DEVICE, save_path=checkpoint_file)

    if args.mode in ["eval", "all"]:
        model = build_model(device=config.DEVICE)
        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        evaluate_model(model, test_loader, device=config.DEVICE)

    if args.mode in ["explain", "all"]:
        model = build_model(device=config.DEVICE)
        if checkpoint_file.exists():
            checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
        mutant_patches = list((data_path / "VHL_Mutant").glob("*.png"))
        if mutant_patches:
            explain_and_visualize(model, mutant_patches[0], device=config.DEVICE)

    print("\n✅ Execution completed successfully.")

if __name__ == "__main__":
    main()

