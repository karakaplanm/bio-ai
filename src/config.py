from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample_patches"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR = BASE_DIR / "reports"

for folder in [DATA_DIR, SAMPLE_DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BACKBONE_NAME = "resnet18"
NUM_CLASSES = 2
CLASS_NAMES = ["VHL_WildType", "VHL_Mutant"]

