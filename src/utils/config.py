from pathlib import Path

# Root del proyecto
BASE_DIR = Path(__file__).resolve().parents[2]

# Carpetas principales
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset
TRAIN_DIR = RAW_DATA_DIR / "training" / "training"
VALIDATION_DIR = RAW_DATA_DIR / "validation" / "validation"
LABELS_FILE = RAW_DATA_DIR / "monkey_labels.txt"

# Modelo
MODEL_NAME = "monkey_species_classifier"
CNN_MODEL_PATH = MODELS_DIR / "cnn_monkey_classifier.pth"
TRANSFER_MODEL_PATH = MODELS_DIR / "efficientnet_monkey_classifier.pth"

# Parámetros de entrenamiento
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Dispositivo
DEVICE = "cuda"  # se ajustará dinámicamente en los scripts si no hay GPU