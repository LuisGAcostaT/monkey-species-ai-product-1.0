import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from tqdm import tqdm

from src.training.preprocessing import create_data_loaders
from src.utils.config import (
    CNN_MODEL_PATH,
    REPORTS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)


def set_seed(seed: int = RANDOM_SEED):
    """
    Fija semillas para hacer el entrenamiento más reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Selecciona GPU si está disponible, de lo contrario CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonkeyCNN(nn.Module):
    """
    CNN propia para clasificación de 10 especies de monos.

    Arquitectura:
    - Bloques Conv2D + BatchNorm + ReLU + MaxPooling
    - Dropout para reducir overfitting
    - Capas fully connected
    - Salida de 10 clases
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(MonkeyCNN, self).__init__()

        self.features = nn.Sequential(
            # Bloque 1: entrada 3x224x224 -> 32x112x112
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: 32x112x112 -> 64x56x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3: 64x56x56 -> 128x28x28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 4: 128x28x28 -> 256x14x14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 5: 256x14x14 -> 512x7x7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Entrena el modelo durante una época.
    """
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro")

    return epoch_loss, epoch_accuracy, epoch_f1


def validate_one_epoch(model, data_loader, criterion, device):
    """
    Evalúa el modelo durante una época.
    """
    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_predictions)
    epoch_f1 = f1_score(all_labels, all_predictions, average="macro")

    return epoch_loss, epoch_accuracy, epoch_f1


def plot_training_curves(history):
    """
    Guarda las curvas de loss, accuracy y macro F1.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["validation_loss"], label="Validation Loss")
    plt.title("CNN propia - Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = REPORTS_DIR / "cnn_loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation Accuracy")
    plt.title("CNN propia - Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    accuracy_path = REPORTS_DIR / "cnn_accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_f1"], label="Train Macro F1")
    plt.plot(epochs, history["validation_f1"], label="Validation Macro F1")
    plt.title("CNN propia - Macro F1")
    plt.xlabel("Época")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.tight_layout()
    f1_path = REPORTS_DIR / "cnn_f1_curve.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    print(f"Curva loss guardada en: {loss_path}")
    print(f"Curva accuracy guardada en: {accuracy_path}")
    print(f"Curva F1 guardada en: {f1_path}")


def save_checkpoint(model, optimizer, epoch, best_validation_f1, class_to_idx, path: Path):
    """
    Guarda checkpoint del modelo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": "MonkeyCNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_f1": best_validation_f1,
        "class_to_idx": class_to_idx,
        "num_classes": NUM_CLASSES,
    }

    torch.save(checkpoint, path)


def main():
    set_seed()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Entrenamiento CNN propia")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}\n")

    train_loader, validation_loader, train_dataset, validation_dataset = create_data_loaders()

    print(f"Imágenes entrenamiento: {len(train_dataset)}")
    print(f"Imágenes validación: {len(validation_dataset)}")
    print(f"Clases: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}\n")

    model = MonkeyCNN(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "validation_f1": [],
    }

    best_validation_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Época {epoch}/{NUM_EPOCHS}")

        train_loss, train_accuracy, train_f1 = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        validation_loss, validation_accuracy, validation_f1 = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["train_f1"].append(train_f1)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["validation_f1"].append(validation_f1)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f} | "
            f"Val F1: {validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_f1=best_validation_f1,
                class_to_idx=train_dataset.class_to_idx,
                path=CNN_MODEL_PATH
            )

            print(f"Nuevo mejor modelo guardado en: {CNN_MODEL_PATH}")

        print("-" * 70)

    history_path = REPORTS_DIR / "cnn_training_history.json"
    history_path.write_text(
        json.dumps(history, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    metrics = {
        "model_name": "MonkeyCNN",
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_train_f1": history["train_f1"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "final_validation_f1": history["validation_f1"][-1],
        "model_path": str(CNN_MODEL_PATH),
    }

    metrics_path = REPORTS_DIR / "cnn_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    plot_training_curves(history)

    print("\nEntrenamiento finalizado.")
    print(f"Mejor época: {best_epoch}")
    print(f"Mejor Validation Macro F1: {best_validation_f1:.4f}")
    print(f"Historial guardado en: {history_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Modelo guardado en: {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()