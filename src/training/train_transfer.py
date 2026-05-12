import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

from src.training.preprocessing import create_data_loaders
from src.utils.config import (
    TRANSFER_MODEL_PATH,
    REPORTS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
)


TRANSFER_EPOCHS = 12
TRANSFER_LEARNING_RATE = 0.0001


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_efficientnet_model(num_classes: int = NUM_CLASSES):
    """
    Crea un modelo EfficientNet-B0 preentrenado en ImageNet
    y reemplaza la última capa para clasificar las 10 especies.
    """

    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(pretrained=True)

    # Congelar extractor de características inicialmente
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def unfreeze_last_blocks(model):
    """
    Libera los últimos bloques de EfficientNet para fine-tuning parcial.
    Esto permite adaptar rasgos más específicos al dataset de monos.
    """

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    return model


def train_one_epoch(model, data_loader, criterion, optimizer, device):
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
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["validation_loss"], label="Validation Loss")
    plt.title("EfficientNet-B0 - Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = REPORTS_DIR / "transfer_loss_curve.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["validation_accuracy"], label="Validation Accuracy")
    plt.title("EfficientNet-B0 - Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    accuracy_path = REPORTS_DIR / "transfer_accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_f1"], label="Train Macro F1")
    plt.plot(epochs, history["validation_f1"], label="Validation Macro F1")
    plt.title("EfficientNet-B0 - Macro F1")
    plt.xlabel("Época")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.tight_layout()
    f1_path = REPORTS_DIR / "transfer_f1_curve.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    print(f"Curva loss guardada en: {loss_path}")
    print(f"Curva accuracy guardada en: {accuracy_path}")
    print(f"Curva F1 guardada en: {f1_path}")


def save_checkpoint(model, optimizer, epoch, best_validation_f1, class_to_idx, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": "EfficientNet-B0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_f1": best_validation_f1,
        "class_to_idx": class_to_idx,
        "num_classes": NUM_CLASSES,
        "input_size": "224x224",
        "framework": "PyTorch",
    }

    torch.save(checkpoint, path)


def main():
    set_seed()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Entrenamiento con Transfer Learning - EfficientNet-B0")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {TRANSFER_EPOCHS}")
    print(f"Learning rate: {TRANSFER_LEARNING_RATE}\n")

    train_loader, validation_loader, train_dataset, validation_dataset = create_data_loaders()

    print(f"Imágenes entrenamiento: {len(train_dataset)}")
    print(f"Imágenes validación: {len(validation_dataset)}")
    print(f"Clases: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}\n")

    model = build_efficientnet_model(num_classes=NUM_CLASSES)
    model = unfreeze_last_blocks(model)
    model = model.to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(trainable_params, lr=TRANSFER_LEARNING_RATE)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

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

    for epoch in range(1, TRANSFER_EPOCHS + 1):
        print(f"Época {epoch}/{TRANSFER_EPOCHS}")

        train_loss, train_accuracy, train_f1 = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy, validation_f1 = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)

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
                path=TRANSFER_MODEL_PATH,
            )

            print(f"Nuevo mejor modelo guardado en: {TRANSFER_MODEL_PATH}")

        print("-" * 70)

    history_path = REPORTS_DIR / "transfer_training_history.json"
    history_path.write_text(
        json.dumps(history, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics = {
        "model_name": "EfficientNet-B0",
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_train_f1": history["train_f1"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "final_validation_f1": history["validation_f1"][-1],
        "model_path": str(TRANSFER_MODEL_PATH),
    }

    metrics_path = REPORTS_DIR / "transfer_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_training_curves(history)

    print("\nEntrenamiento finalizado.")
    print(f"Mejor época: {best_epoch}")
    print(f"Mejor Validation Macro F1: {best_validation_f1:.4f}")
    print(f"Historial guardado en: {history_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Modelo guardado en: {TRANSFER_MODEL_PATH}")


if __name__ == "__main__":
    main()