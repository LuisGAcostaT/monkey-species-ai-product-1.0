from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)


def get_train_transforms():
    """
    Transformaciones para entrenamiento.
    Incluyen data augmentation para reducir overfitting.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_validation_transforms():
    """
    Transformaciones para validación.
    No se aplica augmentation para no alterar la evaluación.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_datasets(
    train_dir: Path = TRAIN_DIR,
    validation_dir: Path = VALIDATION_DIR,
):
    """
    Carga los datasets de entrenamiento y validación usando ImageFolder.

    La estructura esperada es:
    train_dir/
        n0/
        n1/
        ...
        n9/

    validation_dir/
        n0/
        n1/
        ...
        n9/
    """

    if not train_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrenamiento: {train_dir}")

    if not validation_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de validación: {validation_dir}")

    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_train_transforms()
    )

    validation_dataset = datasets.ImageFolder(
        root=str(validation_dir),
        transform=get_validation_transforms()
    )

    return train_dataset, validation_dataset


def create_data_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Crea DataLoaders para entrenamiento y validación.

    En Windows se recomienda num_workers=0 para evitar problemas
    con multiprocessing en algunos entornos.
    """

    torch.manual_seed(RANDOM_SEED)

    train_dataset, validation_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, validation_loader, train_dataset, validation_dataset


def get_class_names(dataset):
    """
    Retorna las clases detectadas por ImageFolder.
    """
    return dataset.classes


def get_class_to_idx(dataset):
    """
    Retorna el diccionario clase -> índice.
    """
    return dataset.class_to_idx


if __name__ == "__main__":
    train_loader, validation_loader, train_dataset, validation_dataset = create_data_loaders()

    print("Preprocesamiento cargado correctamente.\n")

    print(f"Imágenes de entrenamiento: {len(train_dataset)}")
    print(f"Imágenes de validación: {len(validation_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Clases: {get_class_names(train_dataset)}")
    print(f"Class to idx: {get_class_to_idx(train_dataset)}")

    images, labels = next(iter(train_loader))

    print("\nPrimer batch:")
    print(f"Shape imágenes: {images.shape}")
    print(f"Shape etiquetas: {labels.shape}")
    print(f"Etiquetas ejemplo: {labels[:10].tolist()}")