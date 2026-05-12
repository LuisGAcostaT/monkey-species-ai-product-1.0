import random

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.training.preprocessing import get_train_transforms
from src.utils.config import TRAIN_DIR, REPORTS_DIR


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def denormalize_image(tensor_image):
    """
    Revierte la normalización de ImageNet para poder visualizar la imagen.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = tensor_image * std + mean
    image = torch.clamp(image, 0, 1)

    return image.permute(1, 2, 0).numpy()


def get_random_image():
    class_dirs = [path for path in TRAIN_DIR.iterdir() if path.is_dir()]
    selected_class = random.choice(class_dirs)

    image_files = [
        file for file in selected_class.iterdir()
        if file.suffix.lower() in VALID_EXTENSIONS
    ]

    selected_image = random.choice(image_files)

    return selected_class.name, selected_image


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    label, image_path = get_random_image()

    print(f"Imagen seleccionada: {image_path}")
    print(f"Clase: {label}")

    image = Image.open(image_path).convert("RGB")
    transform = get_train_transforms()

    plt.figure(figsize=(12, 8))

    for index in range(8):
        augmented = transform(image)
        augmented_to_show = denormalize_image(augmented)

        plt.subplot(2, 4, index + 1)
        plt.imshow(augmented_to_show)
        plt.title(f"{label} - Aug {index + 1}")
        plt.axis("off")

    plt.suptitle("Ejemplos de Data Augmentation", fontsize=16)
    plt.tight_layout()

    output_path = REPORTS_DIR / "data_augmentation_examples.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Imagen guardada en: {output_path}")


if __name__ == "__main__":
    main()