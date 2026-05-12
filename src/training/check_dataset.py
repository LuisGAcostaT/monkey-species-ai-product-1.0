from pathlib import Path
from PIL import Image

from src.utils.config import TRAIN_DIR, VALIDATION_DIR, LABELS_FILE


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images_by_class(base_dir: Path):
    results = {}

    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la ruta: {base_dir}")

    for class_dir in sorted(base_dir.iterdir()):
        if class_dir.is_dir():
            images = [
                file for file in class_dir.iterdir()
                if file.suffix.lower() in VALID_EXTENSIONS
            ]
            results[class_dir.name] = len(images)

    return results


def inspect_image_sizes(base_dir: Path, max_images_per_class: int = 5):
    sizes = []

    for class_dir in sorted(base_dir.iterdir()):
        if class_dir.is_dir():
            image_files = [
                file for file in class_dir.iterdir()
                if file.suffix.lower() in VALID_EXTENSIONS
            ]

            for image_path in image_files[:max_images_per_class]:
                try:
                    with Image.open(image_path) as img:
                        sizes.append({
                            "class": class_dir.name,
                            "file": image_path.name,
                            "size": img.size
                        })
                except Exception as error:
                    print(f"No se pudo leer {image_path}: {error}")

    return sizes


def main():
    print("Validando dataset...\n")

    print(f"Ruta entrenamiento: {TRAIN_DIR}")
    print(f"Ruta validación: {VALIDATION_DIR}")
    print(f"Archivo labels: {LABELS_FILE}\n")

    train_counts = count_images_by_class(TRAIN_DIR)
    validation_counts = count_images_by_class(VALIDATION_DIR)

    print("Imágenes por clase - Training")
    for label, count in train_counts.items():
        print(f"{label}: {count}")

    print("\nImágenes por clase - Validation")
    for label, count in validation_counts.items():
        print(f"{label}: {count}")

    total_train = sum(train_counts.values())
    total_validation = sum(validation_counts.values())

    print("\nResumen")
    print(f"Total training: {total_train}")
    print(f"Total validation: {total_validation}")
    print(f"Total dataset: {total_train + total_validation}")

    print("\nMuestras de tamaños de imagen")
    image_sizes = inspect_image_sizes(TRAIN_DIR)

    for item in image_sizes[:20]:
        print(f"{item['class']} | {item['file']} | {item['size']}")

    if LABELS_FILE.exists():
        print("\nArchivo monkey_labels.txt encontrado.")
    else:
        print("\nAdvertencia: No se encontró monkey_labels.txt")


if __name__ == "__main__":
    main()