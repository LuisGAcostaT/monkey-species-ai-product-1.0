import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.utils.config import TRAIN_DIR, VALIDATION_DIR, REPORTS_DIR


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files_by_class(base_dir: Path):
    data = {}

    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la ruta: {base_dir}")

    for class_dir in sorted(base_dir.iterdir()):
        if class_dir.is_dir():
            images = [
                file for file in class_dir.iterdir()
                if file.suffix.lower() in VALID_EXTENSIONS
            ]
            data[class_dir.name] = images

    return data


def count_images(data):
    return {label: len(files) for label, files in data.items()}


def collect_image_sizes(data, max_images_per_class=None):
    records = []

    for label, files in data.items():
        selected_files = files

        if max_images_per_class:
            selected_files = files[:max_images_per_class]

        for image_path in selected_files:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    records.append({
                        "label": label,
                        "file": image_path.name,
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(width / height, 4)
                    })
            except Exception as error:
                print(f"No se pudo leer {image_path}: {error}")

    return records


def plot_class_distribution(train_counts, validation_counts):
    labels = sorted(train_counts.keys())
    train_values = [train_counts[label] for label in labels]
    validation_values = [validation_counts[label] for label in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], train_values, width=width, label="Training")
    plt.bar([i + width / 2 for i in x], validation_values, width=width, label="Validation")

    plt.title("Distribución de imágenes por clase")
    plt.xlabel("Clase")
    plt.ylabel("Número de imágenes")
    plt.xticks(list(x), labels)
    plt.legend()
    plt.tight_layout()

    output_path = REPORTS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado: {output_path}")


def plot_train_validation_distribution(total_train, total_validation):
    labels = ["Training", "Validation"]
    values = [total_train, total_validation]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Distribución general Training vs Validation")
    plt.ylabel("Número de imágenes")
    plt.tight_layout()

    output_path = REPORTS_DIR / "train_validation_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado: {output_path}")


def plot_sample_images(train_data, samples_per_class=3):
    labels = sorted(train_data.keys())
    rows = len(labels)
    cols = samples_per_class

    plt.figure(figsize=(cols * 4, rows * 3))

    image_index = 1

    for label in labels:
        files = train_data[label]

        if len(files) == 0:
            continue

        selected_files = random.sample(files, min(samples_per_class, len(files)))

        for image_path in selected_files:
            plt.subplot(rows, cols, image_index)

            try:
                with Image.open(image_path) as img:
                    plt.imshow(img.convert("RGB"))
                    plt.title(label)
                    plt.axis("off")
            except Exception:
                plt.text(0.5, 0.5, "Error leyendo imagen", ha="center")
                plt.axis("off")

            image_index += 1

    plt.suptitle("Ejemplos visuales por clase", fontsize=16)
    plt.tight_layout()

    output_path = REPORTS_DIR / "sample_images_by_class.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado: {output_path}")


def plot_image_sizes_distribution(size_records):
    widths = [record["width"] for record in size_records]
    heights = [record["height"] for record in size_records]

    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.6)
    plt.title("Distribución de tamaños originales de imagen")
    plt.xlabel("Ancho")
    plt.ylabel("Alto")
    plt.tight_layout()

    output_path = REPORTS_DIR / "image_sizes_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Gráfico guardado: {output_path}")


def create_eda_report(summary):
    report_path = REPORTS_DIR / "eda_report.md"

    content = f"""# Análisis Exploratorio del Dataset

## Resumen general

El dataset utilizado corresponde a una tarea de clasificación multiclase de imágenes de monos con 10 clases etiquetadas desde `n0` hasta `n9`.

| Partición | Número de imágenes |
|---|---:|
| Training | {summary["total_train"]} |
| Validation | {summary["total_validation"]} |
| Total | {summary["total_dataset"]} |

## Distribución por clase

### Training

| Clase | Número de imágenes |
|---|---:|
"""

    for label, count in summary["train_counts"].items():
        content += f"| {label} | {count} |\n"

    content += """

### Validation

| Clase | Número de imágenes |
|---|---:|
"""

    for label, count in summary["validation_counts"].items():
        content += f"| {label} | {count} |\n"

    content += f"""

## Tamaños originales de imagen

Las imágenes presentan tamaños originales variables.

| Métrica | Valor |
|---|---:|
| Ancho mínimo | {summary["image_size_stats"]["min_width"]} |
| Ancho máximo | {summary["image_size_stats"]["max_width"]} |
| Alto mínimo | {summary["image_size_stats"]["min_height"]} |
| Alto máximo | {summary["image_size_stats"]["max_height"]} |

Esta variabilidad justifica el redimensionamiento de las imágenes a un tamaño uniforme antes del entrenamiento del modelo.

## Posibles problemas identificados

1. **Tamaño reducido del dataset:** el conjunto completo contiene menos de 1.400 imágenes, lo que incrementa el riesgo de overfitting.
2. **Variabilidad en dimensiones:** las imágenes tienen diferentes resoluciones, por lo que se requiere redimensionamiento.
3. **Similitud visual entre especies:** algunas clases pueden compartir rasgos físicos parecidos, haciendo más difícil la clasificación.
4. **Fondos variables:** las imágenes pueden contener diferentes entornos, iluminación y composición.
5. **Desbalance leve:** aunque las clases son relativamente similares en cantidad, existen pequeñas diferencias entre ellas.
6. **Necesidad de data augmentation:** por el tamaño del dataset, se recomienda aplicar transformaciones como rotación leve, flip horizontal, zoom y ajustes de brillo.

## Archivos generados

- `class_distribution.png`
- `train_validation_distribution.png`
- `sample_images_by_class.png`
- `image_sizes_distribution.png`
- `eda_summary.json`
"""

    report_path.write_text(content, encoding="utf-8")
    print(f"Reporte EDA guardado: {report_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Iniciando análisis exploratorio del dataset...\n")

    train_data = get_image_files_by_class(TRAIN_DIR)
    validation_data = get_image_files_by_class(VALIDATION_DIR)

    train_counts = count_images(train_data)
    validation_counts = count_images(validation_data)

    total_train = sum(train_counts.values())
    total_validation = sum(validation_counts.values())
    total_dataset = total_train + total_validation

    print("Imágenes por clase - Training")
    for label, count in train_counts.items():
        print(f"{label}: {count}")

    print("\nImágenes por clase - Validation")
    for label, count in validation_counts.items():
        print(f"{label}: {count}")

    print("\nResumen")
    print(f"Total training: {total_train}")
    print(f"Total validation: {total_validation}")
    print(f"Total dataset: {total_dataset}")

    all_size_records = collect_image_sizes(train_data) + collect_image_sizes(validation_data)

    widths = [record["width"] for record in all_size_records]
    heights = [record["height"] for record in all_size_records]

    image_size_stats = {
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights)
    }

    summary = {
        "total_train": total_train,
        "total_validation": total_validation,
        "total_dataset": total_dataset,
        "train_counts": train_counts,
        "validation_counts": validation_counts,
        "image_size_stats": image_size_stats,
        "observations": [
            "Dataset pequeño, con riesgo de overfitting.",
            "Las clases están relativamente balanceadas, aunque existen diferencias leves.",
            "Las imágenes tienen tamaños originales variables.",
            "La tarea es fine-grained classification, por lo que puede haber confusión entre especies visualmente similares.",
            "Se recomienda usar data augmentation y transfer learning."
        ]
    }

    summary_path = REPORTS_DIR / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nResumen JSON guardado: {summary_path}")

    plot_class_distribution(train_counts, validation_counts)
    plot_train_validation_distribution(total_train, total_validation)
    plot_sample_images(train_data, samples_per_class=3)
    plot_image_sizes_distribution(all_size_records)
    create_eda_report(summary)

    print("\nAnálisis exploratorio finalizado correctamente.")


if __name__ == "__main__":
    main()