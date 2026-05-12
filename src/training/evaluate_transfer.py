import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import models
import torch.nn as nn

from src.training.preprocessing import create_data_loaders
from src.utils.config import (
    TRANSFER_MODEL_PATH,
    CNN_MODEL_PATH,
    NUM_CLASSES,
    REPORTS_DIR,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_efficientnet_model(num_classes: int = NUM_CLASSES):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def load_transfer_model(device):
    if not TRANSFER_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {TRANSFER_MODEL_PATH}")

    checkpoint = torch.load(TRANSFER_MODEL_PATH, map_location=device)

    model = build_efficientnet_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def get_predictions(model, data_loader, device):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = softmax(outputs)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def calculate_top_k_accuracy(y_true, probabilities, k=3):
    top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]

    correct = 0

    for true_label, top_k in zip(y_true, top_k_predictions):
        if true_label in top_k:
            correct += 1

    return correct / len(y_true)


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("EfficientNet-B0 - Matriz de confusión")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()

    output_path = REPORTS_DIR / "transfer_confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Matriz de confusión guardada en: {output_path}")


def find_most_confused_classes(cm, class_names, top_n=5):
    confused_pairs = []

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "count": int(cm[i, j]),
                })

    confused_pairs = sorted(
        confused_pairs,
        key=lambda item: item["count"],
        reverse=True,
    )

    return confused_pairs[:top_n]


def create_markdown_report(metrics, report_dict, most_confused, class_names, checkpoint):
    report_path = REPORTS_DIR / "transfer_classification_report.md"

    content = f"""# Evaluación de EfficientNet-B0

## Resumen general

| Métrica | Valor |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision Macro | {metrics["precision_macro"]:.4f} |
| Recall Macro | {metrics["recall_macro"]:.4f} |
| Macro F1-score | {metrics["macro_f1"]:.4f} |
| Top-3 Accuracy | {metrics["top_3_accuracy"]:.4f} |

## Información del checkpoint

| Campo | Valor |
|---|---|
| Modelo | {checkpoint.get("model_name", "EfficientNet-B0")} |
| Época guardada | {checkpoint.get("epoch")} |
| Mejor Validation F1 entrenamiento | {checkpoint.get("best_validation_f1")} |
| Framework | {checkpoint.get("framework", "PyTorch")} |
| Input size | {checkpoint.get("input_size", "224x224")} |

## Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
"""

    for class_name in class_names:
        class_metrics = report_dict[class_name]
        content += (
            f"| {class_name} | "
            f"{class_metrics['precision']:.4f} | "
            f"{class_metrics['recall']:.4f} | "
            f"{class_metrics['f1-score']:.4f} | "
            f"{int(class_metrics['support'])} |\n"
        )

    content += """

## Principales confusiones del modelo

| Clase real | Clase predicha | Cantidad |
|---|---|---:|
"""

    if most_confused:
        for item in most_confused:
            content += (
                f"| {item['true_class']} | "
                f"{item['predicted_class']} | "
                f"{item['count']} |\n"
            )
    else:
        content += "| Sin confusiones relevantes | Sin confusiones relevantes | 0 |\n"

    content += """

## Interpretación

EfficientNet-B0 presenta un desempeño superior frente a la CNN propia debido al uso de transfer learning. El modelo aprovecha pesos preentrenados en ImageNet, lo que permite reutilizar patrones visuales generales como bordes, texturas, formas y composiciones.

El alto desempeño debe interpretarse con cautela porque el dataset es pequeño y la validación proviene del mismo conjunto original de Kaggle. Por esta razón, antes de una puesta en producción real sería recomendable validar el modelo con imágenes externas o con un conjunto de prueba independiente.

## Justificación del modelo final

EfficientNet-B0 se selecciona como modelo final porque obtiene mejor desempeño en validación, mejora la generalización frente a la CNN entrenada desde cero y mantiene una arquitectura eficiente para despliegue en backend.
"""

    report_path.write_text(content, encoding="utf-8")
    print(f"Reporte Markdown guardado en: {report_path}")


def load_json_if_exists(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def create_model_comparison(transfer_metrics):
    cnn_metrics_path = REPORTS_DIR / "cnn_evaluation_metrics.json"
    cnn_metrics = load_json_if_exists(cnn_metrics_path)

    comparison = {
        "cnn": cnn_metrics,
        "efficientnet_b0": transfer_metrics,
        "selected_model": "EfficientNet-B0",
        "selection_reason": (
            "EfficientNet-B0 fue seleccionado como modelo final por obtener "
            "mejores métricas de validación y mayor capacidad de generalización "
            "frente a la CNN propia."
        )
    }

    comparison_json_path = REPORTS_DIR / "model_comparison.json"
    comparison_json_path.write_text(
        json.dumps(comparison, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison_md_path = REPORTS_DIR / "model_comparison.md"

    if cnn_metrics:
        cnn_accuracy = cnn_metrics.get("accuracy", 0)
        cnn_f1 = cnn_metrics.get("macro_f1", 0)
        cnn_top3 = cnn_metrics.get("top_3_accuracy", 0)
    else:
        cnn_accuracy = 0
        cnn_f1 = 0
        cnn_top3 = 0

    content = f"""# Comparación de modelos

## Resultados generales

| Modelo | Accuracy | Macro F1-score | Top-3 Accuracy |
|---|---:|---:|---:|
| CNN propia | {cnn_accuracy:.4f} | {cnn_f1:.4f} | {cnn_top3:.4f} |
| EfficientNet-B0 | {transfer_metrics["accuracy"]:.4f} | {transfer_metrics["macro_f1"]:.4f} | {transfer_metrics["top_3_accuracy"]:.4f} |

## Modelo seleccionado

El modelo seleccionado para inferencia y despliegue es **EfficientNet-B0**.

## Justificación

La CNN propia se utilizó como baseline obligatorio para demostrar la implementación de una red convolucional desde cero. Sin embargo, EfficientNet-B0 obtuvo un desempeño significativamente superior gracias al uso de transfer learning.

EfficientNet-B0 aprovecha pesos preentrenados en ImageNet y realiza fine-tuning parcial sobre las últimas capas, lo que permite mejorar la clasificación en un dataset pequeño y de clasificación fina.

## Consideraciones

Aunque las métricas de EfficientNet-B0 son altas, se reconoce que el dataset es reducido. Por lo tanto, para un entorno productivo real se recomienda validar el modelo con imágenes externas adicionales.
"""

    comparison_md_path.write_text(content, encoding="utf-8")
    print(f"Comparación JSON guardada en: {comparison_json_path}")
    print(f"Comparación Markdown guardada en: {comparison_md_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Evaluación completa de EfficientNet-B0")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {TRANSFER_MODEL_PATH}\n")

    _, validation_loader, _, validation_dataset = create_data_loaders()

    class_names = validation_dataset.classes

    model, checkpoint = load_transfer_model(device)

    print(f"Checkpoint cargado desde época: {checkpoint.get('epoch')}")
    print(f"Mejor Validation F1 guardado: {checkpoint.get('best_validation_f1')}\n")

    y_true, y_pred, probabilities = get_predictions(
        model=model,
        data_loader=validation_loader,
        device=device,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    top_3_accuracy = calculate_top_k_accuracy(y_true, probabilities, k=3)

    metrics = {
        "model_name": "EfficientNet-B0",
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(macro_f1),
        "top_3_accuracy": float(top_3_accuracy),
        "model_path": str(TRANSFER_MODEL_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_f1_from_training": float(checkpoint.get("best_validation_f1")),
    }

    print("Métricas generales")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro: {recall_macro:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    most_confused = find_most_confused_classes(
        cm=cm,
        class_names=class_names,
        top_n=5,
    )

    print("\nPrincipales confusiones")
    if most_confused:
        for item in most_confused:
            print(
                f"Real: {item['true_class']} -> "
                f"Predicha: {item['predicted_class']} | "
                f"Cantidad: {item['count']}"
            )
    else:
        print("No se encontraron confusiones relevantes.")

    metrics_path = REPORTS_DIR / "transfer_evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report_json_path = REPORTS_DIR / "transfer_classification_report.json"
    report_json_path.write_text(
        json.dumps(report_dict, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    confused_path = REPORTS_DIR / "transfer_most_confused_classes.json"
    confused_path.write_text(
        json.dumps(most_confused, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    create_markdown_report(
        metrics=metrics,
        report_dict=report_dict,
        most_confused=most_confused,
        class_names=class_names,
        checkpoint=checkpoint,
    )

    create_model_comparison(metrics)

    print("\nEvaluación finalizada correctamente.")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Reporte JSON guardado en: {report_json_path}")
    print(f"Confusiones guardadas en: {confused_path}")


if __name__ == "__main__":
    main()