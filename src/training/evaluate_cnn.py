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

from src.training.preprocessing import create_data_loaders
from src.training.train_cnn import MonkeyCNN
from src.utils.config import CNN_MODEL_PATH, NUM_CLASSES, REPORTS_DIR


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {CNN_MODEL_PATH}")

    checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)

    model = MonkeyCNN(num_classes=NUM_CLASSES).to(device)
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
    plt.title("CNN propia - Matriz de confusión")
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

    output_path = REPORTS_DIR / "cnn_confusion_matrix.png"
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


def create_markdown_report(
    metrics,
    report_dict,
    most_confused,
    class_names,
):
    report_path = REPORTS_DIR / "cnn_classification_report.md"

    content = f"""# Evaluación de la CNN propia

## Resumen general

| Métrica | Valor |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision Macro | {metrics["precision_macro"]:.4f} |
| Recall Macro | {metrics["recall_macro"]:.4f} |
| Macro F1-score | {metrics["macro_f1"]:.4f} |
| Top-3 Accuracy | {metrics["top_3_accuracy"]:.4f} |

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

    for item in most_confused:
        content += (
            f"| {item['true_class']} | "
            f"{item['predicted_class']} | "
            f"{item['count']} |\n"
        )

    content += """

## Interpretación

La CNN propia logra aprender patrones visuales relevantes, pero su rendimiento todavía es limitado frente a la complejidad del problema. Esto es esperable porque el dataset tiene pocas imágenes y la tarea es de clasificación fina, donde varias especies pueden compartir rasgos visuales similares.

Las confusiones pueden deberse a:

- Similitud visual entre especies.
- Variación en iluminación, fondos y encuadres.
- Tamaño reducido del dataset.
- Pérdida de detalle al redimensionar imágenes.
- Capacidad limitada de una CNN entrenada desde cero frente a modelos preentrenados.

"""

    report_path.write_text(content, encoding="utf-8")
    print(f"Reporte Markdown guardado en: {report_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("Evaluación completa de CNN propia")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {CNN_MODEL_PATH}\n")

    _, validation_loader, _, validation_dataset = create_data_loaders()

    class_names = validation_dataset.classes

    model, checkpoint = load_model(device)

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
        "model_name": "MonkeyCNN",
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(macro_f1),
        "top_3_accuracy": float(top_3_accuracy),
        "model_path": str(CNN_MODEL_PATH),
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
    for item in most_confused:
        print(
            f"Real: {item['true_class']} -> "
            f"Predicha: {item['predicted_class']} | "
            f"Cantidad: {item['count']}"
        )

    metrics_path = REPORTS_DIR / "cnn_evaluation_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report_json_path = REPORTS_DIR / "cnn_classification_report.json"
    report_json_path.write_text(
        json.dumps(report_dict, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    confused_path = REPORTS_DIR / "cnn_most_confused_classes.json"
    confused_path.write_text(
        json.dumps(most_confused, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    create_markdown_report(
        metrics=metrics,
        report_dict=report_dict,
        most_confused=most_confused,
        class_names=class_names,
    )

    print("\nEvaluación finalizada correctamente.")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Reporte JSON guardado en: {report_json_path}")
    print(f"Confusiones guardadas en: {confused_path}")


if __name__ == "__main__":
    main()