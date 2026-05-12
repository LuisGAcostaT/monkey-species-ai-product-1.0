import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.inference.labels import IDX_TO_LABEL, MONKEY_LABELS
from src.utils.config import IMAGE_SIZE, NUM_CLASSES, TRANSFER_MODEL_PATH


def get_device():
    """
    Usa GPU si está disponible, de lo contrario CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_efficientnet_model(num_classes: int = NUM_CLASSES):
    """
    Construye EfficientNet-B0 con la misma arquitectura usada en entrenamiento.
    """
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def load_model(
    model_path: Union[str, Path] = TRANSFER_MODEL_PATH,
    device=None,
):
    """
    Carga el checkpoint del modelo EfficientNet-B0 entrenado.
    """
    if device is None:
        device = get_device()

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = build_efficientnet_model(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def get_inference_transform():
    """
    Transformación usada para inferencia.
    Debe coincidir con la validación:
    Resize -> CenterCrop -> Tensor -> Normalización ImageNet.
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


def preprocess_image(image_path: Union[str, Path]):
    """
    Abre y transforma una imagen para inferencia.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)

    return tensor


def format_prediction(label: str, confidence: float) -> Dict:
    """
    Convierte una clase n0-n9 en una respuesta legible.
    """
    label_info = MONKEY_LABELS[label]

    return {
        "label": label,
        "scientific_name": label_info["scientific_name"],
        "common_name": label_info["common_name"],
        "confidence": round(float(confidence), 4)
    }


def predict_image(
    image_path: Union[str, Path],
    model=None,
    device=None,
    top_k: int = 3,
) -> Dict:
    """
    Realiza inferencia sobre una imagen y retorna predicción principal + Top-K.
    """
    if device is None:
        device = get_device()

    if model is None:
        model, _ = load_model(device=device)

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = softmax(outputs)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions: List[Dict] = []

    for idx, probability in zip(top_indices.cpu().tolist(), top_probabilities.cpu().tolist()):
        label = IDX_TO_LABEL[idx]
        top_predictions.append(format_prediction(label, probability))

    predicted = top_predictions[0]

    response = {
        "predicted_label": predicted["label"],
        "scientific_name": predicted["scientific_name"],
        "common_name": predicted["common_name"],
        "confidence": predicted["confidence"],
        "top_predictions": top_predictions
    }

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia local para clasificador de especies de monos."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen a clasificar."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Número de predicciones principales a retornar."
    )

    args = parser.parse_args()

    device = get_device()

    print(f"Dispositivo: {device}")

    model, checkpoint = load_model(device=device)

    print(f"Modelo cargado: {checkpoint.get('model_name')}")
    print(f"Checkpoint época: {checkpoint.get('epoch')}")
    print(f"Mejor Validation F1: {checkpoint.get('best_validation_f1')}")

    result = predict_image(
        image_path=args.image,
        model=model,
        device=device,
        top_k=args.top_k,
    )

    print("\nResultado de inferencia:")
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()