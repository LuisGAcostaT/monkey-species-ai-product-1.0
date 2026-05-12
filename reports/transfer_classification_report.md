# Evaluación de EfficientNet-B0

## Resumen general

| Métrica | Valor |
|---|---:|
| Accuracy | 0.9853 |
| Precision Macro | 0.9853 |
| Recall Macro | 0.9855 |
| Macro F1-score | 0.9852 |
| Top-3 Accuracy | 1.0000 |

## Información del checkpoint

| Campo | Valor |
|---|---|
| Modelo | EfficientNet-B0 |
| Época guardada | 8 |
| Mejor Validation F1 entrenamiento | 0.9852347588452481 |
| Framework | PyTorch |
| Input size | 224x224 |

## Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| n0 | 0.9630 | 1.0000 | 0.9811 | 26 |
| n1 | 1.0000 | 1.0000 | 1.0000 | 28 |
| n2 | 1.0000 | 0.9630 | 0.9811 | 27 |
| n3 | 1.0000 | 0.9667 | 0.9831 | 30 |
| n4 | 1.0000 | 1.0000 | 1.0000 | 26 |
| n5 | 1.0000 | 0.9643 | 0.9818 | 28 |
| n6 | 1.0000 | 1.0000 | 1.0000 | 26 |
| n7 | 1.0000 | 1.0000 | 1.0000 | 28 |
| n8 | 0.9643 | 1.0000 | 0.9818 | 27 |
| n9 | 0.9259 | 0.9615 | 0.9434 | 26 |


## Principales confusiones del modelo

| Clase real | Clase predicha | Cantidad |
|---|---|---:|
| n2 | n8 | 1 |
| n3 | n9 | 1 |
| n5 | n9 | 1 |
| n9 | n0 | 1 |


## Interpretación

EfficientNet-B0 presenta un desempeño superior frente a la CNN propia debido al uso de transfer learning. El modelo aprovecha pesos preentrenados en ImageNet, lo que permite reutilizar patrones visuales generales como bordes, texturas, formas y composiciones.

El alto desempeño debe interpretarse con cautela porque el dataset es pequeño y la validación proviene del mismo conjunto original de Kaggle. Por esta razón, antes de una puesta en producción real sería recomendable validar el modelo con imágenes externas o con un conjunto de prueba independiente.

## Justificación del modelo final

EfficientNet-B0 se selecciona como modelo final porque obtiene mejor desempeño en validación, mejora la generalización frente a la CNN entrenada desde cero y mantiene una arquitectura eficiente para despliegue en backend.
