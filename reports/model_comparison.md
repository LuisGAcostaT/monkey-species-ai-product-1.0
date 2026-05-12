# Comparación de modelos

## Resultados generales

| Modelo | Accuracy | Macro F1-score | Top-3 Accuracy |
|---|---:|---:|---:|
| CNN propia | 0.5882 | 0.5854 | 0.8676 |
| EfficientNet-B0 | 0.9853 | 0.9852 | 1.0000 |

## Modelo seleccionado

El modelo seleccionado para inferencia y despliegue es **EfficientNet-B0**.

## Justificación

La CNN propia se utilizó como baseline obligatorio para demostrar la implementación de una red convolucional desde cero. Sin embargo, EfficientNet-B0 obtuvo un desempeño significativamente superior gracias al uso de transfer learning.

EfficientNet-B0 aprovecha pesos preentrenados en ImageNet y realiza fine-tuning parcial sobre las últimas capas, lo que permite mejorar la clasificación en un dataset pequeño y de clasificación fina.

## Consideraciones

Aunque las métricas de EfficientNet-B0 son altas, se reconoce que el dataset es reducido. Por lo tanto, para un entorno productivo real se recomienda validar el modelo con imágenes externas adicionales.
