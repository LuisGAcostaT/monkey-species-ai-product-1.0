# Evaluación de la CNN propia

## Resumen general

| Métrica | Valor |
|---|---:|
| Accuracy | 0.5882 |
| Precision Macro | 0.6164 |
| Recall Macro | 0.5855 |
| Macro F1-score | 0.5854 |
| Top-3 Accuracy | 0.8676 |

## Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| n0 | 0.5000 | 0.2692 | 0.3500 | 26 |
| n1 | 0.5278 | 0.6786 | 0.5938 | 28 |
| n2 | 0.8571 | 0.8889 | 0.8727 | 27 |
| n3 | 0.8400 | 0.7000 | 0.7636 | 30 |
| n4 | 0.3929 | 0.4231 | 0.4074 | 26 |
| n5 | 0.5652 | 0.4643 | 0.5098 | 28 |
| n6 | 0.8182 | 0.6923 | 0.7500 | 26 |
| n7 | 0.4419 | 0.6786 | 0.5352 | 28 |
| n8 | 0.8000 | 0.4444 | 0.5714 | 27 |
| n9 | 0.4211 | 0.6154 | 0.5000 | 26 |


## Principales confusiones del modelo

| Clase real | Clase predicha | Cantidad |
|---|---|---:|
| n0 | n9 | 13 |
| n8 | n4 | 9 |
| n4 | n7 | 6 |
| n6 | n1 | 6 |
| n1 | n7 | 5 |


## Interpretación

La CNN propia logra aprender patrones visuales relevantes, pero su rendimiento todavía es limitado frente a la complejidad del problema. Esto es esperable porque el dataset tiene pocas imágenes y la tarea es de clasificación fina, donde varias especies pueden compartir rasgos visuales similares.

Las confusiones pueden deberse a:

- Similitud visual entre especies.
- Variación en iluminación, fondos y encuadres.
- Tamaño reducido del dataset.
- Pérdida de detalle al redimensionar imágenes.
- Capacidad limitada de una CNN entrenada desde cero frente a modelos preentrenados.

