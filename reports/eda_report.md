# Análisis Exploratorio del Dataset

## Resumen general

El dataset utilizado corresponde a una tarea de clasificación multiclase de imágenes de monos con 10 clases etiquetadas desde `n0` hasta `n9`.

| Partición | Número de imágenes |
|---|---:|
| Training | 1097 |
| Validation | 272 |
| Total | 1369 |

## Distribución por clase

### Training

| Clase | Número de imágenes |
|---|---:|
| n0 | 105 |
| n1 | 111 |
| n2 | 110 |
| n3 | 122 |
| n4 | 105 |
| n5 | 113 |
| n6 | 106 |
| n7 | 114 |
| n8 | 106 |
| n9 | 105 |


### Validation

| Clase | Número de imágenes |
|---|---:|
| n0 | 26 |
| n1 | 28 |
| n2 | 27 |
| n3 | 30 |
| n4 | 26 |
| n5 | 28 |
| n6 | 26 |
| n7 | 28 |
| n8 | 27 |
| n9 | 26 |


## Tamaños originales de imagen

Las imágenes presentan tamaños originales variables.

| Métrica | Valor |
|---|---:|
| Ancho mínimo | 183 |
| Ancho máximo | 6000 |
| Alto mínimo | 198 |
| Alto máximo | 5472 |

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

## Preprocesamiento definido

Para preparar las imágenes antes del entrenamiento, se definió un pipeline con las siguientes operaciones:

### Entrenamiento

- Redimensionamiento inicial a `256x256`.
- Recorte aleatorio redimensionado a `224x224`.
- Flip horizontal aleatorio.
- Rotación aleatoria leve.
- Ajustes leves de brillo, contraste y saturación.
- Conversión a tensor.
- Normalización con medias y desviaciones estándar de ImageNet.

### Validación

- Redimensionamiento a `256x256`.
- Center crop a `224x224`.
- Conversión a tensor.
- Normalización con medias y desviaciones estándar de ImageNet.

Este pipeline permite estandarizar las dimensiones de entrada del modelo y reducir el riesgo de overfitting mediante data augmentation.