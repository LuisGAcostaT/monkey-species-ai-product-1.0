# Monkey Species AI Product

Sistema de clasificación multiclase de imágenes para identificar especies de monos usando redes neuronales convolucionales, transfer learning y despliegue como producto IA.

El proyecto utiliza el dataset **10 Monkey Species** de Kaggle y cubre el flujo completo de un producto de inteligencia artificial:

- Análisis exploratorio del dataset.
- Preprocesamiento y data augmentation.
- Entrenamiento de una CNN propia.
- Entrenamiento con transfer learning usando EfficientNet-B0.
- Evaluación con métricas completas.
- Backend de inferencia con FastAPI.
- Frontend web con React.
- Dockerización.
- Despliegue público en Google Cloud Run.

---

## Demo público

### Frontend

```txt
https://monkey-species-frontend-992637477682.us-east1.run.app
```

### Backend

```txt
https://monkey-species-backend-992637477682.us-east1.run.app
```

### Documentación interactiva de la API

```txt
https://monkey-species-backend-992637477682.us-east1.run.app/docs
```

---

## Dataset

Se utilizó el dataset **10 Monkey Species** disponible en Kaggle:

```txt
https://www.kaggle.com/datasets/slothkong/10-monkey-species
```

El dataset contiene imágenes JPEG organizadas en carpetas de entrenamiento y validación, con 10 clases etiquetadas desde `n0` hasta `n9`.

| Label | Nombre científico     | Nombre común              |
| ----- | --------------------- | ------------------------- |
| n0    | alouatta_palliata     | mantled_howler            |
| n1    | erythrocebus_patas    | patas_monkey              |
| n2    | cacajao_calvus        | bald_uakari               |
| n3    | macaca_fuscata        | japanese_macaque          |
| n4    | cebuella_pygmea       | pygmy_marmoset            |
| n5    | cebus_capucinus       | white_headed_capuchin     |
| n6    | mico_argentatus       | silvery_marmoset          |
| n7    | saimiri_sciureus      | common_squirrel_monkey    |
| n8    | aotus_nigriceps       | black_headed_night_monkey |
| n9    | trachypithecus_johnii | nilgiri_langur            |

---

## Estructura del proyecto

```txt
monkey-species-ai-product/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── data/
│   └── README.md
│
├── frontend/
│   ├── public/
│   │   └── species/
│   ├── src/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── .env
│
├── models/
│   ├── cnn_monkey_classifier.pth
│   └── efficientnet_monkey_classifier.pth
│
├── reports/
│   ├── eda_report.md
│   ├── cnn_classification_report.md
│   ├── transfer_classification_report.md
│   ├── model_comparison.md
│   └── *.png / *.json
│
├── src/
│   ├── inference/
│   │   ├── labels.py
│   │   └── predict.py
│   │
│   ├── training/
│   │   ├── check_dataset.py
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── train_cnn.py
│   │   ├── evaluate_cnn.py
│   │   ├── train_transfer.py
│   │   └── evaluate_transfer.py
│   │
│   └── utils/
│       └── config.py
│
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Arquitectura general

```txt
Usuario
  │
  ▼
Frontend React
  │
  │ POST /predict
  ▼
Backend FastAPI
  │
  │ Carga modelo EfficientNet-B0
  ▼
Modelo PyTorch
  │
  ▼
Respuesta JSON con predicción, confianza y Top-3 clases
```

La solución está separada en dos servicios principales:

| Componente   | Tecnología                | Descripción                             |
| ------------ | ------------------------- | --------------------------------------- |
| Modelo       | PyTorch + EfficientNet-B0 | Clasificador final de imágenes          |
| Backend      | FastAPI                   | API de inferencia                       |
| Frontend     | React + Vite              | Interfaz web para carga y visualización |
| Contenedores | Docker                    | Empaquetamiento de servicios            |
| Despliegue   | Google Cloud Run          | Publicación del backend y frontend      |

---

## Instalación local desde cero

Esta sección permite ejecutar el proyecto localmente después de clonar el repositorio.

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd monkey-species-ai-product
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual en Windows

```bash
.\venv\Scripts\Activate.ps1
```

### 4. Instalar dependencias del proyecto

```bash
pip install -r requirements.txt
```

Si se desea ejecutar el backend directamente, también se pueden instalar las dependencias del backend:

```bash
pip install -r backend/requirements.txt
```

---

## Configuración de Kaggle para descargar el dataset

El dataset completo **no se versiona dentro del repositorio** debido a su tamaño.

Por esta razón, si se desea ejecutar entrenamiento, evaluación o análisis exploratorio, se debe descargar el dataset desde Kaggle.

### 1. Instalar Kaggle CLI

Con el entorno virtual activo:

```bash
pip install kaggle
```

### 2. Configurar token de Kaggle en PowerShell

Kaggle puede entregar un token tipo API Key. Para usarlo temporalmente en la terminal actual, ejecutar:

```powershell
$env:KAGGLE_API_TOKEN="TU_TOKEN_NUEVO"
```

Ejemplo de formato, sin usar el token real:

```powershell
$env:KAGGLE_API_TOKEN="KGAT_xxxxxxxxxxxxxxxxxxxxxxxx"
```

Este método deja el token activo únicamente en la terminal actual.

### 3. Probar conexión con Kaggle

```bash
kaggle datasets list
```

Si se listan datasets, la configuración quedó funcionando.

### 4. Descargar el dataset

Desde la raíz del proyecto:

```powershell
mkdir data\raw
kaggle datasets download -d slothkong/10-monkey-species -p data/raw
```

Esto descargará un archivo `.zip` dentro de:

```txt
data/raw
```

### 5. Descomprimir el dataset

Ejecutar:

```powershell
Expand-Archive -Path "data/raw/10-monkey-species.zip" -DestinationPath "data/raw" -Force
```

### 6. Validar estructura del dataset

Después de descomprimir, la estructura esperada es:

```txt
monkey-species-ai-product/
├── data/
│   └── raw/
│       ├── training/
│       │   └── training/
│       │       ├── n0/
│       │       ├── n1/
│       │       ├── n2/
│       │       ├── n3/
│       │       ├── n4/
│       │       ├── n5/
│       │       ├── n6/
│       │       ├── n7/
│       │       ├── n8/
│       │       └── n9/
│       │
│       ├── validation/
│       │   └── validation/
│       │       ├── n0/
│       │       ├── n1/
│       │       ├── n2/
│       │       ├── n3/
│       │       ├── n4/
│       │       ├── n5/
│       │       ├── n6/
│       │       ├── n7/
│       │       ├── n8/
│       │       └── n9/
│       │
│       └── monkey_labels.txt
```

Para validar automáticamente:

```bash
python -m src.training.check_dataset
```

Si la carpeta `data/raw/training/training` no existe, los scripts de entrenamiento generarán un error similar a:

```txt
FileNotFoundError: No existe la carpeta de entrenamiento
```

---

## Uso sin dataset local

Para ejecutar el producto web o consumir la API **no es necesario tener el dataset completo localmente**.

Para usar la aplicación solo se requiere:

1. Tener el modelo serializado en:

```txt
models/efficientnet_monkey_classifier.pth
```

2. Ejecutar el backend y el frontend.

3. Usar una imagen propia o las imágenes de prueba precargadas en la interfaz web.

El dataset local solo es necesario si se desea:

- Reentrenar la CNN propia.
- Reentrenar EfficientNet-B0.
- Ejecutar el análisis exploratorio.
- Ejecutar evaluaciones completas sobre training/validation.

---

## Modelo serializado

El modelo final entrenado corresponde a **EfficientNet-B0** y se guarda localmente en la siguiente ruta:

```txt
models/efficientnet_monkey_classifier.pth
```

Este archivo contiene el checkpoint del modelo seleccionado para inferencia y despliegue, incluyendo los pesos entrenados, la arquitectura esperada, la época del mejor checkpoint y las métricas asociadas al mejor desempeño en validación.

Por buenas prácticas, el archivo `.pth` no se versiona directamente en Git debido a su tamaño. Sin embargo, el modelo puede obtenerse de dos formas:

### Opción 1: Regenerar el modelo desde el código

Para regenerar el modelo, primero se debe tener el dataset descargado localmente en la estructura indicada en la sección **Configuración de Kaggle para descargar el dataset**.

Luego ejecutar:

```bash
python -m src.training.train_transfer
```

Al finalizar, se generará el archivo:

```txt
models/efficientnet_monkey_classifier.pth
```

### Opción 2: Descargar el modelo entrenado

El archivo del modelo entrenado puede descargarse desde el siguiente enlace:

```txt
https://github.com/LuisGAcostaT/monkey-species-ai-product-1.0/releases/tag/v1.0.0
```

Después de descargarlo, debe ubicarse en:

```txt
models/efficientnet_monkey_classifier.pth
```

La estructura esperada debe quedar así:

```txt
models/
└── efficientnet_monkey_classifier.pth
```

Este archivo es necesario para ejecutar la inferencia local, levantar el backend y construir la imagen Docker del backend.

---

## Ejecución local sin Docker

### 1. Verificar modelo serializado

Antes de levantar el backend, verificar que exista:

```txt
models/efficientnet_monkey_classifier.pth
```

Si no existe, descargarlo desde la release o regenerarlo con entrenamiento.

### 2. Ejecutar backend

Desde la raíz del proyecto:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend local:

```txt
http://localhost:8000
```

Documentación Swagger:

```txt
http://localhost:8000/docs
```

Endpoints principales:

```txt
GET /health
GET /model-info
POST /predict
```

### 3. Ejecutar frontend

En otra terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend local:

```txt
http://localhost:5173
```

---

## Ejecución local con Docker Compose

Antes de ejecutar Docker Compose, se debe tener el modelo serializado en:

```txt
models/efficientnet_monkey_classifier.pth
```

Luego, desde la raíz del proyecto:

```bash
docker compose up --build
```

Servicios locales:

| Servicio | URL                     |
| -------- | ----------------------- |
| Backend  | `http://localhost:8000` |
| Frontend | `http://localhost:3000` |

Para detener los contenedores:

```bash
docker compose down
```

---

## Entrenamiento

Antes de ejecutar estos comandos, asegúrate de haber descargado el dataset localmente y de que exista la siguiente estructura:

```txt
data/raw/training/training
data/raw/validation/validation
data/raw/monkey_labels.txt
```

### Validar dataset

```bash
python -m src.training.check_dataset
```

### Ejecutar análisis exploratorio

```bash
python -m src.training.eda
```

### Entrenar CNN propia

```bash
python -m src.training.train_cnn
```

### Evaluar CNN propia

```bash
python -m src.training.evaluate_cnn
```

### Entrenar EfficientNet-B0

```bash
python -m src.training.train_transfer
```

### Evaluar EfficientNet-B0

```bash
python -m src.training.evaluate_transfer
```

---

## Inferencia local

Para ejecutar inferencia local se necesita el modelo serializado en:

```txt
models/efficientnet_monkey_classifier.pth
```

Ejemplo usando una imagen del dataset:

```bash
python -m src.inference.predict --image "data/raw/validation/validation/n3/n300.jpg"
```

También se puede usar cualquier imagen local en formato JPG, JPEG o PNG.

Salida esperada:

```json
{
  "predicted_label": "n3",
  "scientific_name": "macaca_fuscata",
  "common_name": "japanese_macaque",
  "confidence": 0.9084,
  "top_predictions": [
    {
      "label": "n3",
      "scientific_name": "macaca_fuscata",
      "common_name": "japanese_macaque",
      "confidence": 0.9084
    }
  ]
}
```

---

## Análisis exploratorio del dataset

El dataset contiene:

| Partición  | Cantidad |
| ---------- | -------: |
| Training   |     1097 |
| Validation |      272 |
| Total      |     1369 |

Durante el análisis exploratorio se identificaron los siguientes puntos:

- Las clases están relativamente balanceadas.
- Las imágenes tienen tamaños originales variables.
- El dataset es pequeño para entrenamiento profundo desde cero.
- Existen especies visualmente similares.
- Hay fondos, poses e iluminación variables.
- Existe riesgo de overfitting si no se aplica data augmentation.

Archivos generados:

```txt
reports/class_distribution.png
reports/train_validation_distribution.png
reports/sample_images_by_class.png
reports/image_sizes_distribution.png
reports/eda_summary.json
reports/eda_report.md
```

---

## Preprocesamiento

El pipeline de preprocesamiento incluye:

### Entrenamiento

- Resize inicial a `256x256`.
- Random resized crop a `224x224`.
- Flip horizontal aleatorio.
- Rotación aleatoria leve.
- Ajustes leves de brillo, contraste y saturación.
- Conversión a tensor.
- Normalización con medias y desviaciones estándar de ImageNet.

### Validación e inferencia

- Resize a `256x256`.
- Center crop a `224x224`.
- Conversión a tensor.
- Normalización con medias y desviaciones estándar de ImageNet.

Este pipeline permite estandarizar el tamaño de entrada y reducir el riesgo de overfitting.

---

## Modelos entrenados

Se entrenaron dos modelos:

### 1. CNN propia

Modelo convolucional implementado desde cero con:

- `Conv2D`
- `BatchNorm2D`
- `ReLU`
- `MaxPooling`
- `Dropout`
- Capas fully connected
- Salida de 10 clases

Este modelo se utilizó como baseline obligatorio para validar el pipeline completo.

### 2. EfficientNet-B0 con transfer learning

Modelo preentrenado en ImageNet, adaptado para clasificar las 10 especies del dataset.

Se aplicó:

- Reemplazo de la capa clasificadora final.
- Fine-tuning parcial sobre los últimos bloques.
- Optimización con Adam.
- Selección del mejor checkpoint por Macro F1 en validación.

---

## Resultados

### Comparación de modelos

| Modelo          | Accuracy | Macro F1-score | Top-3 Accuracy |
| --------------- | -------: | -------------: | -------------: |
| CNN propia      |   0.5882 |         0.5854 |         0.8676 |
| EfficientNet-B0 |   0.9853 |         0.9852 |         1.0000 |

El modelo seleccionado para inferencia y despliegue fue **EfficientNet-B0**.

### Justificación

La CNN propia permitió demostrar la implementación de una red convolucional desde cero. Sin embargo, EfficientNet-B0 obtuvo un desempeño significativamente superior gracias al uso de transfer learning.

EfficientNet-B0 aprovecha pesos preentrenados en ImageNet, reutilizando patrones visuales generales como bordes, texturas, formas y composiciones. Esto resulta especialmente útil en un dataset pequeño y de clasificación fina.

---

## Evaluación del modelo final

Métricas de EfficientNet-B0:

| Métrica         |  Valor |
| --------------- | -----: |
| Accuracy        | 0.9853 |
| Precision Macro | 0.9853 |
| Recall Macro    | 0.9855 |
| Macro F1-score  | 0.9852 |
| Top-3 Accuracy  | 1.0000 |

Principales confusiones identificadas:

| Clase real | Clase predicha | Cantidad |
| ---------- | -------------- | -------: |
| n2         | n8             |        1 |
| n3         | n9             |        1 |
| n5         | n9             |        1 |
| n9         | n0             |        1 |

Archivos generados:

```txt
reports/transfer_confusion_matrix.png
reports/transfer_classification_report.md
reports/transfer_evaluation_metrics.json
reports/model_comparison.md
```

---

## Backend de inferencia

El backend fue desarrollado con **FastAPI**.

### Endpoints disponibles

| Método | Endpoint      | Descripción                       |
| ------ | ------------- | --------------------------------- |
| GET    | `/`           | Información básica del servicio   |
| GET    | `/health`     | Estado del servicio               |
| GET    | `/model-info` | Información del modelo cargado    |
| POST   | `/predict`    | Predicción a partir de una imagen |

### Ejemplo de respuesta de `/predict`

```json
{
  "predicted_label": "n4",
  "scientific_name": "cebuella_pygmea",
  "common_name": "pygmy_marmoset",
  "confidence": 0.9211,
  "top_predictions": [
    {
      "label": "n4",
      "scientific_name": "cebuella_pygmea",
      "common_name": "pygmy_marmoset",
      "confidence": 0.9211
    },
    {
      "label": "n7",
      "scientific_name": "saimiri_sciureus",
      "common_name": "common_squirrel_monkey",
      "confidence": 0.0165
    },
    {
      "label": "n1",
      "scientific_name": "erythrocebus_patas",
      "common_name": "patas_monkey",
      "confidence": 0.016
    }
  ]
}
```

---

## Frontend web

El frontend fue desarrollado con **React + Vite**.

Funcionalidades:

- Carga de imagen desde navegador.
- Validación de formato.
- Vista previa de la imagen.
- Consumo real del endpoint `/predict`.
- Visualización de especie predicha.
- Visualización de nombre científico.
- Visualización de confianza.
- Visualización de Top-3 predicciones.
- Visualización opcional de respuesta JSON.
- Mensajes de error ante archivos inválidos o fallos del backend.
- Atlas visual de especies.
- Imágenes de prueba precargadas para probar el modelo sin dataset local.

---

## Despliegue en Google Cloud Run

La aplicación fue desplegada en Google Cloud Run usando dos servicios:

| Servicio                  | Descripción                               |
| ------------------------- | ----------------------------------------- |
| `monkey-species-backend`  | API FastAPI con el modelo EfficientNet-B0 |
| `monkey-species-frontend` | Aplicación React servida con Nginx        |

### Backend

El backend fue empaquetado en Docker y publicado en Artifact Registry.

Ejemplo de despliegue:

```bash
gcloud run deploy monkey-species-backend \
  --image IMAGE_BACKEND \
  --region us-east1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

### Frontend

El frontend fue compilado con la URL pública del backend:

```bash
docker build -f frontend/Dockerfile \
  --build-arg VITE_API_URL=PEGAR_URL_BACKEND \
  -t IMAGE_FRONTEND .
```

Luego fue desplegado en Cloud Run:

```bash
gcloud run deploy monkey-species-frontend \
  --image IMAGE_FRONTEND \
  --region us-east1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --port 8080
```

---

## Consideraciones sobre producción

Aunque el modelo final obtuvo métricas altas, se reconocen las siguientes limitaciones:

- El dataset es pequeño.
- La validación proviene del mismo dataset original de Kaggle.
- No se cuenta con un conjunto externo independiente.
- Pueden existir imágenes reales con condiciones diferentes a las del dataset.
- El rendimiento en producción puede variar frente a imágenes tomadas en ambientes no controlados.

Por estas razones, antes de una puesta en producción real se recomienda:

- Validar con imágenes externas adicionales.
- Construir un conjunto de test independiente.
- Agregar monitoreo de predicciones.
- Registrar imágenes fallidas para reentrenamiento.
- Implementar versionamiento formal de modelos.
- Implementar CI/CD.
- Agregar autenticación si el servicio se vuelve privado.

---

## Decisiones técnicas

### Uso de CNN propia

Se implementó una CNN desde cero para cumplir el requisito técnico obligatorio y establecer una línea base.

### Uso de transfer learning

Se seleccionó EfficientNet-B0 porque ofrece buen rendimiento en clasificación de imágenes, es eficiente para despliegue y permite aprovechar conocimiento visual preentrenado en ImageNet.

### Uso de FastAPI

FastAPI permite exponer el modelo mediante una API rápida, sencilla de documentar y adecuada para servicios de inferencia.

### Uso de React

React permite construir una interfaz moderna, mantenible y separada del backend.

### Uso de Docker

Docker permite empaquetar backend y frontend para ejecución local y despliegue reproducible.

### Uso de Cloud Run

Cloud Run permite desplegar contenedores HTTP de forma administrada, escalable y accesible mediante URL pública.

---

## Estado del proyecto

- [x] Análisis exploratorio del dataset.
- [x] Preprocesamiento.
- [x] Data augmentation.
- [x] CNN propia.
- [x] Evaluación de CNN propia.
- [x] Transfer learning con EfficientNet-B0.
- [x] Evaluación completa del modelo final.
- [x] Inferencia local.
- [x] Backend FastAPI.
- [x] Frontend React.
- [x] Dockerización.
- [x] Despliegue público en Google Cloud Run.
- [x] Documentación del proyecto.

---