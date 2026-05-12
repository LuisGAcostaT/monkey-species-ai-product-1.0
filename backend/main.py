import os
import sys
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Permite importar módulos desde la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.inference.predict import get_device, load_model, predict_image
from src.utils.config import IMAGE_SIZE, NUM_CLASSES, TRANSFER_MODEL_PATH


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(
    title="Monkey Species Classifier API",
    description="API para clasificar imágenes de especies de monos usando EfficientNet-B0.",
    version="1.0.0",
)

# CORS para que el frontend pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción se recomienda limitar al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
CHECKPOINT = None
DEVICE = None


@app.on_event("startup")
def startup_event():
    """
    Carga el modelo una sola vez al iniciar la API.
    Esto evita cargar el modelo en cada petición.
    """
    global MODEL, CHECKPOINT, DEVICE

    DEVICE = get_device()
    MODEL, CHECKPOINT = load_model(
        model_path=TRANSFER_MODEL_PATH,
        device=DEVICE,
    )

    print("Modelo cargado correctamente.")
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelo: {CHECKPOINT.get('model_name')}")
    print(f"Época checkpoint: {CHECKPOINT.get('epoch')}")
    print(f"Best validation F1: {CHECKPOINT.get('best_validation_f1')}")


@app.get("/")
def root():
    return {
        "message": "Monkey Species Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "monkey_species_classifier_api",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
    }


@app.get("/model-info")
def model_info():
    if CHECKPOINT is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo aún no está cargado.",
        )

    return {
        "model_name": CHECKPOINT.get("model_name", "EfficientNet-B0"),
        "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "classes": NUM_CLASSES,
        "framework": CHECKPOINT.get("framework", "PyTorch"),
        "version": "1.0.0",
        "checkpoint_epoch": CHECKPOINT.get("epoch"),
        "best_validation_f1": CHECKPOINT.get("best_validation_f1"),
        "model_path": str(TRANSFER_MODEL_PATH),
        "device": str(DEVICE),
    }


def validate_upload_file(file: UploadFile):
    """
    Valida extensión y tipo básico del archivo.
    """
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato no válido. Solo se permiten imágenes JPG, JPEG o PNG.",
        )

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Tipo de archivo no válido. Debe ser image/jpeg o image/png.",
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    """
    Recibe una imagen y retorna la especie predicha, confianza y Top-K predicciones.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible.",
        )

    validate_upload_file(file)

    content = await file.read()

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo supera el tamaño máximo permitido de {MAX_FILE_SIZE_MB} MB.",
        )

    suffix = Path(file.filename).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_image_path = temp_file.name

        result = predict_image(
            image_path=temp_image_path,
            model=MODEL,
            device=DEVICE,
            top_k=top_k,
        )

        return JSONResponse(content=result)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(error)}",
        )

    finally:
        if "temp_image_path" in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)