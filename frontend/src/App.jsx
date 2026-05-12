import { useEffect, useMemo, useState } from "react";
import {
  Upload,
  Image as ImageIcon,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Server,
  Brain,
  Database,
  BarChart3,
} from "lucide-react";
import "./App.css";
import { speciesAtlas } from "./data/speciesAtlas";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const allowedTypes = ["image/jpeg", "image/png"];

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatScientificName(value) {
  if (!value) return "";
  return value.replaceAll("_", " ");
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoError, setModelInfoError] = useState("");

  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isCheckingApi, setIsCheckingApi] = useState(false);

  const canPredict = useMemo(() => {
    return selectedFile && !isPredicting;
  }, [selectedFile, isPredicting]);

  const normalizedModelInfo = useMemo(() => {
    return {
      model_name: modelInfo?.model_name || "EfficientNet-B0",
      input_size: modelInfo?.input_size || "224x224",
      classes: modelInfo?.classes || 10,
      framework: modelInfo?.framework || "PyTorch",
      version: modelInfo?.version || "1.0.0",
      val_accuracy: 0.9853,
      macro_f1: 0.9852,
      dataset: "10 Monkey Species",
      device: modelInfo?.device || health?.device || "cpu",
    };
  }, [modelInfo, health]);

  const fetchModelInfo = async () => {
    try {
      setModelInfoError("");

      const response = await fetch(`${API_URL}/model-info`);

      if (!response.ok) {
        throw new Error("No fue posible obtener la información del modelo.");
      }

      const data = await response.json();
      setModelInfo(data);
    } catch (requestError) {
      console.error("Error cargando información del modelo:", requestError);
      setModelInfoError("No se pudo cargar la información del modelo.");
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    setPrediction(null);
    setError("");

    if (!file) {
      setSelectedFile(null);
      setPreviewUrl("");
      return;
    }

    if (!allowedTypes.includes(file.type)) {
      setSelectedFile(null);
      setPreviewUrl("");
      setError("Formato no válido. Solo se permiten imágenes JPG, JPEG o PNG.");
      return;
    }

    const maxSizeMb = 5;
    const maxSizeBytes = maxSizeMb * 1024 * 1024;

    if (file.size > maxSizeBytes) {
      setSelectedFile(null);
      setPreviewUrl("");
      setError(`El archivo supera el tamaño máximo permitido de ${maxSizeMb} MB.`);
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleSampleImageSelect = async (species) => {
  try {
    setError("");
    setPrediction(null);

    const response = await fetch(species.imageUrl);

    if (!response.ok) {
      throw new Error("No fue posible cargar la imagen de prueba.");
    }

    const blob = await response.blob();

    const file = new File([blob], `${species.label}-${species.commonName}.jpg`, {
      type: blob.type || "image/jpeg",
    });

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  } catch (requestError) {
    setError(requestError.message || "No fue posible cargar la imagen de prueba.");
  }
  };

  const checkApiStatus = async () => {
    setIsCheckingApi(true);
    setError("");

    try {
      const [healthResponse, modelInfoResponse] = await Promise.all([
        fetch(`${API_URL}/health`),
        fetch(`${API_URL}/model-info`),
      ]);

      if (!healthResponse.ok || !modelInfoResponse.ok) {
        throw new Error("No fue posible consultar el estado del backend.");
      }

      const healthData = await healthResponse.json();
      const modelInfoData = await modelInfoResponse.json();

      setHealth(healthData);
      setModelInfo(modelInfoData);
    } catch (requestError) {
      setError(
        "No se pudo conectar con el backend. Verifica que FastAPI esté corriendo o que la URL pública esté disponible."
      );
      setHealth(null);
      setModelInfo(null);
    } finally {
      setIsCheckingApi(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Primero selecciona una imagen.");
      return;
    }

    setIsPredicting(true);
    setPrediction(null);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/predict?top_k=3`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Error realizando la predicción.");
      }

      setPrediction(data);
    } catch (requestError) {
      setError(requestError.message || "No fue posible procesar la imagen.");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl("");
    setPrediction(null);
    setError("");
  };

  return (
    <main className="app-shell">
      <section className="hero-section">
        <div className="hero-badge">
          <Brain size={18} />
          Clasificación inteligente de imágenes
        </div>

        <h1>Clasificador de especies de monos</h1>

        <p>
          Sube una imagen y el modelo EfficientNet-B0 entrenado clasificará la
          especie más probable, mostrando la confianza y las tres principales
          predicciones.
        </p>

        <div className="hero-actions">
          <button
            className="secondary-button"
            type="button"
            onClick={checkApiStatus}
            disabled={isCheckingApi}
          >
            {isCheckingApi ? (
              <>
                <Loader2 className="spin" size={18} />
                Verificando API
              </>
            ) : (
              <>
                <Server size={18} />
                Verificar backend
              </>
            )}
          </button>
        </div>
      </section>

      <section className="status-grid">
        <article className="status-card">
          <span>API</span>
          <strong>{health?.status === "ok" ? "Activa" : "Sin verificar"}</strong>
          <small>
            {health?.device
              ? `Dispositivo: ${health.device}`
              : "Ejecuta la verificación"}
          </small>
        </article>

        <article className="status-card">
          <span>Modelo</span>
          <strong>{normalizedModelInfo.model_name}</strong>
          <small>Entrada: {normalizedModelInfo.input_size}</small>
        </article>

        <article className="status-card">
          <span>Clases</span>
          <strong>{normalizedModelInfo.classes}</strong>
          <small>Especies disponibles</small>
        </article>
      </section>

      <section className="model-info-section">
        <div className="section-header">
          <div>
            <h2>Información del modelo</h2>
            <p>
              Resumen técnico del modelo final utilizado para la inferencia en
              producción.
            </p>
          </div>
          <BarChart3 size={24} />
        </div>

        {modelInfoError ? (
          <div className="info-error-card">{modelInfoError}</div>
        ) : (
          <div className="model-info-grid">
            <div className="model-info-card">
              <span className="info-label">Modelo</span>
              <strong>{normalizedModelInfo.model_name}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Input</span>
              <strong>{normalizedModelInfo.input_size}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Clases</span>
              <strong>{normalizedModelInfo.classes}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Framework</span>
              <strong>{normalizedModelInfo.framework}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Versión</span>
              <strong>{normalizedModelInfo.version}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Val Accuracy</span>
              <strong>{formatPercent(normalizedModelInfo.val_accuracy)}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Macro F1</span>
              <strong>{normalizedModelInfo.macro_f1.toFixed(4)}</strong>
            </div>

            <div className="model-info-card">
              <span className="info-label">Dataset</span>
              <strong>{normalizedModelInfo.dataset}</strong>
            </div>
          </div>
        )}
      </section>

      <section className="workspace-grid">
        <article className="upload-card">
          <div className="card-header">
            <div>
              <h2>Cargar imagen</h2>
              <p>Formatos permitidos: JPG, JPEG o PNG. Tamaño máximo: 5 MB.</p>
            </div>
            <Upload size={24} />
          </div>

          <label className="dropzone">
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleFileChange}
            />

            {previewUrl ? (
              <img src={previewUrl} alt="Vista previa" className="preview-image" />
            ) : (
              <div className="dropzone-placeholder">
                <ImageIcon size={42} />
                <strong>Selecciona una imagen</strong>
                <span>Haz clic aquí para cargar una imagen de mono.</span>
              </div>
            )}
          </label>

          {selectedFile && (
            <div className="file-info">
              <span>{selectedFile.name}</span>
              <small>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</small>
            </div>
          )}

          <div className="sample-images-panel">
            <div className="sample-images-header">
              <strong>Imágenes de prueba</strong>
              <span>Selecciona una muestra precargada</span>
            </div>

            <div className="sample-images-grid">
              {speciesAtlas.slice(0, 5).map((species) => (
                <button
                    type="button"
                    className="sample-image-button"
                    key={species.label}
                    onClick={() => handleSampleImageSelect(species)}
                    title={`${species.commonName} - ${species.scientificName}`}
                  >
                    <img src={species.imageUrl} alt={species.commonName} />
                </button>
              ))}
            </div>
          </div>

          <div className="button-row">
            <button
              className="primary-button"
              type="button"
              onClick={handlePredict}
              disabled={!canPredict}
            >
              {isPredicting ? (
                <>
                  <Loader2 className="spin" size={18} />
                  Clasificando
                </>
              ) : (
                <>
                  <Brain size={18} />
                  Clasificar imagen
                </>
              )}
            </button>

            <button className="ghost-button" type="button" onClick={handleReset}>
              Limpiar
            </button>
          </div>

          {error && (
            <div className="alert error-alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </article>

        <article className="result-card">
          <div className="card-header">
            <div>
              <h2>Resultado de predicción</h2>
              <p>Salida generada directamente por el backend FastAPI.</p>
            </div>
            <CheckCircle2 size={24} />
          </div>

          {!prediction ? (
              <div className="empty-result">
                <div className="empty-result-icon">
                  <Brain size={34} />
                </div>

                <div className="empty-result-content">
                  <h3>Sin predicción todavía</h3>
                  <p>
                    Selecciona una imagen desde tu equipo o usa una muestra precargada para
                    ejecutar la clasificación.
                  </p>
                </div>

                <div className="empty-result-preview">
                  <div>
                    <strong>Resultado esperado</strong>
                    <span>Especie predicha</span>
                  </div>
                  <div>
                    <strong>Confianza</strong>
                    <span>Porcentaje del modelo</span>
                  </div>
                  <div>
                    <strong>Top 3</strong>
                    <span>Predicciones principales</span>
                  </div>
                </div>
              </div>
            ) : (
            <div className="prediction-content">
              <div className="main-prediction">
                <span>Especie predicha</span>
                <h3>{prediction.common_name}</h3>
                <p>{formatScientificName(prediction.scientific_name)}</p>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>

                <strong>{formatPercent(prediction.confidence)}</strong>
              </div>

              <div className="top-predictions">
                <h3>Top 3 predicciones</h3>

                {prediction.top_predictions.map((item, index) => (
                  <div className="prediction-row" key={`${item.label}-${index}`}>
                    <div>
                      <strong>
                        {index + 1}. {item.common_name}
                      </strong>
                      <span>{formatScientificName(item.scientific_name)}</span>
                    </div>

                    <span className="prediction-score">
                      {formatPercent(item.confidence)}
                    </span>
                  </div>
                ))}
              </div>

              <details className="json-details">
                <summary>Ver respuesta JSON</summary>
                <pre>{JSON.stringify(prediction, null, 2)}</pre>
              </details>
            </div>
          )}
        </article>
      </section>

      <section className="species-atlas-section">
        <div className="section-header">
          <div>
            <h2>Atlas de especies</h2>
            <p>
              Catálogo de las 10 clases disponibles en el dataset usado para el
              entrenamiento y la inferencia.
            </p>
          </div>
          <Database size={24} />
        </div>

        <div className="species-grid">
          {speciesAtlas.map((species) => (
            <article className="species-card" key={species.label}>
              <div className="species-card-image">
                <img src={species.imageUrl} alt={species.commonName} />
              </div>

              <div className="species-card-content">
                <h3>{species.commonName}</h3>
                <p>{species.scientificName}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

export default App;