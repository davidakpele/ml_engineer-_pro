from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time
from src.models.predict import ModelPredictor
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

from src.models.predict import ModelPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions', ['model', 'status'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction latency')

app = FastAPI(
    title="ML Engineer Portfolio API",
    description="Complete ML Pipeline API",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "best"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    status: str

predictor = ModelPredictor()

@app.on_event("startup")
async def startup_event():
    logger.info("ML API starting up...")
    logger.info(f"Loaded models: {predictor.get_available_models()}")

@app.get("/")
async def root():
    return {
        "message": "ML Engineer Portfolio API",
        "status": "healthy",
        "available_models": predictor.get_available_models(),
        "expected_features": predictor.get_expected_features()
    }
@app.get("/models")
async def list_models():
    """List available model versions"""
    return {"available_models": predictor.get_available_models()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using the ML model"""
    start_time = time.time()
    try:
        PREDICTION_COUNTER.labels(model=request.model_version, status='requested').inc()
        result = predictor.predict(request.features, request.model_version)
        
        PREDICTION_COUNTER.labels(model=result['model_used'], status='success').inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            model_version=result['model_used'],
            status="success"
        )
    
    except Exception as e:
        PREDICTION_COUNTER.labels(model=request.model_version, status='error').inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_expected_features():
    """Get the number of features expected by models"""
    return {
        "expected_features": predictor.get_expected_features(),
        "message": f"Send exactly {predictor.get_expected_features()} features in the 'features' array for predictions"
    }

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(REGISTRY)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = len(predictor.get_available_models()) > 0
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "model_count": len(predictor.get_available_models())
    }

@app.get("/info")
async def api_info():
    """API information"""
    return {
        "api_name": "ML Engineer Portfolio",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root endpoint"},
            {"path": "/predict", "method": "POST", "description": "Make predictions"},
            {"path": "/models", "method": "GET", "description": "List available models"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/metrics", "method": "GET", "description": "Prometheus metrics"}
        ]
    }
