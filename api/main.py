from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import CustomerFeatures, PredictionResponse

MODEL_PATH = Path(__file__).parent.parent / "model"
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python src/train.py` first.")
    _pipeline = mlflow.sklearn.load_model(str(MODEL_PATH))
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability from account features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([features.model_dump()])
    proba = float(_pipeline.predict_proba(df)[0][1])
    prediction = proba >= 0.5
    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk,
    )
