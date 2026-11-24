"""
FastAPI prediction service with Pydantic validation and JSON logging.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------
class JSONLogFormatter(logging.Formatter):
    """Custom log formatter outputting single-line JSON logs to stdout."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_record.update(record.extra_data)
        return json.dumps(log_record)


logger = logging.getLogger("mp138_api")
logger.setLevel(logging.INFO)
# Avoid duplicated handlers if reloaded
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Pydantic Schemas with Client Validation Rules
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """
    Request schema for cash demand prediction.
    Client Rules:
    1. nearby_shops > 250 is impossible -> reject with 422 (le=250).
    2. distance_to_market_km < 0 is always a typing mistake -> reject with 422 (ge=0.0).
    """
    nearby_shops: int = Field(
        ...,
        le=250,
        description="Number of shops within 500 m (max 250)"
    )
    days_to_salary: int = Field(
        ...,
        description="Days to the 1st of the month"
    )
    footfall_index: float = Field(
        ...,
        description="Street footfall index"
    )
    distance_to_market_km: float = Field(
        ...,
        ge=0.0,
        description="Distance to the main market in km (non-negative)"
    )
    avg_ticket_rs: float = Field(
        ...,
        description="Average withdrawal last week in rupees"
    )
    other_atms_1km: int = Field(
        ...,
        description="Count of other ATMs within 1 km"
    )
    atm_type: str = Field(
        ...,
        description="ATM placement category: bank_branch, mall, standalone"
    )
    area: str = Field(
        ...,
        description="Area classification: Market, Residential, Highway"
    )


class PredictResponse(BaseModel):
    prediction: float
    withdrawal_thousand_rs: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Application Initialization & Lazy Model Loading
# ---------------------------------------------------------------------------
model = None


def load_model(model_path: str = "models/model.joblib"):
    global model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    else:
        logger.warning(f"Model file not found at {model_path}")
    return model


def get_model():
    global model
    if model is None:
        model_path = os.getenv("MODEL_PATH", "models/model.joblib")
        load_model(model_path)
    return model


app = FastAPI(
    title="Patna Cooperative Bank - ATM Cash Demand API",
    version="1.0",
    description="Microservice predicting next-day ATM cash withdrawal in thousand rupees."
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint for liveness and readiness probes."""
    m = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=m is not None,
        version="1.0"
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predicts ATM cash withdrawal demand.
    Validates rules via Pydantic and logs prediction in structured JSON format.
    """
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    input_dict = request.model_dump()
    input_df = pd.DataFrame([input_dict])

    pred = m.predict(input_df)[0]
    pred_val = float(round(pred, 2))

    # Log structured prediction event
    log_extra = {
        "event": "prediction",
        "atm_type": request.atm_type,
        "area": request.area,
        "nearby_shops": request.nearby_shops,
        "prediction": pred_val,
    }
    logger.info("Cash demand predicted successfully", extra={"extra_data": log_extra})

    return PredictResponse(
        prediction=pred_val,
        withdrawal_thousand_rs=pred_val
    )
