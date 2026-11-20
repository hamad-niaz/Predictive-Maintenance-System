
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict

from backend.database import get_db
from backend import models

from backend.model_loader import (
    predict_rul,
    detect_anomaly,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH
)


app = FastAPI(
    title="Predictive Maintenance API",
    description="NASA C-MAPSS FD001 based machine health prediction system",
    version="1.0.0"
)


class Reading(BaseModel):
    values: Dict[str, float]


class RULRequest(BaseModel):
    engine_id: int
    readings: List[Reading]


class AnomalyRequest(BaseModel):
    engine_id: int
    reading: Reading


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rul_model": "loaded",
        "anomaly_model": "loaded"
    }


@app.post("/predict/rul")
def predict_engine_rul(
    request: RULRequest,
    db: Session = Depends(get_db)
):

    if len(request.readings) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly {SEQUENCE_LENGTH} readings are required."
        )

    readings = [item.values for item in request.readings]

    for index, reading in enumerate(readings):
        missing = set(FEATURE_COLUMNS) - set(reading.keys())

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Reading {index + 1} is missing features: {sorted(missing)}"
            )

    try:
        # 1. Predict Remaining Useful Life
        rul = predict_rul(readings)

        # 2. Detect anomaly in the latest reading
        anomaly = detect_anomaly(readings[-1])

        # 3. Save all 50 sensor readings
        for reading in readings:
            db.add(
                models.EngineReading(
                    engine_id=request.engine_id,
                    cycle=int(reading["cycle"]),
                    sensor_data=reading
                )
            )

        # 4. Save RUL prediction and anomaly result
        db.add(
            models.PredictionLog(
                engine_id=request.engine_id,
                predicted_rul=round(rul, 2),
                anomaly_status=anomaly["status"],
                anomaly_score=anomaly["anomaly_score"]
            )
        )

        # 5. Commit records
        db.commit()

        return {
            "engine_id": request.engine_id,
            "predicted_rul_cycles": round(rul, 2),
            "unit": "cycles",
            "anomaly_status": anomaly["status"],
            "anomaly_score": anomaly["anomaly_score"],
            "database_saved": True
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post("/detect/anomaly")
def detect_engine_anomaly(
    request: AnomalyRequest,
    db: Session = Depends(get_db)
):

    reading = request.reading.values

    missing = set(FEATURE_COLUMNS) - set(reading.keys())

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing)}"
        )

    try:
        # 1. Detect anomaly
        result = detect_anomaly(reading)

        # 2. Save anomaly result in PostgreSQL
        db.add(
            models.PredictionLog(
                engine_id=request.engine_id,
                predicted_rul=None,
                anomaly_status=result["status"],
                anomaly_score=result["anomaly_score"]
            )
        )

        # 3. Commit database record
        db.commit()

        return {
            "engine_id": request.engine_id,
            **result,
            "database_saved": True
        }

    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )