from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func

from backend.database import Base


class EngineReading(Base):
    __tablename__ = "engine_readings"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer, nullable=False, index=True)
    cycle = Column(Integer, nullable=False)

    sensor_data = Column(JSON, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer, nullable=False, index=True)

    predicted_rul = Column(Float, nullable=True)
    anomaly_status = Column(String(20), nullable=False)
    anomaly_score = Column(Float, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )