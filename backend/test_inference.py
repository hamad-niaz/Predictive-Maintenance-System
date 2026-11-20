
import pandas as pd

from backend.model_loader import (
    predict_rul,
    detect_anomaly,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH
)

# Load NASA FD001 training dataset
df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

# Assign original NASA column names
df.columns = (
    ["engine_id", "cycle",
     "operational_setting_1",
     "operational_setting_2",
     "operational_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Select Engine 1
engine = df[df["engine_id"] == 1].sort_values("cycle")

# Take its first 50 consecutive readings
sample = engine.head(SEQUENCE_LENGTH)

if len(sample) != SEQUENCE_LENGTH:
    raise ValueError("Not enough readings for prediction.")

# Prepare model input
readings = sample[FEATURE_COLUMNS].to_dict(orient="records")

# Predict RUL
predicted_rul = predict_rul(readings)

# Calculate known RUL for comparison
actual_rul = (
    engine["cycle"].max() - sample["cycle"].iloc[-1]
)

# Detect anomaly in the latest reading
anomaly_result = detect_anomaly(readings[-1])

print("\n===== RUL PREDICTION TEST =====")
print("Engine ID:", 1)
print("Readings used:", len(readings))
print("Last cycle:", sample["cycle"].iloc[-1])
print("Actual RUL:", actual_rul)
print("Predicted RUL:", round(predicted_rul, 2))

print("\n===== ANOMALY DETECTION TEST =====")
print(anomaly_result)
