
import pandas as pd
import requests

# Load NASA FD001 dataset
df = pd.read_csv(
    "data/train_FD001.txt",
    sep=r"\s+",
    header=None
)

# Assign column names
df.columns = (
    ["engine_id", "cycle",
     "operational_setting_1",
     "operational_setting_2",
     "operational_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Select Engine 1 and its first 50 readings
engine = df[df["engine_id"] == 1].sort_values("cycle")
sample = engine.head(50)

# Prepare readings in the exact model feature order
feature_columns = [
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

readings = sample[feature_columns].to_dict(orient="records")

# Prepare API request
payload = {
    "engine_id": 1,
    "readings": [
        {"values": reading}
        for reading in readings
    ]
}

# Send request to FastAPI
response = requests.post(
    "http://127.0.0.1:8000/predict/rul",
    json=payload
)

print("HTTP Status:", response.status_code)
print("API Response:", response.json())

# Calculate actual RUL
actual_rul = engine["cycle"].max() - sample["cycle"].iloc[-1]

print("Actual RUL:", actual_rul)




# Prepare anomaly detection request
latest_reading = readings[-1]

anomaly_payload = {
    "engine_id": 1,
    "reading": {
        "values": latest_reading
    }
}

# Send request to anomaly detection API
anomaly_response = requests.post(
    "http://127.0.0.1:8000/detect/anomaly",
    json=anomaly_payload
)

print("\n===== ANOMALY API TEST =====")
print("HTTP Status:", anomaly_response.status_code)
print("API Response:", anomaly_response.json())
