
from pathlib import Path

import joblib
import torch
import torch.nn as nn


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Device
DEVICE = torch.device("cpu")


# LSTM architecture (same as training)
class RUL_LSTM(nn.Module):
    def __init__(
        self,
        input_size=19,
        hidden_size=128,
        num_layers=1,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        x = hidden[-1]
        x = self.dropout(x)

        return self.fc(x).squeeze(-1)


# Load saved LSTM checkpoint
checkpoint_path = MODEL_DIR / "final_lstm_model.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)


# Recreate model using saved configuration
rul_model = RUL_LSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
)

rul_model.load_state_dict(checkpoint["model_state_dict"])
rul_model.to(DEVICE)
rul_model.eval()


# Load feature scaler
scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")


# Load Isolation Forest
anomaly_model = joblib.load(MODEL_DIR / "isolation_forest.pkl")


# Model metadata
FEATURE_COLUMNS = checkpoint["feature_columns"]
SEQUENCE_LENGTH = checkpoint["sequence_length"]


print("Models loaded successfully!")
print("Device:", DEVICE)
print("Input features:", len(FEATURE_COLUMNS))
print("Sequence length:", SEQUENCE_LENGTH)
print("LSTM model: Ready")
print("Isolation Forest: Ready")



def predict_rul(readings):
    """
    Predict Remaining Useful Life using the last 50 readings.
    """

    if len(readings) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Exactly {SEQUENCE_LENGTH} readings are required."
        )

    # Convert readings into DataFrame
    import pandas as pd

    input_data = pd.DataFrame(readings)

    # Ensure correct feature order
    input_data = input_data[FEATURE_COLUMNS]

    # Scale features using training scaler
    scaled_data = scaler.transform(input_data)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # Generate prediction
    with torch.no_grad():
        prediction = rul_model(input_tensor)

    return max(0.0, float(prediction.item()))





def detect_anomaly(reading):
    """
    Detect unusual sensor readings using Isolation Forest.
    """

    import pandas as pd

    # Keep original feature order
    input_data = pd.DataFrame([reading])
    input_data = input_data[FEATURE_COLUMNS]

    # Isolation Forest was trained on raw, unscaled features
    prediction = anomaly_model.predict(input_data)[0]

    anomaly_score = anomaly_model.decision_function(input_data)[0]

    return {
        "status": "anomaly" if prediction == -1 else "normal",
        "anomaly_score": float(anomaly_score)
    }
    