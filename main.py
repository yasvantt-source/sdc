from datetime import datetime, timedelta

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="WattWise API")

model = joblib.load("model.joblib")

# Default threshold in kW; replace with the 90th percentile
# calculated from your training dataset.
DEFAULT_THRESHOLD_KW = 2.5


class ForecastRequest(BaseModel):
    last_hour_timestamp: str
    previous_24_hours_kw: list[float]
    threshold_kw: float = Field(
        default=DEFAULT_THRESHOLD_KW,
        gt=0
    )


@app.get("/")
def home():
    return {
        "message": "Welcome to WattWise API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(request: ForecastRequest):

    values = request.previous_24_hours_kw
    threshold = request.threshold_kw

    # Validate the input readings
    if (
        len(values) != 24
        or not np.isfinite(values).all()
        or any(v < 0 for v in values)
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 24 valid, non-negative hourly readings."
        )

    # Validate the timestamp
    try:
        last_hour = datetime.strptime(
            request.last_hour_timestamp,
            "%Y-%m-%d %H:%M"
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Timestamp must look like 2007-01-01 23:00."
        )

    if last_hour.hour != 23 or last_hour.minute != 0:
        raise HTTPException(
            status_code=400,
            detail="The last input reading must be at 23:00."
        )

    # Generate the next 24-hour forecast
    predictions = model.predict(
        np.array(values).reshape(1, 24)
    )[0]

    predictions = np.maximum(predictions, 0)

    if len(predictions) != 24 or not np.isfinite(predictions).all():
        raise HTTPException(
            status_code=500,
            detail="The model did not return 24 valid predictions."
        )

    # Identify the highest predicted consumption
    peak_index = int(np.argmax(predictions))

    peak_load = float(predictions[peak_index])

    peak_time = (
        last_hour + timedelta(hours=peak_index + 1)
    ).strftime("%Y-%m-%d %H:%M")

    # Classify the peak-load risk
    if peak_load >= threshold:
        risk_level = "HIGH"

    elif peak_load >= threshold * 0.8:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # Prepare all 24 hourly predictions
    forecast = [
        {
            "timestamp": (
                last_hour + timedelta(hours=i + 1)
            ).strftime("%Y-%m-%d %H:%M"),
            "predicted_power_kw": float(predictions[i])
        }
        for i in range(24)
    ]

    return {
        "forecast": forecast,
        "peak_load": {
            "predicted_power_kw": round(peak_load, 3),
            "peak_timestamp": peak_time
        },
        "risk_assessment": {
            "risk_level": risk_level,
            "threshold_kw": threshold,
            "medium_risk_threshold_kw": round(threshold * 0.8, 3)
        }
    }